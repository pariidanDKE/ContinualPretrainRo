import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

# ==============================================================================
# SHARED: Tag replacement and detection
# ==============================================================================

LLAMA_SPECIAL_MAP = {
    "<sistem>": "<|reserved_special_token_0|>",
    "</sistem>": "<|reserved_special_token_1|>",
    "<utilizator>": "<|reserved_special_token_2|>",
    "</utilizator>": "<|reserved_special_token_3|>",
    "<asistent>": "<|reserved_special_token_4|>",
    "</asistent>": "<|reserved_special_token_5|>",
}

def detect_is_llama(tokenizer):
    """Check if tokenizer is Llama-based."""
    return "<|reserved_special_token_0|>" in tokenizer.get_vocab()

def replace_tags(text, tag_map):
    """Replace custom tags with reserved tokens."""
    for tag, tok in tag_map.items():
        text = text.replace(tag, tok)
    return text


# ==============================================================================
# SETUP 1: PACKED SEQUENCE (Custom Collator + Preprocessing)
# ==============================================================================

def make_preprocess_packed(tokenizer):
    """Preprocessing for packed sequences (no attention mask needed)."""
    IS_LLAMA = detect_is_llama(tokenizer)
    
    def preprocess(example):
        text = example["formatted_text"]

        if IS_LLAMA:
            text = replace_tags(text, LLAMA_SPECIAL_MAP)

        text = tokenizer.bos_token + text + tokenizer.eos_token

        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False  # No mask in preprocessing
        )["input_ids"]

        return {
            "input_ids": ids,
            "labels": ids.copy()
        }

    return preprocess


def get_attention_mask_for_packed_sequence(x, token_id, eos: bool = True):
    """Generate attention mask and position IDs for packed sequences."""
    B, T = x.shape
    
    # Find EOS token positions
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat([eos_idx, torch.arange(0, B*T+1, T)]).unique().sort()[0]
    
    # Normalize indices to per-row positions
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    
    # Calculate repetitions (sequence lengths)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    
    # Create repeated boundary indices
    repeated_idx = torch.repeat_interleave(normalized_idx[1:], reps).view(B, 1, T).expand(-1, T, -1)
    
    # Create mask indices (column positions)
    mask_indices = torch.arange(T).view(1, -1, 1).expand(B, -1, T)
    
    # Create causal mask and block cross-sequence attention
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, -1, -1)
    mask = mask.masked_fill(mask_indices >= repeated_idx, False)
    mask = mask.unsqueeze(1)  # Add head dimension [B, 1, T, T]
    
    # Generate position IDs that reset at each sequence
    pos_ids = (torch.arange(B*T) - torch.repeat_interleave(eos_idx_expanded[:-1], reps)).view(B, T)
    
    return mask, pos_ids


@dataclass
class PackedSequenceDataCollator:
    """Custom collator that packs multiple sequences per row."""
    tokenizer: any
    pack_length: int
    eos_token_id: int

    def __call__(self, features):
        sequences = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        packed_rows_x = []
        packed_rows_y = []

        cur_tokens = []
        cur_labels = []
        cur_len = 0

        for seq, lab in zip(sequences, labels):
            seq_len = len(seq)

            # Truncate if too long
            if seq_len > self.pack_length:
                seq = seq[:self.pack_length]
                lab = lab[:self.pack_length]
                seq[-1] = self.eos_token_id
                lab[-1] = self.eos_token_id
                seq_len = self.pack_length

            # Flush current packed row if adding this sequence would exceed limit
            if cur_len + seq_len > self.pack_length:
                pad_needed = self.pack_length - len(cur_tokens)
                if pad_needed > 0:
                    cur_tokens += [self.eos_token_id] * pad_needed
                    cur_labels += [self.eos_token_id] * pad_needed

                packed_rows_x.append(torch.tensor(cur_tokens))
                packed_rows_y.append(torch.tensor(cur_labels))

                cur_tokens = []
                cur_labels = []
                cur_len = 0

            cur_tokens.extend(seq.tolist())
            cur_labels.extend(lab.tolist())
            cur_len += seq_len

        # Flush last row
        if cur_tokens:
            pad_needed = self.pack_length - len(cur_tokens)
            if pad_needed > 0:
                cur_tokens += [self.eos_token_id] * pad_needed
                cur_labels += [self.eos_token_id] * pad_needed

            packed_rows_x.append(torch.tensor(cur_tokens))
            packed_rows_y.append(torch.tensor(cur_labels))

        padded_x = torch.stack(packed_rows_x)
        padded_labels = torch.stack(packed_rows_y)

        # Generate custom attention mask and position IDs
        mask, pos_ids = get_attention_mask_for_packed_sequence(
            padded_x, self.eos_token_id
        )

        return {
            "input_ids": padded_x,
            "labels": padded_labels,
            "attention_mask": mask,
            "position_ids": pos_ids,
        }


# ==============================================================================
# SETUP 2: STANDARD (Default Collator + Preprocessing)
# ==============================================================================

def make_preprocess_standard(tokenizer):
    """Standard preprocessing with attention mask."""
    IS_LLAMA = detect_is_llama(tokenizer)
    
    def preprocess(example):
        text = example["formatted_text"]

        if IS_LLAMA:
            text = replace_tags(text, LLAMA_SPECIAL_MAP)

        text = tokenizer.bos_token + text + tokenizer.eos_token

        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=True  # Standard mask
        )

        return {
            "input_ids": enc["input_ids"],
            "labels": enc["input_ids"].copy(),
            "attention_mask": enc["attention_mask"]
        }

    return preprocess


# ==============================================================================
# MAIN COMPARISON FUNCTION
# ==============================================================================

def run_comparison(
    model_name: str = "meta-llama/Llama-3.2-1B",
    data_path: str = "../data/formatted_data/norobots",
    num_samples: int = 100,
    pack_length: int = 256,
    batch_size: int = 1,
    gradient_accumulation: int = 2,
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
    use_fp16: bool = True,
    run_type="both"
):
    """
    Compare packed vs standard training setups.
    
    Args:
        model_name: HuggingFace model identifier
        data_path: Path to formatted dataset
        num_samples: Number of samples to use (for testing)
        pack_length: Maximum sequence length for packing
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        use_fp16: Use float16 for training
    """
    
    print("="*80)
    print("LOADING MODEL AND DATA")
    print("="*80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    #ds =  Dataset.load_from_disk(data_path)
    ds = load_dataset("danp27/norobots_sft").select(range(100))


    if run_type != "default":
        
        # -------------------------------------------------------------------------
        # SETUP 1: PACKED SEQUENCES
        # -------------------------------------------------------------------------
        print("\n" + "="*80)
        print("SETUP 1: PACKED SEQUENCES (Custom Collator)")
        print("="*80)
        
        # Load model for packed training
        model_packed = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        
        # Preprocess dataset
        preprocess_packed = make_preprocess_packed(tokenizer)
        tokenized_ds_packed = ds.map(
            preprocess_packed,
            remove_columns=[k for k in ds.features.keys()]
        )
        
        # Create collator
        collator_packed = PackedSequenceDataCollator(
            tokenizer=tokenizer,
            pack_length=pack_length,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Training arguments
        training_args_packed = TrainingArguments(
            output_dir="output_packed",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            save_strategy="no",
            fp16=use_fp16,
        )
        
        # Create trainer
        trainer_packed = Trainer(
            model=model_packed,
            train_dataset=tokenized_ds_packed,
            data_collator=collator_packed,
            args=training_args_packed
        )
        
        print("\nStarting packed training...")
        trainer_packed.train()
    
    if run_type !="packed":
        # -------------------------------------------------------------------------
        # SETUP 2: STANDARD (NO PACKING)
        # -------------------------------------------------------------------------
        print("\n" + "="*80)
        print("SETUP 2: STANDARD (Default Collator)")
        print("="*80)
        
        # Load fresh model for standard training
        model_standard = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        
        # Preprocess dataset
        preprocess_standard = make_preprocess_standard(tokenizer)
        tokenized_ds_standard = ds.map(
            preprocess_standard,
            remove_columns=[k for k in ds.features.keys()]
        )
        
        
        # Training arguments
        training_args_standard = TrainingArguments(
            output_dir="output_standard",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            save_strategy="no",
            fp16=use_fp16,
        )
        tokenizer.pad_token = tokenizer.eos_token
        # Create trainer
        trainer_standard = Trainer(
            model=model_standard,
            train_dataset=tokenized_ds_standard,
            args=training_args_standard
        )
        
        print("\nStarting standard training...")
        trainer_standard.train()
    
    # -------------------------------------------------------------------------
    # COMPARISON SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nPacked training completed: {training_args_packed.output_dir}")
    print(f"Standard training completed: {training_args_standard.output_dir}")
    print("\nCheck logs for performance metrics.")


if __name__ == "__main__":
    run_comparison(
        model_name="meta-llama/Llama-3.2-1B",
        data_path="../data/formatted_data/norobots",
        num_samples=100,
        pack_length=256,
        batch_size=1,
        gradient_accumulation=1,
        learning_rate=2e-5,
        num_epochs=1,
        use_fp16=False,
        run_type = "packed"
    )