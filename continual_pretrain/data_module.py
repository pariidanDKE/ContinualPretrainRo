from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

class DataPreprocessor:
    # ---- Special token maps ----
    LLAMA_SPECIAL_MAP = {
        "<sistem>": "<|reserved_special_token_0|>",
        "</sistem>": "<|reserved_special_token_1|>",
        "<utilizator>": "<|reserved_special_token_2|>",
        "</utilizator>": "<|reserved_special_token_3|>",
        "<asistent>": "<|reserved_special_token_4|>",
        "</asistent>": "<|reserved_special_token_5|>",
    }

    GEMMA_SPECIAL_MAP = {
        "<sistem>": "<unused0>",
        "</sistem>": "<unused1>",
        "<utilizator>": "<unused2>",
        "</utilizator>": "<unused3>",
        "<asistent>": "<unused4>",
        "</asistent>": "<unused5>",
    }

    SPECIAL_MAPS = {
        "llama": LLAMA_SPECIAL_MAP,
        "gemma": GEMMA_SPECIAL_MAP,
    }

    def __init__(self, tokenizer, text_field: str = "formatted_text", add_bos_eos: bool = True):
        """
        tokenizer: HF tokenizer
        text_field: name of the field in the dataset that contains the formatted dialogue text
        add_bos_eos: whether to wrap the text with bos/eos tokens
        """
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.add_bos_eos = add_bos_eos

        self.family = self._detect_model_family()
        print(f"[DataPreprocessor] Detected model family: {self.family}")
        self.special_map = self.SPECIAL_MAPS.get(self.family, None)

    # ---- Model family detection ----
    def _detect_model_family(self):
        vocab = self.tokenizer.get_vocab()

        # LLaMA-style special reserved tokens
        if "<|reserved_special_token_0|>" in vocab:
            return "llama"

        # Gemma-style unused tokens
        if "<unused0>" in vocab:
            return "gemma"

        return None  # fallback if neither is detected

    # ---- Generic tag replacer ----
    @staticmethod
    def _replace_tags(text, tag_map):
        for tag, tok in tag_map.items():
            text = text.replace(tag, tok)
        return text

    # ---- Unified loss mask (system + user masked, assistant unmasked) ----
    def _apply_loss_mask(self, tokens, token_ids):
        """Return label tensor with system/user/pad masked (-100) and assistant unmasked."""
        special_map = self.special_map
        tokenizer = self.tokenizer

        SYS_OPEN = special_map["<sistem>"]
        SYS_CLOSE = special_map["</sistem>"]
        USER_OPEN = special_map["<utilizator>"]
        USER_CLOSE = special_map["</utilizator>"]

        BOS = tokenizer.bos_token or None
        EOS = tokenizer.eos_token or None

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

        labels = []
        in_system = False
        in_user = False

        for tok, tid in zip(tokens, token_ids):
            # ----- system section -----
            if tok == SYS_OPEN:
                in_system = True
                labels.append(-100)
                continue
            if tok == SYS_CLOSE:
                labels.append(-100)
                in_system = False
                continue

            # ----- user section -----
            if tok == USER_OPEN:
                in_user = True
                labels.append(-100)
                continue
            if tok == USER_CLOSE:
                labels.append(-100)
                in_user = False
                continue

            # Inside system/user → masked
            if in_system or in_user:
                labels.append(-100)
                continue

            # Mask BOS/EOS
            if tok == BOS or tok == EOS:
                labels.append(-100)
                continue

            # Padding should be masked
            if tid == pad_id:
                labels.append(-100)
                continue

            # Otherwise assistant text → unmasked (model predicts these)
            labels.append(tid)

        return labels

    # ---- Main preprocessing (callable for ds.map) ----
    def __call__(self, example):
        """
        Example is a single row from the dataset.
        Expects example[self.text_field] to contain the formatted conversation text.
        """
        original = example[self.text_field]

        # Replace <sistem>/<utilizator>/<asistent> tags with model-specific tokens
        if self.special_map is not None:
            original_local = self._replace_tags(original, self.special_map)
        else:
            # If unknown family, just use raw text
            original_local = original

        # Add BOS/EOS
        if self.add_bos_eos:
            bos = self.tokenizer.bos_token or ""
            eos = self.tokenizer.eos_token or ""
            text = bos + original_local + eos
        else:
            text = original_local

        # Tokenize
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        ids = encoded["input_ids"]

        # Convert ids → tokens for boundary detection
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        # Build labels using unified masking logic
        if self.special_map is not None:
            labels = self._apply_loss_mask(tokens, ids)
        else:
            # Fallback: no masking, train on everything except pad
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -1
            labels = [tid if tid != pad_id else -100 for tid in ids]

        # loss_mask: 1 where we compute loss, 0 where ignored (optional)
        loss_mask = [0 if l == -100 else 1 for l in labels]

        return {
            "input_ids": ids,
            "labels": labels,
            # uncomment if you want these:
            # "loss_mask": loss_mask,
            "attention_mask": encoded["attention_mask"],
        }

################## DATA COLLATOR ###########################

def get_attention_mask_for_packed_sequence(x, token_id, eos: bool = True):
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat([eos_idx, torch.arange(0,B*T+1,T)]).unique().sort()[0]
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    repeated_idx = torch.repeat_interleave(normalized_idx[1:], reps).view(B,1,T).expand(-1,T,-1)
    mask_indices = torch.arange(T).view(1,-1,1).expand(B, -1, T)
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, -1, -1) # SWITCH THIS BACKKK
    mask = mask.masked_fill(mask_indices >= repeated_idx, False)
    #mask = (~mask).float() * -1e9
    mask = mask.unsqueeze(1)


    # get position ids for packed sequence
    pos_ids = (torch.arange(B*T) - torch.repeat_interleave(eos_idx_expanded[:-1], reps)).view(B,T)
    return mask, pos_ids


@dataclass
class PackedSequenceDataCollator:
    tokenizer: any
    pack_length: int
    eos_token_id: int

    def __call__(self, features):
        sequences = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels    = [torch.tensor(f["labels"],    dtype=torch.long) for f in features]

        packed_rows_x = []
        packed_rows_y = []

        cur_tokens = []
        cur_labels = []
        cur_len = 0

        for seq, lab in zip(sequences, labels):
            seq_len = len(seq)

            # TRUNCATION RULE
            if seq_len > self.pack_length:
                seq = seq[:self.pack_length]
                lab = lab[:self.pack_length]
                seq[-1] = self.eos_token_id
                lab[-1] = self.eos_token_id
                seq_len = self.pack_length

            if cur_len + seq_len > self.pack_length:
                # flush with exact pack_length padding
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

        # flush last row (also exact pack_length)
        if cur_tokens:
            pad_needed = self.pack_length - len(cur_tokens)
            if pad_needed > 0:
                cur_tokens += [self.eos_token_id] * pad_needed
                cur_labels += [self.eos_token_id] * pad_needed

            packed_rows_x.append(torch.tensor(cur_tokens))
            packed_rows_y.append(torch.tensor(cur_labels))

        padded_x = torch.stack(packed_rows_x)
        padded_labels = torch.stack(packed_rows_y)

        mask4, pos_ids = get_attention_mask_for_packed_sequence(
            padded_x, self.eos_token_id
        )

        #print("Created mask with shape:", mask4.shape)
        return {
            "input_ids": padded_x,
            "labels": padded_labels,
            "attention_mask": mask4,
            "position_ids": pos_ids,
        }


class SimplePaddingCollator:
    def __init__(self, tokenizer, label_pad_token_id: int = -100, max_length: int = 512):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.max_length = max_length

    def _truncate(self, seq):
        if len(seq) <= self.max_length:
            return seq
        return seq[: self.max_length]

    def __call__(self, features):
        pad_inputs = []
        for f in features:
            input_ids = self._truncate(f["input_ids"])
            attention_mask = f.get("attention_mask")
            if attention_mask is not None:
                attention_mask = self._truncate(attention_mask)

            pad_inputs.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

        batch = self.tokenizer.pad(
            pad_inputs,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        padded_labels = []
        for f in features:
            labels = self._truncate(f["labels"])
            remainder = self.max_length - len(labels)
            if remainder > 0:
                labels = labels + [self.label_pad_token_id] * remainder
            padded_labels.append(labels)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch