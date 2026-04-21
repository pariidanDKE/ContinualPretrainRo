from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    # ---- Special token maps ----
    # Map Romanian tags to native instruction format for each model family
    # Works for BOTH base and instruct models - base models will learn, instruct models already know

    LLAMA_SPECIAL_MAP = {
        "<sistem>": "<|start_header_id|>system<|end_header_id|>\n\n",
        "</sistem>": "<|eot_id|>",
        "<utilizator>": "<|start_header_id|>user<|end_header_id|>\n\n",
        "</utilizator>": "<|eot_id|>",
        "<asistent>": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "</asistent>": "<|eot_id|>",
    }

    GEMMA_SPECIAL_MAP = {
        "<sistem>": "<start_of_turn>user\n",  # Gemma handles system prompts as user context
        "</sistem>": "\n\n",
        "<utilizator>": "<start_of_turn>user\n",
        "</utilizator>": "<end_of_turn>\n",
        "<asistent>": "<start_of_turn>model\n",
        "</asistent>": "<end_of_turn>\n",
    }

    QWEN_SPECIAL_MAP = {
        "<sistem>": "<|im_start|>system\n",
        "</sistem>": "<|im_end|>\n",
        "<utilizator>": "<|im_start|>user\n",
        "</utilizator>": "<|im_end|>\n",
        "<asistent>": "<|im_start|>assistant\n",
        "</asistent>": "<|im_end|>\n",
    }

    SPECIAL_MAPS = {
        "llama": LLAMA_SPECIAL_MAP,
        "gemma": GEMMA_SPECIAL_MAP,
        "qwen": QWEN_SPECIAL_MAP,
    }

    def __init__(self, tokenizer, text_field: str = "formatted_text", add_bos_eos: bool = True, tokenize: bool = False, use_sft: bool = True):
        """
        tokenizer: HF tokenizer
        text_field: name of the field in the dataset that contains the formatted dialogue text
        add_bos_eos: whether to wrap the text with bos/eos tokens
        tokenize: if True, tokenize here and return input_ids/labels; if False, return formatted_text for Trainer
        use_sft: if True, use SFT mode (Romanian tags → model tokens, loss masking); if False, CPT mode (raw text, no masking)
        """
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.add_bos_eos = add_bos_eos
        self.tokenize = tokenize
        self.use_sft = use_sft

        self.family = self._detect_model_family()
        # For CPT mode (use_sft=False), disable tag replacement and use fallback loss masking
        self.special_map = None if not use_sft else self.SPECIAL_MAPS.get(self.family, None)


    # ---- Model family detection ----
    def _detect_model_family(self):
        vocab = self.tokenizer.get_vocab()
        model_name = getattr(self.tokenizer, 'name_or_path', '').lower()

        # Detect Llama (both base and instruct models have these tokens)
        if "<|start_header_id|>" in vocab or "<|reserved_special_token_0|>" in vocab:
            return "llama"

        # Detect Gemma (both base and IT models have these tokens)
        if "<start_of_turn>" in vocab or "<unused0>" in vocab:
            return "gemma"

        # Detect Qwen (both base and instruct models have these tokens)
        if "<|im_start|>" in vocab or "qwen" in model_name:
            return "qwen"

        return None  # fallback if no family detected

    # ---- Chat template generation ----
    def get_chat_template(self) -> str:
        """
        Generate a Jinja2 chat template dynamically based on the detected model family.
        This template uses the special tokens mapped for Romanian dialogue tags.

        Templates are simplified versions that match native templates but without:
        - Date/time stamps (Llama)
        - Other metadata that doesn't affect Romanian evaluation

        Returns:
            A Jinja2 template string compatible with HuggingFace tokenizers
        """
        if self.special_map is None:
            raise ValueError("Cannot generate chat template: model family not detected")

        # Build family-specific templates
        if self.family == "llama":
            # Simplified Llama template with BOS but without date stamps
            template = (
                "{{ bos_token }}"  # Add BOS token at start
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
                "{% endif %}"
                "{% endfor %}"
            )

        elif self.family == "gemma":
            # Simplified Gemma template with BOS
            # Gemma merges system message into first user message if present
            template = (
                "{{ bos_token }}"
                "{% if messages[0]['role'] == 'system' %}"
                "{{ '<start_of_turn>user\\n' + messages[0]['content'] + '\\n\\n' }}"
                "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                "{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )

        elif self.family == "qwen":
            # Qwen template (already matches perfectly)
            template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )

        else:
            raise ValueError(f"Unsupported model family: {self.family}")

        return template

    # ---- Generic tag replacer ----
    @staticmethod
    def _replace_tags(text, tag_map):
        for tag, tok in tag_map.items():
            text = text.replace(tag, tok)
        return text

    # ---- Unified loss mask (system + user masked, assistant unmasked) ----
    def _apply_loss_mask(self, token_ids):
        """Return label tensor with system/user/pad masked (-100) and assistant unmasked."""
        special_map = self.special_map
        tokenizer = self.tokenizer

        # Encode the special token strings to get their token ID sequences
        # These can be multi-token sequences (e.g., "<|start_header_id|>system<|end_header_id|>\n\n" = 4 tokens)
        sys_open_ids = tokenizer.encode(special_map["<sistem>"], add_special_tokens=False)
        sys_close_ids = tokenizer.encode(special_map["</sistem>"], add_special_tokens=False)
        user_open_ids = tokenizer.encode(special_map["<utilizator>"], add_special_tokens=False)
        user_close_ids = tokenizer.encode(special_map["</utilizator>"], add_special_tokens=False)
        asst_open_ids = tokenizer.encode(special_map["<asistent>"], add_special_tokens=False)
        asst_close_ids = tokenizer.encode(special_map["</asistent>"], add_special_tokens=False)

        BOS = tokenizer.bos_token_id or None
        EOS = tokenizer.eos_token_id or None
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

        # Helper function to check if a sequence matches at position i
        def matches_sequence(tokens, seq, pos):
            if pos + len(seq) > len(tokens):
                return False
            return tokens[pos:pos+len(seq)] == seq

        labels = []
        i = 0
        in_system = False
        in_user = False
        in_assistant = False

        while i < len(token_ids):
            tid = token_ids[i]

            # Check for role transitions - opening a new role closes the previous one
            # System role
            if matches_sequence(token_ids, sys_open_ids, i):
                in_system = True
                in_user = False
                in_assistant = False
                for _ in range(len(sys_open_ids)):
                    labels.append(-100)
                i += len(sys_open_ids)
                continue

            # User role
            if matches_sequence(token_ids, user_open_ids, i):
                in_system = False
                in_user = True
                in_assistant = False
                for _ in range(len(user_open_ids)):
                    labels.append(-100)
                i += len(user_open_ids)
                continue

            # Assistant role
            if matches_sequence(token_ids, asst_open_ids, i):
                in_system = False
                in_user = False
                in_assistant = True
                for _ in range(len(asst_open_ids)):
                    labels.append(-100)
                i += len(asst_open_ids)
                continue

            # Close tags (like <|eot_id|>, <end_of_turn>, <|im_end|>) - mask them but don't change state
            # The state only changes when a NEW role opens
            if (matches_sequence(token_ids, sys_close_ids, i) or
                matches_sequence(token_ids, user_close_ids, i) or
                matches_sequence(token_ids, asst_close_ids, i)):

                # Determine length of closing sequence
                close_len = 0
                if matches_sequence(token_ids, sys_close_ids, i):
                    close_len = len(sys_close_ids)
                elif matches_sequence(token_ids, user_close_ids, i):
                    close_len = len(user_close_ids)
                elif matches_sequence(token_ids, asst_close_ids, i):
                    close_len = len(asst_close_ids)

                # Mask the closing tokens
                for _ in range(close_len):
                    labels.append(-100)
                i += close_len
                continue

            # Handle content based on current role
            if in_system or in_user:
                # Mask system and user content
                labels.append(-100)
            elif in_assistant:
                # Keep assistant content (this is what we train on)
                labels.append(tid)
            else:
                # Outside any role, mask BOS/EOS/PAD
                if tid == BOS or tid == EOS or tid == pad_id:
                    labels.append(-100)
                else:
                    # Default: mask unknown content
                    labels.append(-100)

            i += 1

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

        if not self.tokenize:
            # Return text for SFTTrainer/UnslothTrainer to handle tokenization

            
            return {
                f'{self.text_field}' : text
            }
        else: 

            # Tokenize
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=True,
            )
            ids = encoded["input_ids"]

            # Build labels using unified masking logic
            if self.special_map is not None:
                labels = self._apply_loss_mask(ids)
            else:
                # Fallback: no masking, train on everything except pad
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -1
                labels = [tid if tid != pad_id else -100 for tid in ids]

            # loss_mask: 1 where we compute loss, 0 where ignored (optional)
            loss_mask = [0 if l == -100 else 1 for l in labels]
          
            return {
                "input_ids": ids,
                "labels": labels,
            
                "attention_mask": encoded["attention_mask"],
            }
