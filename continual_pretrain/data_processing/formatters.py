"""
Formatting functions for different dataset structures.
Each formatter takes the data from specified columns and formats it with special tokens.
"""


def format_messages_standard(messages, tokenizer=None, user_token="<utilizator>", 
                             assistant_token="<asistent>", system_token="<sistem>"):
    """
    Standard formatting for datasets with 'messages' column.
    Used for: OpenLLM-Ro/ro_sft_norobots
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: Optional HuggingFace tokenizer
        user_token: Token for user messages
        assistant_token: Token for assistant messages
        system_token: Token for system messages
    
    Returns:
        Formatted string with special tokens
    """
    if tokenizer is not None:
        bos_token = tokenizer.bos_token or "<s>"
        eos_token = tokenizer.eos_token or "</s>"
    else:
        bos_token = ""
        eos_token = ""
    
    role_token_map = {
        'system': system_token,
        'user': user_token,
        'assistant': assistant_token
    }
    
    formatted_text = bos_token + "\n"
    
    for message in messages:
        role = message['role']
        content = message['content'].strip()
        
        role_token = role_token_map.get(role, f"<{role}>")
        formatted_text += f"{role_token}\n{content}\n"
    
    formatted_text += eos_token
    
    return formatted_text


def format_dolly_context_instruction(data_dict, tokenizer=None, user_token="<utilizator>", 
                                     assistant_token="<asistent>", system_token="<sistem>"):
    """
    Formatting for OpenLLM-Ro/ro_sft_dolly dataset.
    Combines 'context' and 'instruction' into user message, 'response' as assistant message.
    
    Args:
        data_dict: Dict with 'context', 'instruction', and 'response' keys
        tokenizer: Optional HuggingFace tokenizer
        user_token: Token for user messages
        assistant_token: Token for assistant messages
        system_token: Token for system messages (unused)
    
    Returns:
        Formatted string with special tokens
    """
    if tokenizer is not None:
        bos_token = tokenizer.bos_token or "<s>"
        eos_token = tokenizer.eos_token or "</s>"
    else:
        bos_token = ""
        eos_token = ""
    
    formatted_text = "\n"
    
    # Build user message
    user_message = ""
    
    context = data_dict.get('context', '').strip()
    instruction = data_dict.get('instruction', '').strip()
    response = data_dict.get('response', '').strip()
    
    # Format: instruction first, then context if present
    if instruction:
        user_message = instruction
    
    if context:
        if user_message:
            user_message += "\n\nContext: " + context
        else:
            user_message = context
    
    # Add user message
    if user_message:
        formatted_text += f"{user_token}\n{user_message}\n"
    
    # Add assistant response
    if response:
        formatted_text += f"{assistant_token}\n{response}\n"
    
    #formatted_text += eos_token
    return formatted_text