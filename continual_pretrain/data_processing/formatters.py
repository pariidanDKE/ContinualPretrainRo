"""
Formatting functions for different dataset structures.
Each formatter takes the data from specified columns and formats it with special tokens.
"""


def format_messages_standard(messages, tokenizer=None, user_token="<utilizator>", 
                             assistant_token="<asistent>", system_token="<sistem>", logger=None):
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
        'system': (system_token, '</sistem>'),
        'user': (user_token, '</utilizator>'),
        'assistant': (assistant_token, '</asistent>')
    }
    
    formatted_text = ""
    
    for message in messages:
        role = message['role']
        content = message['content'].strip()
        
        open_tag, close_tag = role_token_map.get(role, (f"<{role}>", f"</{role}>"))
        formatted_text += f"{open_tag}\n{content}\n{close_tag}\n"
    
    return formatted_text


def format_dolly_context_instruction(data_dict, tokenizer=None, user_token="<utilizator>", 
                                     assistant_token="<asistent>", system_token="<sistem>", logger = None):
    if tokenizer is not None:
        bos_token = tokenizer.bos_token or "<s>"
        eos_token = tokenizer.eos_token or "</s>"
    else:
        bos_token = ""
        eos_token = ""
    
    formatted_text = ""
    
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
        formatted_text += f"{user_token}\n{user_message}\n</utilizator>\n"
    
    # Add assistant response
    if response:
        formatted_text += f"{assistant_token}\n{response}\n</asistent>\n"
    
    return formatted_text


def format_camel_instructs(data_dict, tokenizer=None, user_token="<utilizator>", 
                           assistant_token="<asistent>", system_token="<sistem>", logger=None):
    if tokenizer is not None:
        bos_token = tokenizer.bos_token or ""
        eos_token = tokenizer.eos_token or ""
    else:
        bos_token = ""
        eos_token = ""
    
    formatted_text = ""
    
    # Get messages
    message_1 = data_dict.get('message_1', '').strip()
    message_2 = data_dict.get('message_2', '').strip()
    
    # Add user message (message_1)
    if message_1:
        formatted_text += f"{user_token}\n{message_1}\n</utilizator>\n"
    
    # Add assistant message (message_2)
    if message_2:
        formatted_text += f"{assistant_token}\n{message_2}\n</asistent>\n"
    
    return formatted_text

    
def format_orca_messages(messages, tokenizer=None, user_token="<utilizator>", 
                        assistant_token="<asistent>", system_token="<sistem>", logger=None):
    if tokenizer is not None:
        bos_token = tokenizer.bos_token or ""
        eos_token = tokenizer.eos_token or ""
    else:
        bos_token = ""
        eos_token = ""
    
    # Map "from" values to role tokens with closing tags
    role_token_map = {
        'system': (system_token, '</sistem>'),
        'human': (user_token, '</utilizator>'),
        'gpt': (assistant_token, '</asistent>')
    }
    
    formatted_text = ""
    
    for message in messages:
        from_role = message.get('from', '').strip()
        content = message.get('value', '').strip()
        
        # Get the appropriate role tokens
        open_tag, close_tag = role_token_map.get(from_role, (f"<{from_role}>", f"</{from_role}>"))
        
        if content:
            formatted_text += f"{open_tag}\n{content}\n{close_tag}\n"
    
    return formatted_text

def format_oasst_messages(messages, tokenizer=None, user_token="<utilizator>", 
                         assistant_token="<asistent>", system_token="<sistem>", logger=None):

    if tokenizer is not None:
        bos_token = tokenizer.bos_token or ""
        eos_token = tokenizer.eos_token or ""
    else:
        bos_token = ""
        eos_token = ""
    
    # Map OASST role names to our tokens with closing tags
    role_token_map = {
        'prompter': (user_token, '</utilizator>'),
        'assistant': (assistant_token, '</asistent>'),
        'system': (system_token, '</sistem>')
    }
    
    formatted_text = ""
    
    for message in messages:
        role = message.get('role', '').strip()
        content = message.get('content', '').strip()
        
        # Get the appropriate role tokens
        open_tag, close_tag = role_token_map.get(role, (f"<{role}>", f"</{role}>"))
        
        if content:
            formatted_text += f"{open_tag}\n{content}\n{close_tag}\n"
    
    return formatted_text

def format_magpie_conversations(conversations, tokenizer=None, user_token="<utilizator>", 
                                assistant_token="<asistent>", system_token="<sistem>", logger=None):
    if tokenizer is not None:
        bos_token = tokenizer.bos_token or ""
        eos_token = tokenizer.eos_token or ""
    else:
        bos_token = ""
        eos_token = ""
    
    # Map "from" values to role tokens with closing tags
    role_token_map = {
        'system': (system_token, '</sistem>'),
        'human': (user_token, '</utilizator>'),
        'gpt': (assistant_token, '</asistent>')
    }
    
    formatted_text = ""
    
    for message in conversations:
        from_role = message.get('from', '').strip()
        content = message.get('value', '').strip()
        
        # Get the appropriate role tokens
        open_tag, close_tag = role_token_map.get(from_role, (f"<{from_role}>", f"</{from_role}>"))
        
        if content:
            formatted_text += f"{open_tag}\n{content}\n{close_tag}\n"
    
    return formatted_text