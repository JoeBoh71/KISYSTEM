def extract_code(text):
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        return '\n'.join(lines).strip()
    return text
