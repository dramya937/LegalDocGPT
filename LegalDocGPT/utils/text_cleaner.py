def clean_text(text):
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())
    return text
