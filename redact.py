import re


def redact(t):
    t = str(t)
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", t)
    t = re.sub(r"\+?\d[\d\s().-]{8,}\d", "[PHONE]", t)
    return t
