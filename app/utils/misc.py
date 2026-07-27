from collections import Counter

def looks_like_gibberish(text: str) -> bool:
    if not text:
        return True

    s = text.strip()
    if len(s) < 40:
        return True

    good_chars = sum(
        1 for ch in s
        if ch.isalnum() or ch.isspace() or ch in ".,;:!?()-[]"
    )
    if good_chars / max(1, len(s)) < 0.6:
        return True

    cnt = Counter(s)
    most_common = cnt.most_common(1)[0][1]
    if most_common / len(s) > 0.25:
        return True

    return False
