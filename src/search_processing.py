import hashlib
import html
import re


def clean_web_text(text):
    text = str(text or "")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<(script|style|noscript|svg)\b[^>]*>.*?</\1\s*>",
                  " ", text, flags=re.I | re.S)
    text = re.sub(r"</?(?:p|div|li|tr|h[1-6]|br)\b[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    lines, seen = [], set()
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or re.fullmatch(r"(?:登录|注册|返回顶部|接受所有Cookie|拒绝Cookie)", line, re.I):
            continue
        # Preserve short numeric rows and repeated facts; deduplicate long boilerplate only.
        if len(line) > 30 and line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def query_terms(query):
    words = set(re.findall(r"[a-zA-Z0-9]{2,}", query.lower()))
    for phrase in re.findall(r"[\u4e00-\u9fff]+", query):
        words.update(phrase[i:i+2] for i in range(len(phrase)-1))
    return words


def select_windows(text, query, max_chars=700, window_chars=400, overlap=80):
    if max_chars < 1 or window_chars < 1 or not 0 <= overlap < window_chars:
        raise ValueError("invalid window budget or overlap")
    if len(text) <= max_chars:
        return text
    terms = query_terms(query)
    windows = []
    for start in range(0, len(text), window_chars-overlap):
        fragment = text[start:start+window_chars]
        score = sum(min(fragment.lower().count(term), 3) for term in terms)
        windows.append((score, start, min(start+window_chars, len(text))))
    # Select by query salience, then restore document order. Merge overlap so
    # repeated characters are not charged to the context budget twice.
    selected, used = [], 0
    for _, start, end in sorted(windows, key=lambda w: (-w[0], w[1])):
        uncovered = [(start, end)]
        for a, b in selected:
            uncovered = [(x, y) for x, y in uncovered if y <= a or x >= b] + [
                part for x, y in uncovered if not (y <= a or x >= b)
                for part in ((x, min(y, a)), (max(x, b), y)) if part[0] < part[1]]
        for a, b in uncovered:
            remaining = max_chars-used-(3 if selected else 0)
            if remaining <= 0:
                break
            b = min(b, a+remaining)
            selected.append((a, b))
            used += b-a+(3 if len(selected) > 1 else 0)
        if used >= max_chars:
            break
    return " … ".join(text[a:b] for a, b in sorted(selected))[:max_chars]


def process_search_response(data, query, max_source_chars=700, enabled=True):
    if "error" in data:
        return dict(data)
    results, raw_chars, cleaned_chars, kept_chars = [], 0, 0, 0
    for item in data.get("results", []):
        raw = str(item.get("raw_content") or item.get("content") or "")
        cleaned = clean_web_text(raw) if enabled else raw
        selected = select_windows(cleaned, query, max_chars=max_source_chars) if enabled else raw
        url = str(item.get("url") or "")
        title = clean_web_text(item.get("title"))
        source_id = "S" + hashlib.sha256((url or title+raw).encode()).hexdigest()[:12]
        results.append({"source_id": source_id, "title": title, "url": url,
                        "content": selected})
        raw_chars += len(raw)
        cleaned_chars += len(cleaned)
        kept_chars += len(selected)
    return {"answer": select_windows(clean_web_text(data.get("answer")), query, 300)
            if data.get("answer") else "", "results": results,
            "response_time": data.get("response_time"),
            "processing": {"raw_chars": raw_chars, "cleaned_chars": cleaned_chars,
                           "kept_chars": kept_chars,
                           "body_reduction": 1-kept_chars/raw_chars if raw_chars else 0.0,
                           "enabled": enabled}}
