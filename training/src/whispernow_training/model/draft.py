"""The draft sent with each request for speculative decoding: the transcript, pre-cleaned by code.

A patched `llama-server` (`serving/llama-server-draft-text.patch`) takes it as the request field
`draft_text`. Each pass it finds the output's last words in the draft and proposes the words after
them; the model checks them all in one pass and keeps the ones it would have written itself, so the
output is unchanged up to near-ties. A stock server ignores the
field. The closer the draft is to the cleaned text, the more each pass keeps, so the draft does
what code can: drops fillers, writes spoken punctuation and numbers as symbols and digits, joins
spelled-out letters and capitalizes sentences.

`serving/prompt_golden.json` pins `draft_text` for each case; the app's port
(`whispernow-core`, `llm/draft.rs`) is tested against the same file.
"""

from __future__ import annotations

FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm"}
SPOKEN = {"period": ".", "comma": ",", "question mark": "?", "colon": ":", "exclamation point": "!", "full stop": "."}
UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}


def _digit(word: str) -> bool:
    return word in UNITS and UNITS[word] < 10


def _number(words: list[str], i: int) -> tuple[str, int] | None:
    """A spoken number starting at lowercase `words[i]` → (digits, words used)."""
    if words[i] not in UNITS and words[i] not in TENS:
        return None
    # A run of single digits ("zero four seven nine") is read digit by digit.
    j = i
    while j < len(words) and _digit(words[j]):
        j += 1
    if j - i >= 3 or (j - i >= 2 and words[i] == "zero"):
        return "".join(str(UNITS[w]) for w in words[i:j]), j - i
    total, current, j = 0, 0, i
    while j < len(words):
        w = words[j]
        if w in UNITS and current % 100 and (current % 100 < 20 or current % 10):
            break  # "nineteen nine" is two numbers
        if w in UNITS:
            current += UNITS[w]
        elif w in TENS:
            current += TENS[w]
        elif w in SCALES and (current or total):
            if SCALES[w] == 100:
                current *= 100
            else:
                total += current * SCALES[w]
                current = 0
        elif w == "and" and j + 1 < len(words) and (words[j + 1] in UNITS or words[j + 1] in TENS) and current >= 100:
            pass
        else:
            break
        j += 1
    text = str(total + current)
    if j + 1 < len(words) and words[j] == "point" and _digit(words[j + 1]):
        k = j + 1
        while k < len(words) and _digit(words[k]):
            k += 1
        text += "." + "".join(str(UNITS[w]) for w in words[j + 1 : k])
        j = k
    return text, j - i


def preclean(transcript: str) -> str:
    """The transcript as code can clean it: the draft the server proposes words from."""
    words = [w for w in transcript.split() if w.lower() not in FILLERS]
    lower = [w.lower() for w in words]
    out: list[str] = []
    i = 0
    while i < len(words):
        w = lower[i]
        two = " ".join(lower[i : i + 2])
        if two in SPOKEN or w in SPOKEN:
            mark = SPOKEN.get(two) or SPOKEN[w]
            if out:
                out[-1] += mark
            i += 2 if two in SPOKEN else 1
        elif (n := _number(lower, i)) is not None:
            out.append(n[0])
            i += n[1]
        elif len(w) == 1 and w.isalpha() and w not in "ai" and i + 1 < len(words) and len(words[i + 1]) == 1:
            j = i
            while j < len(words) and len(words[j]) == 1 and words[j].isalpha():
                j += 1
            out.append("".join(words[i:j]).upper())
            i = j
        else:
            out.append("I" if w == "i" else ("I" + words[i][1:] if w.startswith("i'") else words[i]))
            i += 1
    # Sentence case: the first word and every word after a sentence end.
    for k, w in enumerate(out):
        if k == 0 or out[k - 1][-1] in ".?!":
            out[k] = w[:1].upper() + w[1:]
    return " ".join(out)
