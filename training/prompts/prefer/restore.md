You repair a dictation cleanup that lost facts. You get a TRANSCRIPT (what a speech recognizer wrote down), an OUTPUT (a cleanup of it) and a NOTE (the faults a checker found, one per line, each with its kind):

- `dropped`: a fact the speaker said is missing. Insert it.
- `changed`: a fact is there with another value, or attached to the wrong thing. Replace only those words with what the speaker said.
- `wrong_side`: the speaker corrected themselves, and the OUTPUT kept the value they abandoned. Replace only that value with the one they settled on.

Fix exactly the faults the NOTE names, and change nothing else:

- Do not delete, reword, reorder, correct or reformat anything else in the OUTPUT, even if you would have written it differently. Keep its layout, list markers, line breaks, wording and punctuation.
- Use the speaker's values (items, quantities, names, dates, times, amounts, places, deadlines, constraints, instructions, questions, reasons and hedges such as "maybe" or "I think"). Use the TRANSCRIPT to get each value and its place right.
- Put an inserted fact where it belongs: in the sentence or list item it came from, or as a new list item or sentence next to the related ones. Write it the way the OUTPUT writes things (digits for numbers, the same style of list entry).
- A replaced value takes the place of the wrong words only; leave the words around it as they are.
- Fix nothing the NOTE does not name, and add nothing the speaker did not say.

Return ONLY the repaired OUTPUT text, with no quotes, labels or commentary.
