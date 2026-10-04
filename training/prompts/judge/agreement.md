You compare two cleanups of the same dictation transcript. REFERENCE is the target style. CANDIDATE was written by another model. Decide whether CANDIDATE agrees with REFERENCE on three points. Wording, punctuation and small phrasing differences do not matter.

1. layout_agrees: both use the same kind of layout: plain sentences, paragraphs, a list (with or without lead-in and section labels), or prose mixed with a list. Bullets versus numbers for the same list counts as agreeing only when the order does not matter. FAIL when one is a list and the other prose, or one has paragraph breaks where the other is a single block of more than a few sentences.
2. lead_in_agrees: when REFERENCE has a lead-in, title or group labels before list items, CANDIDATE has an equivalent one; when REFERENCE has none, CANDIDATE adds none beyond words from the transcript. PASS when neither has a list.
3. facts_agree: CANDIDATE keeps every fact REFERENCE keeps (names, numbers, amounts, dates, items, quantities, brands, sizes, constraints, reasons, and hedges such as "maybe", "probably", "I think", "I guess" and "I believe") and adds none that REFERENCE lacks, apart from facts REFERENCE wrongly dropped from the transcript. Numbers must have the same value. A hedge or reason that belonged only to the abandoned side of a correction is not a fact CANDIDATE must keep.

Return ONLY a JSON object, no prose and no code fence:
{"results": [{"i": 0, "layout_agrees": true, "lead_in_agrees": true, "facts_agree": true, "note": ""}]}

Include one entry per item, with its index as "i". For a failing point, the note says briefly what differs.
