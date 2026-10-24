You compare two candidate reference outputs for the same dictated transcript in the training and test data of a dictation cleanup model, under the editing contract below. For Disfl-QA rows you also see the ORIGINAL written question the transcript was made from, as evidence of what the speaker abandoned; the speaker's final spoken version still decides the wording.

For each row, say which candidate is the correct cleanup of the TRANSCRIPT under the contract: "A", "B", or "same" when both are equally acceptable. Judge self-corrections (abandoned words gone, the final version kept), meaning, and literal code and commands. Ignore differences that do not matter under the contract.

{contract}

Return ONLY a JSON object, no prose and no code fence:
{"results": [{"i": 0, "better": "A", "note": "why, briefly"}]}
Include one entry per row, with the row's index as "i".