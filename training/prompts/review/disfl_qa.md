You check reference outputs in the training and test data of a dictation cleanup model. A reference is what the model should output for a transcript under the editing contract below.

Every row here comes from Disfl-QA: someone took a written question (ORIGINAL) and wrote a spoken version of it with a false start or a self-correction (TRANSCRIPT). The REFERENCE is the cleaned transcript.

Decide for each row whether the REFERENCE handles the self-correction correctly:
- It drops every word the speaker abandoned: the false start, the corrected part, and the cue ("no", "sorry", "or rather", "make that", "I mean"). Use the ORIGINAL as evidence of what the speaker meant: transcript words that are missing from the ORIGINAL and sit in the stretch the speaker restarted were abandoned, also when no cue word marks the restart ("What is the system um mechanisms do many species not utilize?" restarts "What mechanisms…", so "system" goes).
- It keeps the speaker's final version as spoken, even where that differs from the ORIGINAL ("…or actually tell me who announced…" stays "Tell me who announced…"). Never bring in words, typos or word order from the ORIGINAL that the speaker did not say in their final version.
- "A or uh B" with no other cue: drop A when the restart is clear (A and B cannot both be meant, or the question does not read with both). Keep "A or B" only when a listener would reasonably take it as a real choice between two alternatives, since the contract keeps words when it is not clear they are a correction.
- When the final version refers back with a pronoun to something named only in the abandoned part ("Why did Harvard reintroduce… no sorry why did they end…"), a reference that names it ("Why did Harvard end…") is fine, and so is one that smooths the restart into a plain question ("Fresno City College… I mean it is located at…" as "Fresno City College is located at…").
Fix only clear errors: a content word the speaker abandoned is still there, a word of the final version is missing or changed, or the question no longer asks what the speaker asked. Leave everything else in the REFERENCE as it is: its spelling, capitalization, punctuation and quotation marks, and any wording the self-correction does not touch. Return "ok" when the REFERENCE is a reasonable cleanup; differences from the ORIGINAL alone are not a reason to fix.

{contract}

Return ONLY a JSON object, no prose and no code fence:
{"results": [{"i": 0, "verdict": "ok"}, {"i": 1, "verdict": "fix", "output": "the whole corrected reference", "note": "what changed and why"}]}
Include one entry per row, with the row's index as "i".