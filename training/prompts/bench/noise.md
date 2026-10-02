You are writing held-out test transcripts that each show one hesitation or correction pattern, for a dictation cleanup model: what a speech recognizer wrote down while a person talked into a dictation app. Write only the transcripts. Someone else cleans them up, and a checker rejects any that read like written text.

Each transcript is 15 to 140 words of ordinary dictation: a message, a note to self, a list, a plan, a question, an instruction. Vary the length across the batch, with most under 80 words. Apart from the pattern asked for, the speaker talks naturally: casual, with a few other stumbles here and there, never polished prose with fillers sprinkled in. The pattern shows up clearly at least once.

One block of text: no bullets, headings, numbering or line breaks. Every transcript in a batch is different: its opening, its topic, its names and numbers.

Follow the transcript style given below in every transcript.

Return ONLY a JSON object, no prose and no code fence:
{"items": [{"input": "the transcript", "evidence": ["the pattern, exactly as written in the input"], "layout": "prose", "tags": ["short tag"]}]}

"layout" is what a careful cleanup should produce: "list" when the speaker is making a list of things to use or dictates list items, "mixed" when only part of it becomes a list, and "prose" otherwise.
