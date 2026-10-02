You are writing held-out test transcripts for a dictation cleanup model: what a speech recognizer wrote down while a person talked into a dictation app. Write only the transcripts. Someone else cleans them up, and a checker rejects any that read like written text.

How people actually sound when they dictate:
- They talk, they don't write. Long sentences chained with "and", "so", "but"; ideas that wander and return; things added late ("oh, also"); sentences that trail off.
- Hesitation comes where thinking happens: before a number or a name, at a topic change, when unsure. It is uneven: some stretches are fluent, others stumble.
- Hesitations bunch up and fall mid-phrase: "we need uh um maybe four", "book the er the bigger room".
- Words get said twice or three times: "they they", "that's that's", "we should we should ask", "so so so".
- Phrases get dropped and restarted without warning: "send her the we need to send her the draft", or a stray word left behind: "she tomorrow she's off".
- Changes of mind. One short cue or none at all; nobody says "no wait, sorry, I mean" for one change.
  - A bare swap: "Friday, sorry, Saturday".
  - A fresh decision in a full clause: "the pasta or hmm no, let's do the risotto".
  - The reason in the middle and "instead" at the end: "the big van, except well it won't fit in the garage, so let's just get the small one instead".
- Talking to themselves: "okay what else", "hmm, where was I", "right, so".
- Casual stays casual. Never polished prose with a few "um"s dropped in.
- One block of text: no bullets, headings, numbering or line breaks. Layout only when spoken ("new line", "bullet").
- Every transcript in a batch is different: its opening, its names, its places, its numbers, how messy it is.

Follow the transcript style given below in every transcript.

Return ONLY a JSON object, no prose and no code fence:
{"items": [{"input": "the transcript", "layout": "prose", "tags": ["short tag"]}]}

"layout" is what a careful cleanup should produce: "list" when the speaker is making a list of things to use (shopping, errands, to-dos, packing, steps, a recipe) or dictates list items, "mixed" when only part of it becomes a list, and "prose" otherwise. "tags" names the hard cases in the transcript, from the category guidance.
