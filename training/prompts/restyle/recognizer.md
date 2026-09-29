You rewrite speech-recognizer transcripts from one output style into another. The input transcripts were written by an older recognizer: all lowercase, almost no punctuation, numbers spelled out as words. Rewrite each one as a modern recognizer such as Parakeet TDT or Whisper would have written the same audio.

Keep every spoken word, in order. This is a formatting change only:
- Keep all fillers ("um", "uh", "like", "you know"), repeated words ("the the"), false starts, self-corrections and spoken commands ("period", "comma", "new line", "new paragraph", "bullet point", "dot", "slash", "dash"). Write spoken commands as the words, never as the symbol, and keep spoken emails and web addresses as words ("orders at breadlab dot com"). Keep slang as said ("fifty bucks", not "$50"). Do not drop, add, merge, split, reorder, correct or expand any word, and do not write a contraction for two words or two words for a contraction.
- Do not clean up, rephrase or fix grammar. The result is still a raw transcript, just cased and punctuated.

How the recognizer writes:
- Sentences start with a capital letter and usually end with a period; a question gets a question mark. Sentences run long, about 16 words in the middle and often 25 or more: speakers who chain clauses with "and" or "so" get long sentences with commas, not a period at every clause. Expect about five periods and six commas per hundred words.
- Commas mark some pauses, list items and clause boundaries, but not all of them.
- Names, places, brands, the pronoun "I" and acronyms are capitalized ("HRT", "CPU", "Q3"). Letters spelled out one by one ("h r t") become the acronym.
- Fillers are written as plain words, mostly without commas around them ("so um we need", "and uh the"). A filler that starts a sentence is capitalized, sometimes with a comma after it ("Um, about eight months ago").
- Spoken commands stay words, and a real mark sometimes lands next to them where the speaker paused ("Dear Admissions Office, comma new paragraph.", "two days period she's got").
- Numbers are inconsistent, even within one transcript: about half become digits and half stay words. Percentages, money with a dollar amount, measurements and years lean to digits ("20%", "$4,200", "5 milligrams", "2024"); small counts, round phrases and long spoken amounts often stay words ("three times", "two weeks", "four hundred and sixty dollars", "one point two million"). Do not make every number the same way.

Return JSON only: {"items": [{"n": <n>, "text": "<rewritten transcript>"}, ...]}, one item per input item, in the same order, with the same n.
