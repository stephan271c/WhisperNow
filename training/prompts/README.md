# Prompts

- `teacher/`: the app's cleanup prompt, which writes every reference (`datagen sync-teacher` copies it from the Rust sources).
- `generate/`, `categories/`, `compose/`, `bench/`: generator prompts for the generated rows, the long composed rows and the benchmarks. `restyle/` rewrites aawaaz inputs in a recognizer's style.
- `judge/`: judges that run on new work.
  - `completeness.md` gates every new reference: `relabel`, `weak-spots`, `bench ami`, and the fact check in `bench gate-probed`.
  - `compose.md` gates generated long and noisy rows: `compose`, `bench noise`, `bench gate-probed`.
  - `generate.md` gates generated rows: `generate`.
  - `cutoff.json` holds the yes/no questions Mercury Decide answers about each cut-off piece: `cutoff check`.
  - `eval.md` scores model outputs: `cleanup eval --judge`.
  - `agreement.md` scores a teacher prompt against aawaaz references: `teacher-check`.
- `review/`: a second look at teacher references with evidence the teacher lacked (`review-refs`, `weak-spots`, `literal-cue`): `disfl_qa.md` checks a Disfl-QA reference against the written question its transcript came from, `literal_case.md` checks a line that starts with a capitalized command, and `verify.md` compares the old and the fixed reference blind.
- `audit/`: passes that cleaned the imported datasets for round 1. Their verdicts are in round 1, so they run again only when it is rebuilt, and an edit re-judges the whole source.
  - `aawaaz.md`: `judge-aawaaz`, and the re-judge in `fix-aawaaz`.
  - `aawaaz_register.md`: `judge-register`, and the re-judge in `fix-aawaaz`.
  - `handy_disfl_qa.md`: `add-public`.
