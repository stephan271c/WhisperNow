# Prompts

- `teacher/`: the app's cleanup prompt, which writes every reference.
- `generate/`, `categories/`, `compose/`, `bench/`: generator prompts for stage 3 transcripts, composed rows and the benchmarks. `restyle/` rewrites aawaaz inputs in a recognizer's style.
- `judge/`: judges that run on new work.
  - `completeness.md` gates every new reference: `relabel`, `bench ami`, and the fact check in `bench gate-probed`.
  - `compose.md` gates generated long and noisy rows: `compose`, `bench noise`, `bench gate-probed`.
  - `generate.md` gates stage 3 rows: `generate`.
  - `eval.md` scores model outputs: `cleanup eval --judge`.
  - `agreement.md` scores a teacher prompt against aawaaz references: `teacher-check`.
- `audit/`: one-off passes that cleaned the imported datasets. Their verdicts are in the build, so they run again only when that stage is rebuilt, and an edit re-judges the whole stage.
  - `aawaaz.md`: `judge-aawaaz`, and the re-judge in `fix-aawaaz`.
  - `aawaaz_register.md`: `judge-register`, and the re-judge in `fix-aawaaz`.
  - `handy_disfl_qa.md`: `add`.
