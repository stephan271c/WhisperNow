# Teacher prompt

The teacher is `LLM_MODEL` sent exactly what the app sends for Clean Up & Format. These files are copies of the Rust sources:

- `system.txt`: `CLEANUP_SYSTEM_CONTRACT` in `crates/whispernow-core/src/llm/prompt.rs`.
- `edit_request.txt`: the `fix_grammar` prompt text in `crates/whispernow-domain/src/prompts.rs`.

The user message wraps them as `build_cleanup_prompt` does for `PromptKind::Cleanup`; `datagen/teacher.py` rebuilds it. After changing either Rust string, run `uv run datagen sync-teacher`. `tests/test_teacher.py` fails while the copies and the Rust source differ.
