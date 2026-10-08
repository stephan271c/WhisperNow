# Teacher prompt

The teacher is `LLM_MODEL` sent exactly what the app sends a cleanup provider. These files are copies of the Rust sources in `crates/whispernow-core/src/llm/prompt.rs`:

- `system.txt`: `CLEANUP_SYSTEM_CONTRACT`.
- `edit_request.txt`: `CLEANUP_PROMPT`.

The user message wraps them as `build_cleanup_prompt` does; `datagen/teacher.py` rebuilds it. After changing either Rust string, run `uv run datagen sync-teacher`. `tests/test_teacher.py` fails while the copies and the Rust source differ.
