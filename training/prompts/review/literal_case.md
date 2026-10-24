You check reference outputs in the training and test data of a dictation cleanup model. A reference is what the model should output for a transcript under the editing contract below.

In each row, the FLAGGED line of the REFERENCE starts with a capitalized word that may be a literal command or file name. The contract keeps code, commands, file names, paths and identifiers exactly as dictated, so a command the speaker means to be typed keeps its literal lowercase even at the start of a line or sentence ("docker compose up -d", "kill -9 4512", "cargo.toml"), while prose that names a tool or product keeps its usual capitalization ("Terraform import is for…", "Git stash is more flexible than…", "README.md", "Next.js").

Lowercase the first word only when the FLAGGED line or sentence is itself the command or path to be typed, with its arguments ("Docker compose up -d." as "docker compose up -d.", "Source .venv/bin/activate and run…" as "source .venv/bin/activate and run…"). A sentence that talks about a command or tool stays in sentence case, even when it starts with the command's name ("Terraform destroy is the command to tear down…", "Docker exec for running commands in a container.", "Git revert is different from git reset…"). Library and product names keep their usual spelling ("Node.js", "Three.js", "Chart.js").

Return the whole REFERENCE with only that first word's case fixed when it needs fixing, otherwise "ok". Change nothing else.

{contract}

Return ONLY a JSON object, no prose and no code fence:
{"results": [{"i": 0, "verdict": "ok"}, {"i": 1, "verdict": "fix", "output": "the whole corrected reference", "note": "what changed and why"}]}
Include one entry per row, with the row's index as "i".