# Resume after a missing generator API key

A generator run whose batch log contains the exact error
`unreadable generator reply: 'LLM_API_KEY'` can resume after `LLM_API_KEY` is
configured. Run the same generation command again. No log or output-file edits
are required.

The generation, composed-task, and noise workflows restore one batch of budget
for each such record only when `returned`, `passed`, and `relabeled` are all zero
and `notes` is empty. These configuration failures remain in the log, with their
original batch numbers. Completed rows, rejections, response-cache entries, and
all other batch records remain intact. This also handles missing-key records
interleaved with later completed work.

Genuine malformed model replies still consume the bounded batch budget.
Similar-looking errors with another missing key, a different message, nonzero
row counts, or feedback notes receive no budget credit. Client failures during
new requests propagate and write no completed batch, so correcting the client
configuration allows the same request to resume.
