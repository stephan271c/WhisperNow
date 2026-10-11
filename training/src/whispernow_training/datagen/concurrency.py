from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


@asynccontextmanager
async def task_group() -> AsyncIterator[asyncio.TaskGroup]:
    try:
        async with asyncio.TaskGroup() as group:
            yield group
    except ExceptionGroup as errors:
        error: BaseException = errors
        # Preserve the client exception callers catch when one request failed.
        while isinstance(error, ExceptionGroup) and len(error.exceptions) == 1:
            error = error.exceptions[0]
        raise error from None
