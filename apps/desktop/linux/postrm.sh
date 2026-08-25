#!/bin/sh
# Forgets the removed uinput rule.
if command -v udevadm >/dev/null 2>&1; then
    udevadm control --reload-rules || true
fi
exit 0
