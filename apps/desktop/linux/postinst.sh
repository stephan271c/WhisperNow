#!/bin/sh
# Makes the key release and paste helper setgid input, so it alone can read
# keyboards and create its paste keyboard, and applies the uinput rule now.
set -e
helper=/usr/libexec/whispernow/whispernow-input-helper
if getent group input >/dev/null 2>&1; then
    chgrp input "$helper"
    chmod 2755 "$helper"
else
    chmod 0755 "$helper"
fi
if command -v udevadm >/dev/null 2>&1; then
    udevadm control --reload-rules || true
    udevadm trigger --action=change --sysname-match=uinput || true
fi
exit 0
