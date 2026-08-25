#!/bin/sh
# Installs the shortcut release and paste helper setgid `input`, so it alone can
# read keyboards and create the virtual keyboard that presses the paste
# shortcut, with a udev rule that gives the input group `/dev/uinput`. The
# .deb and .rpm packages do this themselves; this is for AppImage and source
# builds. Build it first with `cargo build --release -p whispernow-input-helper`.
# Run as root: sudo sh scripts/install_input_helper.sh
# Undo with: sudo rm -r /usr/local/libexec/whispernow /etc/udev/rules.d/60-whispernow-uinput.rules
set -eu

here="$(dirname "$0")"
if [ -f "$here/whispernow-input-helper" ]; then
    default_binary="$here/whispernow-input-helper"
else
    default_binary="$here/../target/release/whispernow-input-helper"
fi
source_binary="${1:-$default_binary}"
install -d -o root -g root -m 0755 /usr/local/libexec/whispernow
install -o root -g input -m 2755 "$source_binary" \
  /usr/local/libexec/whispernow/whispernow-input-helper
# Matches apps/desktop/linux/60-whispernow-uinput.rules, which the packages install.
cat > /etc/udev/rules.d/60-whispernow-uinput.rules <<'RULE'
# Lets the input group create virtual keyboards. WhisperNow's paste helper is
# setgid input and presses only the paste shortcut on its keyboard.
KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"
RULE
udevadm control --reload-rules
udevadm trigger --action=change --sysname-match=uinput
ls -l /usr/local/libexec/whispernow/whispernow-input-helper /dev/uinput
echo "Restart WhisperNow to use the helper."
