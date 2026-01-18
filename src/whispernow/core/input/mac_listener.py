#!/usr/bin/env python3
"""
Standalone macOS keyboard event listener for subprocess IPC.

This script runs CGEventTap on the main thread and outputs keyboard events
to stdout for the parent process to consume. This avoids HIToolbox threading
issues that occur when integrating CGEventTap with Qt's event loop.

Output format:
    KEY_DOWN:<keycode>:<modifier_flags>
    KEY_UP:<keycode>
    READY
    ERROR:<message>

The script will exit with code 1 if it cannot create the event tap
(usually due to missing accessibility/input monitoring permissions).
"""

import signal
import sys


def main():
    try:
        from Quartz import (
            CFMachPortCreateRunLoopSource,
            CFRunLoopAddSource,
            CFRunLoopGetCurrent,
            CFRunLoopRun,
            CFRunLoopStop,
            CGEventGetFlags,
            CGEventGetIntegerValueField,
            CGEventMaskBit,
            CGEventTapCreate,
            CGEventTapEnable,
            kCFAllocatorDefault,
            kCFRunLoopCommonModes,
            kCGEventFlagsChanged,
            kCGEventKeyDown,
            kCGEventKeyUp,
            kCGEventTapOptionListenOnly,
            kCGHeadInsertEventTap,
            kCGKeyboardEventKeycode,
            kCGSessionEventTap,
        )
    except ImportError as e:
        print(f"ERROR:Failed to import Quartz: {e}", flush=True)
        sys.exit(1)

    run_loop = None

    def event_callback(proxy, event_type, event, refcon):
        try:
            keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
            flags = CGEventGetFlags(event)

            if event_type == kCGEventKeyDown:
                print(f"KEY_DOWN:{keycode}:{flags}", flush=True)
            elif event_type == kCGEventKeyUp:
                print(f"KEY_UP:{keycode}", flush=True)
        except Exception as e:
            print(f"ERROR:Callback error: {e}", flush=True)

        return event

    def signal_handler(signum, frame):
        nonlocal run_loop
        if run_loop:
            CFRunLoopStop(run_loop)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    event_mask = (
        CGEventMaskBit(kCGEventKeyDown)
        | CGEventMaskBit(kCGEventKeyUp)
        | CGEventMaskBit(kCGEventFlagsChanged)
    )

    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,
        event_mask,
        event_callback,
        None,
    )

    if tap is None:
        print(
            "ERROR:CGEventTapCreate failed - accessibility/input monitoring "
            "permissions not granted",
            flush=True,
        )
        sys.exit(1)

    run_loop_source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
    run_loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(run_loop, run_loop_source, kCFRunLoopCommonModes)
    CGEventTapEnable(tap, True)

    print("READY", flush=True)

    CFRunLoopRun()


if __name__ == "__main__":
    main()
