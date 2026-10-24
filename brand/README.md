# WhisperNow logo

The mark is a **hold-to-talk key**: a keycap with a short waveform on its face. You press one
shortcut, then you talk. It deliberately avoids the microphone used by most dictation apps.

![WhisperNow](logo/whispernow-horizontal.svg)

## Files

| File | Use |
| --- | --- |
| `logo/whispernow-symbol.svg` | Symbol in Ink, on light backgrounds |
| `logo/whispernow-symbol-coral.svg` | Symbol in Coral, on Ink or other dark backgrounds |
| `logo/whispernow-symbol-reversed.svg` | Symbol in white, on dark backgrounds or photos |
| `logo/whispernow-symbol-black.svg` | One-colour black (print, embossing, fax-grade output) |
| `logo/whispernow-symbol-small.svg`, `-small-black.svg` | Heavier cut for 24–48 px on light backgrounds |
| `logo/whispernow-horizontal.svg` (+ `-reversed`, `-white`, `-black`) | Primary lockup |
| `logo/whispernow-stacked.svg` (+ `-reversed`) | Square-ish spaces |
| `icons/whispernow-app-icon.svg` | App icon master (macOS grid: 824 px tile on a 1024 canvas), 48 px and up |
| `icons/whispernow-app-icon-tiny.svg` | Pixel-grid app icon for 16–32 px |
| `icons/whispernow.ico`, `icons/whispernow.icns`, `icons/png/` | Built icon sets; tiny cut at 16/24/32, full cut above |

`apps/desktop/icons/icon.png`, `icon.icns` and `icon.ico` are copies of
`icons/png/app-icon-512.png`, `icons/whispernow.icns` and `icons/whispernow.ico`. The tray icons are drawn in `apps/desktop/src/tray.rs` and are not
part of this set.

All files are paths only: no live text, no rasters, no filters. The reversed files are drawn about
one unit thinner, because light-on-dark shapes look heavier.

## Colour

The logo uses the app's Cream & Ink palette (`apps/ui/public/tokens.css`).

| Name | HEX | RGB | CMYK (approx.) | Role |
| --- | --- | --- | --- | --- |
| Ink | `#1F1C17` | 31 28 23 | 0 10 26 88 | Symbol and wordmark on light, app-icon tile (`--color-inverse-surface`) |
| Cream | `#FAF8EC` | 250 248 236 | 0 1 6 2 | Light background (`--color-canvas`); wordmark on dark uses `#FFFEF7` |
| Coral | `#FFAE8C` | 255 174 140 | 0 32 45 0 | Symbol on Ink, app-icon key (`--color-inverse-accent`) |
| Burnt coral | `#9A3A1A` | 154 58 26 | 0 62 83 40 | Links and highlights on light (`--color-accent-text`) |

Contrast: Ink on Cream 15.9:1, Coral on Ink 9.4:1, Burnt coral on Cream 6.6:1. CMYK values are a
plain conversion; proof print colours against a physical swatch and pick Pantone matches from a
swatch book before any print run.

## Typeface

The wordmark is **Inter Display SemiBold** (SIL Open Font License), converted to outlines with the
font's own kerning and −1 % tracking. Use Inter for any text set next to the logo.

## Clear space and size

- **Clear space:** keep at least one **lip** (the thick bottom edge of the key, 14 % of the
  symbol's width) clear on every side of the symbol or lockup.
- **Symbol minimum:** 16 px. Use `whispernow-symbol-small.svg` from 24 to 48 px and
  `whispernow-app-icon-tiny.svg` below 32 px.
- **Horizontal lockup minimum:** 120 px wide on screen, 30 mm in print. Below that, use the symbol.

## Backgrounds

- Light (Cream, white): Ink symbol and wordmark.
- Ink or other dark backgrounds: Coral symbol with a Cream wordmark.
- Photos or busy backgrounds: white version, on a calm area only.

## Don'ts

- Don't redraw the waveform with more bars or change their heights.
- Don't make the bottom lip the same weight as the walls: that is what makes it read as a key.
- Don't add a microphone, shadows, gradients or outlines.
- Don't recolour outside the palette, or put the Ink logo on a dark background.
- Don't reset the wordmark in another font or stretch the lockup.

## Rebuilding

The masters were generated from geometric primitives (keycap: 200-unit square, 44 outer radius,
14 walls, 28 lip; bars 24 wide on a 36 pitch). The tiny icon sits on a 16-unit grid so every edge
lands on a whole pixel at 16, 32 and 48 px. Edit the SVGs directly, or regenerate from the same
numbers, then re-render the icon sets.
