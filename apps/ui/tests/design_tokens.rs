//! Keeps the stylesheets in `public/` within the design-system contract.
//!
//! Colour tokens are read from `tokens.css` itself, so a palette edit that breaks WCAG 2.2
//! contrast (4.5:1 for text, 3:1 for control boundaries and focus) fails here.

use std::collections::BTreeMap;

const TOKENS: &str = include_str!("../public/tokens.css");
const STYLESHEETS: [(&str, &str); 4] = [
    ("tokens.css", TOKENS),
    ("ui.css", include_str!("../public/ui.css")),
    ("settings.css", include_str!("../public/settings.css")),
    ("overlay.css", include_str!("../public/overlay.css")),
];

const LIGHT: &str = ":root";
const SYSTEM_DARK: &str = ":root:not([data-theme=\"light\"])";
const EXPLICIT_DARK: &str = ":root[data-theme=\"dark\"]";

/// Every background text and controls are drawn on.
const BACKGROUNDS: [&str; 9] = [
    "canvas",
    "surface",
    "surface-raised",
    "surface-sunken",
    "accent-soft",
    "success-soft",
    "warning-soft",
    "danger-soft",
    "info-soft",
];
const SURFACES: [&str; 4] = ["canvas", "surface", "surface-raised", "surface-sunken"];
const TEXT: [&str; 7] = [
    "text-primary",
    "text-secondary",
    "accent-text",
    "success",
    "warning",
    "danger",
    "info",
];
/// Control boundaries, the focus ring and filled accents (toggle, progress, primary button).
const NON_TEXT: [&str; 3] = ["border-control", "focus", "accent"];
/// The recording overlay's theme-independent palette; its pill and hovered controls.
const INVERSE_SURFACES: [&str; 2] = ["inverse-surface", "inverse-surface-raised"];
const INVERSE_TEXT: [&str; 4] = [
    "inverse-text",
    "inverse-text-secondary",
    "inverse-accent",
    "inverse-danger",
];

type Palette = BTreeMap<String, [u8; 3]>;

/// The declarations inside the first `selector { … }` block.
fn block<'a>(css: &'a str, selector: &str) -> &'a str {
    let opening = format!("{selector} {{");
    let start = css
        .find(&opening)
        .unwrap_or_else(|| panic!("tokens.css declares {selector}"))
        + opening.len();
    let end = start + css[start..].find('}').expect("closed block");
    &css[start..end]
}

fn palette(block: &str) -> Palette {
    block
        .lines()
        .filter_map(|line| line.trim().strip_prefix("--color-"))
        .map(|declaration| {
            let (name, value) = declaration.split_once(':').expect("colour declaration");
            let value = value.trim().trim_end_matches(';');
            (name.to_owned(), hex(name, value))
        })
        .collect()
}

fn hex(name: &str, value: &str) -> [u8; 3] {
    let digits = value
        .strip_prefix('#')
        .filter(|digits| digits.len() == 6)
        .unwrap_or_else(|| panic!("--color-{name} must be #rrggbb, found {value}"));
    let channel = |index: usize| {
        u8::from_str_radix(&digits[index..index + 2], 16)
            .unwrap_or_else(|_| panic!("--color-{name} has invalid hex {value}"))
    };
    [channel(0), channel(2), channel(4)]
}

fn luminance(colour: [u8; 3]) -> f64 {
    let linear = |channel: u8| {
        let value = f64::from(channel) / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * linear(colour[0]) + 0.7152 * linear(colour[1]) + 0.0722 * linear(colour[2])
}

fn contrast(first: [u8; 3], second: [u8; 3]) -> f64 {
    let (first, second) = (luminance(first), luminance(second));
    (first.max(second) + 0.05) / (first.min(second) + 0.05)
}

fn themes() -> [(&'static str, Palette); 2] {
    [
        ("light", palette(block(TOKENS, LIGHT))),
        ("dark", palette(block(TOKENS, EXPLICIT_DARK))),
    ]
}

/// Contrast failures of each `foregrounds` × `backgrounds` pair below `minimum`.
fn failures(
    theme: &str,
    palette: &Palette,
    foregrounds: &[&str],
    backgrounds: &[&str],
    minimum: f64,
) -> Vec<String> {
    let colour = |name: &str| {
        *palette
            .get(name)
            .unwrap_or_else(|| panic!("{theme} palette defines --color-{name}"))
    };
    foregrounds
        .iter()
        .flat_map(|foreground| {
            backgrounds
                .iter()
                .map(move |background| (foreground, background))
        })
        .filter_map(|(foreground, background)| {
            let ratio = contrast(colour(foreground), colour(background));
            (ratio < minimum)
                .then(|| format!("{theme}: {foreground} on {background} is {ratio:.2}:1"))
        })
        .collect()
}

#[test]
fn contrast_calculation_matches_wcag_reference_values() {
    assert!((contrast([0, 0, 0], [255, 255, 255]) - 21.0).abs() < 1e-9);
    assert!((contrast([0x76, 0x76, 0x76], [255, 255, 255]) - 4.54).abs() < 0.01);
}

fn declarations(block: &str) -> Vec<&str> {
    block
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect()
}

#[test]
fn both_dark_declarations_match_and_every_theme_defines_the_same_colours() {
    let system_dark = block(TOKENS, SYSTEM_DARK);
    let explicit_dark = block(TOKENS, EXPLICIT_DARK);
    assert_eq!(declarations(system_dark), declarations(explicit_dark));

    let [(_, light), (_, dark)] = themes();
    assert!(light.len() >= BACKGROUNDS.len() + TEXT.len());
    assert_eq!(
        light.keys().collect::<Vec<_>>(),
        dark.keys().collect::<Vec<_>>()
    );
}

#[test]
fn text_colours_meet_4_5_to_1_on_every_background_in_both_themes() {
    let mut problems = Vec::new();
    for (theme, palette) in themes() {
        problems.extend(failures(theme, &palette, &TEXT, &BACKGROUNDS, 4.5));
        problems.extend(failures(
            theme,
            &palette,
            &["on-accent"],
            &["accent", "accent-hover"],
            4.5,
        ));
        problems.extend(failures(
            theme,
            &palette,
            &INVERSE_TEXT,
            &INVERSE_SURFACES,
            4.5,
        ));
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

#[test]
fn controls_focus_and_accents_meet_3_to_1_on_every_surface_in_both_themes() {
    let mut problems = Vec::new();
    for (theme, palette) in themes() {
        problems.extend(failures(theme, &palette, &NON_TEXT, &SURFACES, 3.0));
        problems.extend(failures(
            theme,
            &palette,
            &["inverse-accent", "inverse-border"],
            &INVERSE_SURFACES,
            3.0,
        ));
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

/// Stylesheet text without comments.
fn without_comments(css: &str) -> String {
    let mut output = String::with_capacity(css.len());
    let mut rest = css;
    while let Some(start) = rest.find("/*") {
        output.push_str(&rest[..start]);
        let end = rest[start..].find("*/").expect("closed comment");
        rest = &rest[start + end + 2..];
    }
    output.push_str(rest);
    output
}

/// Every selector list or at-rule prelude, i.e. the text before each `{`.
fn preludes(css: &str) -> Vec<String> {
    let mut preludes = Vec::new();
    let mut current = String::new();
    for character in css.chars() {
        match character {
            '{' => preludes.push(std::mem::take(&mut current).trim().to_owned()),
            '}' | ';' => current.clear(),
            _ => current.push(character),
        }
    }
    preludes
}

fn names_element(selector: &str, element: &str) -> bool {
    selector.match_indices(element).any(|(index, _)| {
        let before = selector[..index].chars().next_back();
        let after = selector[index + element.len()..].chars().next();
        let part_of_name =
            |character: char| character.is_ascii_alphanumeric() || "-_.#".contains(character);
        !before.is_some_and(part_of_name) && !after.is_some_and(part_of_name)
    })
}

#[test]
fn stylesheets_avoid_important_and_style_buttons_only_by_class() {
    for (name, css) in STYLESHEETS {
        let css = without_comments(css);
        assert!(!css.contains("!important"), "{name} uses !important");
        for prelude in preludes(&css) {
            assert!(
                !names_element(&prelude, "button"),
                "{name} styles the button element: {prelude}"
            );
        }
    }
}

#[test]
fn no_text_is_smaller_than_12px() {
    for (name, css) in STYLESHEETS {
        let css = without_comments(css);
        for declaration in css.split([';', '{', '}']) {
            let Some((property, value)) = declaration.split_once(':') else {
                continue;
            };
            let property = property.trim();
            if property != "font-size" && !property.starts_with("--font-size") {
                continue;
            }
            for number in value.split("px").filter_map(|part| {
                part.rsplit(|character: char| !character.is_ascii_digit() && character != '.')
                    .next()
                    .and_then(|digits| digits.parse::<f64>().ok())
            }) {
                assert!(number >= 12.0, "{name}: {property}: {value}");
            }
        }
    }
}

#[test]
fn inverse_colours_are_the_same_in_every_theme() {
    let [(_, light), (_, dark)] = themes();
    let inverse = |palette: &Palette| {
        palette
            .iter()
            .filter(|(name, _)| name.starts_with("inverse-"))
            .map(|(name, colour)| (name.clone(), *colour))
            .collect::<Vec<_>>()
    };
    assert!(inverse(&light).len() >= INVERSE_SURFACES.len() + INVERSE_TEXT.len());
    assert_eq!(inverse(&light), inverse(&dark));
}

/// The body of the `@media (prefers-reduced-motion: reduce)` block in `css`.
fn reduced_motion_block(css: &str) -> &str {
    let opening = "@media (prefers-reduced-motion: reduce) {";
    let start = css.find(opening).expect("a reduced-motion block") + opening.len();
    let mut depth = 1;
    for (offset, character) in css[start..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &css[start..start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unclosed reduced-motion block")
}

#[test]
fn reduced_motion_stops_every_overlay_animation_and_transition() {
    let css = without_comments(include_str!("../public/overlay.css"));
    let reduced = reduced_motion_block(&css);
    let (selectors, declarations) = reduced.split_once('{').expect("a reduced-motion rule");
    assert!(declarations.contains("animation: none"), "{declarations}");
    assert!(declarations.contains("transition: none"), "{declarations}");
    let stopped: Vec<&str> = selectors.split(',').map(str::trim).collect();

    let outside = css.replace(reduced, "");
    let mut moving = Vec::new();
    let mut rest = outside.as_str();
    while let Some(open) = rest.find('{') {
        let selector = rest[..open].rsplit(['}', ';']).next().unwrap_or("").trim();
        let close = open + rest[open..].find('}').expect("closed rule");
        let body = &rest[open + 1..close];
        if !selector.starts_with('@')
            && body.lines().any(|line| {
                let line = line.trim();
                line.starts_with("animation:") || line.starts_with("transition:")
            })
        {
            moving.push(selector.to_owned());
        }
        rest = &rest[close + 1..];
    }
    assert!(!moving.is_empty(), "the overlay animates");
    for selector in moving {
        assert!(
            stopped.contains(&selector.as_str()),
            "{selector} keeps moving under reduced motion"
        );
    }
}
