/// Whole-segment phrases Whisper produces for silence or noise, compared case-insensitively
/// without trailing punctuation.
const SILENCE_HALLUCINATIONS: &[&str] = &[
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "thank you for watching and see you next time",
    "thanks for watching and see you next time",
    "please subscribe",
    "like and subscribe",
    "subtitles by the amara.org community",
    "[blank_audio]",
    "[silence]",
    "[music]",
    "(music)",
];

const SENTENCE_ENDINGS: &[char] = &['.', '!', '?', '…', '。', '！', '？'];
const CLOSING_MARKS: &[char] = &['"', '\'', ')', ']', '”', '’', '»'];

fn is_silence_hallucination(text: &str) -> bool {
    let normalized = text
        .trim()
        .trim_end_matches(|character: char| {
            SENTENCE_ENDINGS.contains(&character) || character.is_whitespace()
        })
        .to_lowercase();
    SILENCE_HALLUCINATIONS.contains(&normalized.as_str())
}

/// Joins segment texts with single spaces, dropping blank segments and silence hallucinations.
/// A segment that continues a sentence starts in lower case unless its first word is `I` or an
/// acronym.
#[must_use]
pub fn join_segment_texts<'text>(texts: impl IntoIterator<Item = &'text str>) -> String {
    let mut joined = String::new();
    for text in texts
        .into_iter()
        .map(str::trim)
        .filter(|text| !text.is_empty() && !is_silence_hallucination(text))
    {
        if joined.is_empty() {
            joined.push_str(text);
            continue;
        }
        let continues_sentence = !joined
            .trim_end_matches(CLOSING_MARKS)
            .ends_with(SENTENCE_ENDINGS);
        joined.push(' ');
        if continues_sentence && !keeps_capital(text) {
            let mut characters = text.chars();
            if let Some(first) = characters.next() {
                joined.extend(first.to_lowercase());
                joined.push_str(characters.as_str());
            }
        } else {
            joined.push_str(text);
        }
    }
    joined
}

fn keeps_capital(text: &str) -> bool {
    let word = text
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_end_matches(|character: char| !character.is_alphanumeric());
    if word == "I" || word.starts_with("I'") || word.starts_with("I’") {
        return true;
    }
    let mut letters = word.chars().filter(|character| character.is_alphabetic());
    letters.clone().count() >= 2 && letters.all(char::is_uppercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuing_segments_start_in_lower_case() {
        assert_eq!(
            join_segment_texts(["So we went", "To the store.", "Then home"]),
            "So we went to the store. Then home"
        );
        assert_eq!(
            join_segment_texts(["He said \"stop.\"", "Nobody moved"]),
            "He said \"stop.\" Nobody moved"
        );
    }

    #[test]
    fn i_and_acronyms_keep_their_capitals() {
        assert_eq!(
            join_segment_texts(["and then", "I left", "I'm back", "NASA called", "A plan"]),
            "and then I left I'm back NASA called a plan"
        );
    }

    #[test]
    fn segments_join_with_single_spaces_and_blanks_are_skipped() {
        assert_eq!(
            join_segment_texts(["  one.  ", "", " ", "Two."]),
            "one. Two."
        );
        assert_eq!(join_segment_texts(Vec::<&str>::new()), "");
    }

    #[test]
    fn whole_segment_hallucinations_are_recognized_case_insensitively() {
        assert!(is_silence_hallucination(" Thank you for watching! "));
        assert!(is_silence_hallucination("THANKS FOR WATCHING."));
        assert!(is_silence_hallucination("[BLANK_AUDIO]"));
        assert!(is_silence_hallucination(
            "Subtitles by the Amara.org community"
        ));
        assert!(!is_silence_hallucination(
            "Thank you for watching the kids today."
        ));
        assert!(!is_silence_hallucination("Thank you."));
    }

    #[test]
    fn joining_drops_hallucinated_and_blank_segments() {
        assert_eq!(
            join_segment_texts([
                "Hello there",
                "Thanks for watching!",
                "   ",
                "General Kenobi."
            ]),
            "Hello there general Kenobi."
        );
    }
}
