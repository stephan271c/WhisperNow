//! The word aligner that turns the pre-cleaned transcript (see `draft`) into drafts for
//! speculative decoding.
//!
//! The words generated so far are matched against the draft's words, ignoring case and
//! punctuation, and the words after the match are the draft. The pointer is only the last
//! match: each pass re-derives it from the output, so a rejected draft needs no undo.
//!
//! A byte-for-byte port of `server_text_draft` in the training side's
//! `serving/llama-server-draft-text.patch`, so the in-app engine drafts exactly what the
//! patched `llama-server` drafts. Like the C++, it works on bytes with the C locale's
//! character classes: ASCII letters and digits are word characters, and so is every byte of
//! a multi-byte character.

/// How many words a draft carries at most, before the engine cuts it to its token budget.
const DRAFT_WORDS: usize = 24;
/// How far behind the pointer a match may be found before the aligner counts as lost.
const LOOK_BEHIND: usize = 8;
/// How far ahead of the pointer a match may be found before the aligner counts as lost.
const LOOK_AHEAD: usize = 60;

#[derive(Clone, Debug, Default)]
pub struct WordAligner {
    words: Vec<Vec<u8>>,
    keys: Vec<Vec<u8>>,
    pointer: usize,
}

impl WordAligner {
    #[must_use]
    pub fn new(text: &str) -> Self {
        let words = split(text.as_bytes());
        let keys = words.iter().map(|word| norm(word)).collect();
        Self {
            words,
            keys,
            pointer: 0,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The draft text to follow `output`, the reply generated so far, or nothing when the
    /// output's last words are not found in the draft.
    pub fn draft(&mut self, output: &[u8]) -> Vec<u8> {
        let Some(&last_byte) = output.last() else {
            return self.join(0, self.words.len(), false);
        };
        let parts = split(output);
        let Some(last_part) = parts.last() else {
            return Vec::new();
        };
        let keys: Vec<Vec<u8>> = parts
            .iter()
            .map(|part| norm(part))
            .filter(|key| !key.is_empty())
            .collect();
        // The last word taken as complete: draft the words after it.
        if let Some(position) = self
            .find(&keys)
            .filter(|&position| position < self.words.len())
        {
            self.pointer = position;
            let mut rest = self.join(position, position + DRAFT_WORDS, false);
            // C's `strchr` also finds the terminating NUL.
            let ends_sentence = last_part
                .last()
                .is_some_and(|byte| b".?!:\0".contains(byte));
            if ends_sentence && let Some(first) = rest.first_mut() {
                first.make_ascii_uppercase();
            }
            if last_byte == b' ' || last_byte == b'\n' {
                return rest;
            }
            let mut draft = Vec::with_capacity(rest.len() + 1);
            draft.push(b' ');
            draft.append(&mut rest);
            return draft;
        }
        // The last word unfinished: complete it from the next word.
        if last_byte.is_ascii_alphanumeric() && keys.len() > 1 {
            let position = self
                .find(&keys[..keys.len() - 1])
                .filter(|&position| position < self.words.len());
            if let Some(position) = position {
                self.pointer = position;
                let word = &self.words[position];
                if word.len() > last_part.len()
                    && word[..last_part.len()].eq_ignore_ascii_case(last_part)
                {
                    let mut draft = word[last_part.len()..].to_vec();
                    draft.extend(self.join(position + 1, position + DRAFT_WORDS, true));
                    return draft;
                }
            }
        }
        Vec::new()
    }

    /// The position after the match of the last words of `keys`: three, two or one words
    /// near the pointer, else three or two words anywhere, nearest the pointer.
    fn find(&self, keys: &[Vec<u8>]) -> Option<usize> {
        let low = self.pointer.saturating_sub(LOOK_BEHIND);
        let high = self.keys.len().min(self.pointer + LOOK_AHEAD);
        for length in [3, 2, 1] {
            if keys.len() < length {
                continue;
            }
            if let Some(position) = self.nearest(keys, length, low, high) {
                return Some(position + length);
            }
        }
        for length in [3, 2] {
            if keys.len() < length {
                continue;
            }
            if let Some(position) = self.nearest(keys, length, 0, self.keys.len()) {
                return Some(position + length);
            }
        }
        None
    }

    /// The match of the last `length` keys within `[low, high)` nearest the pointer, the
    /// earliest on a tie.
    fn nearest(&self, keys: &[Vec<u8>], length: usize, low: usize, high: usize) -> Option<usize> {
        let tail = &keys[keys.len() - length..];
        let mut best: Option<usize> = None;
        let mut position = low;
        while position + length <= high {
            if self.keys[position..position + length] == *tail
                && best.is_none_or(|best| {
                    position.abs_diff(self.pointer) < best.abs_diff(self.pointer)
                })
            {
                best = Some(position);
            }
            position += 1;
        }
        best
    }

    fn join(&self, from: usize, to: usize, lead: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for index in from..to.min(self.words.len()) {
            if lead || index > from {
                out.push(b' ');
            }
            out.extend_from_slice(&self.words[index]);
        }
        out
    }
}

/// A word's key: its letters, digits, underscores and non-ASCII bytes, lowercased.
fn norm(word: &[u8]) -> Vec<u8> {
    word.iter()
        .filter(|byte| byte.is_ascii_alphanumeric() || **byte == b'_' || **byte >= 0x80)
        .map(u8::to_ascii_lowercase)
        .collect()
}

/// Splits on C's `isspace`: space, tab, newline, vertical tab, form feed and carriage return.
fn split(text: &[u8]) -> Vec<Vec<u8>> {
    text.split(|byte| matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r'))
        .filter(|word| !word.is_empty())
        .map(<[u8]>::to_vec)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::WordAligner;

    fn draft(aligner: &mut WordAligner, output: &str) -> String {
        String::from_utf8(aligner.draft(output.as_bytes())).expect("drafts stay UTF-8")
    }

    #[test]
    fn empty_inputs_keep_their_distinct_draft_behavior() {
        let mut aligner = WordAligner::new("  So can you\tsend it ");
        assert_eq!(draft(&mut aligner, ""), "So can you send it");

        let mut aligner = WordAligner::new(" \n ");
        assert!(aligner.is_empty());
        assert_eq!(draft(&mut aligner, ""), "");
        assert_eq!(draft(&mut aligner, "So"), "");

        let mut aligner = WordAligner::new("So can you send it");
        assert_eq!(draft(&mut aligner, " \n"), "");
    }

    #[test]
    fn the_words_after_the_match_follow_with_a_space() {
        let mut aligner = WordAligner::new("So can you send me the report by Friday");
        assert_eq!(
            draft(&mut aligner, "So, can you"),
            " send me the report by Friday"
        );
        assert_eq!(
            draft(&mut aligner, "So, can you send "),
            "me the report by Friday"
        );
        assert_eq!(
            draft(&mut aligner, "So, can you send\n"),
            "me the report by Friday"
        );
    }

    #[test]
    fn case_and_punctuation_are_ignored_when_matching() {
        let mut aligner = WordAligner::new("the API's limit is 5 per hour");
        assert_eq!(draft(&mut aligner, "The APIs limit"), " is 5 per hour");
    }

    #[test]
    fn a_sentence_end_capitalizes_the_draft() {
        let mut aligner = WordAligner::new("we met today the team agreed");
        assert_eq!(draft(&mut aligner, "We met today."), " The team agreed");
        assert_eq!(draft(&mut aligner, "We met today:"), " The team agreed");
        assert_eq!(draft(&mut aligner, "We met today,"), " the team agreed");
    }

    #[test]
    fn an_unfinished_word_is_completed_from_the_next_word() {
        let mut aligner = WordAligner::new("please schedule the meeting tomorrow");
        // "sched" is not a word of the text, so the words before it place the pointer.
        assert_eq!(
            draft(&mut aligner, "Please sched"),
            "ule the meeting tomorrow"
        );
    }

    #[test]
    fn an_unfinished_word_that_is_not_a_prefix_drafts_nothing() {
        let mut aligner = WordAligner::new("please schedule the meeting tomorrow");
        assert_eq!(draft(&mut aligner, "Please book"), "");
        // An output ending in punctuation is never completed.
        assert_eq!(draft(&mut aligner, "Please sched-"), "");
    }

    #[test]
    fn the_last_match_counts_and_the_end_of_the_text_drafts_nothing() {
        let mut aligner = WordAligner::new("one two three");
        assert_eq!(draft(&mut aligner, "one two three"), "");
    }

    #[test]
    fn drafts_carry_at_most_24_words() {
        let text: Vec<String> = (0..40).map(|index| format!("w{index}")).collect();
        let mut aligner = WordAligner::new(&text.join(" "));
        let drafted = draft(&mut aligner, "w0");
        assert_eq!(drafted, format!(" {}", text[1..25].join(" ")));
    }

    #[test]
    fn repeated_phrases_match_nearest_the_pointer() {
        let mut aligner = WordAligner::new("a b c x a b c y a b c z");
        assert_eq!(draft(&mut aligner, "a b c"), " x a b c y a b c z");
        assert_eq!(draft(&mut aligner, "a b c x a b c"), " y a b c z");
        // The pointer is now after the second "a b c"; the third is nearer than the first.
        assert_eq!(draft(&mut aligner, "a b c y a b c"), " z");
    }

    #[test]
    fn a_lost_aligner_searches_the_whole_text_on_two_words() {
        let mut words: Vec<String> = (0..100).map(|index| format!("w{index}")).collect();
        words[90] = "far".to_owned();
        words[91] = "away".to_owned();
        let mut aligner = WordAligner::new(&words.join(" "));
        // One word beyond the look-ahead is not enough...
        assert_eq!(draft(&mut aligner, "zzz far"), "");
        // ...two are.
        assert_eq!(
            draft(&mut aligner, "far away"),
            " w92 w93 w94 w95 w96 w97 w98 w99"
        );
    }

    #[test]
    fn non_ascii_bytes_are_word_characters() {
        let mut aligner = WordAligner::new("café olé naïve test");
        assert_eq!(draft(&mut aligner, "Café olé"), " naïve test");
        // Non-ASCII letters are not lowercased, as in the C locale.
        assert_eq!(draft(&mut aligner, "CAFÉ"), "");
    }
}
