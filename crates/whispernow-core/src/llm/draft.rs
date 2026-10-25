//! The draft sent with each local cleanup request for speculative decoding: the transcript as
//! code can clean it (fillers dropped, spoken punctuation and numbers written as symbols and
//! digits, spelled-out letters joined, sentences capitalized). `llama-server` built with
//! `serving/llama-server-draft-text.patch` takes it as `draft_text`; a stock server ignores it.
//!
//! Must agree with the training drafter; both are tested against `training/serving/prompt_golden.json`.

const FILLERS: [&str; 8] = ["um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm"];
const SPOKEN: [(&str, &str); 6] = [
    ("period", "."),
    ("comma", ","),
    ("question mark", "?"),
    ("colon", ":"),
    ("exclamation point", "!"),
    ("full stop", "."),
];
const UNITS: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 8] = [
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

#[must_use]
pub(crate) fn preclean(transcript: &str) -> String {
    let (words, lower): (Vec<&str>, Vec<String>) = transcript
        .split_whitespace()
        .map(|word| (word, word.to_lowercase()))
        .filter(|(_, lower)| !FILLERS.contains(&lower.as_str()))
        .unzip();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        let word = lower[index].as_str();
        let two_word_mark = lower.get(index + 1).and_then(|next| spoken(&[word, next]));
        if let Some(mark) = two_word_mark.or_else(|| spoken(&[word])) {
            if let Some(last) = out.last_mut() {
                last.push_str(mark);
            }
            index += if two_word_mark.is_some() { 2 } else { 1 };
        } else if let Some((digits, word_count)) = number(&lower, index) {
            out.push(digits);
            index += word_count;
        } else if is_letter(word)
            && word != "a"
            && word != "i"
            && words
                .get(index + 1)
                .is_some_and(|next| next.chars().count() == 1)
        {
            let letter_count = words[index..]
                .iter()
                .take_while(|word| is_letter(word))
                .count()
                .max(1);
            out.push(words[index..index + letter_count].concat().to_uppercase());
            index += letter_count;
        } else {
            out.push(if word == "i" {
                "I".to_owned()
            } else if word.starts_with("i'") {
                format!("I{}", after_first_character(words[index]))
            } else {
                words[index].to_owned()
            });
            index += 1;
        }
    }
    let mut sentence_start = true;
    for word in &mut out {
        if sentence_start {
            let mut characters = word.chars();
            if let Some(first) = characters.next() {
                *word = first.to_uppercase().chain(characters).collect();
            }
        }
        sentence_start = word.ends_with(['.', '?', '!']);
    }
    out.join(" ")
}

/// A spoken number starting at lowercase `words[start]`: its digits and the words it used.
fn number(words: &[String], start: usize) -> Option<(String, usize)> {
    let first = words[start].as_str();
    if unit(first).is_none() && tens(first).is_none() {
        return None;
    }
    // A run of single digits ("zero four seven nine") is read digit by digit.
    let digit_run: String = words[start..]
        .iter()
        .map_while(|word| digit(word))
        .collect();
    if digit_run.len() >= 3 || (digit_run.len() >= 2 && first == "zero") {
        let word_count = digit_run.len();
        return Some((digit_run, word_count));
    }
    let (mut total, mut current, mut end) = (0_u64, 0_u64, start);
    while let Some(word) = words.get(end) {
        let word = word.as_str();
        if let Some(value) = unit(word) {
            // "nineteen nine" is two numbers
            if current % 100 != 0 && (current % 100 < 20 || current % 10 != 0) {
                break;
            }
            current = current.saturating_add(value);
        } else if let Some(value) = tens(word) {
            current = current.saturating_add(value);
        } else if let Some(scale) = scale(word).filter(|_| current != 0 || total != 0) {
            if scale == 100 {
                current = current.saturating_mul(100);
            } else {
                total = total.saturating_add(current.saturating_mul(scale));
                current = 0;
            }
        } else if word != "and"
            || current < 100
            || !words
                .get(end + 1)
                .is_some_and(|next| unit(next).is_some() || tens(next).is_some())
        {
            // "and" carries on only after a hundred: "one hundred and five"
            break;
        }
        end += 1;
    }
    let mut text = total.saturating_add(current).to_string();
    if words.get(end).is_some_and(|word| word == "point") {
        let decimals: String = words[end + 1..]
            .iter()
            .map_while(|word| digit(word))
            .collect();
        if !decimals.is_empty() {
            text.push('.');
            text.push_str(&decimals);
            end += 1 + decimals.len();
        }
    }
    Some((text, end - start))
}

fn spoken(words: &[&str]) -> Option<&'static str> {
    SPOKEN
        .iter()
        .find(|(spoken, _)| spoken.split(' ').eq(words.iter().copied()))
        .map(|(_, mark)| *mark)
}

fn unit(word: &str) -> Option<u64> {
    UNITS
        .iter()
        .position(|unit| *unit == word)
        .and_then(|value| u64::try_from(value).ok())
}

fn tens(word: &str) -> Option<u64> {
    TENS.iter()
        .position(|tens| *tens == word)
        .and_then(|index| u64::try_from(index + 2).ok())
        .map(|tens| tens * 10)
}

fn scale(word: &str) -> Option<u64> {
    match word {
        "hundred" => Some(100),
        "thousand" => Some(1_000),
        "million" => Some(1_000_000),
        _ => None,
    }
}

fn digit(word: &str) -> Option<char> {
    unit(word)
        .and_then(|value| u32::try_from(value).ok())
        .and_then(|value| char::from_digit(value, 10))
}

fn is_letter(word: &str) -> bool {
    let mut characters = word.chars();
    matches!((characters.next(), characters.next()), (Some(character), None) if character.is_alphabetic())
}

fn after_first_character(word: &str) -> &str {
    word.char_indices()
        .nth(1)
        .map_or("", |(index, _)| &word[index..])
}

#[cfg(test)]
mod tests {
    use super::preclean;

    #[test]
    fn numbers_too_large_for_u64_saturate_instead_of_overflowing() {
        let hundreds = "one".to_owned() + &" hundred".repeat(20);
        assert_eq!(preclean(&hundreds), u64::MAX.to_string());
    }

    #[test]
    fn a_point_without_a_digit_after_it_stays_a_word() {
        assert_eq!(preclean("five point eleven"), "5 point 11");
    }
}
