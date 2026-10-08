//! A shipped prompt the user never edited is upgraded to the newest default on
//! load. An edited one is kept, and records which shipped default it was last
//! reconciled with so the user can be told when a newer default exists.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CleanupPrompt {
    pub id: String,
    pub title: String,
    pub prompt: String,
    /// Lowercase SHA-256 of the shipped default text this prompt was last
    /// reconciled with. Absent for custom prompts and legacy documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_hash: Option<String>,
}

impl CleanupPrompt {
    /// Returns the current shipped text for a built-in prompt the user edited
    /// before that text shipped and has not reviewed since.
    #[must_use]
    pub fn newer_default(&self) -> Option<&'static str> {
        let built_in = built_in(&self.id)?;
        let current_hash = prompt_hash(built_in.prompt);
        (self.prompt != built_in.prompt && self.default_hash.as_ref() != Some(&current_hash))
            .then_some(built_in.prompt)
    }

    /// Whether the prompt asks for a rewrite in a new tone rather than a cleanup: true only for
    /// the built-in Make Professional prompt.
    #[must_use]
    pub fn rewrites(&self) -> bool {
        self.id == PROFESSIONAL_PROMPT_ID
    }

    /// Records that the user reviewed the current shipped default, keeping
    /// their own text. Custom prompts have no default and are unchanged.
    pub fn acknowledge_current_default(&mut self) {
        if let Some(built_in) = built_in(&self.id) {
            self.default_hash = Some(prompt_hash(built_in.prompt));
        }
    }
}

struct BuiltInPrompt {
    id: &'static str,
    title: &'static str,
    prompt: &'static str,
    /// Hashes of every earlier shipped text, so an unedited copy of any of
    /// them is recognized and upgraded.
    earlier_hashes: &'static [&'static str],
}

/// The built-in Clean Up & Format prompt, the one edit the local cleanup model does. Still
/// `fix_grammar`, so saved and imported settings keep pointing at it.
pub const CLEAN_UP_PROMPT_ID: &str = "fix_grammar";

const PROFESSIONAL_PROMPT_ID: &str = "professional";

const BUILT_IN_PROMPTS: [BuiltInPrompt; 2] = [
    BuiltInPrompt {
        id: CLEAN_UP_PROMPT_ID,
        title: "Clean Up & Format",
        prompt: "Clean up the dictation and lay it out the way the speaker would have written it. Remove filler words and verbal tics, such as 'um', 'uh', 'like', 'you know', 'I mean', 'basically', 'actually', 'honestly', 'literally', 'oh my god', 'dude', and 'so' or 'okay' opening a sentence, along with stutters, repeated words, and false starts, and write 'I was like' as 'I said'. Write 'wanna', 'gonna', 'gotta', 'kinda', and 'sorta' out as 'want to', 'going to', 'got to', 'kind of', and 'sort of'. When the speaker clearly corrects themselves, keep only the corrected version, and drop the correction cue, such as 'wait', 'oh wait', 'no', 'scratch that', or 'actually', together with the abandoned words. Keep hedges, such as 'maybe', 'probably', 'I think', 'I guess', and 'I believe', with whoever the speaker says they came from. Keep the speaker's reasons, such as 'because', 'since', or 'so that' and what follows, including the reason they give for the option they corrected to. Put quoted speech and the titles of talks, books, and the like in quotation marks, such as 'my partner said, \"We're astronauts.\"' and 'Topic is \"Intro to Kubernetes for Non-Engineers.\"'. Fix grammar, spelling, punctuation, capitalization, and obvious misheard words, but keep the speaker's own words, tone, and register: do not make casual speech formal, and do not shorten sentences into notes. Choose the layout from what the speaker is making. When they are making something to use, such as a shopping, errand, to-do, or packing list, steps, or a recipe, lay it out as that thing, even with only two items. This holds inside a message too: when a message asks someone to get or bring several things, lay out those things as a list and keep the rest of the message as sentences. Put a short lead-in or label taken from their words on its own line before the list, such as 'We need:' or 'Grocery list:', and use separate labels when they group items, such as 'Pharmacy:'. Start each item with '- ', or number ordered steps '1.', '2.', '3.'. A recipe may get a plain title line and 'Ingredients:' and 'Instructions:' sections, keeping anything else the speaker says about it as sentences. When a long list holds different kinds of things, such as food, gear, and toiletries for a trip, group them under short labels, such as 'Food:' and 'Gear:'. Meeting notes get a title line, an 'Attendees:' line that includes the speaker, a short label line for each topic, and '- ' items for action items. Tidy each list item into a short entry and move details into parentheses: 'the Tillamook cheddar sharp cheddar I mean the two pound block' becomes 'Tillamook sharp cheddar (2 lb block)'. Put the reason for an item in parentheses after it, such as '3 bags of carrots (Joaquin is bringing his kids)'. Asides with no content, such as mic checks, thinking aloud, and 'anyway', may go, but keep every item, quantity, brand, size, store, deadline, constraint, hedge, and reason. When the speaker goes through several separate points of the same kind, such as requirements, findings, or measurements in a report, a clinical note, or a meeting, put them as '- ' items under a short label, such as 'User requirements:' or 'Growth parameters:'. Everything else stays in sentences: messages, emails, stories, opinions, and explanations, even when they mention several things in passing or give several details about one thing. Break a long dictation into paragraphs only where it moves to a different topic, and keep a story or message about one thing as one paragraph. Lay out a dictated poem or spoken-word piece in verse lines. Write amounts, counts, quantities, measurements, dates, times, and percentages in digits, such as '$500' for 'five hundred dollars', '2 bags of ice' for 'two bags of ice', '12 chef knives' for 'twelve chef knives', and 'the 15th' for 'the fifteenth'. Turn spoken punctuation, layout commands such as 'new paragraph' or 'bullet point', and spelled-out symbols in code and commands into the real thing, such as '--help' for 'dash dash help' and '{' for 'curly brace'. Write names of brands, products, and organizations with their usual spelling and capitalization, such as 'CeraVe' and 'DevOps', write acronyms in capitals, such as 'SKU' and 'ETA', and write units, symbols, and technical notation the way they are written, such as '0.979 Å' and '25 °C'. Keep code, paths, and URLs exactly. Do not use em dashes: use a comma, period, colon, or parentheses instead, though hyphens in compound words and en dashes in number ranges are fine. Do not use bold, italics, backticks, '#' headings, or code fences, and add nothing the speaker did not say. If nothing needs fixing, return the text unchanged.",
        earlier_hashes: &[
            // The 0.1.x Python release text.
            "cefcd98f60fafef6f58072f72224418153be3558b0edd229af816f73c94d9489",
        ],
    },
    BuiltInPrompt {
        id: PROFESSIONAL_PROMPT_ID,
        title: "Make Professional",
        prompt: "Rewrite the following text in a professional, formal tone suitable for business communication. Remove all filler words (such as 'um', 'uh', 'like') and resolve any misspoken phrases or self-corrections to retain only the final intended information. Preserve the original meaning. Only output the rewritten text, nothing else.",
        earlier_hashes: &[],
    },
];

fn built_in(id: &str) -> Option<&'static BuiltInPrompt> {
    BUILT_IN_PROMPTS.iter().find(|prompt| prompt.id == id)
}

fn prompt_hash(text: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    Sha256::digest(text.as_bytes())
        .iter()
        .fold(String::with_capacity(64), |mut encoded, byte| {
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
            encoded
        })
}

#[must_use]
pub fn default_prompts() -> Vec<CleanupPrompt> {
    BUILT_IN_PROMPTS
        .iter()
        .map(|built_in| CleanupPrompt {
            id: built_in.id.to_owned(),
            title: built_in.title.to_owned(),
            prompt: built_in.prompt.to_owned(),
            default_hash: Some(prompt_hash(built_in.prompt)),
        })
        .collect()
}

pub(crate) fn upgrade_unedited_defaults(prompts: &mut [CleanupPrompt]) {
    for prompt in prompts {
        let Some(built_in) = built_in(&prompt.id) else {
            continue;
        };
        if prompt.prompt == built_in.prompt
            || built_in
                .earlier_hashes
                .contains(&prompt_hash(&prompt.prompt).as_str())
        {
            built_in.title.clone_into(&mut prompt.title);
            built_in.prompt.clone_into(&mut prompt.prompt);
            prompt.default_hash = Some(prompt_hash(built_in.prompt));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CleanupPrompt, default_prompts, upgrade_unedited_defaults};

    fn prompt(id: &str, text: &str, default_hash: Option<String>) -> CleanupPrompt {
        CleanupPrompt {
            id: id.to_owned(),
            title: "Title".to_owned(),
            prompt: text.to_owned(),
            default_hash,
        }
    }

    #[test]
    fn edited_default_reports_a_newer_default_until_acknowledged() {
        let current = default_prompts().remove(0);
        let mut edited = prompt(&current.id, "My own cleanup rules.", None);
        upgrade_unedited_defaults(std::slice::from_mut(&mut edited));

        assert_eq!(edited.prompt, "My own cleanup rules.");
        assert_eq!(edited.newer_default(), Some(current.prompt.as_str()));
        edited.acknowledge_current_default();
        assert_eq!(edited.newer_default(), None);
        assert_eq!(edited.default_hash, current.default_hash);
    }

    #[test]
    fn current_and_custom_prompts_never_report_a_newer_default() {
        let current = default_prompts().remove(0);
        let unreconciled = prompt(&current.id, &current.prompt, None);
        let mut custom = prompt("meeting_notes", "Turn this into notes.", None);
        custom.acknowledge_current_default();

        assert_eq!(unreconciled.newer_default(), None);
        assert_eq!(custom.newer_default(), None);
        assert_eq!(custom.default_hash, None);
    }
}
