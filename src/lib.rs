use std::collections::HashMap;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Violation {
    #[serde(rename = "type")]
    pub violation_type: String,
    pub rule: String,
    #[serde(rename = "match")]
    pub match_text: String,
    pub context: String,
    pub penalty: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisResult {
    pub score: i32,
    pub band: String,
    pub word_count: usize,
    pub violations: Vec<Violation>,
    pub counts: HashMap<String, usize>,
    pub total_penalty: i32,
    pub weighted_sum: f64,
    pub density: f64,
    pub advice: Vec<String>,
}

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct Hyperparameters {
    concentration_alpha: f64,
    decay_lambda: f64,
    claude_categories: &'static [&'static str],
    context_window_chars: usize,
    short_text_word_count: usize,
    repeated_ngram_min_n: usize,
    repeated_ngram_max_n: usize,
    repeated_ngram_min_count: usize,
    slop_word_penalty: i32,
    slop_phrase_penalty: i32,
    structural_bold_header_min: usize,
    structural_bold_header_penalty: i32,
    structural_bullet_run_min: usize,
    structural_bullet_run_penalty: i32,
    triadic_record_cap: usize,
    triadic_penalty: i32,
    triadic_advice_min: usize,
    tone_penalty: i32,
    sentence_opener_penalty: i32,
    weasel_penalty: i32,
    ai_disclosure_penalty: i32,
    placeholder_penalty: i32,
    rhythm_min_sentences: usize,
    rhythm_cv_threshold: f64,
    rhythm_penalty: i32,
    em_dash_words_basis: f64,
    em_dash_density_threshold: f64,
    em_dash_penalty: i32,
    contrast_record_cap: usize,
    contrast_penalty: i32,
    contrast_advice_min: usize,
    setup_resolution_record_cap: usize,
    setup_resolution_penalty: i32,
    colon_words_basis: f64,
    colon_density_threshold: f64,
    colon_density_penalty: i32,
    pithy_max_sentence_words: usize,
    pithy_record_cap: usize,
    pithy_penalty: i32,
    bullet_density_threshold: f64,
    bullet_density_penalty: i32,
    blockquote_min_lines: usize,
    blockquote_free_lines: usize,
    blockquote_cap: usize,
    blockquote_penalty_step: i32,
    bold_bullet_run_min: usize,
    bold_bullet_run_penalty: i32,
    horizontal_rule_min: usize,
    horizontal_rule_penalty: i32,
    phrase_reuse_record_cap: usize,
    phrase_reuse_penalty: i32,
    density_words_basis: f64,
    score_min: i32,
    score_max: i32,
    band_clean_min: i32,
    band_light_min: i32,
    band_moderate_min: i32,
    band_heavy_min: i32,
}

static HP: Hyperparameters = Hyperparameters {
    concentration_alpha: 2.5,
    decay_lambda: 0.04,
    claude_categories: &["contrast_pairs", "pithy_fragment", "setup_resolution"],
    context_window_chars: 60,
    short_text_word_count: 10,
    repeated_ngram_min_n: 4,
    repeated_ngram_max_n: 8,
    repeated_ngram_min_count: 3,
    slop_word_penalty: -2,
    slop_phrase_penalty: -3,
    structural_bold_header_min: 3,
    structural_bold_header_penalty: -5,
    structural_bullet_run_min: 6,
    structural_bullet_run_penalty: -3,
    triadic_record_cap: 5,
    triadic_penalty: -1,
    triadic_advice_min: 3,
    tone_penalty: -3,
    sentence_opener_penalty: -2,
    weasel_penalty: -2,
    ai_disclosure_penalty: -10,
    placeholder_penalty: -5,
    rhythm_min_sentences: 5,
    rhythm_cv_threshold: 0.3,
    rhythm_penalty: -5,
    em_dash_words_basis: 150.0,
    em_dash_density_threshold: 1.0,
    em_dash_penalty: -3,
    contrast_record_cap: 5,
    contrast_penalty: -1,
    contrast_advice_min: 2,
    setup_resolution_record_cap: 5,
    setup_resolution_penalty: -3,
    colon_words_basis: 150.0,
    colon_density_threshold: 1.5,
    colon_density_penalty: -3,
    pithy_max_sentence_words: 6,
    pithy_record_cap: 3,
    pithy_penalty: -2,
    bullet_density_threshold: 0.40,
    bullet_density_penalty: -8,
    blockquote_min_lines: 3,
    blockquote_free_lines: 2,
    blockquote_cap: 4,
    blockquote_penalty_step: -3,
    bold_bullet_run_min: 3,
    bold_bullet_run_penalty: -5,
    horizontal_rule_min: 4,
    horizontal_rule_penalty: -3,
    phrase_reuse_record_cap: 5,
    phrase_reuse_penalty: -1,
    density_words_basis: 1000.0,
    score_min: 0,
    score_max: 100,
    band_clean_min: 80,
    band_light_min: 60,
    band_moderate_min: 40,
    band_heavy_min: 20,
};

// ---------------------------------------------------------------------------
// Compiled patterns
// ---------------------------------------------------------------------------

static SLOP_WORD_RE: Lazy<Regex> = Lazy::new(|| {
    let words = [
        // Adjectives
        "crucial",
        "groundbreaking",
        "pivotal",
        "paramount",
        "seamless",
        "holistic",
        "multifaceted",
        "meticulous",
        "profound",
        "comprehensive",
        "invaluable",
        "notable",
        "noteworthy",
        "game-changing",
        "revolutionary",
        "pioneering",
        "visionary",
        "formidable",
        "quintessential",
        "unparalleled",
        "stunning",
        "breathtaking",
        "captivating",
        "nestled",
        "robust",
        "innovative",
        "cutting-edge",
        "impactful",
        // Verbs
        "delve",
        "delves",
        "delved",
        "delving",
        "embark",
        "embrace",
        "elevate",
        "foster",
        "harness",
        "unleash",
        "unlock",
        "orchestrate",
        "streamline",
        "transcend",
        "navigate",
        "underscore",
        "showcase",
        "leverage",
        "ensuring",
        "highlighting",
        "emphasizing",
        "reflecting",
        // Nouns
        "landscape",
        "tapestry",
        "journey",
        "paradigm",
        "testament",
        "trajectory",
        "nexus",
        "symphony",
        "spectrum",
        "odyssey",
        "pinnacle",
        "realm",
        "intricacies",
        // Hedging
        "notably",
        "importantly",
        "furthermore",
        "additionally",
        "particularly",
        "significantly",
        "interestingly",
        "remarkably",
        "surprisingly",
        "fascinatingly",
        "moreover",
        "however",
        "overall",
    ];
    let alt = words
        .iter()
        .map(|w| regex::escape(w))
        .collect::<Vec<_>>()
        .join("|");
    Regex::new(&format!("(?i)\\b({alt})\\b")).unwrap()
});

static SLOP_PHRASES: Lazy<Vec<Regex>> = Lazy::new(|| {
    let phrases = [
        "it's worth noting",
        "it's important to note",
        "this is where things get interesting",
        "here's the thing",
        "at the end of the day",
        "in today's fast-paced",
        "as technology continues to",
        "something shifted",
        "everything changed",
        "the answer? it's simpler than you think",
        "what makes this work is",
        "this is exactly",
        "let's break this down",
        "let's dive in",
        "in this post, we'll explore",
        "in this article, we'll",
        "let me know if",
        "would you like me to",
        "i hope this helps",
        "as mentioned earlier",
        "as i mentioned",
        "without further ado",
        "on the other hand",
        "in addition",
        "in summary",
        "in conclusion",
        "you might be wondering",
        "the obvious question is",
        "no discussion would be complete",
        "great question",
        "that's a great",
        "if you want, i can",
        "i can adapt this",
        "i can make this",
        "here are some options",
        "here are a few options",
        "would you prefer",
        "shall i",
        "if you'd like, i can",
        "i can also",
        "in other words",
        "put differently",
        "that is to say",
        "to put it simply",
        "to put it another way",
        "what this means is",
        "the takeaway is",
        "the bottom line is",
        "the key takeaway",
        "the key insight",
    ];
    phrases
        .iter()
        .map(|p| Regex::new(&format!("(?i){}", regex::escape(p))).unwrap())
        .collect()
});

static NOT_JUST_BUT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)not (just|only) .{1,40}, but (also )?").unwrap());

static BOLD_HEADER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\*\*[^*]+[.:]\*\*\s+\S").unwrap());

static BULLET_LINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\s*[-*]\s|\s*\d+\.\s)").unwrap());

static TRIADIC_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\w+, \w+, and \w+").unwrap());

static META_COMM_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"(?i)would you like").unwrap(),
        Regex::new(r"(?i)let me know if").unwrap(),
        Regex::new(r"(?i)as mentioned").unwrap(),
        Regex::new(r"(?i)i hope this").unwrap(),
        Regex::new(r"(?i)feel free to").unwrap(),
        Regex::new(r"(?i)don't hesitate to").unwrap(),
    ]
});

static FALSE_NARRATIVITY_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"(?i)then something interesting happened").unwrap(),
        Regex::new(r"(?i)this is where things get interesting").unwrap(),
        Regex::new(r"(?i)that's when everything changed").unwrap(),
    ]
});

static SENTENCE_OPENER_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"(?im)(?:^|[.!?]\s+)(certainly[,! ])").unwrap(),
        Regex::new(r"(?im)(?:^|[.!?]\s+)(absolutely[,! ])").unwrap(),
    ]
});

static WEASEL_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"(?i)\bsome critics argue\b").unwrap(),
        Regex::new(r"(?i)\bmany believe\b").unwrap(),
        Regex::new(r"(?i)\bexperts suggest\b").unwrap(),
        Regex::new(r"(?i)\bstudies show\b").unwrap(),
        Regex::new(r"(?i)\bsome argue\b").unwrap(),
        Regex::new(r"(?i)\bit is widely believed\b").unwrap(),
        Regex::new(r"(?i)\bresearch suggests\b").unwrap(),
    ]
});

static AI_DISCLOSURE_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"(?i)\bas an ai\b").unwrap(),
        Regex::new(r"(?i)\bas a language model\b").unwrap(),
        Regex::new(r"(?i)\bi don't have personal\b").unwrap(),
        Regex::new(r"(?i)\bi cannot browse\b").unwrap(),
        Regex::new(r"(?i)\bup to my last training\b").unwrap(),
        Regex::new(r"(?i)\bas of my (last |knowledge )?cutoff\b").unwrap(),
        Regex::new(r"(?i)\bi'm just an? ai\b").unwrap(),
    ]
});

static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)\[insert [^\]]*\]|\[describe [^\]]*\]|\[url [^\]]*\]|\[your [^\]]*\]|\[todo[^\]]*\]",
    )
    .unwrap()
});

static SENTENCE_SPLIT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"[.!?]["'\u{201D}\u{2019})\]]*(?:\s|$)"#).unwrap());

static EM_DASH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\u{2014}| -- ").unwrap());

static CONTRAST_PAIR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\w+), not (\w+)\b").unwrap());

static SETUP_RESOLUTION_A_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)\b(this|that|these|those|it|they|we)\s+",
        r"(isn't|aren't|wasn't|weren't|doesn't|don't|didn't|hasn't|haven't|won't|can't|couldn't|shouldn't",
        r"|is\s+not|are\s+not|was\s+not|were\s+not|does\s+not|do\s+not|did\s+not",
        r"|has\s+not|have\s+not|will\s+not|cannot|could\s+not|should\s+not)\b",
        r".{0,80}[.;:,]\s*",
        r"(it's|they're|that's|he's|she's|we're|it\s+is|they\s+are|that\s+is|this\s+is",
        r"|these\s+are|those\s+are|he\s+is|she\s+is|we\s+are|what's|what\s+is",
        r"|the\s+real|the\s+actual|instead|rather)",
    ))
    .unwrap()
});

static SETUP_RESOLUTION_B_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"(?i)\b(it's|that's|this\s+is|they're|he's|she's|we're)\s+not\b",
        r".{0,80}[.;:,]\s*",
        r"(it's|they're|that's|he's|she's|we're|it\s+is|they\s+are|that\s+is|this\s+is",
        r"|these\s+are|those\s+are|what's|what\s+is|the\s+real|the\s+actual|instead|rather)",
    ))
    .unwrap()
});

static ELABORATION_COLON_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r": [a-z]").unwrap());

static FENCED_CODE_BLOCK_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```.*?```").unwrap());

static JSON_COLON_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#": ["{\[\d]|: true|: false|: null"#).unwrap());

static PITHY_PIVOT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i),\s+(?:but|yet|and|not|or)\b").unwrap());

static BULLET_DENSITY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*[-*]\s|^\s*\d+[.)]\s").unwrap());

static BOLD_TERM_BULLET_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*[-*]\s+\*\*|^\s*\d+[.)]\s+\*\*").unwrap());

static HORIZONTAL_RULE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*(?:---+|\*\*\*+|___+)\s*$").unwrap());

static PUNCT_STRIP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[^\w]+|[^\w]+$").unwrap());

// We need a per-line header check (non-multiline)
static MD_HEADER_LINE_SINGLE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*#").unwrap());

// ---------------------------------------------------------------------------
// Stopwords
// ---------------------------------------------------------------------------

static STOPWORDS: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is", "it",
        "that", "this", "with", "as", "by", "from", "was", "were", "are", "be", "been", "has",
        "have", "had", "not", "no", "do", "does", "did", "will", "would", "could", "should", "can",
        "may", "might", "if", "then", "than", "so", "up", "out", "about", "into", "over", "after",
        "before", "between", "through", "just", "also", "very", "more", "most", "some", "any",
        "each", "every", "all", "both", "few", "other", "such", "only", "own", "same", "too",
        "how", "what", "which", "who", "when", "where", "why",
    ]
    .into_iter()
    .collect()
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn context_around(text: &str, start: usize, end: usize, width: usize) -> String {
    let mid = (start + end) / 2;
    let half = width / 2;
    let ctx_start = mid.saturating_sub(half);
    let ctx_end = std::cmp::min(text.len(), mid + half);

    // Ensure we don't slice in the middle of a multi-byte char
    let ctx_start = snap_to_char_boundary(text, ctx_start, false);
    let ctx_end = snap_to_char_boundary(text, ctx_end, true);

    let snippet = text[ctx_start..ctx_end].replace('\n', " ");
    let prefix = if ctx_start > 0 { "..." } else { "" };
    let suffix = if ctx_end < text.len() { "..." } else { "" };
    format!("{prefix}{snippet}{suffix}")
}

/// Snap a byte offset to a valid char boundary.
/// If `forward` is true, snap forward; otherwise snap backward.
fn snap_to_char_boundary(text: &str, pos: usize, forward: bool) -> usize {
    if pos >= text.len() {
        return text.len();
    }
    if text.is_char_boundary(pos) {
        return pos;
    }
    if forward {
        let mut p = pos;
        while p < text.len() && !text.is_char_boundary(p) {
            p += 1;
        }
        p
    } else {
        let mut p = pos;
        while p > 0 && !text.is_char_boundary(p) {
            p -= 1;
        }
        p
    }
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn strip_code_blocks(text: &str) -> String {
    FENCED_CODE_BLOCK_RE.replace_all(text, "").into_owned()
}

#[derive(Debug)]
struct NgramResult {
    phrase: String,
    count: usize,
    n: usize,
}

fn find_repeated_ngrams(text: &str) -> Vec<NgramResult> {
    let min_n = HP.repeated_ngram_min_n;
    let max_n = HP.repeated_ngram_max_n;
    let min_count = HP.repeated_ngram_min_count;

    // Tokenize
    let raw_tokens: Vec<&str> = text.split_whitespace().collect();
    let tokens: Vec<String> = raw_tokens
        .iter()
        .filter_map(|t| {
            let stripped = PUNCT_STRIP_RE.replace_all(t, "").to_lowercase();
            if stripped.is_empty() {
                None
            } else {
                Some(stripped)
            }
        })
        .collect();

    if tokens.len() < min_n {
        return vec![];
    }

    // Count n-grams
    let mut ngram_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for n in min_n..=max_n {
        if tokens.len() < n {
            continue;
        }
        for i in 0..=tokens.len() - n {
            let gram: Vec<String> = tokens[i..i + n].to_vec();
            *ngram_counts.entry(gram).or_insert(0) += 1;
        }
    }

    // Filter: count >= min_count AND not all stopwords
    let repeated: HashMap<Vec<String>, usize> = ngram_counts
        .into_iter()
        .filter(|(gram, count)| {
            *count >= min_count && !gram.iter().all(|w| STOPWORDS.contains(w.as_str()))
        })
        .collect();

    if repeated.is_empty() {
        return vec![];
    }

    // Suppress sub-n-grams
    let mut sorted_grams: Vec<(&Vec<String>, &usize)> = repeated.iter().collect();
    sorted_grams.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut to_remove: std::collections::HashSet<Vec<String>> = std::collections::HashSet::new();
    for i in 0..sorted_grams.len() {
        let longer = sorted_grams[i].0;
        let longer_str = longer.join(" ");
        let longer_count = *sorted_grams[i].1;
        for &(shorter, &shorter_count) in sorted_grams.iter().skip(i + 1) {
            if to_remove.contains(shorter) {
                continue;
            }
            let shorter_str = shorter.join(" ");
            if longer_str.contains(&shorter_str) && longer_count >= shorter_count {
                to_remove.insert(shorter.clone());
            }
        }
    }

    // Sort by (-length, -count)
    let mut result_grams: Vec<(&Vec<String>, &usize)> = repeated
        .iter()
        .filter(|(gram, _)| !to_remove.contains(*gram))
        .collect();
    result_grams.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| b.1.cmp(a.1)));

    result_grams
        .into_iter()
        .map(|(gram, &count)| NgramResult {
            phrase: gram.join(" "),
            count,
            n: gram.len(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Initial counts
// ---------------------------------------------------------------------------

fn initial_counts() -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for key in &[
        "slop_words",
        "slop_phrases",
        "structural",
        "tone",
        "weasel",
        "ai_disclosure",
        "placeholder",
        "rhythm",
        "em_dash",
        "contrast_pairs",
        "colon_density",
        "pithy_fragment",
        "setup_resolution",
        "bullet_density",
        "blockquote_density",
        "bold_bullet_list",
        "horizontal_rules",
        "phrase_reuse",
    ] {
        m.insert(key.to_string(), 0);
    }
    m
}

// ---------------------------------------------------------------------------
// Rule implementations
// ---------------------------------------------------------------------------

struct RuleOutput {
    violations: Vec<Violation>,
    advice: Vec<String>,
    count_deltas: HashMap<String, usize>,
}

impl RuleOutput {
    fn new() -> Self {
        Self {
            violations: Vec::new(),
            advice: Vec::new(),
            count_deltas: HashMap::new(),
        }
    }

    fn inc(&mut self, key: &str) {
        *self.count_deltas.entry(key.to_string()).or_insert(0) += 1;
    }
}

fn rule_slop_words(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;
    for m in SLOP_WORD_RE.find_iter(text) {
        let word = m.as_str().to_lowercase();
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "slop_word".to_string(),
            match_text: word.clone(),
            context: context_around(text, m.start(), m.end(), width),
            penalty: HP.slop_word_penalty,
        });
        out.advice.push(format!(
            "Replace '{word}' \u{2014} what specifically do you mean?"
        ));
        out.inc("slop_words");
    }
    out
}

fn rule_slop_phrases(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;

    for pat in SLOP_PHRASES.iter() {
        for m in pat.find_iter(text) {
            let phrase = m.as_str().to_lowercase();
            out.violations.push(Violation {
                violation_type: "Violation".to_string(),
                rule: "slop_phrase".to_string(),
                match_text: phrase.clone(),
                context: context_around(text, m.start(), m.end(), width),
                penalty: HP.slop_phrase_penalty,
            });
            out.advice.push(format!(
                "Cut '{phrase}' \u{2014} just state the point directly."
            ));
            out.inc("slop_phrases");
        }
    }

    for m in NOT_JUST_BUT_RE.find_iter(text) {
        let phrase = m.as_str().trim().to_lowercase();
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "slop_phrase".to_string(),
            match_text: phrase.clone(),
            context: context_around(text, m.start(), m.end(), width),
            penalty: HP.slop_phrase_penalty,
        });
        out.advice.push(format!(
            "Cut '{phrase}' \u{2014} just state the point directly."
        ));
        out.inc("slop_phrases");
    }
    out
}

fn rule_structural(text: &str, lines: &[&str]) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;

    // Bold headers
    let bold_matches: Vec<_> = BOLD_HEADER_RE.find_iter(text).collect();
    if bold_matches.len() >= HP.structural_bold_header_min {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "structural".to_string(),
            match_text: "bold_header_explanation".to_string(),
            context: format!(
                "Found {} instances of **Bold.** pattern",
                bold_matches.len()
            ),
            penalty: HP.structural_bold_header_penalty,
        });
        out.advice.push(format!(
            "Vary paragraph structure \u{2014} {} bold-header-explanation blocks in a row reads as LLM listicle.",
            bold_matches.len()
        ));
        out.inc("structural");
    }

    // Bullet runs
    let mut run_length: usize = 0;
    for line in lines {
        if BULLET_LINE_RE.is_match(line) {
            run_length += 1;
        } else {
            if run_length >= HP.structural_bullet_run_min {
                out.violations.push(Violation {
                    violation_type: "Violation".to_string(),
                    rule: "structural".to_string(),
                    match_text: "excessive_bullets".to_string(),
                    context: format!("Run of {run_length} consecutive bullet lines"),
                    penalty: HP.structural_bullet_run_penalty,
                });
                out.advice.push(format!(
                    "Consider prose instead of this {run_length}-item bullet list."
                ));
                out.inc("structural");
            }
            run_length = 0;
        }
    }
    if run_length >= HP.structural_bullet_run_min {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "structural".to_string(),
            match_text: "excessive_bullets".to_string(),
            context: format!("Run of {run_length} consecutive bullet lines"),
            penalty: HP.structural_bullet_run_penalty,
        });
        out.advice.push(format!(
            "Consider prose instead of this {run_length}-item bullet list."
        ));
        out.inc("structural");
    }

    // Triadic
    let triadic_matches: Vec<_> = TRIADIC_RE.find_iter(text).collect();
    let triadic_count = triadic_matches.len();
    for m in triadic_matches.iter().take(HP.triadic_record_cap) {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "structural".to_string(),
            match_text: "triadic".to_string(),
            context: context_around(text, m.start(), m.end(), width),
            penalty: HP.triadic_penalty,
        });
        out.inc("structural");
    }
    if triadic_count >= HP.triadic_advice_min {
        out.advice.push(format!(
            "{triadic_count} triadic structures ('X, Y, and Z') \u{2014} vary your list cadence."
        ));
    }

    out
}

fn rule_tone(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;

    for pat in META_COMM_PATTERNS.iter() {
        for m in pat.find_iter(text) {
            let phrase = m.as_str().to_lowercase();
            out.violations.push(Violation {
                violation_type: "Violation".to_string(),
                rule: "tone".to_string(),
                match_text: phrase.clone(),
                context: context_around(text, m.start(), m.end(), width),
                penalty: HP.tone_penalty,
            });
            out.advice.push(format!(
                "Remove '{phrase}' \u{2014} this is a direct AI tell."
            ));
            out.inc("tone");
        }
    }

    for pat in FALSE_NARRATIVITY_PATTERNS.iter() {
        for m in pat.find_iter(text) {
            let phrase = m.as_str().to_lowercase();
            out.violations.push(Violation {
                violation_type: "Violation".to_string(),
                rule: "tone".to_string(),
                match_text: phrase.clone(),
                context: context_around(text, m.start(), m.end(), width),
                penalty: HP.tone_penalty,
            });
            out.advice
                .push(format!("Cut '{phrase}' \u{2014} announce less, show more."));
            out.inc("tone");
        }
    }

    for pat in SENTENCE_OPENER_PATTERNS.iter() {
        for caps in pat.captures_iter(text) {
            let full = caps.get(0).unwrap();
            let word_match = caps.get(1).unwrap();
            let word = word_match
                .as_str()
                .trim_matches(|c: char| c == ',' || c == '!' || c == ' ')
                .to_lowercase();
            out.violations.push(Violation {
                violation_type: "Violation".to_string(),
                rule: "tone".to_string(),
                match_text: word.clone(),
                context: context_around(text, full.start(), full.end(), width),
                penalty: HP.sentence_opener_penalty,
            });
            out.advice.push(format!(
                "'{word}' as a sentence opener is an AI tell \u{2014} just make the point."
            ));
            out.inc("tone");
        }
    }

    out
}

fn rule_weasel(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;

    for pat in WEASEL_PATTERNS.iter() {
        for m in pat.find_iter(text) {
            let phrase = m.as_str().to_lowercase();
            out.violations.push(Violation {
                violation_type: "Violation".to_string(),
                rule: "weasel".to_string(),
                match_text: phrase.clone(),
                context: context_around(text, m.start(), m.end(), width),
                penalty: HP.weasel_penalty,
            });
            out.advice.push(format!(
                "Cut '{phrase}' \u{2014} either cite a source or own the claim."
            ));
            out.inc("weasel");
        }
    }
    out
}

fn rule_ai_disclosure(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;

    for pat in AI_DISCLOSURE_PATTERNS.iter() {
        for m in pat.find_iter(text) {
            let phrase = m.as_str().to_lowercase();
            out.violations.push(Violation {
                violation_type: "Violation".to_string(),
                rule: "ai_disclosure".to_string(),
                match_text: phrase.clone(),
                context: context_around(text, m.start(), m.end(), width),
                penalty: HP.ai_disclosure_penalty,
            });
            out.advice.push(format!(
                "Remove '{phrase}' \u{2014} AI self-disclosure in authored prose is a critical tell."
            ));
            out.inc("ai_disclosure");
        }
    }
    out
}

fn rule_placeholder(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;

    for m in PLACEHOLDER_RE.find_iter(text) {
        let match_text = m.as_str().to_lowercase();
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "placeholder".to_string(),
            match_text: match_text.clone(),
            context: context_around(text, m.start(), m.end(), width),
            penalty: HP.placeholder_penalty,
        });
        out.advice.push(format!(
            "Remove placeholder '{match_text}' \u{2014} this is unfinished template text."
        ));
        out.inc("placeholder");
    }
    out
}

fn rule_rhythm(sentences: &[String]) -> RuleOutput {
    let mut out = RuleOutput::new();

    if sentences.len() < HP.rhythm_min_sentences {
        return out;
    }

    let lengths: Vec<f64> = sentences
        .iter()
        .map(|s| s.split_whitespace().count() as f64)
        .collect();
    let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
    if mean <= 0.0 {
        return out;
    }

    let variance = lengths.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lengths.len() as f64;
    let std = variance.sqrt();
    let cv = std / mean;

    if cv < HP.rhythm_cv_threshold {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "rhythm".to_string(),
            match_text: "monotonous_rhythm".to_string(),
            context: format!(
                "CV={cv:.2} across {} sentences (mean {mean:.1} words)",
                sentences.len()
            ),
            penalty: HP.rhythm_penalty,
        });
        out.advice.push(format!(
            "Sentence lengths are too uniform (CV={cv:.2}) \u{2014} vary short and long."
        ));
        out.inc("rhythm");
    }
    out
}

fn rule_em_dash_density(text: &str, wc: usize) -> RuleOutput {
    let mut out = RuleOutput::new();
    if wc == 0 {
        return out;
    }

    let em_dash_count = EM_DASH_RE.find_iter(text).count();
    let ratio_per_150 = (em_dash_count as f64 / wc as f64) * HP.em_dash_words_basis;
    if ratio_per_150 > HP.em_dash_density_threshold {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "em_dash".to_string(),
            match_text: "em_dash_density".to_string(),
            context: format!(
                "{em_dash_count} em dashes in {wc} words ({ratio_per_150:.1} per 150 words)"
            ),
            penalty: HP.em_dash_penalty,
        });
        out.advice.push(format!(
            "Too many em dashes ({em_dash_count} in {wc} words) \u{2014} use other punctuation."
        ));
        out.inc("em_dash");
    }
    out
}

fn rule_contrast_pairs(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;

    let matches: Vec<_> = CONTRAST_PAIR_RE.find_iter(text).collect();
    let count = matches.len();

    for m in matches.iter().take(HP.contrast_record_cap) {
        let matched = m.as_str().to_string();
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "contrast_pair".to_string(),
            match_text: matched.clone(),
            context: context_around(text, m.start(), m.end(), width),
            penalty: HP.contrast_penalty,
        });
        out.advice.push(format!(
            "'{matched}' \u{2014} 'X, not Y' contrast \u{2014} consider rephrasing to avoid the Claude pattern."
        ));
        out.inc("contrast_pairs");
    }

    if count >= HP.contrast_advice_min {
        out.advice.push(format!(
            "{count} 'X, not Y' contrasts \u{2014} this is a Claude rhetorical tic. Vary your phrasing."
        ));
    }
    out
}

fn rule_setup_resolution(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();
    let width = HP.context_window_chars;

    let mut setup_res_recorded = 0usize;
    for pat in [&*SETUP_RESOLUTION_A_RE, &*SETUP_RESOLUTION_B_RE] {
        for m in pat.find_iter(text) {
            if setup_res_recorded < HP.setup_resolution_record_cap {
                let matched = m.as_str().to_string();
                out.violations.push(Violation {
                    violation_type: "Violation".to_string(),
                    rule: "setup_resolution".to_string(),
                    match_text: matched.clone(),
                    context: context_around(text, m.start(), m.end(), width),
                    penalty: HP.setup_resolution_penalty,
                });
                out.advice.push(format!(
                    "'{matched}' \u{2014} setup-and-resolution is a Claude rhetorical tic. Just state the point directly."
                ));
                setup_res_recorded += 1;
            }
            out.inc("setup_resolution");
        }
    }
    out
}

fn rule_colon_density(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();

    let stripped_text = strip_code_blocks(text);
    let mut colon_count = 0usize;

    for line in stripped_text.split('\n') {
        if MD_HEADER_LINE_SINGLE_RE.is_match(line) {
            continue;
        }
        for cm in ELABORATION_COLON_RE.find_iter(line) {
            let col_pos = cm.start();
            let before = &line[..col_pos + 1];
            if before.ends_with("http:") || before.ends_with("https:") {
                continue;
            }
            let snippet_end = std::cmp::min(col_pos + 10, line.len());
            let snippet_end = snap_to_char_boundary(line, snippet_end, true);
            let snippet = &line[col_pos..snippet_end];
            if JSON_COLON_RE.is_match(snippet) {
                continue;
            }
            colon_count += 1;
        }
    }

    let stripped_wc = word_count(&stripped_text);
    if stripped_wc == 0 {
        return out;
    }

    let colon_ratio_per_150 = (colon_count as f64 / stripped_wc as f64) * HP.colon_words_basis;
    if colon_ratio_per_150 > HP.colon_density_threshold {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "colon_density".to_string(),
            match_text: "colon_density".to_string(),
            context: format!(
                "{colon_count} elaboration colons in {stripped_wc} words ({colon_ratio_per_150:.1} per 150 words)"
            ),
            penalty: HP.colon_density_penalty,
        });
        out.advice.push(format!(
            "Too many elaboration colons ({colon_count} in {stripped_wc} words) \u{2014} use periods or restructure sentences."
        ));
        out.inc("colon_density");
    }
    out
}

fn rule_pithy_fragments(sentences: &[String]) -> RuleOutput {
    let mut out = RuleOutput::new();
    let mut pithy_count = 0usize;

    for sent in sentences {
        let s = sent.trim();
        if s.is_empty() {
            continue;
        }
        let sent_words: Vec<&str> = s.split_whitespace().collect();
        if sent_words.len() > HP.pithy_max_sentence_words {
            continue;
        }
        if PITHY_PIVOT_RE.is_match(s) {
            if pithy_count < HP.pithy_record_cap {
                out.violations.push(Violation {
                    violation_type: "Violation".to_string(),
                    rule: "pithy_fragment".to_string(),
                    match_text: s.to_string(),
                    context: s.to_string(),
                    penalty: HP.pithy_penalty,
                });
                out.advice.push(format!(
                    "'{s}' \u{2014} pithy evaluative fragments are a Claude tell. Expand or cut."
                ));
            }
            pithy_count += 1;
            out.inc("pithy_fragment");
        }
    }
    out
}

fn rule_bullet_density(lines: &[&str]) -> RuleOutput {
    let mut out = RuleOutput::new();

    let non_empty: Vec<&&str> = lines.iter().filter(|l| !l.trim().is_empty()).collect();
    let total_non_empty = non_empty.len();
    if total_non_empty == 0 {
        return out;
    }

    let bullet_count = non_empty
        .iter()
        .filter(|l| BULLET_DENSITY_RE.is_match(l))
        .count();
    let bullet_ratio = bullet_count as f64 / total_non_empty as f64;
    if bullet_ratio > HP.bullet_density_threshold {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "structural".to_string(),
            match_text: "bullet_density".to_string(),
            context: format!(
                "{bullet_count} of {total_non_empty} non-empty lines are bullets ({:.0}%)",
                bullet_ratio * 100.0
            ),
            penalty: HP.bullet_density_penalty,
        });
        out.advice.push(format!(
            "Over {:.0}% of lines are bullets \u{2014} write prose instead of lists.",
            bullet_ratio * 100.0
        ));
        out.inc("bullet_density");
    }
    out
}

fn rule_blockquote_density(lines: &[&str]) -> RuleOutput {
    let mut out = RuleOutput::new();

    let mut in_code_block = false;
    let mut blockquote_count = 0usize;
    for line in lines {
        if line.trim().starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if !in_code_block && line.starts_with('>') {
            blockquote_count += 1;
        }
    }

    if blockquote_count >= HP.blockquote_min_lines {
        let excess = blockquote_count - HP.blockquote_free_lines;
        let capped = std::cmp::min(excess, HP.blockquote_cap);
        let bq_penalty = HP.blockquote_penalty_step * capped as i32;
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "structural".to_string(),
            match_text: "blockquote_density".to_string(),
            context: format!(
                "{blockquote_count} blockquote lines \u{2014} Claude uses these as thesis statements"
            ),
            penalty: bq_penalty,
        });
        out.advice.push(format!(
            "{blockquote_count} blockquotes \u{2014} integrate key claims into prose instead of pulling them out as blockquotes."
        ));
        out.inc("blockquote_density");
    }
    out
}

fn rule_bold_bullet_runs(lines: &[&str]) -> RuleOutput {
    let mut out = RuleOutput::new();

    let mut bold_bullet_run = 0usize;
    for line in lines {
        if BOLD_TERM_BULLET_RE.is_match(line) {
            bold_bullet_run += 1;
            continue;
        }
        if bold_bullet_run >= HP.bold_bullet_run_min {
            out.violations.push(Violation {
                violation_type: "Violation".to_string(),
                rule: "structural".to_string(),
                match_text: "bold_bullet_list".to_string(),
                context: format!("Run of {bold_bullet_run} bold-term bullets"),
                penalty: HP.bold_bullet_run_penalty,
            });
            out.advice.push(format!(
                "Run of {bold_bullet_run} bold-term bullets \u{2014} this is an LLM listicle pattern. Use varied paragraph structure."
            ));
            out.inc("bold_bullet_list");
        }
        bold_bullet_run = 0;
    }
    if bold_bullet_run >= HP.bold_bullet_run_min {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "structural".to_string(),
            match_text: "bold_bullet_list".to_string(),
            context: format!("Run of {bold_bullet_run} bold-term bullets"),
            penalty: HP.bold_bullet_run_penalty,
        });
        out.advice.push(format!(
            "Run of {bold_bullet_run} bold-term bullets \u{2014} this is an LLM listicle pattern. Use varied paragraph structure."
        ));
        out.inc("bold_bullet_list");
    }
    out
}

fn rule_horizontal_rules(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();

    let hr_count = HORIZONTAL_RULE_RE.find_iter(text).count();
    if hr_count >= HP.horizontal_rule_min {
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "structural".to_string(),
            match_text: "horizontal_rules".to_string(),
            context: format!("{hr_count} horizontal rules \u{2014} excessive section dividers"),
            penalty: HP.horizontal_rule_penalty,
        });
        out.advice.push(format!(
            "{hr_count} horizontal rules \u{2014} section headers alone are sufficient, dividers are a crutch."
        ));
        out.inc("horizontal_rules");
    }
    out
}

fn rule_phrase_reuse(text: &str) -> RuleOutput {
    let mut out = RuleOutput::new();

    let repeated = find_repeated_ngrams(text);
    for (recorded, ng) in repeated.iter().enumerate() {
        if recorded >= HP.phrase_reuse_record_cap {
            break;
        }
        out.violations.push(Violation {
            violation_type: "Violation".to_string(),
            rule: "phrase_reuse".to_string(),
            match_text: ng.phrase.clone(),
            context: format!(
                "'{}' ({}-word phrase) appears {} times",
                ng.phrase, ng.n, ng.count
            ),
            penalty: HP.phrase_reuse_penalty,
        });
        out.advice.push(format!(
            "'{}' appears {} times \u{2014} vary your phrasing to avoid repetition.",
            ng.phrase, ng.count
        ));
        out.inc("phrase_reuse");
    }
    out
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

fn compute_weighted_sum(violations: &[Violation], counts: &HashMap<String, usize>) -> f64 {
    let mut weighted_sum = 0.0f64;
    for v in violations {
        let penalty = v.penalty.unsigned_abs() as f64;
        let rule = &v.rule;

        // Check if rule or rule+"s" is in claude_categories
        let cat_key = if HP.claude_categories.contains(&rule.as_str()) {
            Some(rule.as_str())
        } else {
            let with_s = format!("{rule}s");
            if HP.claude_categories.iter().any(|c| *c == with_s) {
                Some("")
            } else {
                None
            }
        };

        // Get the count for the rule
        let cat_count = counts
            .get(rule.as_str())
            .copied()
            .unwrap_or(0)
            .max(counts.get(&format!("{rule}s")).copied().unwrap_or(0));

        if cat_key.is_some() && cat_count > 1 {
            let weight = penalty * (1.0 + HP.concentration_alpha * (cat_count as f64 - 1.0));
            weighted_sum += weight;
        } else {
            weighted_sum += penalty;
        }
    }
    weighted_sum
}

fn band_for_score(score: i32) -> &'static str {
    if score >= HP.band_clean_min {
        "clean"
    } else if score >= HP.band_light_min {
        "light"
    } else if score >= HP.band_moderate_min {
        "moderate"
    } else if score >= HP.band_heavy_min {
        "heavy"
    } else {
        "saturated"
    }
}

fn deduplicate_advice(advice: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for item in advice {
        if seen.insert(item.clone()) {
            unique.push(item);
        }
    }
    unique
}

// ---------------------------------------------------------------------------
// Merge helper
// ---------------------------------------------------------------------------

fn merge_output(
    violations: &mut Vec<Violation>,
    advice: &mut Vec<String>,
    counts: &mut HashMap<String, usize>,
    out: RuleOutput,
) {
    violations.extend(out.violations);
    advice.extend(out.advice);
    for (key, delta) in out.count_deltas {
        if delta > 0 {
            *counts.entry(key).or_insert(0) += delta;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn analyze(text: &str) -> AnalysisResult {
    let wc = word_count(text);
    let counts_init = initial_counts();

    if wc < HP.short_text_word_count {
        return AnalysisResult {
            score: HP.score_max,
            band: "clean".to_string(),
            word_count: wc,
            violations: vec![],
            counts: counts_init,
            total_penalty: 0,
            weighted_sum: 0.0,
            density: 0.0,
            advice: vec![],
        };
    }

    let lines: Vec<&str> = text.split('\n').collect();
    let sentences: Vec<String> = SENTENCE_SPLIT_RE
        .split(text)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut violations: Vec<Violation> = Vec::new();
    let mut advice: Vec<String> = Vec::new();
    let mut counts = counts_init;

    // 1. Slop words
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_slop_words(text),
    );
    // 2. Slop phrases
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_slop_phrases(text),
    );
    // 3. Structural
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_structural(text, &lines),
    );
    // 4. Tone
    merge_output(&mut violations, &mut advice, &mut counts, rule_tone(text));
    // 5. Weasel
    merge_output(&mut violations, &mut advice, &mut counts, rule_weasel(text));
    // 6. AI disclosure
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_ai_disclosure(text),
    );
    // 7. Placeholder
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_placeholder(text),
    );
    // 8. Rhythm
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_rhythm(&sentences),
    );
    // 9. Em dash density
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_em_dash_density(text, wc),
    );
    // 10. Contrast pairs
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_contrast_pairs(text),
    );
    // 11. Setup-resolution
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_setup_resolution(text),
    );
    // 12. Colon density
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_colon_density(text),
    );
    // 13. Pithy fragments
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_pithy_fragments(&sentences),
    );
    // 14. Bullet density
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_bullet_density(&lines),
    );
    // 15. Blockquote density
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_blockquote_density(&lines),
    );
    // 16. Bold bullet runs
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_bold_bullet_runs(&lines),
    );
    // 17. Horizontal rules
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_horizontal_rules(text),
    );
    // 18. Phrase reuse
    merge_output(
        &mut violations,
        &mut advice,
        &mut counts,
        rule_phrase_reuse(text),
    );

    let total_penalty: i32 = violations.iter().map(|v| v.penalty).sum();
    let weighted_sum = compute_weighted_sum(&violations, &counts);
    let density = if wc > 0 {
        weighted_sum / (wc as f64 / HP.density_words_basis)
    } else {
        0.0
    };
    let raw_score = HP.score_max as f64 * (-HP.decay_lambda * density).exp();
    let score = (raw_score.round() as i32).clamp(HP.score_min, HP.score_max);
    let band = band_for_score(score).to_string();

    AnalysisResult {
        score,
        band,
        word_count: wc,
        violations,
        counts,
        total_penalty,
        weighted_sum: (weighted_sum * 100.0).round() / 100.0,
        density: (density * 100.0).round() / 100.0,
        advice: deduplicate_advice(advice),
    }
}
