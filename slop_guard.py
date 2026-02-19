# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp"]
# ///
"""MCP server for prose linting — detects AI slop patterns in text.

Single tool: check_slop(text) returns a JSON score with specific violations.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

MCP_SERVER_NAME = "slop-guard"

# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------

_ALPHA = 2.5    # concentration multiplier for Claude-specific patterns
_LAMBDA = 0.04  # exponential decay rate for scoring

# Categories where repeat occurrences get concentration-amplified
_CLAUDE_CATEGORIES = {"contrast_pairs", "pithy_fragment", "setup_resolution"}

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# --- 1. Slop words ---

_SLOP_ADJECTIVES = [
    "crucial", "groundbreaking", "pivotal", "paramount", "seamless", "holistic",
    "multifaceted", "meticulous", "profound", "comprehensive", "invaluable",
    "notable", "noteworthy", "game-changing", "revolutionary", "pioneering",
    "visionary", "formidable", "quintessential", "unparalleled",
    "stunning", "breathtaking", "captivating", "nestled", "robust",
    "innovative", "cutting-edge", "impactful",
]

_SLOP_VERBS = [
    "delve", "delves", "delved", "delving", "embark", "embrace", "elevate",
    "foster", "harness", "unleash", "unlock", "orchestrate", "streamline",
    "transcend", "navigate", "underscore", "showcase", "leverage",
    "ensuring", "highlighting", "emphasizing", "reflecting",
]

_SLOP_NOUNS = [
    "landscape", "tapestry", "journey", "paradigm", "testament", "trajectory",
    "nexus", "symphony", "spectrum", "odyssey", "pinnacle", "realm", "intricacies",
]

_SLOP_HEDGE = [
    "notably", "importantly", "furthermore", "additionally", "particularly",
    "significantly", "interestingly", "remarkably", "surprisingly", "fascinatingly",
    "moreover", "however", "overall",
]

_ALL_SLOP_WORDS = _SLOP_ADJECTIVES + _SLOP_VERBS + _SLOP_NOUNS + _SLOP_HEDGE

# Single compiled regex: word boundary, case-insensitive, alternation
_SLOP_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _ALL_SLOP_WORDS) + r")\b",
    re.IGNORECASE,
)

# --- 2. Slop phrases ---

_SLOP_PHRASES_LITERAL = [
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
    # Rule 10: Menu-of-options / offer to rewrite
    "if you want, i can",
    "i can adapt this",
    "i can make this",
    "here are some options",
    "here are a few options",
    "would you prefer",
    "shall i",
    "if you'd like, i can",
    "i can also",
    # Rule 11: Restatement transition phrases
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
]

_SLOP_PHRASES_RE_LIST: list[re.Pattern[str]] = [
    re.compile(re.escape(p), re.IGNORECASE) for p in _SLOP_PHRASES_LITERAL
]

# "not just X, but" regex pattern
_NOT_JUST_BUT_RE = re.compile(
    r"not (just|only) .{1,40}, but (also )?", re.IGNORECASE
)

# --- 3. Structural patterns ---

# **Bold.** or **Bold:** followed by more text
_BOLD_HEADER_RE = re.compile(r"\*\*[^*]+[.:]\*\*\s+\S")

# Bullet lines: - item, * item, or 1. item
_BULLET_LINE_RE = re.compile(r"^(\s*[-*]\s|\s*\d+\.\s)")

# Triadic: "X, Y, and Z"
_TRIADIC_RE = re.compile(r"\w+, \w+, and \w+", re.IGNORECASE)

# --- 4. Tone markers ---

_META_COMM_PATTERNS = [
    re.compile(r"would you like", re.IGNORECASE),
    re.compile(r"let me know if", re.IGNORECASE),
    re.compile(r"as mentioned", re.IGNORECASE),
    re.compile(r"i hope this", re.IGNORECASE),
    re.compile(r"feel free to", re.IGNORECASE),
    re.compile(r"don't hesitate to", re.IGNORECASE),
]

_FALSE_NARRATIVITY_PATTERNS = [
    re.compile(r"then something interesting happened", re.IGNORECASE),
    re.compile(r"this is where things get interesting", re.IGNORECASE),
    re.compile(r"that's when everything changed", re.IGNORECASE),
]

# --- 4b. Sentence-opener tells ---

_SENTENCE_OPENER_PATTERNS = [
    re.compile(r"(?:^|[.!?]\s+)(certainly[,! ])", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|[.!?]\s+)(absolutely[,! ])", re.IGNORECASE | re.MULTILINE),
]

# --- 4c. Weasel phrases ---

_WEASEL_PATTERNS = [
    re.compile(r"\bsome critics argue\b", re.IGNORECASE),
    re.compile(r"\bmany believe\b", re.IGNORECASE),
    re.compile(r"\bexperts suggest\b", re.IGNORECASE),
    re.compile(r"\bstudies show\b", re.IGNORECASE),
    re.compile(r"\bsome argue\b", re.IGNORECASE),
    re.compile(r"\bit is widely believed\b", re.IGNORECASE),
    re.compile(r"\bresearch suggests\b", re.IGNORECASE),
]

# --- 4d. AI self-disclosure ---

_AI_DISCLOSURE_PATTERNS = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bas a language model\b", re.IGNORECASE),
    re.compile(r"\bi don't have personal\b", re.IGNORECASE),
    re.compile(r"\bi cannot browse\b", re.IGNORECASE),
    re.compile(r"\bup to my last training\b", re.IGNORECASE),
    re.compile(r"\bas of my (last |knowledge )?cutoff\b", re.IGNORECASE),
    re.compile(r"\bi'm just an? ai\b", re.IGNORECASE),
]

# --- 4e. Placeholder text ---

_PLACEHOLDER_RE = re.compile(
    r"\[insert [^\]]*\]|\[describe [^\]]*\]|\[url [^\]]*\]|\[your [^\]]*\]|\[todo[^\]]*\]",
    re.IGNORECASE,
)

# --- 5. Rhythm ---

_SENTENCE_SPLIT_RE = re.compile(r"[.!?][\"'\u201D\u2019)\]]*(?:\s|$)")

# --- 6. Em dash ---

_EM_DASH_RE = re.compile(r"\u2014| -- ")

# --- 7. "X, not Y" contrast pattern ---

_CONTRAST_PAIR_RE = re.compile(r"\b(\w+), not (\w+)\b")

# --- 7b. Setup-and-resolution ("This isn't X. It's Y.") ---

# Form A: "This isn't X. It's Y." — pronoun + negative verb + content + separator + positive restatement
_SETUP_RESOLUTION_A_RE = re.compile(
    r"\b(this|that|these|those|it|they|we)\s+"
    r"(isn't|aren't|wasn't|weren't|doesn't|don't|didn't|hasn't|haven't|won't|can't|couldn't|shouldn't"
    r"|is\s+not|are\s+not|was\s+not|were\s+not|does\s+not|do\s+not|did\s+not"
    r"|has\s+not|have\s+not|will\s+not|cannot|could\s+not|should\s+not)\b"
    r".{0,80}[.;:,]\s*"
    r"(it's|they're|that's|he's|she's|we're|it\s+is|they\s+are|that\s+is|this\s+is"
    r"|these\s+are|those\s+are|he\s+is|she\s+is|we\s+are|what's|what\s+is"
    r"|the\s+real|the\s+actual|instead|rather)",
    re.IGNORECASE,
)

# Form B: "It's not X. It's Y." — positive contraction + "not" + content + separator + positive restatement
_SETUP_RESOLUTION_B_RE = re.compile(
    r"\b(it's|that's|this\s+is|they're|he's|she's|we're)\s+not\b"
    r".{0,80}[.;:,]\s*"
    r"(it's|they're|that's|he's|she's|we're|it\s+is|they\s+are|that\s+is|this\s+is"
    r"|these\s+are|those\s+are|what's|what\s+is|the\s+real|the\s+actual|instead|rather)",
    re.IGNORECASE,
)

# --- 8. Colon density (elaboration colons) ---

# Matches a colon followed by space + lowercase letter (mid-sentence elaboration)
_ELABORATION_COLON_RE = re.compile(r": [a-z]")
# Fenced code block removal (greedy across lines)
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# URL colon exclusion (http: or https:)
_URL_COLON_RE = re.compile(r"https?:")
# Markdown header line
_MD_HEADER_LINE_RE = re.compile(r"^\s*#", re.MULTILINE)
# JSON-like colon contexts
_JSON_COLON_RE = re.compile(r': ["{\[\d]|: true|: false|: null')

# --- 9. Pithy evaluative fragments ---

_PITHY_PIVOT_RE = re.compile(r",\s+(?:but|yet|and|not|or)\b", re.IGNORECASE)

# --- 12. Bullet density ---

_BULLET_DENSITY_RE = re.compile(r"^\s*[-*]\s|^\s*\d+[.)]\s", re.MULTILINE)

# --- 13. Blockquote-as-thesis ---

_BLOCKQUOTE_LINE_RE = re.compile(r"^>", re.MULTILINE)

# --- 14. Bold-term bullet runs ---

_BOLD_TERM_BULLET_RE = re.compile(r"^\s*[-*]\s+\*\*|^\s*\d+[.)]\s+\*\*")

# --- 15. Horizontal rule overuse ---

_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:---+|\*\*\*+|___+)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _context_around(text: str, start: int, end: int, width: int = 60) -> str:
    """Extract ~width chars of surrounding text centered on [start, end]."""
    mid = (start + end) // 2
    half = width // 2
    ctx_start = max(0, mid - half)
    ctx_end = min(len(text), mid + half)
    snippet = text[ctx_start:ctx_end].replace("\n", " ")
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _word_count(text: str) -> int:
    return len(text.split())


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code block contents for analyses that shouldn't count them."""
    return _FENCED_CODE_BLOCK_RE.sub("", text)


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "it", "that", "this", "with", "as", "by", "from", "was", "were", "are",
    "be", "been", "has", "have", "had", "not", "no", "do", "does", "did", "will",
    "would", "could", "should", "can", "may", "might", "if", "then", "than", "so",
    "up", "out", "about", "into", "over", "after", "before", "between", "through",
    "just", "also", "very", "more", "most", "some", "any", "each", "every", "all",
    "both", "few", "other", "such", "only", "own", "same", "too", "how", "what",
    "which", "who", "when", "where", "why",
})

_PUNCT_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


def _find_repeated_ngrams(
    text: str, min_n: int = 4, max_n: int = 8
) -> list[dict]:
    """Find multi-word phrases repeated 2+ times, returning only the longest.

    Returns a list of dicts with keys: phrase, count, n.
    """
    # Tokenize: split on whitespace, strip punctuation from edges, lowercase
    raw_tokens = text.split()
    tokens = [_PUNCT_STRIP_RE.sub("", t).lower() for t in raw_tokens]
    tokens = [t for t in tokens if t]  # drop empty after stripping

    if len(tokens) < min_n:
        return []

    # Count n-grams for each size
    ngram_counts: dict[tuple[str, ...], int] = {}
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

    # Filter: 2+ occurrences, not all stopwords
    repeated = {
        gram: count
        for gram, count in ngram_counts.items()
        if count >= 3 and not all(w in _STOPWORDS for w in gram)
    }

    if not repeated:
        return []

    # Suppress sub-n-grams: if a longer n-gram is repeated, remove shorter
    # n-grams that are fully contained within it.
    to_remove: set[tuple[str, ...]] = set()
    # Sort by length descending so longer grams suppress shorter ones
    sorted_grams = sorted(repeated.keys(), key=len, reverse=True)
    for i, longer in enumerate(sorted_grams):
        longer_str = " ".join(longer)
        for shorter in sorted_grams[i + 1 :]:
            if shorter in to_remove:
                continue
            shorter_str = " ".join(shorter)
            if shorter_str in longer_str and repeated[longer] >= repeated[shorter]:
                to_remove.add(shorter)

    results = []
    for gram in sorted(repeated.keys(), key=lambda g: (-len(g), -repeated[g])):
        if gram not in to_remove:
            results.append({
                "phrase": " ".join(gram),
                "count": repeated[gram],
                "n": len(gram),
            })

    return results


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _analyze(text: str) -> dict:
    word_count = _word_count(text)

    # Short-circuit for very short text
    if word_count < 10:
        return {
            "score": 100,
            "band": "clean",
            "word_count": word_count,
            "violations": [],
            "counts": {
                "slop_words": 0, "slop_phrases": 0, "structural": 0,
                "tone": 0, "weasel": 0, "ai_disclosure": 0, "placeholder": 0,
                "rhythm": 0, "em_dash": 0,
                "contrast_pairs": 0, "colon_density": 0, "pithy_fragment": 0,
                "setup_resolution": 0,
                "bullet_density": 0, "blockquote_density": 0,
                "bold_bullet_list": 0, "horizontal_rules": 0,
                "phrase_reuse": 0,
            },
            "total_penalty": 0,
            "weighted_sum": 0.0,
            "density": 0.0,
            "advice": [],
        }

    violations: list[dict] = []
    advice: list[str] = []
    counts = {
        "slop_words": 0, "slop_phrases": 0, "structural": 0,
        "tone": 0, "weasel": 0, "ai_disclosure": 0, "placeholder": 0,
        "rhythm": 0, "em_dash": 0,
        "contrast_pairs": 0, "colon_density": 0, "pithy_fragment": 0,
        "setup_resolution": 0,
        "bullet_density": 0, "blockquote_density": 0,
        "bold_bullet_list": 0, "horizontal_rules": 0,
        "phrase_reuse": 0,
    }

    # --- 1. Slop words ---
    for m in _SLOP_WORD_RE.finditer(text):
        word = m.group(0)
        violations.append({
            "rule": "slop_word",
            "match": word.lower(),
            "context": _context_around(text, m.start(), m.end()),
            "penalty": -2,
        })
        advice.append(f"Replace '{word.lower()}' \u2014 what specifically do you mean?")
        counts["slop_words"] += 1

    # --- 2. Slop phrases ---
    for pat in _SLOP_PHRASES_RE_LIST:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append({
                "rule": "slop_phrase",
                "match": phrase.lower(),
                "context": _context_around(text, m.start(), m.end()),
                "penalty": -3,
            })
            advice.append(f"Cut '{phrase.lower()}' \u2014 just state the point directly.")
            counts["slop_phrases"] += 1

    # "not just X, but" regex
    for m in _NOT_JUST_BUT_RE.finditer(text):
        phrase = m.group(0)
        violations.append({
            "rule": "slop_phrase",
            "match": phrase.strip().lower(),
            "context": _context_around(text, m.start(), m.end()),
            "penalty": -3,
        })
        advice.append(f"Cut '{phrase.strip().lower()}' \u2014 just state the point directly.")
        counts["slop_phrases"] += 1

    # --- 3. Structural patterns ---

    # Bold-header-explanation
    bold_matches = list(_BOLD_HEADER_RE.finditer(text))
    if len(bold_matches) >= 3:
        violations.append({
            "rule": "structural",
            "match": "bold_header_explanation",
            "context": f"Found {len(bold_matches)} instances of **Bold.** pattern",
            "penalty": -5,
        })
        advice.append(
            f"Vary paragraph structure \u2014 {len(bold_matches)} bold-header-explanation "
            f"blocks in a row reads as LLM listicle."
        )
        counts["structural"] += 1

    # Excessive consecutive bullets
    lines = text.split("\n")
    run_length = 0
    for line in lines:
        if _BULLET_LINE_RE.match(line):
            run_length += 1
        else:
            if run_length >= 6:
                violations.append({
                    "rule": "structural",
                    "match": "excessive_bullets",
                    "context": f"Run of {run_length} consecutive bullet lines",
                    "penalty": -3,
                })
                advice.append(
                    f"Consider prose instead of this {run_length}-item bullet list."
                )
                counts["structural"] += 1
            run_length = 0
    # Check trailing run
    if run_length >= 6:
        violations.append({
            "rule": "structural",
            "match": "excessive_bullets",
            "context": f"Run of {run_length} consecutive bullet lines",
            "penalty": -3,
        })
        advice.append(
            f"Consider prose instead of this {run_length}-item bullet list."
        )
        counts["structural"] += 1

    # Triadic structures
    triadic_matches = list(_TRIADIC_RE.finditer(text))
    triadic_count = len(triadic_matches)
    triadic_penalty = min(triadic_count, 5)  # -1 each, cap at 5
    for m in triadic_matches[:5]:  # only record up to 5 violations
        violations.append({
            "rule": "structural",
            "match": "triadic",
            "context": _context_around(text, m.start(), m.end()),
            "penalty": -1,
        })
        counts["structural"] += 1
    if triadic_count >= 3:
        advice.append(
            f"{triadic_count} triadic structures ('X, Y, and Z') \u2014 "
            f"vary your list cadence."
        )

    # --- 4. Tone markers ---

    # Meta-communication
    for pat in _META_COMM_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append({
                "rule": "tone",
                "match": phrase.lower(),
                "context": _context_around(text, m.start(), m.end()),
                "penalty": -3,
            })
            advice.append(f"Remove '{phrase.lower()}' \u2014 this is a direct AI tell.")
            counts["tone"] += 1

    # False narrativity
    for pat in _FALSE_NARRATIVITY_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append({
                "rule": "tone",
                "match": phrase.lower(),
                "context": _context_around(text, m.start(), m.end()),
                "penalty": -3,
            })
            advice.append(f"Cut '{phrase.lower()}' \u2014 announce less, show more.")
            counts["tone"] += 1

    # Sentence-opener tells
    for pat in _SENTENCE_OPENER_PATTERNS:
        for m in pat.finditer(text):
            word = m.group(1).strip(" ,!")
            violations.append({
                "rule": "tone",
                "match": word.lower(),
                "context": _context_around(text, m.start(), m.end()),
                "penalty": -2,
            })
            advice.append(
                f"'{word.lower()}' as a sentence opener is an AI tell "
                f"\u2014 just make the point."
            )
            counts["tone"] += 1

    # --- 4b. Weasel phrases ---
    for pat in _WEASEL_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append({
                "rule": "weasel",
                "match": phrase.lower(),
                "context": _context_around(text, m.start(), m.end()),
                "penalty": -2,
            })
            advice.append(
                f"Cut '{phrase.lower()}' \u2014 either cite a source or own the claim."
            )
            counts["weasel"] += 1

    # --- 4c. AI self-disclosure ---
    for pat in _AI_DISCLOSURE_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append({
                "rule": "ai_disclosure",
                "match": phrase.lower(),
                "context": _context_around(text, m.start(), m.end()),
                "penalty": -10,
            })
            advice.append(
                f"Remove '{phrase.lower()}' \u2014 AI self-disclosure in authored "
                f"prose is a critical tell."
            )
            counts["ai_disclosure"] += 1

    # --- 4d. Placeholder text ---
    for m in _PLACEHOLDER_RE.finditer(text):
        match_text = m.group(0)
        violations.append({
            "rule": "placeholder",
            "match": match_text.lower(),
            "context": _context_around(text, m.start(), m.end()),
            "penalty": -5,
        })
        advice.append(
            f"Remove placeholder '{match_text.lower()}' \u2014 this is unfinished "
            f"template text."
        )
        counts["placeholder"] += 1

    # --- 5. Rhythm analysis ---
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) >= 5:
        lengths = [len(s.split()) for s in sentences]
        mean = sum(lengths) / len(lengths)
        if mean > 0:
            variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
            std = math.sqrt(variance)
            cv = std / mean
            if cv < 0.3:
                violations.append({
                    "rule": "rhythm",
                    "match": "monotonous_rhythm",
                    "context": f"CV={cv:.2f} across {len(sentences)} sentences "
                               f"(mean {mean:.1f} words)",
                    "penalty": -5,
                })
                advice.append(
                    f"Sentence lengths are too uniform (CV={cv:.2f}) \u2014 "
                    f"vary short and long."
                )
                counts["rhythm"] += 1

    # --- 6. Em dash density ---
    em_dash_matches = list(_EM_DASH_RE.finditer(text))
    em_dash_count = len(em_dash_matches)
    if word_count > 0:
        ratio_per_150 = (em_dash_count / word_count) * 150
        if ratio_per_150 > 1.0:
            violations.append({
                "rule": "em_dash",
                "match": "em_dash_density",
                "context": f"{em_dash_count} em dashes in {word_count} words "
                           f"({ratio_per_150:.1f} per 150 words)",
                "penalty": -3,
            })
            advice.append(
                f"Too many em dashes ({em_dash_count} in {word_count} words) \u2014 "
                f"use other punctuation."
            )
            counts["em_dash"] += 1

    # --- 7. "X, not Y" contrast pairs ---
    contrast_matches = list(_CONTRAST_PAIR_RE.finditer(text))
    contrast_count = len(contrast_matches)
    for m in contrast_matches[:5]:  # cap at 5 violations recorded
        violations.append({
            "rule": "contrast_pair",
            "match": m.group(0),
            "context": _context_around(text, m.start(), m.end()),
            "penalty": -1,
        })
        advice.append(
            f"'{m.group(0)}' \u2014 'X, not Y' contrast \u2014 consider "
            f"rephrasing to avoid the Claude pattern."
        )
        counts["contrast_pairs"] += 1
    if contrast_count >= 2:
        advice.append(
            f"{contrast_count} 'X, not Y' contrasts \u2014 this is a Claude "
            f"rhetorical tic. Vary your phrasing."
        )

    # --- 7b. Setup-and-resolution ("This isn't X. It's Y.") ---
    setup_res_recorded = 0
    for pat in (_SETUP_RESOLUTION_A_RE, _SETUP_RESOLUTION_B_RE):
        for m in pat.finditer(text):
            if setup_res_recorded < 5:
                matched = m.group(0)
                violations.append({
                    "rule": "setup_resolution",
                    "match": matched,
                    "context": _context_around(text, m.start(), m.end()),
                    "penalty": -3,
                })
                advice.append(
                    f"'{matched}' \u2014 setup-and-resolution is a Claude "
                    f"rhetorical tic. Just state the point directly."
                )
                setup_res_recorded += 1
            counts["setup_resolution"] += 1

    # --- 8. Colon density (elaboration colons) ---
    stripped_text = _strip_code_blocks(text)
    # Process line by line to exclude headers
    colon_count = 0
    for line in stripped_text.split("\n"):
        # Skip markdown header lines
        if _MD_HEADER_LINE_RE.match(line):
            continue
        # Find all elaboration colons in this line
        for cm in _ELABORATION_COLON_RE.finditer(line):
            col_pos = cm.start()
            # Exclude URL colons (http: or https: immediately before)
            before = line[:col_pos + 1]
            if before.endswith("http:") or before.endswith("https:"):
                continue
            # Exclude JSON-like contexts
            snippet = line[col_pos:col_pos + 10]
            if _JSON_COLON_RE.match(snippet):
                continue
            colon_count += 1

    stripped_word_count = _word_count(stripped_text)
    if stripped_word_count > 0:
        colon_ratio_per_150 = (colon_count / stripped_word_count) * 150
        if colon_ratio_per_150 > 1.5:
            violations.append({
                "rule": "colon_density",
                "match": "colon_density",
                "context": f"{colon_count} elaboration colons in {stripped_word_count} "
                           f"words ({colon_ratio_per_150:.1f} per 150 words)",
                "penalty": -3,
            })
            advice.append(
                f"Too many elaboration colons ({colon_count} in "
                f"{stripped_word_count} words) \u2014 use periods or "
                f"restructure sentences."
            )
            counts["colon_density"] += 1

    # --- 9. Pithy evaluative fragments ---
    pithy_count = 0
    for sent in sentences:
        sent_stripped = sent.strip()
        if not sent_stripped:
            continue
        sent_words = sent_stripped.split()
        if len(sent_words) > 6:
            continue
        if _PITHY_PIVOT_RE.search(sent_stripped):
            if pithy_count < 3:  # cap at 3 violations
                violations.append({
                    "rule": "pithy_fragment",
                    "match": sent_stripped,
                    "context": sent_stripped,
                    "penalty": -2,
                })
                advice.append(
                    f"'{sent_stripped}' \u2014 pithy evaluative fragments are "
                    f"a Claude tell. Expand or cut."
                )
            pithy_count += 1
            counts["pithy_fragment"] += 1

    # --- 12. Bullet density ---
    non_empty_lines = [l for l in lines if l.strip()]
    total_non_empty = len(non_empty_lines)
    if total_non_empty > 0:
        bullet_count = sum(1 for l in non_empty_lines if _BULLET_DENSITY_RE.match(l))
        bullet_ratio = bullet_count / total_non_empty
        if bullet_ratio > 0.40:
            violations.append({
                "rule": "structural",
                "match": "bullet_density",
                "context": f"{bullet_count} of {total_non_empty} non-empty lines are bullets ({bullet_ratio:.0%})",
                "penalty": -8,
            })
            advice.append(
                f"Over {bullet_ratio:.0%} of lines are bullets \u2014 write prose instead of lists."
            )
            counts["bullet_density"] += 1

    # --- 13. Blockquote-as-thesis ---
    # Count blockquote lines outside fenced code blocks
    in_code_block = False
    blockquote_count = 0
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block and line.startswith(">"):
            blockquote_count += 1
    if blockquote_count >= 3:
        excess = blockquote_count - 2
        capped = min(excess, 4)
        bq_penalty = -3 * capped
        violations.append({
            "rule": "structural",
            "match": "blockquote_density",
            "context": f"{blockquote_count} blockquote lines \u2014 Claude uses these as thesis statements",
            "penalty": bq_penalty,
        })
        advice.append(
            f"{blockquote_count} blockquotes \u2014 integrate key claims into prose instead of pulling them out as blockquotes."
        )
        counts["blockquote_density"] += 1

    # --- 14. Bold-term bullet runs ---
    bold_bullet_run = 0
    for line in lines:
        if _BOLD_TERM_BULLET_RE.match(line):
            bold_bullet_run += 1
        else:
            if bold_bullet_run >= 3:
                violations.append({
                    "rule": "structural",
                    "match": "bold_bullet_list",
                    "context": f"Run of {bold_bullet_run} bold-term bullets",
                    "penalty": -5,
                })
                advice.append(
                    f"Run of {bold_bullet_run} bold-term bullets \u2014 this is an LLM listicle pattern. Use varied paragraph structure."
                )
                counts["bold_bullet_list"] += 1
            bold_bullet_run = 0
    # Check trailing run
    if bold_bullet_run >= 3:
        violations.append({
            "rule": "structural",
            "match": "bold_bullet_list",
            "context": f"Run of {bold_bullet_run} bold-term bullets",
            "penalty": -5,
        })
        advice.append(
            f"Run of {bold_bullet_run} bold-term bullets \u2014 this is an LLM listicle pattern. Use varied paragraph structure."
        )
        counts["bold_bullet_list"] += 1

    # --- 15. Horizontal rule overuse ---
    hr_count = len(_HORIZONTAL_RULE_RE.findall(text))
    if hr_count >= 4:
        violations.append({
            "rule": "structural",
            "match": "horizontal_rules",
            "context": f"{hr_count} horizontal rules \u2014 excessive section dividers",
            "penalty": -3,
        })
        advice.append(
            f"{hr_count} horizontal rules \u2014 section headers alone are sufficient, dividers are a crutch."
        )
        counts["horizontal_rules"] += 1

    # --- 16. Phrase reuse ---
    repeated_ngrams = _find_repeated_ngrams(text)
    phrase_reuse_recorded = 0
    for ng in repeated_ngrams:
        if phrase_reuse_recorded >= 5:
            break
        phrase = ng["phrase"]
        count = ng["count"]
        n = ng["n"]
        violations.append({
            "rule": "phrase_reuse",
            "match": phrase,
            "context": f"'{phrase}' ({n}-word phrase) appears {count} times",
            "penalty": -1,
        })
        advice.append(
            f"'{phrase}' appears {count} times \u2014 vary your phrasing "
            f"to avoid repetition."
        )
        counts["phrase_reuse"] += 1
        phrase_reuse_recorded += 1

    # --- Compute score ---
    total_penalty = sum(v["penalty"] for v in violations)

    # Exponential decay scoring with concentration multiplier
    weighted_sum = 0.0
    for v in violations:
        rule = v["rule"]
        penalty = abs(v["penalty"])
        # Map rule name to counts key: try exact, then with trailing "s"
        cat_count = counts.get(rule, 0) or counts.get(rule + "s", 0)
        # Check if this rule's count key is a Claude-specific category
        count_key = rule if rule in _CLAUDE_CATEGORIES else (rule + "s" if (rule + "s") in _CLAUDE_CATEGORIES else None)
        if count_key and count_key in _CLAUDE_CATEGORIES and cat_count > 1:
            weight = penalty * (1 + _ALPHA * (cat_count - 1))
        else:
            weight = penalty
        weighted_sum += weight

    density = weighted_sum / (word_count / 1000) if word_count > 0 else 0.0
    raw_score = 100 * math.exp(-_LAMBDA * density)
    score = max(0, min(100, round(raw_score)))

    if score >= 80:
        band = "clean"
    elif score >= 60:
        band = "light"
    elif score >= 40:
        band = "moderate"
    elif score >= 20:
        band = "heavy"
    else:
        band = "saturated"

    # Deduplicate advice
    seen_advice: set[str] = set()
    unique_advice: list[str] = []
    for a in advice:
        if a not in seen_advice:
            seen_advice.add(a)
            unique_advice.append(a)

    return {
        "score": score,
        "band": band,
        "word_count": word_count,
        "violations": violations,
        "counts": counts,
        "total_penalty": total_penalty,
        "weighted_sum": round(weighted_sum, 2),
        "density": round(density, 2),
        "advice": unique_advice,
    }


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp_server = FastMCP(MCP_SERVER_NAME)


@mcp_server.tool()
def check_slop(text: str) -> str:
    """Analyze text for AI slop patterns.

    Returns a JSON object with a score (0-100), band label, list of specific
    violations with context, and actionable advice for each issue found.
    """
    result = _analyze(text)
    return json.dumps(result, indent=2)


@mcp_server.tool()
def check_slop_file(file_path: str) -> str:
    """Analyze a file for AI slop patterns.

    Reads the file at the given path and runs the same analysis as check_slop.
    Returns a JSON object with a score (0-100), band label, list of specific
    violations with context, and actionable advice for each issue found.
    """
    p = Path(file_path)
    if not p.is_file():
        return json.dumps({"error": f"File not found: {file_path}"})
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": f"Could not read file: {e}"})
    result = _analyze(text)
    result["file"] = file_path
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp_server.run()
