# Project Context — slop-guard

## Overview
A CLI tool and Rust library that scores prose 0-100 for ~80 regex-based rules detecting common LLM writing tics ("AI slop").

## Architecture
- **Single-file library** (`src/lib.rs`, ~1700 lines): all rules, scoring, and data structures live here. No module splits.
- **18 rule functions** (`rule_slop_words`, `rule_slop_phrases`, `rule_structural`, `rule_tone`, `rule_weasel`, `rule_ai_disclosure`, `rule_placeholder`, `rule_rhythm`, `rule_em_dash_density`, `rule_contrast_pairs`, `rule_setup_resolution`, `rule_colon_density`, `rule_pithy_fragments`, `rule_bullet_density`, `rule_blockquote_density`, `rule_bold_bullet_runs`, `rule_horizontal_rules`, `rule_phrase_reuse`) — each returns `RuleOutput` containing violations and counts.
- **Scoring pipeline**: `analyze_with_config()` runs all rules, sums weighted penalties, computes density (penalty/word_count), maps to 0-100 score and band (clean/light/moderate/heavy/saturated).
- **Hyperparameters struct**: ~45 tunable fields (penalties, thresholds, caps) with `Default` impl and JSON override support via `HyperparametersOverride`.
- **Regex patterns**: compiled once via `once_cell::sync::Lazy` statics (`SLOP_WORD_RE`, `SLOP_PHRASE_RE`, `TONE_RE`, etc.).

Data flow: text in → split into sentences/lines → each rule function scans for patterns → violations aggregated → weighted sum → density → clamped score → JSON output.

## Key Files
- `src/lib.rs` — Core library: all rule functions, scoring logic, data types, regex patterns, hyperparameters
- `src/main.rs` — CLI entry point: reads files or stdin, outputs JSON analysis
- `src/bin/calibrate.rs` — Calibration binary: runs corpus through rules, outputs adjusted penalty config
- `src/rule_tests.rs` — Unit tests for individual rule functions (included via `#[cfg(test)] mod rule_tests`)
- `tests/integration.rs` — Integration tests: end-to-end scoring, band assignment, violation detection
- `Cargo.toml` — Package config, dependencies, binary targets
- `.github/workflows/ci.yml` — CI: test (3 OS matrix), lint (fmt + clippy)
- `.github/workflows/release.yml` — Release: cross-platform builds, .deb, Homebrew tap, crates.io publish
- `hooks/pre-commit` — Local pre-commit: fmt check, clippy, tests
- `skills/slop-guard/SKILL.md` — Claude Code skill definition for running slop-guard interactively
- `homebrew-tap/Formula/slop-guard.rb` — Homebrew formula template

## Build & Test
- **Language**: Rust, edition 2021
- **Package manager**: Cargo
- **Build**: `cargo build --release`
- **Test**: `cargo test --all-features`
- **Lint**: `cargo clippy --all-targets --all-features -- -D warnings`
- **Format**: `cargo fmt --all -- --check`
- **Type check**: Compiler (Rust is statically typed; no separate type checker)
- **Pre-commit**: `hooks/pre-commit` runs fmt check → clippy → tests (not installed via git hooks by default; manual symlink)
- **Quirks**: Single lib.rs is large (~1700 lines); all rule tests use `#[cfg(test)] mod rule_tests` with `use super::*` to access private functions. CI tests on ubuntu/macos/windows matrix.

## Conventions
- All code in a single `lib.rs` — no sub-modules. Rule tests in a separate file included as `mod rule_tests`.
- Rule functions are private `fn rule_<name>(...) -> RuleOutput` — only `analyze` and `analyze_with_config` are public.
- Regex patterns use `once_cell::sync::Lazy<Regex>` statics for compile-once semantics.
- Violations carry `violation_type`, `rule`, `match_text`, `context`, and `penalty` — JSON field names use `#[serde(rename)]`.
- Hyperparameters use a paired `Hyperparameters` / `HyperparametersOverride` pattern (all-Option fields) with `with_overrides()` merge.
- Tests follow 2+2 pattern per rule: 2 positive detection tests, 2 negative (clean text) tests.
- `RULE_NAMES` constant enumerates all rule identifiers for calibration and count maps.
- No async code. No unsafe. No feature flags.
- CLI outputs `serde_json::to_string_pretty` JSON to stdout; errors to stderr.

## Dependencies & Integration
- `regex` (1.x) — all pattern matching
- `once_cell` (1.x) — lazy static regex compilation
- `serde` + `serde_json` (1.x) — JSON serialization of results and hyperparameter config
- `clap` (4.x, derive feature) — CLI argument parsing for both binaries
- No external APIs, databases, or network calls. Pure text-in, JSON-out.
- Published to crates.io as `slop-guard`. Distributed via Homebrew tap (`tnguyen21/homebrew-slop-guard`) and GitHub releases (Linux .deb, macOS universal, Windows).

## Gotchas
- `lib.rs` is a monolith — any structural change touches one large file. Read the section headers to navigate.
- Private rule functions are only testable via the `rule_tests` module (uses `super::*`). Integration tests can only exercise `analyze()`.
- The `calibrate` binary shares `Hyperparameters` but uses a macro (`calibrate_penalty!`) to scale fields — adding a new penalty field requires updating both the struct and the calibrate macro.
- Adding a new rule requires: (1) add rule function, (2) add to `RULE_NAMES`, (3) call it in `analyze_with_config`, (4) add penalty field to `Hyperparameters` + `HyperparametersOverride` + `Default` + `with_overrides`, (5) add calibrate macro call, (6) add tests in `rule_tests.rs`.
- Score bands are configurable via hyperparameters but defaults are: clean ≥ 80, light ≥ 60, moderate ≥ 40, heavy ≥ 20, saturated < 20.
- The `hooks/pre-commit` file exists but isn't auto-installed — must be manually symlinked to `.git/hooks/pre-commit`.
