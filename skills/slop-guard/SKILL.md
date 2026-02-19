---
name: slop-guard
description: Run slop-guard on files or the text you just wrote to detect AI writing patterns, then revise to eliminate violations. Use when writing prose, documentation, blog posts, READMEs, or any user-facing text. Also use when the user says "slop-guard", "deslop", or asks to check for AI slop.
argument-hint: "[file paths or 'last' to check your most recent output]"
allowed-tools: Bash(slop-guard *), Bash(which *), Bash(cargo install *), Bash(curl *), Bash(sh *), Read, Edit, Write, Grep
---

# De-Slop Skill

You have access to `slop-guard`, a CLI tool that detects AI writing patterns in prose. Use it to analyze and then revise text to sound less like AI output.

## Step 0: Ensure slop-guard is installed

Before anything else, check if `slop-guard` is on PATH:

```bash
which slop-guard
```

If not found, check for cargo:

```bash
which cargo
```

- If `cargo` exists: ask the user for permission, then run `cargo install slop-guard`
- If `cargo` is missing: ask the user for permission to install Rust via rustup, then install the crate:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  # reload PATH, then:
  cargo install slop-guard
  ```

Do NOT proceed until slop-guard is available.

## Workflow

1. **Identify the target text.** Either:
   - The user provides file path(s) as `$ARGUMENTS` — analyze those files
   - The user says "last" or provides no arguments — analyze the most recent prose you wrote in this conversation
   - The user pastes text directly — write it to a temp file first

2. **Run slop-guard** on the target:
   ```bash
   slop-guard <file>
   ```
   The output is JSON with: `score` (0-100, higher=cleaner), `band` (clean/light/moderate/heavy/saturated), `violations` (each with rule, match, context, penalty), and `advice` (actionable suggestions).

3. **Report the results concisely:**
   - State the score and band
   - List the top violations grouped by category (don't dump the full JSON)
   - Quote the specific advice items that matter most

4. **Revise the text** to fix the violations:
   - Replace slop words (crucial, groundbreaking, delve, tapestry, etc.) with specific, concrete alternatives
   - Cut slop phrases ("it's worth noting", "let's dive in") — just state the point
   - Break up structural patterns: no bold-header listicles, no 6+ bullet runs, vary paragraph structure
   - Remove tone markers: no "feel free to", "would you like", "certainly" as opener
   - Remove weasel phrases: either cite a source or own the claim
   - Remove AI disclosure: no "as an AI", "as a language model"
   - Vary sentence rhythm: mix short and long sentences
   - Reduce em dash and colon density
   - Eliminate "X, not Y" contrast pairs and "This isn't X. It's Y." setup-resolution flips
   - Cut pithy evaluative fragments ("Simple, but effective.")
   - Vary repeated multi-word phrases

5. **Re-run slop-guard** on the revised text to verify improvement. Target score >= 80 (clean band). If still below, iterate.

6. **Show the user the before/after scores** and a brief summary of what changed.

## Important

- Don't over-correct into bland prose. The goal is natural human writing, not sterile text.
- Preserve the author's voice and intent. Fix the patterns, not the ideas.
- If the score is already >= 80 (clean), say so and only suggest minor tweaks if any.
- When revising files, use Edit to make targeted changes rather than rewriting everything.
