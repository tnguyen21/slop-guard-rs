# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-19

### Added
- Initial release
- ~80 regex-based rules detecting AI slop patterns in prose
- Scoring engine: exponential decay with concentration amplification
- Score bands: clean (80-100), light (60-79), moderate (40-59), heavy (20-39), saturated (0-19)
- CLI reads from stdin or file paths, outputs JSON
- Library exposes `analyze()` function for embedding
- Rules cover: slop words, slop phrases, structural patterns, tone markers, weasel phrases, AI disclosure, placeholder text, rhythm monotony, em dash density, contrast pairs, setup-resolution flips, colon density, pithy fragments, bullet density, blockquote density, bold bullet runs, horizontal rules, phrase reuse
