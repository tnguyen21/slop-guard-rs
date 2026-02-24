use super::*;

fn hp() -> Hyperparameters {
    Hyperparameters::default()
}

// ---------------------------------------------------------------------------
// rule_slop_words
// ---------------------------------------------------------------------------

#[test]
fn slop_words_detects_crucial() {
    let out = rule_slop_words("This is a crucial development in the field.", &hp());
    assert!(!out.violations.is_empty());
    assert!(out.violations.iter().any(|v| v.match_text == "crucial"));
}

#[test]
fn slop_words_detects_delve() {
    let out = rule_slop_words("Let us delve deeper into this topic here.", &hp());
    assert!(!out.violations.is_empty());
    assert!(out.violations.iter().any(|v| v.match_text == "delve"));
}

#[test]
fn slop_words_clean_no_violations() {
    let out = rule_slop_words("The committee met on Tuesday to discuss plans.", &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn slop_words_clean_common_prose() {
    let out = rule_slop_words("She walked to the store and bought some apples.", &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_slop_phrases
// ---------------------------------------------------------------------------

#[test]
fn slop_phrases_detects_worth_noting() {
    let out = rule_slop_phrases("It's worth noting that this matters a great deal.", &hp());
    assert!(!out.violations.is_empty());
    assert!(out
        .violations
        .iter()
        .any(|v| v.match_text == "it's worth noting"));
}

#[test]
fn slop_phrases_detects_not_just_but() {
    let out = rule_slop_phrases("This is not just fast, but also reliable.", &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn slop_phrases_clean_no_violations() {
    let out = rule_slop_phrases("The report was filed last week and reviewed.", &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn slop_phrases_clean_normal_sentence() {
    let out = rule_slop_phrases("Rain fell through the morning hours.", &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_structural
// ---------------------------------------------------------------------------

#[test]
fn structural_detects_bold_headers() {
    let text = "**Intro.** First section.\n**Body.** Second section.\n**End.** Third section.";
    let lines = split_lines(text);
    let out = rule_structural(text, &lines, &hp());
    assert!(out
        .violations
        .iter()
        .any(|v| v.match_text == "bold_header_explanation"));
}

#[test]
fn structural_detects_bullet_runs() {
    let text = "- one\n- two\n- three\n- four\n- five\n- six\n- seven";
    let lines = split_lines(text);
    let out = rule_structural(text, &lines, &hp());
    assert!(out
        .violations
        .iter()
        .any(|v| v.match_text == "excessive_bullets"));
}

#[test]
fn structural_clean_no_bold_headers() {
    let text = "First paragraph here.\n\nSecond paragraph here.";
    let lines = split_lines(text);
    let out = rule_structural(text, &lines, &hp());
    assert!(!out
        .violations
        .iter()
        .any(|v| v.match_text == "bold_header_explanation"));
}

#[test]
fn structural_clean_short_bullet_list() {
    let text = "- one\n- two\n- three";
    let lines = split_lines(text);
    let out = rule_structural(text, &lines, &hp());
    assert!(!out
        .violations
        .iter()
        .any(|v| v.match_text == "excessive_bullets"));
}

// ---------------------------------------------------------------------------
// rule_tone
// ---------------------------------------------------------------------------

#[test]
fn tone_detects_would_you_like() {
    let out = rule_tone("Would you like me to explain more about this?", &hp());
    assert!(!out.violations.is_empty());
    assert!(out
        .violations
        .iter()
        .any(|v| v.match_text == "would you like"));
}

#[test]
fn tone_detects_feel_free() {
    let out = rule_tone("Feel free to reach out with any questions.", &hp());
    assert!(!out.violations.is_empty());
    assert!(out
        .violations
        .iter()
        .any(|v| v.match_text == "feel free to"));
}

#[test]
fn tone_clean_no_violations() {
    let out = rule_tone("The project deadline is next Friday.", &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn tone_clean_neutral_prose() {
    let out = rule_tone("He closed the door and sat down at the desk.", &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_weasel
// ---------------------------------------------------------------------------

#[test]
fn weasel_detects_many_believe() {
    let out = rule_weasel("Many believe that the earth is round.", &hp());
    assert!(!out.violations.is_empty());
    assert!(out
        .violations
        .iter()
        .any(|v| v.match_text == "many believe"));
}

#[test]
fn weasel_detects_studies_show() {
    let out = rule_weasel("Studies show a strong correlation here.", &hp());
    assert!(!out.violations.is_empty());
    assert!(out
        .violations
        .iter()
        .any(|v| v.match_text == "studies show"));
}

#[test]
fn weasel_clean_no_violations() {
    let out = rule_weasel("The data shows a 15% increase over last quarter.", &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn weasel_clean_attributed_claim() {
    let out = rule_weasel("According to Smith (2020), the rate doubled.", &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_ai_disclosure
// ---------------------------------------------------------------------------

#[test]
fn ai_disclosure_detects_as_an_ai() {
    let out = rule_ai_disclosure("As an AI, I have some limitations.", &hp());
    assert!(!out.violations.is_empty());
    assert!(out.violations.iter().any(|v| v.match_text == "as an ai"));
}

#[test]
fn ai_disclosure_detects_language_model() {
    let out = rule_ai_disclosure("As a language model, I can help with that.", &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn ai_disclosure_clean_no_violations() {
    let out = rule_ai_disclosure("The author discussed the theory in chapter three.", &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn ai_disclosure_clean_normal_prose() {
    let out = rule_ai_disclosure("She opened her laptop and began typing.", &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_placeholder
// ---------------------------------------------------------------------------

#[test]
fn placeholder_detects_insert() {
    let out = rule_placeholder("Visit [insert URL here] for details.", &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn placeholder_detects_your() {
    let out = rule_placeholder("Contact [your name] for more info.", &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn placeholder_clean_no_violations() {
    let out = rule_placeholder("The link is https://example.com for reference.", &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn placeholder_clean_normal_brackets() {
    let out = rule_placeholder("See section [3] for the full table of results.", &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_rhythm
// ---------------------------------------------------------------------------

#[test]
fn rhythm_detects_monotonous() {
    // All sentences of roughly equal length
    let sentences: Vec<String> = vec![
        "The cat sat on the mat today.",
        "The dog ran in the park then.",
        "The bird flew over the lake now.",
        "The fish swam in the pond here.",
        "The cow stood in the field too.",
        "The hen pecked at the ground once.",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();
    let out = rule_rhythm(&sentences, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn rhythm_detects_uniform_lengths() {
    let sentences: Vec<String> = vec![
        "Five words in this one.",
        "Five words in that one.",
        "Five words in each one.",
        "Five words in last one.",
        "Five words in next one.",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();
    let out = rule_rhythm(&sentences, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn rhythm_clean_varied_lengths() {
    let sentences: Vec<String> = vec![
        "Short.",
        "A medium-length sentence that says something.",
        "Yes.",
        "This one is quite a bit longer because it has many words in it to vary the rhythm.",
        "Another short one.",
        "And yet another sentence that is somewhere in the middle of the length spectrum to add variety.",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();
    let out = rule_rhythm(&sentences, &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn rhythm_clean_too_few_sentences() {
    let sentences: Vec<String> = vec!["Just one sentence.".to_string()];
    let out = rule_rhythm(&sentences, &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_em_dash_density
// ---------------------------------------------------------------------------

#[test]
fn em_dash_detects_high_density() {
    // 4 em dashes in ~30 words should trigger
    let text = "The plan \u{2014} if you can call it that \u{2014} was bold. \
                The team \u{2014} all ten members \u{2014} agreed on the approach. \
                Results came quickly and exceeded expectations.";
    let wc = word_count(text);
    let out = rule_em_dash_density(text, wc, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn em_dash_detects_double_dash() {
    let text = "The idea -- controversial as it was -- gained traction. \
                The result -- surprising to many -- was clear. \
                This short text has many dashes.";
    let wc = word_count(text);
    let out = rule_em_dash_density(text, wc, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn em_dash_clean_no_dashes() {
    let text = "The committee met and reviewed the proposals carefully. They selected the second option after discussion.";
    let wc = word_count(text);
    let out = rule_em_dash_density(text, wc, &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn em_dash_clean_few_dashes() {
    // 1 em dash in ~150 words should be well under the 1.0 per 150 threshold
    let text = "The plan \u{2014} bold as it was worked out fine in the end and everyone was satisfied with the result. \
                The committee reviewed proposals and selected the second option after a long discussion that covered each one in detail. \
                Implementation begins next month with full support from the board and the executive team across the entire organization. \
                Results will be shared in the quarterly report that goes out to all stakeholders in every department worldwide. \
                The finance team will oversee the transition from the old system to the new one over the coming months ahead. \
                Each department submitted their estimates last week covering projected costs for the next two fiscal quarters of the year. \
                No objections were raised during the meeting. The chair adjourned at four and thanked everyone for their thoughtful work. \
                Minutes were distributed by email to all attendees and interested parties throughout the company and beyond its walls. \
                A follow-up meeting is scheduled for March with an expanded agenda that includes vendor selection and timeline review.";
    let wc = word_count(text);
    let out = rule_em_dash_density(text, wc, &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_contrast_pairs
// ---------------------------------------------------------------------------

#[test]
fn contrast_pairs_detects_x_not_y() {
    let out = rule_contrast_pairs("clarity, not complexity, is the goal here.", &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn contrast_pairs_detects_multiple() {
    let out = rule_contrast_pairs(
        "We want speed, not accuracy. We want clarity, not confusion.",
        &hp(),
    );
    assert!(out.violations.len() >= 2);
}

#[test]
fn contrast_pairs_clean_no_violations() {
    let out = rule_contrast_pairs("The project finished on time and under budget.", &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn contrast_pairs_clean_normal_comma() {
    let out = rule_contrast_pairs(
        "We bought apples, oranges, and bananas at the store.",
        &hp(),
    );
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_setup_resolution
// ---------------------------------------------------------------------------

#[test]
fn setup_resolution_detects_pattern_a() {
    let out = rule_setup_resolution(
        "This isn't about speed. It's about getting things done right.",
        &hp(),
    );
    assert!(!out.violations.is_empty());
}

#[test]
fn setup_resolution_detects_pattern_b() {
    let out = rule_setup_resolution(
        "It's not a simple fix. It's a complete overhaul of the system.",
        &hp(),
    );
    assert!(!out.violations.is_empty());
}

#[test]
fn setup_resolution_clean_no_violations() {
    let out = rule_setup_resolution(
        "The software update was released last week by the team.",
        &hp(),
    );
    assert!(out.violations.is_empty());
}

#[test]
fn setup_resolution_clean_simple_negation() {
    let out = rule_setup_resolution("This is not the final version of the document.", &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_colon_density
// ---------------------------------------------------------------------------

#[test]
fn colon_density_detects_high() {
    // Many elaboration colons in a short text
    let text = "the goal: simplicity. the method: iteration. the result: success. \
                the plan: execution. the outcome: growth.";
    let out = rule_colon_density(text, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn colon_density_detects_excessive_colons() {
    let text = "first point: explanation here. second point: more details. \
                third point: additional context. fourth point: final note.";
    let out = rule_colon_density(text, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn colon_density_clean_no_colons() {
    let out = rule_colon_density(
        "The meeting lasted two hours. Everyone agreed on the plan.",
        &hp(),
    );
    assert!(out.violations.is_empty());
}

#[test]
fn colon_density_clean_urls_ignored() {
    let out = rule_colon_density(
        "Visit https://example.com and http://test.org for details about the project.",
        &hp(),
    );
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_pithy_fragments
// ---------------------------------------------------------------------------

#[test]
fn pithy_detects_short_pivot() {
    let sentences = vec!["Simple, but effective.".to_string()];
    let out = rule_pithy_fragments(&sentences, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn pithy_detects_yet_pivot() {
    let sentences = vec!["Brief, yet powerful.".to_string()];
    let out = rule_pithy_fragments(&sentences, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn pithy_clean_long_sentence() {
    let sentences = vec![
        "This is a rather long sentence that discusses many things, but it goes on for a while."
            .to_string(),
    ];
    let out = rule_pithy_fragments(&sentences, &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn pithy_clean_no_pivot() {
    let sentences = vec!["The meeting ended early.".to_string()];
    let out = rule_pithy_fragments(&sentences, &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_bullet_density
// ---------------------------------------------------------------------------

#[test]
fn bullet_density_detects_high() {
    let text = "Intro line\n- a\n- b\n- c\n- d\n- e\n- f\n- g\n- h\n- i";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_bullet_density(&line_refs, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn bullet_density_detects_numbered() {
    let text = "Intro\n1. a\n2. b\n3. c\n4. d\n5. e\n6. f\n7. g\n8. h\n9. i";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_bullet_density(&line_refs, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn bullet_density_clean_few_bullets() {
    let text = "Paragraph one.\n- item a\n- item b\nParagraph two.\nParagraph three.";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_bullet_density(&line_refs, &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn bullet_density_clean_all_prose() {
    let text = "First paragraph here.\nSecond paragraph here.\nThird paragraph here.";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_bullet_density(&line_refs, &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_blockquote_density
// ---------------------------------------------------------------------------

#[test]
fn blockquote_detects_excessive() {
    let text = "> quote one\n> quote two\n> quote three\n> quote four";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_blockquote_density(&line_refs, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn blockquote_detects_many_lines() {
    let text = "> a\n> b\n> c\nsome text\n> d\n> e\n> f";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_blockquote_density(&line_refs, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn blockquote_clean_few_quotes() {
    let text = "> one quote\nsome prose here\nmore prose below";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_blockquote_density(&line_refs, &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn blockquote_clean_no_quotes() {
    let text = "Just normal prose.\nAnother line here.\nAnd a third line.";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_blockquote_density(&line_refs, &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_bold_bullet_runs
// ---------------------------------------------------------------------------

#[test]
fn bold_bullets_detects_run() {
    let text =
        "- **Term1** description one\n- **Term2** description two\n- **Term3** description three";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_bold_bullet_runs(&line_refs, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn bold_bullets_detects_numbered_run() {
    let text = "1. **Alpha** first item\n2. **Beta** second item\n3. **Gamma** third item";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_bold_bullet_runs(&line_refs, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn bold_bullets_clean_no_bold() {
    let text = "- plain item one\n- plain item two\n- plain item three";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_bold_bullet_runs(&line_refs, &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn bold_bullets_clean_mixed() {
    let text = "- **Bold** one\nPlain line\n- **Bold** two";
    let lines = split_lines(text);
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
    let out = rule_bold_bullet_runs(&line_refs, &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_horizontal_rules
// ---------------------------------------------------------------------------

#[test]
fn horizontal_rules_detects_excessive() {
    let text = "Section 1\n---\nSection 2\n---\nSection 3\n---\nSection 4\n---\nSection 5";
    let out = rule_horizontal_rules(text, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn horizontal_rules_detects_asterisks() {
    let text = "Part A\n***\nPart B\n***\nPart C\n***\nPart D\n***\nPart E";
    let out = rule_horizontal_rules(text, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn horizontal_rules_clean_few() {
    let text = "Section 1\n---\nSection 2\n---\nSection 3";
    let out = rule_horizontal_rules(text, &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn horizontal_rules_clean_none() {
    let text = "Just paragraphs of text.\n\nAnother paragraph here.";
    let out = rule_horizontal_rules(text, &hp());
    assert!(out.violations.is_empty());
}

// ---------------------------------------------------------------------------
// rule_phrase_reuse
// ---------------------------------------------------------------------------

#[test]
fn phrase_reuse_detects_repeated() {
    let text = "the quick brown fox jumped over the fence. \
                the quick brown fox ran through the field. \
                the quick brown fox slept under the tree. \
                the quick brown fox ate some food.";
    let out = rule_phrase_reuse(text, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn phrase_reuse_detects_long_ngram() {
    let text = "according to the latest report we found something. \
                according to the latest report we saw a trend. \
                according to the latest report we noted a change.";
    let out = rule_phrase_reuse(text, &hp());
    assert!(!out.violations.is_empty());
}

#[test]
fn phrase_reuse_clean_varied_text() {
    let text = "The morning started with rain. By noon the sun came out. \
                Afternoon brought warm winds. Evening settled in quietly.";
    let out = rule_phrase_reuse(text, &hp());
    assert!(out.violations.is_empty());
}

#[test]
fn phrase_reuse_clean_short_text() {
    let text = "Hello world.";
    let out = rule_phrase_reuse(text, &hp());
    assert!(out.violations.is_empty());
}
