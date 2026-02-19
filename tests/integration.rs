use slop_guard::analyze;

#[test]
fn clean_text_scores_high() {
    let text = "The committee met on Tuesday. \
                They reviewed three proposals and selected the second one after a long discussion that covered each option in detail. \
                Implementation begins next month. \
                Results will be shared in the quarterly report that goes out to all stakeholders across the organization. \
                The finance team will oversee the transition. \
                Each department submitted their estimates last week, covering projected costs for the next two fiscal quarters. \
                No objections were raised. \
                The chair adjourned the meeting at four o'clock and thanked everyone for their thoughtful contributions to the process. \
                Minutes were distributed by email. \
                A follow-up meeting is scheduled for March, with an expanded agenda that includes vendor selection and timeline review.";
    let result = analyze(text);
    assert!(
        result.score >= 80,
        "Clean text should score >= 80, got {}",
        result.score
    );
    assert_eq!(result.band, "clean");
}

#[test]
fn slop_heavy_text_scores_low() {
    let text = "Let me delve into this crucial and groundbreaking journey. \
                It's worth noting that this seamless paradigm is pivotal. \
                Furthermore, this comprehensive tapestry showcases the holistic landscape. \
                As an AI, I hope this helps. Let me know if you need more. \
                Feel free to explore this innovative realm. \
                The answer? It's simpler than you think. \
                Let's dive in and embrace this revolutionary odyssey.";
    let result = analyze(text);
    assert!(
        result.score < 40,
        "Slop-heavy text should score < 40, got {}",
        result.score
    );
}

#[test]
fn short_text_returns_100() {
    let result = analyze("Hello world.");
    assert_eq!(result.score, 100);
    assert_eq!(result.band, "clean");
    assert!(result.violations.is_empty());
    assert!(result.advice.is_empty());
}

#[test]
fn detects_slop_words() {
    let text = "This is a crucial development in a groundbreaking field. \
                The seamless integration is paramount to success. \
                We must delve deeper into this pivotal matter.";
    let result = analyze(text);
    let slop_word_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.rule == "slop_word")
        .collect();
    assert!(!slop_word_violations.is_empty(), "Should detect slop words");
    let words: Vec<&str> = slop_word_violations
        .iter()
        .map(|v| v.match_text.as_str())
        .collect();
    assert!(words.contains(&"crucial"));
    assert!(words.contains(&"groundbreaking"));
    assert!(words.contains(&"seamless"));
    assert!(words.contains(&"paramount"));
    assert!(words.contains(&"delve"));
    assert!(words.contains(&"pivotal"));
}

#[test]
fn detects_slop_phrases() {
    let text = "It's worth noting that this approach works well. \
                At the end of the day, results matter most. \
                Let me know if you have any questions about this.";
    let result = analyze(text);
    let phrase_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.rule == "slop_phrase")
        .collect();
    assert!(!phrase_violations.is_empty(), "Should detect slop phrases");
    let phrases: Vec<&str> = phrase_violations
        .iter()
        .map(|v| v.match_text.as_str())
        .collect();
    assert!(phrases.contains(&"it's worth noting"));
    assert!(phrases.contains(&"at the end of the day"));
    assert!(phrases.contains(&"let me know if"));
}

#[test]
fn detects_structural_patterns() {
    let text = "\
**Introduction.** This is the first section about the topic.\n\
**Background.** This is the second section about background.\n\
**Methods.** This is the third section about methods.\n\
**Results.** This is the fourth section about results.";
    let result = analyze(text);
    let structural_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.rule == "structural" && v.match_text == "bold_header_explanation")
        .collect();
    assert!(
        !structural_violations.is_empty(),
        "Should detect bold header structural pattern"
    );
}

#[test]
fn json_output_is_valid() {
    let text = "This is a test of the analysis system for detecting AI slop patterns in text.";
    let result = analyze(text);
    let json = serde_json::to_string_pretty(&result).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.get("score").is_some());
    assert!(parsed.get("band").is_some());
    assert!(parsed.get("word_count").is_some());
    assert!(parsed.get("violations").is_some());
    assert!(parsed.get("counts").is_some());
    assert!(parsed.get("total_penalty").is_some());
    assert!(parsed.get("weighted_sum").is_some());
    assert!(parsed.get("density").is_some());
    assert!(parsed.get("advice").is_some());
}

#[test]
fn detects_ai_disclosure() {
    let text = "As an AI, I don't have personal opinions on this matter. \
                I cannot browse the internet for current information. \
                Up to my last training data, this was the accepted view.";
    let result = analyze(text);
    let ai_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.rule == "ai_disclosure")
        .collect();
    assert!(
        !ai_violations.is_empty(),
        "Should detect AI disclosure patterns"
    );
}

#[test]
fn detects_tone_markers() {
    let text = "Would you like me to explain further about this topic? \
                Let me know if you have any other questions. \
                I hope this explanation has been helpful to you. \
                Feel free to ask about anything else.";
    let result = analyze(text);
    let tone_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.rule == "tone")
        .collect();
    assert!(!tone_violations.is_empty(), "Should detect tone markers");
}

#[test]
fn detects_weasel_phrases() {
    let text = "Many believe that this approach is correct. \
                Experts suggest there are better alternatives. \
                Studies show that the results are promising. \
                Research suggests a different conclusion entirely.";
    let result = analyze(text);
    let weasel_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.rule == "weasel")
        .collect();
    assert!(
        !weasel_violations.is_empty(),
        "Should detect weasel phrases"
    );
}

#[test]
fn detects_placeholder_text() {
    let text = "Please visit [insert URL here] for more information. \
                You can also [describe your experience] in the form below. \
                Contact [your name] for additional details about the program.";
    let result = analyze(text);
    let placeholder_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.rule == "placeholder")
        .collect();
    assert!(
        !placeholder_violations.is_empty(),
        "Should detect placeholder text"
    );
}

#[test]
fn detects_bullet_runs() {
    let text = "Here is a list:\n\
                - First item in the list\n\
                - Second item in the list\n\
                - Third item in the list\n\
                - Fourth item in the list\n\
                - Fifth item in the list\n\
                - Sixth item in the list\n\
                - Seventh item in the list\n\
                And that concludes the list.";
    let result = analyze(text);
    let bullet_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.rule == "structural" && v.match_text == "excessive_bullets")
        .collect();
    assert!(
        !bullet_violations.is_empty(),
        "Should detect excessive bullet runs"
    );
}
