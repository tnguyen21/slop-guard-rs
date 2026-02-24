use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;
use slop_guard::{analyze_with_config, Hyperparameters, RULE_NAMES};

#[derive(Parser)]
#[command(
    name = "slop-guard-calibrate",
    about = "Calibrate slop-guard penalties against a human-written corpus"
)]
struct Cli {
    /// Directory containing .txt corpus files
    corpus_dir: PathBuf,

    /// Output path for calibrated config JSON
    #[arg(long, default_value = "calibrated.json")]
    output: PathBuf,

    /// Minimum word count to include a document
    #[arg(long, default_value_t = 50)]
    min_words: usize,
}

fn main() {
    let cli = Cli::parse();

    let hp = Hyperparameters::default();

    // Collect .txt files
    let entries: Vec<PathBuf> = std::fs::read_dir(&cli.corpus_dir)
        .unwrap_or_else(|e| {
            eprintln!("Error reading corpus dir {:?}: {e}", cli.corpus_dir);
            std::process::exit(1);
        })
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if entries.is_empty() {
        eprintln!("No .txt files found in {:?}", cli.corpus_dir);
        std::process::exit(1);
    }

    // Per-rule support tracking: how many docs trigger each rule
    let mut rule_support: HashMap<String, usize> = HashMap::new();
    for name in RULE_NAMES {
        rule_support.insert(name.to_string(), 0);
    }
    let mut total_docs = 0usize;

    for path in &entries {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Skipping {:?}: {e}", path);
                continue;
            }
        };

        let wc = text.split_whitespace().count();
        if wc < cli.min_words {
            continue;
        }

        total_docs += 1;
        let result = analyze_with_config(&text, &hp);

        for name in RULE_NAMES {
            if let Some(&count) = result.counts.get(*name) {
                if count > 0 {
                    *rule_support.get_mut(*name).unwrap() += 1;
                }
            }
        }
    }

    if total_docs == 0 {
        eprintln!(
            "No documents met the minimum word count ({}).",
            cli.min_words
        );
        std::process::exit(1);
    }

    // Print support table to stderr
    eprintln!("Corpus: {} documents from {:?}", total_docs, cli.corpus_dir);
    eprintln!("{:<25} {:>8} {:>10}", "Rule", "Support", "Ratio");
    eprintln!("{}", "-".repeat(45));

    for name in RULE_NAMES {
        let support = *rule_support.get(*name).unwrap();
        let ratio = support as f64 / total_docs as f64;
        eprintln!("{:<25} {:>8} {:>9.2}%", name, support, ratio * 100.0);
    }

    // Apply fit_penalty: scale = clamp(1.5 - support_ratio, 0.5, 1.75)
    // magnitude = max(1, round(|base_penalty| * scale))
    // Result is always negative (penalty).
    let mut calibrated = hp.clone();

    macro_rules! calibrate_penalty {
        ($field:ident, $rule:expr) => {
            let support_ratio = *rule_support.get($rule).unwrap() as f64 / total_docs as f64;
            let scale = (1.5 - support_ratio).clamp(0.5, 1.75);
            let base = calibrated.$field.unsigned_abs() as f64;
            let magnitude = (base * scale).round().max(1.0) as i32;
            calibrated.$field = -magnitude;
        };
    }

    calibrate_penalty!(slop_word_penalty, "slop_words");
    calibrate_penalty!(slop_phrase_penalty, "slop_phrases");
    calibrate_penalty!(structural_bold_header_penalty, "structural");
    calibrate_penalty!(structural_bullet_run_penalty, "structural");
    calibrate_penalty!(triadic_penalty, "structural");
    calibrate_penalty!(tone_penalty, "tone");
    calibrate_penalty!(sentence_opener_penalty, "tone");
    calibrate_penalty!(weasel_penalty, "weasel");
    calibrate_penalty!(ai_disclosure_penalty, "ai_disclosure");
    calibrate_penalty!(placeholder_penalty, "placeholder");
    calibrate_penalty!(rhythm_penalty, "rhythm");
    calibrate_penalty!(em_dash_penalty, "em_dash");
    calibrate_penalty!(contrast_penalty, "contrast_pairs");
    calibrate_penalty!(setup_resolution_penalty, "setup_resolution");
    calibrate_penalty!(colon_density_penalty, "colon_density");
    calibrate_penalty!(pithy_penalty, "pithy_fragment");
    calibrate_penalty!(bullet_density_penalty, "bullet_density");
    calibrate_penalty!(blockquote_penalty_step, "blockquote_density");
    calibrate_penalty!(bold_bullet_run_penalty, "bold_bullet_list");
    calibrate_penalty!(horizontal_rule_penalty, "horizontal_rules");
    calibrate_penalty!(phrase_reuse_penalty, "phrase_reuse");

    // Write calibrated config
    let json = serde_json::to_string_pretty(&calibrated).unwrap();
    std::fs::write(&cli.output, &json).unwrap_or_else(|e| {
        eprintln!("Error writing {:?}: {e}", cli.output);
        std::process::exit(1);
    });

    eprintln!("\nCalibrated config written to {:?}", cli.output);
}
