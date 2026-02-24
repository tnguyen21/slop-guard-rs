use std::io::Read;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "slop-guard",
    about = "Detect AI slop patterns in prose",
    version
)]
struct Cli {
    /// File paths to analyze (reads stdin if none provided)
    files: Vec<String>,

    /// Path to a JSON config file with hyperparameter overrides
    #[arg(long)]
    config: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let hp = if let Some(config_path) = &cli.config {
        let json = std::fs::read_to_string(config_path).unwrap_or_else(|e| {
            eprintln!("Error reading config {config_path}: {e}");
            std::process::exit(1);
        });
        let overrides: slop_guard::HyperparametersOverride = serde_json::from_str(&json)
            .unwrap_or_else(|e| {
                eprintln!("Error parsing config {config_path}: {e}");
                std::process::exit(1);
            });
        slop_guard::Hyperparameters::default().with_overrides(&overrides)
    } else {
        slop_guard::Hyperparameters::default()
    };

    if cli.files.is_empty() {
        let mut input = String::new();
        std::io::stdin()
            .read_to_string(&mut input)
            .expect("Failed to read stdin");
        let result = slop_guard::analyze_with_config(&input, &hp);
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        for path in &cli.files {
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
                eprintln!("Error reading {path}: {e}");
                std::process::exit(1);
            });
            let result = slop_guard::analyze_with_config(&text, &hp);
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
    }
}
