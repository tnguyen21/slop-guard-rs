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
}

fn main() {
    let cli = Cli::parse();

    if cli.files.is_empty() {
        let mut input = String::new();
        std::io::stdin()
            .read_to_string(&mut input)
            .expect("Failed to read stdin");
        let result = slop_guard::analyze(&input);
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        for path in &cli.files {
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
                eprintln!("Error reading {path}: {e}");
                std::process::exit(1);
            });
            let result = slop_guard::analyze(&text);
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
    }
}
