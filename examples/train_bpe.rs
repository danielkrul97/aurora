#![allow(clippy::cast_precision_loss)]

//! Sprint 26: train a BPE tokenizer on a corpus.
//!
//! Reads a file, runs greedy byte-pair merging until reaching the
//! requested vocab size, saves the merge list to disk, and reports
//! compression ratio.
//!
//! Usage:
//!   cargo run --release --example train_bpe -- \
//!     <corpus_path> [vocab_size] [out_path]
//!
//! Defaults: vocab_size=4096, out_path=`<corpus_basename>.bpe`.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use aurora::bpe::BpeTokenizer;
use aurora::text_io::load_corpus;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "data/tiny.txt".into());
    let vocab: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let out: PathBuf = args.get(3).map_or_else(
        || {
            let stem = std::path::Path::new(&path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("corpus");
            PathBuf::from(format!("data/{stem}_{vocab}.bpe"))
        },
        PathBuf::from,
    );

    let corpus = match load_corpus(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to read corpus from {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("corpus: {path} ({} bytes)", corpus.len());
    println!("target vocab: {vocab}");
    println!("training BPE…");

    let t0 = Instant::now();
    let tok = BpeTokenizer::train(&corpus, vocab);
    let elapsed = t0.elapsed().as_secs_f64();

    let tokens = tok.encode(&corpus);
    let ratio = (corpus.len() as f32) / (tokens.len() as f32);

    println!();
    println!("training done in {elapsed:.1}s");
    println!("vocab size: {}", tok.vocab_size());
    println!("merges: {}", tok.merges().len());
    println!(
        "encoded length: {} tokens ({ratio:.2}× compression vs {} bytes)",
        tokens.len(),
        corpus.len()
    );

    if let Err(e) = tok.save(&out) {
        eprintln!("save failed: {e}");
        return ExitCode::FAILURE;
    }
    println!("saved to {}", out.display());

    // Quick demo: print first few merges (longest token expansions).
    let mut merges_sorted: Vec<(u32, &[u8])> = (256..tok.vocab_size() as u32)
        .map(|id| (id, tok.token_bytes(id)))
        .collect();
    merges_sorted.sort_by_key(|(_, bytes)| std::cmp::Reverse(bytes.len()));
    println!();
    println!("top 10 longest merged tokens:");
    for (id, bytes) in merges_sorted.iter().take(10) {
        let text = String::from_utf8_lossy(bytes);
        let display: String = text.chars().take(40).collect();
        println!("  id={id:>5} len={:>2}  {display:?}", bytes.len());
    }

    ExitCode::SUCCESS
}
