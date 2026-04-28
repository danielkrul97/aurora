//! Sprint 26: BPE tokenizer smoke tests.

use aurora::bpe::BpeTokenizer;

#[test]
fn empty_corpus_yields_byte_level() {
    let tok = BpeTokenizer::train(&[], 256);
    assert_eq!(tok.vocab_size(), 256);
    assert!(tok.merges().is_empty());
}

#[test]
fn round_trip_byte_level() {
    let tok = BpeTokenizer::new_byte_level();
    let bytes = b"Hello, world!\n\xff\x00\x7f";
    let tokens = tok.encode(bytes);
    assert_eq!(tokens.len(), bytes.len());
    let decoded = tok.decode(&tokens);
    assert_eq!(decoded, bytes);
}

#[test]
fn training_creates_merges() {
    let corpus = b"abcabcabcabc".repeat(8);
    let tok = BpeTokenizer::train(&corpus, 270);
    assert!(!tok.merges().is_empty());
    assert!(tok.vocab_size() <= 270);

    // After training, encoded length must be ≤ raw byte length.
    let tokens = tok.encode(&corpus);
    assert!(
        tokens.len() < corpus.len(),
        "BPE must compress repeating corpus; got {} tokens for {} bytes",
        tokens.len(),
        corpus.len()
    );

    // Decoding still recovers exact bytes.
    let decoded = tok.decode(&tokens);
    assert_eq!(decoded, corpus);
}

#[test]
fn save_load_round_trip() {
    let corpus = b"The quick brown fox jumps over the lazy dog. ".repeat(20);
    let tok = BpeTokenizer::train(&corpus, 320);

    let serialized = tok.to_serialized();
    let loaded = BpeTokenizer::from_string(&serialized).unwrap();

    assert_eq!(loaded.vocab_size(), tok.vocab_size());
    assert_eq!(loaded.merges(), tok.merges());

    // Encode produces same token stream.
    let original = tok.encode(&corpus);
    let from_loaded = loaded.encode(&corpus);
    assert_eq!(original, from_loaded);
}

#[test]
fn compression_ratio_on_natural_text() {
    let corpus = b"the quick brown fox jumps over the lazy dog the quick brown fox";
    let big_corpus: Vec<u8> = corpus
        .iter()
        .copied()
        .chain(std::iter::repeat_n(b' ', 100))
        .cycle()
        .take(8_000)
        .collect();
    let tok = BpeTokenizer::train(&big_corpus, 512);
    let tokens = tok.encode(&big_corpus);
    let ratio = (big_corpus.len() as f32) / (tokens.len() as f32);
    eprintln!(
        "BPE compression: {} bytes → {} tokens (ratio {ratio:.2}×)",
        big_corpus.len(),
        tokens.len()
    );
    assert!(
        ratio >= 1.5,
        "expected ≥1.5× compression on repeating text, got {ratio:.2}×"
    );
}
