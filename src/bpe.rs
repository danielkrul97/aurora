#![allow(
    clippy::cast_possible_truncation,
    clippy::must_use_candidate,
    clippy::missing_const_for_fn,
    clippy::format_push_string,
    clippy::items_after_statements,
    clippy::single_match_else,
    clippy::should_implement_trait
)]

//! Byte-pair encoding (BPE) tokenizer.
//!
//! Trained offline on a corpus by greedy byte-pair merging: count all
//! adjacent token pairs in the current token stream, merge the most
//! frequent pair into a new token, repeat until reaching the target
//! vocabulary size. The first 256 tokens are reserved for raw bytes
//! (so any byte sequence remains encodable). Subsequent tokens are
//! merge results.
//!
//! The merge list IS the model — encoding is a deterministic priority
//! queue over merges. Decoding is a flat lookup table from token id
//! to its byte expansion.
//!
//! Storage format is plain text: one merge per line `"left right"`,
//! preceded by a `vocab_size=N` header line.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    /// Merge table indexed by rank (lower = higher priority).
    /// Element `i` is `(left_token, right_token)` whose merge produces
    /// the new token at id `256 + i`.
    merges: Vec<(u32, u32)>,
    /// `vocab[token_id]` is the byte expansion of `token_id`.
    /// `vocab[0..256]` is the literal bytes.
    vocab: Vec<Vec<u8>>,
}

impl BpeTokenizer {
    /// Empty tokenizer with byte-level vocab (256 tokens, no merges).
    pub fn new_byte_level() -> Self {
        let vocab = (0..256_u32).map(|b| vec![b as u8]).collect();
        Self {
            merges: Vec::new(),
            vocab,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    /// Look up the byte expansion of a token id.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.vocab[id as usize]
    }

    /// Train BPE on `corpus` until `target_vocab_size` is reached or
    /// no more profitable merge is found (best-pair count drops to 1).
    ///
    /// Uses a doubly-linked list of tokens (prev/next pointers) plus
    /// incremental pair-count updates: merging pair `(L, R) → N` only
    /// needs to scan known match positions and locally fix the four
    /// affected pair counts per match. Total work per merge is
    /// proportional to the number of matches, not the corpus length.
    ///
    /// # Panics
    /// Panics if `target_vocab_size < 256`.
    pub fn train(corpus: &[u8], target_vocab_size: usize) -> Self {
        assert!(
            target_vocab_size >= 256,
            "target_vocab_size must be ≥ 256 (byte-level minimum)"
        );
        let mut tok = Self::new_byte_level();
        if corpus.is_empty() {
            return tok;
        }

        // Linked list of tokens: token[i] holds the value, prev[i] /
        // next[i] are doubly-linked indices. Sentinels at i = -1 are
        // represented by `usize::MAX` (= NIL).
        const NIL: usize = usize::MAX;
        let n = corpus.len();
        let mut value: Vec<u32> = corpus.iter().map(|&b| u32::from(b)).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { NIL } else { i - 1 }).collect();
        let mut next: Vec<usize> =
            (0..n).map(|i| if i + 1 == n { NIL } else { i + 1 }).collect();

        // Pair counts + the inverted index of which positions hold each
        // pair. (Position `i` "holds" pair `(value[i], value[next[i]])`.)
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
        let mut positions: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for i in 0..n {
            let j = next[i];
            if j == NIL {
                continue;
            }
            let pair = (value[i], value[j]);
            *counts.entry(pair).or_insert(0) += 1;
            positions.entry(pair).or_default().push(i);
        }

        while tok.vocab_size() < target_vocab_size {
            // Find the best pair (highest count). Ties broken by
            // first-seen ordering thanks to HashMap iteration being
            // unspecified — that's fine, BPE doesn't require a unique
            // canonical pick.
            let best = counts
                .iter()
                .filter(|&(_, &c)| c >= 2)
                .max_by_key(|(_, &c)| c)
                .map(|(p, &c)| (*p, c));
            let Some((best_pair, _best_count)) = best else {
                break;
            };
            let new_id = tok.vocab_size() as u32;

            // Apply merge at every position that currently holds
            // best_pair. The inverted index may contain stale entries
            // (positions that were the right-half of an earlier merge
            // and got deleted), so we re-validate each one.
            let pos_list = positions.remove(&best_pair).unwrap_or_default();
            let mut applied = 0_usize;
            for &i in &pos_list {
                if value.get(i).copied() == Some(u32::MAX) {
                    continue; // already deleted
                }
                let j = next[i];
                if j == NIL {
                    continue;
                }
                if !(value[i] == best_pair.0 && value[j] == best_pair.1) {
                    continue; // stale: position no longer holds this pair
                }

                // Drop the right-half pair counts that touched j.
                let k = next[j];
                if k != NIL {
                    let right_pair = (value[j], value[k]);
                    if let Some(c) = counts.get_mut(&right_pair) {
                        *c = c.saturating_sub(1);
                    }
                }

                // Drop the left-half pair counts that touched i.
                let h = prev[i];
                if h != NIL {
                    let left_pair = (value[h], value[i]);
                    if let Some(c) = counts.get_mut(&left_pair) {
                        *c = c.saturating_sub(1);
                    }
                }

                // Drop the merged pair itself.
                if let Some(c) = counts.get_mut(&best_pair) {
                    *c = c.saturating_sub(1);
                }

                // Replace value[i] with new_id, splice out j.
                value[i] = new_id;
                let after_j = next[j];
                next[i] = after_j;
                if after_j != NIL {
                    prev[after_j] = i;
                }
                value[j] = u32::MAX; // tombstone

                // Add new pair counts around the merged position.
                if h != NIL {
                    let new_left = (value[h], new_id);
                    *counts.entry(new_left).or_insert(0) += 1;
                    positions.entry(new_left).or_default().push(h);
                }
                if after_j != NIL {
                    let new_right = (new_id, value[after_j]);
                    *counts.entry(new_right).or_insert(0) += 1;
                    positions.entry(new_right).or_default().push(i);
                }

                applied += 1;
            }

            if applied == 0 {
                // Stale entries only — drop the count to 0 and retry.
                counts.insert(best_pair, 0);
                continue;
            }

            // Record the merge + extend vocab table.
            let mut merged_bytes = tok.vocab[best_pair.0 as usize].clone();
            merged_bytes.extend_from_slice(&tok.vocab[best_pair.1 as usize]);
            tok.vocab.push(merged_bytes);
            tok.merges.push(best_pair);
        }

        tok
    }

    /// Encode a byte stream into token ids using rank-priority merges.
    /// Greedily applies the lowest-rank merge anywhere in the stream
    /// until no more merges apply.
    ///
    /// Uses a doubly-linked list of token positions plus a binary
    /// min-heap keyed by (rank, position): popping the heap yields
    /// the next merge to apply, which is then a constant-time list
    /// splice. Stale heap entries (where the position no longer holds
    /// the expected pair) are filtered when popped. Total cost
    /// `O(N log N)` for input length N — viable for multi-MB corpora.
    pub fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        if self.merges.is_empty() {
            return bytes.iter().map(|&b| u32::from(b)).collect();
        }
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }

        let mut rank: HashMap<(u32, u32), usize> = HashMap::with_capacity(self.merges.len());
        for (i, &m) in self.merges.iter().enumerate() {
            rank.insert(m, i);
        }

        const NIL: usize = usize::MAX;
        let mut value: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { NIL } else { i - 1 }).collect();
        let mut next: Vec<usize> =
            (0..n).map(|i| if i + 1 == n { NIL } else { i + 1 }).collect();

        // Min-heap of (rank, position). Lower rank first, lower
        // position breaks ties (so we apply leftmost first
        // deterministically). We use std's BinaryHeap which is a
        // max-heap — invert ordering with `Reverse`.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::with_capacity(n);
        for i in 0..n {
            let j = next[i];
            if j == NIL {
                continue;
            }
            if let Some(&r) = rank.get(&(value[i], value[j])) {
                heap.push(Reverse((r, i)));
            }
        }

        let mut head = 0_usize;
        while let Some(Reverse((r, i))) = heap.pop() {
            // Validate entry — position may have been invalidated by
            // an earlier merge that consumed this slot or its right
            // neighbor.
            if value.get(i).copied() == Some(u32::MAX) {
                continue;
            }
            let j = next[i];
            if j == NIL {
                continue;
            }
            // Confirm the pair at position i still has rank r.
            if rank.get(&(value[i], value[j])).copied() != Some(r) {
                continue;
            }

            // Apply merge: replace value[i] with new id, splice out j.
            let new_id = (256 + r) as u32;
            value[i] = new_id;
            let after_j = next[j];
            next[i] = after_j;
            if after_j != NIL {
                prev[after_j] = i;
            }
            value[j] = u32::MAX; // tombstone

            // Push potential new pairs: (prev[i], i) and (i, after_j).
            let h = prev[i];
            if h != NIL {
                if let Some(&nr) = rank.get(&(value[h], new_id)) {
                    heap.push(Reverse((nr, h)));
                }
            }
            if after_j != NIL {
                if let Some(&nr) = rank.get(&(new_id, value[after_j])) {
                    heap.push(Reverse((nr, i)));
                }
            }

            // If we just deleted the head, advance.
            if head == j {
                head = next[head];
            }
        }

        // Walk the linked list from head, collecting surviving tokens.
        let mut out = Vec::new();
        let mut p = head;
        while p != NIL {
            if value[p] != u32::MAX {
                out.push(value[p]);
            }
            p = next[p];
        }
        out
    }

    /// Decode token ids back to bytes via vocab lookup.
    pub fn decode(&self, tokens: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &t in tokens {
            if let Some(bytes) = self.vocab.get(t as usize) {
                out.extend_from_slice(bytes);
            }
        }
        out
    }

    /// Serialize merges to a string. Format:
    /// ```text
    /// vocab_size=NNNN
    /// L_0 R_0
    /// L_1 R_1
    /// ...
    /// ```
    /// Each merge line lists the (left, right) token ids. The full
    /// vocab is reconstructable from the merge list.
    pub fn to_serialized(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("vocab_size={}\n", self.vocab_size()));
        for &(l, r) in &self.merges {
            s.push_str(&format!("{l} {r}\n"));
        }
        s
    }

    /// Inverse of `to_serialized`.
    ///
    /// # Errors
    /// Returns `Err` if a merge line references an unknown token id
    /// or has malformed numbers.
    pub fn from_string(s: &str) -> Result<Self, String> {
        let mut tok = Self::new_byte_level();
        for (line_no, line) in s.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("vocab_size=") || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(format!("line {}: expected 'L R', got {line:?}", line_no + 1));
            }
            let l: u32 = parts[0]
                .parse()
                .map_err(|e| format!("line {}: parse left: {e}", line_no + 1))?;
            let r: u32 = parts[1]
                .parse()
                .map_err(|e| format!("line {}: parse right: {e}", line_no + 1))?;
            if l as usize >= tok.vocab.len() || r as usize >= tok.vocab.len() {
                return Err(format!(
                    "line {}: merge references token >= current vocab",
                    line_no + 1
                ));
            }
            let mut merged_bytes = tok.vocab[l as usize].clone();
            merged_bytes.extend_from_slice(&tok.vocab[r as usize]);
            tok.vocab.push(merged_bytes);
            tok.merges.push((l, r));
        }
        Ok(tok)
    }

    /// Save tokenizer to a file in the text format.
    ///
    /// # Errors
    /// Propagates IO errors.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_serialized())
    }

    /// Load tokenizer from a file.
    ///
    /// # Errors
    /// Returns IO error on missing/unreadable file or `Err(String)`
    /// (boxed) on malformed merge list.
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let s = std::fs::read_to_string(path)?;
        Self::from_string(&s).map_err(std::convert::Into::into)
    }
}
