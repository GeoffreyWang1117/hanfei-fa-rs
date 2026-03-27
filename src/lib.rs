//! # hanfei-fa 法
//!
//! ML model weight integrity verification via hierarchical Merkle trees.
//!
//! > 法不阿贵，绳不挠曲。
//! > *The law does not favor the noble; the plumb line does not bend for the crooked.*
//!
//! ## Features
//!
//! - **O(1) root hash** — single comparison verifies the entire model
//! - **O(k log C) diff** — layer-aware diff without scanning unchanged weights
//! - **4-level hierarchy** — Model > Layer > Parameter > Chunk
//! - **5 hash algorithms** — SHA-256, SHA-512, SHA3-256, BLAKE2b, BLAKE3
//! - **Merkle proofs** — cryptographic proof that a chunk belongs to the tree
//! - **Streaming** — O(chunk_size) memory for multi-GB files
//!
//! ## Quick Start
//!
//! ```rust
//! use hanfei_fa::{MerkleTree, HashAlgorithm, compute_hash, chunk_bytes};
//!
//! let data = b"model weight bytes here...";
//! let chunks = chunk_bytes(data, 16384);
//! let hashes: Vec<String> = chunks.iter().map(|c| compute_hash(c, HashAlgorithm::Sha256)).collect();
//! let tree = MerkleTree::new(hashes, 16384);
//! println!("root: {}", tree.root_hash());
//! ```
//!
//! Part of the [HanFei (韩非) verifiable AI series](https://github.com/GeoffreyWang1117/hanfei-shu).

pub mod hasher;
pub mod chunking;
pub mod merkle_tree;
pub mod comparison;
pub mod strategies;

pub use hasher::{HashAlgorithm, compute_hash, hash_pair, verify_hash};
pub use chunking::{chunk_bytes, chunk_bytes_iter, estimate_chunk_count, DEFAULT_CHUNK_SIZE};
pub use merkle_tree::{
    MerkleNode, MerkleProof, MerkleTree,
    LayerMerkleTree, ModelMerkleTree,
    build_model_merkle_tree,
};
pub use comparison::{ComparisonResult, compare_model_trees, estimate_sync_savings};
pub use strategies::{ChunkStrategy, FixedChunkStrategy, AdaptiveChunkStrategy};
