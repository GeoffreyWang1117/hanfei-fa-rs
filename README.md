<div align="center">

# hanfei-fa 法

**Verify ML model weights. Know exactly what changed.**

[![Crates.io](https://img.shields.io/crates/v/hanfei-fa.svg)](https://crates.io/crates/hanfei-fa)
[![docs.rs](https://docs.rs/hanfei-fa/badge.svg)](https://docs.rs/hanfei-fa)
[![PyPI](https://img.shields.io/pypi/v/hanfei-fa.svg)](https://pypi.org/project/hanfei-fa/)

> **法不阿贵，绳不挠曲。**
> *The law does not favor the noble; the plumb line does not bend for the crooked.*
> — 韩非子

</div>

---

Hierarchical Merkle tree verification for ML model weights in Rust.

- **O(1) root hash** — single comparison verifies the entire model
- **O(k log C) diff** — layer-aware diff without scanning unchanged weights
- **4-level hierarchy** — Model > Layer > Parameter > Chunk
- **5 hash algorithms** — SHA-256, SHA-512, SHA3-256, BLAKE2b, BLAKE3
- **Merkle proofs** — cryptographic proof that a chunk belongs to the tree
- **Streaming** — `from_reader()` with O(chunk_size) memory for multi-GB files
- **Parallel** — rayon-based parallel hashing

Also available as a Python package: `pip install hanfei-fa`

## Install

```toml
[dependencies]
hanfei-fa = "0.2"
```

## Quick Start

```rust
use hanfei_fa::{MerkleTree, HashAlgorithm, compute_hash, chunk_bytes};

// Hash a file's chunks
let data = std::fs::read("model.safetensors").unwrap();
let chunks = chunk_bytes(&data, 16384);
let hashes: Vec<String> = chunks.iter()
    .map(|c| compute_hash(c, HashAlgorithm::Sha256))
    .collect();

let tree = MerkleTree::new(hashes, 16384);
println!("root: {}", tree.root_hash());
```

### Stream from file (constant memory)

```rust
use hanfei_fa::{MerkleTree, HashAlgorithm};
use std::fs::File;

let mut f = File::open("70b-model.safetensors").unwrap();
let tree = MerkleTree::from_reader(&mut f, 16384, HashAlgorithm::Blake3).unwrap();
println!("root: {}", tree.root_hash());
```

### Diff two model versions

```rust
use hanfei_fa::{build_model_merkle_tree, compare_model_trees, HashAlgorithm};
use std::collections::BTreeMap;

let blobs_v1: BTreeMap<String, Vec<u8>> = /* load base model */;
let blobs_v2: BTreeMap<String, Vec<u8>> = /* load fine-tuned model */;

let tree_v1 = build_model_merkle_tree(&blobs_v1, 16384, "v1", HashAlgorithm::Sha256, true);
let tree_v2 = build_model_merkle_tree(&blobs_v2, 16384, "v2", HashAlgorithm::Sha256, true);

let result = compare_model_trees(&tree_v1, &tree_v2, true);
println!("Changed: {} params, {:.1}%", result.changed_params.len(), result.change_percentage);
```

### Merkle proofs

```rust
let proof = tree.get_proof(42);
assert!(proof.verify(HashAlgorithm::Sha256));
// Proof is portable — verifier needs only the proof + expected root hash
```

## Why this exists

Every existing tool hashes model files as opaque blobs. No tool in any ecosystem provides structure-aware verification at the layer/tensor/chunk level:

| Tool | Knows model structure? | Layer-aware diff? |
|------|----------------------|-------------------|
| HuggingFace Hub / Xet | No | No |
| Sigstore Model Signing | No | No |
| DVC / W&B / MLflow | No | No |
| safetensors / PyTorch | No integrity check | No |
| **hanfei-fa** | **Yes** | **O(k log C)** |

## Part of the HanFei (韩非) series

| Project | Role | Install |
|---------|------|---------|
| [**hanfei-shu 术**](https://github.com/GeoffreyWang1117/hanfei-shu) | GPU-accelerated MSM for ZK proofs | `cargo add hanfei-shu` |
| **hanfei-fa 法** (this) | Model weight integrity verification | `cargo add hanfei-fa` / `pip install hanfei-fa` |

## Contributing

Contributions welcome. Fork, branch, PR. We especially welcome:
- safetensors/GGUF native parsing in Rust
- PyO3 bindings for the Python package's native backend
- WASM compilation for browser-based verification
- Additional model architecture patterns in `extract_layer_name()`

**Found this useful?** Please star the repo and cite:

```bibtex
@software{hanfei_fa,
  author = {Geoffrey Wang},
  title = {hanfei-fa: ML Model Weight Integrity Verification via Hierarchical Merkle Trees},
  year = {2026},
  url = {https://github.com/GeoffreyWang1117/hanfei-fa-rs},
}
```

## License

Apache-2.0 — Copyright 2026 Geoffrey Wang
