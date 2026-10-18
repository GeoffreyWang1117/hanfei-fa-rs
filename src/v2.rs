//! Manifest format v2.
//!
//! The byte-level specification is `docs/MANIFEST_V2.md` in the Python package
//! (`merkle-weight-verify` / `hanfei-fa` on PyPI); both implementations are
//! checked against the same vectors (`tests/vectors/format_v2.json`).
//!
//! * Chunk trees are RFC 6962 Merkle trees: leaves `H(0x00 || chunk)`, interior
//!   nodes `H(0x01 || left || right)` over raw digests, an odd node carried up.
//! * A tensor root binds the tensor's name, dtype, shape, byte length and chunk size.
//! * The model root is `H(0x04 || u64(P) || MTH(tensor roots sorted by name))`.
//!
//! The v1 types in [`crate::merkle_tree`] are unchanged.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Read;

use digest::Digest;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Sha256, Sha512};
use sha3::Sha3_256;
use blake2::Blake2b512;

use crate::hasher::HashAlgorithm;

pub const FORMAT_NAME: &str = "merkle-weight-verify";
pub const FORMAT_VERSION: u64 = 2;

const TAG_LEAF: u8 = 0x00;
const TAG_NODE: u8 = 0x01;
const TAG_TENSOR: u8 = 0x02;
const TAG_LAYER: u8 = 0x03;
const TAG_MODEL: u8 = 0x04;

// ---------------------------------------------------------------------------
// Hashing primitives
// ---------------------------------------------------------------------------

/// Raw digest of the concatenation of `parts`.
pub fn digest_parts(parts: &[&[u8]], algorithm: HashAlgorithm) -> Vec<u8> {
    fn run<D: Digest>(parts: &[&[u8]]) -> Vec<u8> {
        let mut h = D::new();
        for p in parts {
            h.update(p);
        }
        h.finalize().to_vec()
    }
    match algorithm {
        HashAlgorithm::Sha256 => run::<Sha256>(parts),
        HashAlgorithm::Sha512 => run::<Sha512>(parts),
        HashAlgorithm::Sha3_256 => run::<Sha3_256>(parts),
        HashAlgorithm::Blake2b => run::<Blake2b512>(parts),
        HashAlgorithm::Blake3 => {
            let mut h = blake3::Hasher::new();
            for p in parts {
                h.update(p);
            }
            h.finalize().as_bytes().to_vec()
        }
    }
}

/// Parse an algorithm name as written in manifests (`"sha256"`, `"blake3"`, ...).
pub fn parse_algorithm(name: &str) -> Option<HashAlgorithm> {
    match name.to_ascii_lowercase().as_str() {
        "sha256" => Some(HashAlgorithm::Sha256),
        "sha512" => Some(HashAlgorithm::Sha512),
        "sha3_256" => Some(HashAlgorithm::Sha3_256),
        "blake2b" => Some(HashAlgorithm::Blake2b),
        "blake3" => Some(HashAlgorithm::Blake3),
        _ => None,
    }
}

/// Chunk leaf: `H(0x00 || chunk)`.
pub fn hash_leaf(chunk: &[u8], algorithm: HashAlgorithm) -> Vec<u8> {
    digest_parts(&[&[TAG_LEAF], chunk], algorithm)
}

/// Interior node: `H(0x01 || left || right)`.
pub fn hash_node(left: &[u8], right: &[u8], algorithm: HashAlgorithm) -> Vec<u8> {
    digest_parts(&[&[TAG_NODE], left, right], algorithm)
}

fn next_level(level: &[Vec<u8>], algorithm: HashAlgorithm) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = level
        .chunks_exact(2)
        .map(|pair| hash_node(&pair[0], &pair[1], algorithm))
        .collect();
    if level.len() % 2 == 1 {
        out.push(level[level.len() - 1].clone()); // carried up unchanged
    }
    out
}

/// RFC 6962 Merkle Tree Hash over already-hashed leaves. `None` for an empty list.
pub fn mth(leaves: &[Vec<u8>], algorithm: HashAlgorithm) -> Option<Vec<u8>> {
    if leaves.is_empty() {
        return None;
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        level = next_level(&level, algorithm);
    }
    level.pop()
}

/// RFC 9162 inclusion path (siblings from leaf to root) for `leaves[index]`.
pub fn inclusion_path(leaves: &[Vec<u8>], index: usize, algorithm: HashAlgorithm) -> Option<Vec<Vec<u8>>> {
    if index >= leaves.len() {
        return None;
    }
    let mut path = Vec::new();
    let mut level = leaves.to_vec();
    let mut i = index;
    while level.len() > 1 {
        if i % 2 == 1 {
            path.push(level[i - 1].clone());
        } else if i + 1 < level.len() {
            path.push(level[i + 1].clone());
        }
        level = next_level(&level, algorithm);
        i /= 2;
    }
    Some(path)
}

/// Fold an inclusion path (RFC 9162 §2.1.3.2). `None` if the path does not fit `index`/`size`.
pub fn root_from_inclusion_path(
    leaf: &[u8],
    index: u64,
    size: u64,
    path: &[Vec<u8>],
    algorithm: HashAlgorithm,
) -> Option<Vec<u8>> {
    if index >= size {
        return None;
    }
    let (mut fn_, mut sn) = (index, size - 1);
    let mut r = leaf.to_vec();
    for p in path {
        if sn == 0 {
            return None;
        }
        if fn_ & 1 == 1 || fn_ == sn {
            r = hash_node(p, &r, algorithm);
            if fn_ & 1 == 0 {
                while fn_ & 1 == 0 && fn_ != 0 {
                    fn_ >>= 1;
                    sn >>= 1;
                }
            }
        } else {
            r = hash_node(&r, p, algorithm);
        }
        fn_ >>= 1;
        sn >>= 1;
    }
    if sn != 0 {
        return None;
    }
    Some(r)
}

// ---------------------------------------------------------------------------
// Tensor, layer and model roots
// ---------------------------------------------------------------------------

/// Tensor dtype (safetensors code, e.g. `"BF16"`) and shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<u64>,
}

fn enc_str(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = (b.len() as u32).to_be_bytes().to_vec();
    out.extend_from_slice(b);
    out
}

fn enc_meta(meta: Option<&TensorMeta>) -> Vec<u8> {
    match meta {
        None => vec![0u8],
        Some(m) => {
            let mut out = vec![1u8];
            out.extend(enc_str(&m.dtype));
            out.extend_from_slice(&(m.shape.len() as u32).to_be_bytes());
            for d in &m.shape {
                out.extend_from_slice(&d.to_be_bytes());
            }
            out
        }
    }
}

/// `H(0x02 || str(name) || meta || u64(num_bytes) || u64(chunk_size) || chunk_root)`.
pub fn tensor_root(
    name: &str,
    meta: Option<&TensorMeta>,
    num_bytes: u64,
    chunk_size: u64,
    chunk_root: &[u8],
    algorithm: HashAlgorithm,
) -> Vec<u8> {
    digest_parts(
        &[
            &[TAG_TENSOR],
            &enc_str(name),
            &enc_meta(meta),
            &num_bytes.to_be_bytes(),
            &chunk_size.to_be_bytes(),
            chunk_root,
        ],
        algorithm,
    )
}

/// Layer index digest (not part of the model root); `tensor_roots` in name order.
pub fn layer_digest(layer_name: &str, tensor_roots: &[Vec<u8>], algorithm: HashAlgorithm) -> Vec<u8> {
    let mut parts: Vec<Vec<u8>> = vec![vec![TAG_LAYER], enc_str(layer_name),
                                       (tensor_roots.len() as u64).to_be_bytes().to_vec()];
    parts.extend(tensor_roots.iter().cloned());
    let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
    digest_parts(&refs, algorithm)
}

/// `H(0x04 || u64(P) || tensor_tree_root)`.
pub fn wrap_model_root(tensor_tree_root: &[u8], num_tensors: u64, algorithm: HashAlgorithm) -> Vec<u8> {
    digest_parts(&[&[TAG_MODEL], &num_tensors.to_be_bytes(), tensor_tree_root], algorithm)
}

/// Model root over tensor roots already sorted by name. `None` for zero tensors.
pub fn model_root(tensor_roots: &[Vec<u8>], algorithm: HashAlgorithm) -> Option<Vec<u8>> {
    let tree = mth(tensor_roots, algorithm)?;
    Some(wrap_model_root(&tree, tensor_roots.len() as u64, algorithm))
}

// ---------------------------------------------------------------------------
// Chunk trees
// ---------------------------------------------------------------------------

/// Proof that one chunk belongs to a chunk tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkProof {
    pub index: u64,
    pub num_chunks: u64,
    pub leaf: Vec<u8>,
    pub path: Vec<Vec<u8>>,
    pub root: Vec<u8>,
}

impl ChunkProof {
    pub fn verify(&self, algorithm: HashAlgorithm) -> bool {
        root_from_inclusion_path(&self.leaf, self.index, self.num_chunks, &self.path, algorithm)
            .map_or(false, |r| r == self.root)
    }

    pub fn verify_chunk(&self, chunk: &[u8], algorithm: HashAlgorithm) -> bool {
        hash_leaf(chunk, algorithm) == self.leaf && self.verify(algorithm)
    }
}

/// A v2 chunk tree, stored as its levels (leaves first).
#[derive(Debug, Clone)]
pub struct ChunkTree {
    algorithm: HashAlgorithm,
    chunk_size: u64,
    num_bytes: u64,
    levels: Vec<Vec<Vec<u8>>>,
}

impl ChunkTree {
    /// Build from leaf digests (already `hash_leaf`ed).
    pub fn from_leaves(leaves: Vec<Vec<u8>>, chunk_size: u64, num_bytes: u64, algorithm: HashAlgorithm) -> Option<Self> {
        if leaves.is_empty() || chunk_size == 0 {
            return None;
        }
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let next = next_level(levels.last().unwrap(), algorithm);
            levels.push(next);
        }
        Some(Self { algorithm, chunk_size, num_bytes, levels })
    }

    /// Chunk and hash `data`. Zero-length data is one empty chunk.
    pub fn from_bytes(data: &[u8], chunk_size: u64, algorithm: HashAlgorithm) -> Self {
        assert!(chunk_size > 0, "chunk_size must be positive");
        let mut leaves: Vec<Vec<u8>> = data
            .chunks(chunk_size as usize)
            .map(|c| hash_leaf(c, algorithm))
            .collect();
        if leaves.is_empty() {
            leaves.push(hash_leaf(&[], algorithm));
        }
        Self::from_leaves(leaves, chunk_size, data.len() as u64, algorithm).unwrap()
    }

    /// Stream from a reader with O(chunk_size) memory.
    pub fn from_reader<R: Read>(reader: &mut R, chunk_size: u64, algorithm: HashAlgorithm) -> std::io::Result<Self> {
        assert!(chunk_size > 0, "chunk_size must be positive");
        let mut buf = vec![0u8; chunk_size as usize];
        let mut leaves = Vec::new();
        let mut total = 0u64;
        loop {
            let mut filled = 0;
            while filled < buf.len() {
                let n = reader.read(&mut buf[filled..])?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            if filled == 0 {
                break;
            }
            total += filled as u64;
            leaves.push(hash_leaf(&buf[..filled], algorithm));
            if filled < buf.len() {
                break;
            }
        }
        if leaves.is_empty() {
            leaves.push(hash_leaf(&[], algorithm));
        }
        Ok(Self::from_leaves(leaves, chunk_size, total, algorithm).unwrap())
    }

    pub fn root(&self) -> &[u8] {
        &self.levels.last().unwrap()[0]
    }
    pub fn root_hex(&self) -> String {
        hex::encode(self.root())
    }
    pub fn leaves(&self) -> &[Vec<u8>] {
        &self.levels[0]
    }
    pub fn num_chunks(&self) -> u64 {
        self.levels[0].len() as u64
    }
    pub fn chunk_size(&self) -> u64 {
        self.chunk_size
    }
    pub fn num_bytes(&self) -> u64 {
        self.num_bytes
    }
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    pub fn proof(&self, index: usize) -> Option<ChunkProof> {
        if index >= self.levels[0].len() {
            return None;
        }
        let mut path = Vec::new();
        let mut i = index;
        for level in &self.levels[..self.levels.len() - 1] {
            if i % 2 == 1 {
                path.push(level[i - 1].clone());
            } else if i + 1 < level.len() {
                path.push(level[i + 1].clone());
            }
            i /= 2;
        }
        Some(ChunkProof {
            index: index as u64,
            num_chunks: self.num_chunks(),
            leaf: self.levels[0][index].clone(),
            path,
            root: self.root().to_vec(),
        })
    }

    /// Changed chunk indices and the number of hash comparisons.
    pub fn diff(&self, other: &ChunkTree) -> Result<(Vec<usize>, usize), String> {
        if self.algorithm != other.algorithm {
            return Err(format!("cannot diff trees hashed with {} and {}", self.algorithm, other.algorithm));
        }
        if self.chunk_size != other.chunk_size {
            return Err(format!("cannot diff trees with chunk sizes {} and {}", self.chunk_size, other.chunk_size));
        }
        let (a, b) = (&self.levels, &other.levels);
        if a[0].len() != b[0].len() {
            let common = a[0].len().min(b[0].len());
            let mut changed: Vec<usize> = (0..common).filter(|&i| a[0][i] != b[0][i]).collect();
            changed.extend(common..a[0].len().max(b[0].len()));
            return Ok((changed, common));
        }
        let mut changed = Vec::new();
        let mut comparisons = 0;
        let mut stack = vec![(a.len() - 1, 0usize)];
        while let Some((mut k, mut j)) = stack.pop() {
            comparisons += 1;
            if a[k][j] == b[k][j] {
                continue;
            }
            while k > 0 && 2 * j + 1 >= a[k - 1].len() {
                k -= 1;
                j *= 2;
            }
            if k == 0 {
                changed.push(j);
                continue;
            }
            stack.push((k - 1, 2 * j + 1));
            stack.push((k - 1, 2 * j));
        }
        changed.sort_unstable();
        Ok((changed, comparisons))
    }
}

// ---------------------------------------------------------------------------
// Model trees and manifests
// ---------------------------------------------------------------------------

/// One tensor of a model: its metadata and chunk tree (or only its chunk root).
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub meta: Option<TensorMeta>,
    pub num_bytes: u64,
    pub chunk_size: u64,
    pub num_chunks: u64,
    pub chunk_root: Vec<u8>,
    /// `None` for tensors loaded from a compact manifest.
    pub tree: Option<ChunkTree>,
}

impl TensorEntry {
    pub fn from_tree(meta: Option<TensorMeta>, tree: ChunkTree) -> Self {
        Self {
            meta,
            num_bytes: tree.num_bytes(),
            chunk_size: tree.chunk_size(),
            num_chunks: tree.num_chunks(),
            chunk_root: tree.root().to_vec(),
            tree: Some(tree),
        }
    }

    pub fn tensor_root(&self, name: &str, algorithm: HashAlgorithm) -> Vec<u8> {
        tensor_root(name, self.meta.as_ref(), self.num_bytes, self.chunk_size, &self.chunk_root, algorithm)
    }
}

/// Proof that one tensor is committed by a model root.
#[derive(Debug, Clone)]
pub struct TensorProof {
    pub name: String,
    pub meta: Option<TensorMeta>,
    pub num_bytes: u64,
    pub chunk_size: u64,
    pub chunk_root: Vec<u8>,
    pub index: u64,
    pub num_tensors: u64,
    pub path: Vec<Vec<u8>>,
    pub model_root: Vec<u8>,
}

impl TensorProof {
    pub fn verify(&self, algorithm: HashAlgorithm) -> bool {
        let leaf = tensor_root(&self.name, self.meta.as_ref(), self.num_bytes, self.chunk_size,
                               &self.chunk_root, algorithm);
        match root_from_inclusion_path(&leaf, self.index, self.num_tensors, &self.path, algorithm) {
            Some(tree_root) => wrap_model_root(&tree_root, self.num_tensors, algorithm) == self.model_root,
            None => false,
        }
    }
}

/// Errors from building or loading a v2 model tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestError(pub String);

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ManifestError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ManifestError> {
    Err(ManifestError(msg.into()))
}

/// A v2 model: tensors keyed (and ordered) by name. `String` ordering is byte
/// order of the UTF-8 encoding, as the specification requires.
#[derive(Debug, Clone)]
pub struct ModelTree {
    pub algorithm: HashAlgorithm,
    pub model_name: String,
    pub tensors: BTreeMap<String, TensorEntry>,
}

impl ModelTree {
    pub fn new(model_name: &str, algorithm: HashAlgorithm) -> Self {
        Self { algorithm, model_name: model_name.to_string(), tensors: BTreeMap::new() }
    }

    /// Hash and add one tensor.
    pub fn insert(&mut self, name: &str, meta: Option<TensorMeta>, data: &[u8], chunk_size: u64) {
        let tree = ChunkTree::from_bytes(data, chunk_size, self.algorithm);
        self.tensors.insert(name.to_string(), TensorEntry::from_tree(meta, tree));
    }

    pub fn tensor_roots(&self) -> Vec<Vec<u8>> {
        self.tensors.iter().map(|(n, e)| e.tensor_root(n, self.algorithm)).collect()
    }

    pub fn model_root(&self) -> Option<Vec<u8>> {
        model_root(&self.tensor_roots(), self.algorithm)
    }

    pub fn model_root_hex(&self) -> Option<String> {
        self.model_root().map(hex::encode)
    }

    pub fn tensor_proof(&self, name: &str) -> Option<TensorProof> {
        let index = self.tensors.keys().position(|n| n == name)?;
        let entry = &self.tensors[name];
        let roots = self.tensor_roots();
        Some(TensorProof {
            name: name.to_string(),
            meta: entry.meta.clone(),
            num_bytes: entry.num_bytes,
            chunk_size: entry.chunk_size,
            chunk_root: entry.chunk_root.clone(),
            index: index as u64,
            num_tensors: roots.len() as u64,
            path: inclusion_path(&roots, index, self.algorithm)?,
            model_root: self.model_root()?,
        })
    }

    /// Serialize to the v2 manifest JSON layout.
    pub fn to_manifest(&self, include_chunks: bool) -> Value {
        let mut tensors = serde_json::Map::new();
        for (name, e) in &self.tensors {
            let mut entry = json!({
                "dtype": e.meta.as_ref().map(|m| m.dtype.clone()),
                "shape": e.meta.as_ref().map(|m| m.shape.clone()),
                "num_bytes": e.num_bytes,
                "chunk_size": e.chunk_size,
                "num_chunks": e.num_chunks,
                "chunk_root": hex::encode(&e.chunk_root),
                "tensor_root": hex::encode(e.tensor_root(name, self.algorithm)),
            });
            if include_chunks {
                if let Some(tree) = &e.tree {
                    entry["chunk_hashes"] = json!(tree.leaves().iter().map(hex::encode).collect::<Vec<_>>());
                }
            }
            tensors.insert(name.clone(), entry);
        }
        json!({
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "algorithm": self.algorithm.to_string(),
            "model_name": self.model_name,
            "model_root": self.model_root_hex(),
            "num_tensors": self.tensors.len(),
            "created_by": concat!("hanfei-fa (Rust) ", env!("CARGO_PKG_VERSION")),
            "tensors": Value::Object(tensors),
        })
    }

    /// Load a v2 manifest, re-deriving every root; rejects inconsistent manifests.
    pub fn from_manifest(value: &Value) -> Result<Self, ManifestError> {
        let get = |k: &str| value.get(k).ok_or_else(|| ManifestError(format!("missing field {k}")));
        if value.get("format").and_then(Value::as_str).unwrap_or(FORMAT_NAME) != FORMAT_NAME {
            return err("not a merkle-weight-verify manifest");
        }
        if get("format_version")?.as_u64() != Some(FORMAT_VERSION) {
            return err("unsupported manifest format version");
        }
        let algorithm = get("algorithm")?
            .as_str()
            .and_then(parse_algorithm)
            .ok_or_else(|| ManifestError("unknown algorithm".into()))?;
        let model_name = value.get("model_name").and_then(Value::as_str).unwrap_or("").to_string();
        let mut model = ModelTree::new(&model_name, algorithm);

        let tensors = get("tensors")?.as_object().ok_or_else(|| ManifestError("tensors must be an object".into()))?;
        for (name, e) in tensors {
            let field_u64 = |k: &str| {
                e.get(k).and_then(Value::as_u64)
                    .ok_or_else(|| ManifestError(format!("manifest is inconsistent: {name:?} {k}")))
            };
            let num_bytes = field_u64("num_bytes")?;
            let chunk_size = field_u64("chunk_size")?;
            if chunk_size == 0 {
                return err(format!("manifest is inconsistent: {name:?} chunk_size"));
            }
            let expected_chunks = num_bytes.div_ceil(chunk_size).max(1);
            if field_u64("num_chunks")? != expected_chunks {
                return err(format!("manifest is inconsistent: {name:?} num_chunks"));
            }
            let meta = match (e.get("dtype"), e.get("shape")) {
                (Some(Value::String(d)), Some(Value::Array(s))) => {
                    let shape: Option<Vec<u64>> = s.iter().map(Value::as_u64).collect();
                    let shape = shape.ok_or_else(|| ManifestError(format!("manifest is inconsistent: {name:?} shape")))?;
                    Some(TensorMeta { dtype: d.clone(), shape })
                }
                (None | Some(Value::Null), None | Some(Value::Null)) => None,
                _ => return err(format!("manifest is inconsistent: {name:?} dtype/shape")),
            };
            let stored_root = hex_field(e, "chunk_root", name)?;
            let entry = match e.get("chunk_hashes") {
                Some(Value::Array(hs)) => {
                    let leaves: Result<Vec<Vec<u8>>, _> = hs.iter()
                        .map(|h| h.as_str().and_then(|s| hex::decode(s).ok())
                            .ok_or_else(|| ManifestError(format!("manifest is inconsistent: {name:?} chunk_hashes"))))
                        .collect();
                    let tree = ChunkTree::from_leaves(leaves?, chunk_size, num_bytes, algorithm)
                        .ok_or_else(|| ManifestError(format!("manifest is inconsistent: {name:?} chunk_hashes")))?;
                    if tree.num_chunks() != expected_chunks || tree.root() != stored_root.as_slice() {
                        return err(format!("manifest is inconsistent: {name:?} chunk_hashes"));
                    }
                    TensorEntry::from_tree(meta, tree)
                }
                None => TensorEntry {
                    meta, num_bytes, chunk_size, num_chunks: expected_chunks,
                    chunk_root: stored_root, tree: None,
                },
                Some(_) => return err(format!("manifest is inconsistent: {name:?} chunk_hashes")),
            };
            if entry.tensor_root(name, algorithm) != hex_field(e, "tensor_root", name)? {
                return err(format!("manifest is inconsistent: {name:?} tensor_root"));
            }
            model.tensors.insert(name.clone(), entry);
        }
        if model.tensors.is_empty() {
            return err("a model needs at least one tensor");
        }
        let stored_model_root = get("model_root")?.as_str().and_then(|s| hex::decode(s).ok());
        if model.model_root() != stored_model_root {
            return err("manifest is inconsistent: model_root");
        }
        if let Some(n) = value.get("num_tensors") {
            if n.as_u64() != Some(model.tensors.len() as u64) {
                return err("manifest is inconsistent: num_tensors");
            }
        }
        Ok(model)
    }

    /// Group tensor names into layers (same heuristic as the Python package) and
    /// return each layer's index digest. Layer digests are not part of the model root.
    pub fn layer_digests(&self) -> BTreeMap<String, Vec<u8>> {
        let mut layers: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();
        for (name, e) in &self.tensors {
            layers.entry(crate::merkle_tree::extract_layer_name(name))
                .or_default()
                .push(e.tensor_root(name, self.algorithm));
        }
        layers.into_iter()
            .map(|(layer, roots)| { let d = layer_digest(&layer, &roots, self.algorithm); (layer, d) })
            .collect()
    }
}

fn hex_field(e: &Value, key: &str, name: &str) -> Result<Vec<u8>, ManifestError> {
    e.get(key)
        .and_then(Value::as_str)
        .and_then(|s| hex::decode(s).ok())
        .ok_or_else(|| ManifestError(format!("manifest is inconsistent: {name:?} {key}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALG: HashAlgorithm = HashAlgorithm::Sha256;

    fn rfc6962(leaves: &[Vec<u8>]) -> Vec<u8> {
        let n = leaves.len();
        if n == 1 {
            return leaves[0].clone();
        }
        let mut k = 1;
        while k * 2 < n {
            k *= 2;
        }
        hash_node(&rfc6962(&leaves[..k]), &rfc6962(&leaves[k..]), ALG)
    }

    #[test]
    fn bottom_up_equals_rfc6962_split() {
        for n in 1..100usize {
            let leaves: Vec<Vec<u8>> = (0..n).map(|i| hash_leaf(&i.to_le_bytes(), ALG)).collect();
            assert_eq!(mth(&leaves, ALG).unwrap(), rfc6962(&leaves), "n = {n}");
        }
    }

    #[test]
    fn inclusion_paths_verify_and_bind_index() {
        for n in 1..40usize {
            let leaves: Vec<Vec<u8>> = (0..n).map(|i| hash_leaf(&[i as u8], ALG)).collect();
            let root = mth(&leaves, ALG).unwrap();
            for i in 0..n {
                let path = inclusion_path(&leaves, i, ALG).unwrap();
                assert_eq!(root_from_inclusion_path(&leaves[i], i as u64, n as u64, &path, ALG), Some(root.clone()));
                if i + 1 < n {
                    assert_ne!(root_from_inclusion_path(&leaves[i], i as u64 + 1, n as u64, &path, ALG), Some(root.clone()));
                }
            }
        }
    }

    #[test]
    fn duplicated_last_chunk_changes_root() {
        let l: Vec<Vec<u8>> = [b"a", b"b", b"c"].iter().map(|c| hash_leaf(*c, ALG)).collect();
        let mut l4 = l.clone();
        l4.push(l[2].clone());
        assert_ne!(mth(&l, ALG), mth(&l4, ALG));
    }

    #[test]
    fn diff_matches_linear_scan() {
        for n in [1usize, 2, 3, 7, 8, 33, 100] {
            let a: Vec<Vec<u8>> = (0..n).map(|i| hash_leaf(&[i as u8, 0], ALG)).collect();
            let mut b = a.clone();
            let changed: Vec<usize> = (0..n).filter(|i| i % 7 == 3).collect();
            for &i in &changed {
                b[i] = hash_leaf(b"x", ALG);
            }
            let ta = ChunkTree::from_leaves(a, 4, 0, ALG).unwrap();
            let tb = ChunkTree::from_leaves(b, 4, 0, ALG).unwrap();
            assert_eq!(ta.diff(&tb).unwrap().0, changed);
        }
    }

    #[test]
    fn streaming_matches_in_memory() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 7) as u8).collect();
        let mut cursor = std::io::Cursor::new(&data);
        let streamed = ChunkTree::from_reader(&mut cursor, 1000, ALG).unwrap();
        assert_eq!(streamed.root(), ChunkTree::from_bytes(&data, 1000, ALG).root());
        let empty = ChunkTree::from_reader(&mut std::io::Cursor::new(&[] as &[u8]), 16, ALG).unwrap();
        assert_eq!(empty.root(), hash_leaf(&[], ALG).as_slice());
    }
}
