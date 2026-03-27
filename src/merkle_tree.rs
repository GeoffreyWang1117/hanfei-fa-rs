use std::collections::BTreeMap;
use std::io::Read;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

use crate::hasher::{compute_hash, hash_pair, HashAlgorithm};
use crate::chunking::chunk_bytes;

// ── Nodes ──────────────────────────────────────────────────────────────

/// A node in a Merkle tree.
#[derive(Debug, Clone)]
pub struct MerkleNode {
    pub hash_value: String,
    pub left: Option<Box<MerkleNode>>,
    pub right: Option<Box<MerkleNode>>,
    pub is_leaf: bool,
    pub chunk_start: usize,
    pub chunk_count: usize,
    is_dup_right: bool,
}

/// Merkle proof for a single chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub chunk_index: usize,
    pub chunk_hash: String,
    /// (sibling_hash, position) where position is "left" or "right".
    pub sibling_hashes: Vec<(String, String)>,
    pub root_hash: String,
}

impl MerkleProof {
    /// Verify this proof reconstructs the expected root hash.
    pub fn verify(&self, algorithm: HashAlgorithm) -> bool {
        let mut current = self.chunk_hash.clone();
        for (sibling, position) in &self.sibling_hashes {
            current = if position == "left" {
                hash_pair(sibling, &current, algorithm)
            } else {
                hash_pair(&current, sibling, algorithm)
            };
        }
        current == self.root_hash
    }
}

// ── MerkleTree ─────────────────────────────────────────────────────────

/// Merkle tree for a sequence of chunks.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    chunk_hashes: Vec<String>,
    chunk_size: usize,
    root: MerkleNode,
    algorithm: HashAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTreeData {
    pub root_hash: String,
    pub num_chunks: usize,
    pub chunk_size: usize,
    pub chunk_hashes: Vec<String>,
}

impl MerkleTree {
    /// Build a Merkle tree from pre-computed chunk hashes.
    pub fn new(chunk_hashes: Vec<String>, chunk_size: usize) -> Self {
        Self::with_algorithm(chunk_hashes, chunk_size, HashAlgorithm::Sha256)
    }

    pub fn with_algorithm(chunk_hashes: Vec<String>, chunk_size: usize, algorithm: HashAlgorithm) -> Self {
        assert!(!chunk_hashes.is_empty(), "cannot build tree from empty hash list");
        let root = Self::build_tree(&chunk_hashes, algorithm);
        Self { chunk_hashes, chunk_size, root, algorithm }
    }

    /// Build by streaming from a reader. O(chunk_size) memory.
    pub fn from_reader<R: Read>(reader: &mut R, chunk_size: usize, algorithm: HashAlgorithm) -> std::io::Result<Self> {
        let mut chunk_hashes = Vec::new();
        let mut buf = vec![0u8; chunk_size];
        loop {
            let mut filled = 0;
            while filled < chunk_size {
                match reader.read(&mut buf[filled..]) {
                    Ok(0) => break,
                    Ok(n) => filled += n,
                    Err(e) => return Err(e),
                }
            }
            if filled == 0 { break; }
            chunk_hashes.push(compute_hash(&buf[..filled], algorithm));
        }
        if chunk_hashes.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "empty input"));
        }
        Ok(Self::with_algorithm(chunk_hashes, chunk_size, algorithm))
    }

    pub fn root_hash(&self) -> &str { &self.root.hash_value }
    pub fn num_chunks(&self) -> usize { self.chunk_hashes.len() }
    pub fn chunk_size(&self) -> usize { self.chunk_size }
    pub fn algorithm(&self) -> HashAlgorithm { self.algorithm }

    // ── build ──

    fn build_tree(hashes: &[String], algo: HashAlgorithm) -> MerkleNode {
        if hashes.len() == 1 {
            return MerkleNode {
                hash_value: hashes[0].clone(),
                left: None, right: None,
                is_leaf: true, chunk_start: 0, chunk_count: 1,
                is_dup_right: false,
            };
        }

        let mut nodes: Vec<MerkleNode> = hashes.iter().enumerate().map(|(i, h)| MerkleNode {
            hash_value: h.clone(),
            left: None, right: None,
            is_leaf: true, chunk_start: i, chunk_count: 1,
            is_dup_right: false,
        }).collect();

        while nodes.len() > 1 {
            let mut next = Vec::with_capacity((nodes.len() + 1) / 2);
            let mut i = 0;
            while i < nodes.len() {
                let left = nodes[i].clone();
                let (right, parent_count, is_dup) = if i + 1 < nodes.len() {
                    let r = nodes[i + 1].clone();
                    let c = left.chunk_count + r.chunk_count;
                    (r, c, false)
                } else {
                    (left.clone(), left.chunk_count, true)
                };
                let parent_hash = hash_pair(&left.hash_value, &right.hash_value, algo);
                next.push(MerkleNode {
                    hash_value: parent_hash,
                    chunk_start: left.chunk_start,
                    chunk_count: parent_count,
                    left: Some(Box::new(left)),
                    right: Some(Box::new(right)),
                    is_leaf: false,
                    is_dup_right: is_dup,
                });
                i += 2;
            }
            nodes = next;
        }
        nodes.into_iter().next().unwrap()
    }

    // ── proof ──

    /// Generate a Merkle proof for a given chunk index.
    pub fn get_proof(&self, chunk_index: usize) -> MerkleProof {
        assert!(chunk_index < self.chunk_hashes.len(), "chunk index out of range");

        let mut siblings = Vec::new();
        let mut idx = chunk_index;
        let mut level_size = self.chunk_hashes.len();

        while level_size > 1 {
            let level_hashes = self.get_level_hashes(level_size);
            if idx % 2 == 0 {
                let sibling = if idx + 1 < level_size {
                    &level_hashes[idx + 1]
                } else {
                    &level_hashes[idx] // odd: self-sibling
                };
                siblings.push((sibling.clone(), "right".to_string()));
            } else {
                siblings.push((level_hashes[idx - 1].clone(), "left".to_string()));
            }
            idx /= 2;
            level_size = (level_size + 1) / 2;
        }

        MerkleProof {
            chunk_index,
            chunk_hash: self.chunk_hashes[chunk_index].clone(),
            sibling_hashes: siblings,
            root_hash: self.root.hash_value.clone(),
        }
    }

    fn get_level_hashes(&self, level_size: usize) -> Vec<String> {
        if level_size == self.chunk_hashes.len() {
            return self.chunk_hashes.clone();
        }
        let mut current = self.chunk_hashes.clone();
        while current.len() > level_size {
            let mut next = Vec::with_capacity((current.len() + 1) / 2);
            let mut i = 0;
            while i < current.len() {
                let left = &current[i];
                let right = if i + 1 < current.len() { &current[i + 1] } else { &current[i] };
                next.push(hash_pair(left, right, self.algorithm));
                i += 2;
            }
            current = next;
        }
        current
    }

    // ── diff ──

    /// Find changed chunk indices by walking both trees. Returns (changed_indices, comparisons).
    pub fn diff_tree(&self, other: &MerkleTree) -> (Vec<usize>, usize) {
        let mut changed = Vec::new();
        let mut comparisons = 0usize;
        Self::diff_walk(&self.root, &other.root, &mut changed, &mut comparisons);

        let min_c = self.num_chunks().min(other.num_chunks());
        let max_c = self.num_chunks().max(other.num_chunks());
        if min_c != max_c {
            for i in min_c..max_c {
                if !changed.contains(&i) { changed.push(i); }
            }
        }
        changed.sort();
        (changed, comparisons)
    }

    fn diff_walk(
        a: &MerkleNode, b: &MerkleNode,
        changed: &mut Vec<usize>, comparisons: &mut usize,
    ) {
        *comparisons += 1;
        if a.hash_value == b.hash_value { return; }

        if a.is_leaf && b.is_leaf {
            changed.push(a.chunk_start);
            return;
        }
        if a.is_leaf || b.is_leaf {
            let wider = if a.chunk_count >= b.chunk_count { a } else { b };
            for i in wider.chunk_start..wider.chunk_start + wider.chunk_count {
                changed.push(i);
            }
            return;
        }

        // Recurse left
        if let (Some(la), Some(lb)) = (&a.left, &b.left) {
            Self::diff_walk(la, lb, changed, comparisons);
        }

        // Recurse right, handling duplication
        match (a.is_dup_right, b.is_dup_right) {
            (true, true) => {} // both duplicated — no new chunks
            (true, false) => {
                if let Some(rb) = &b.right {
                    for i in rb.chunk_start..rb.chunk_start + rb.chunk_count {
                        changed.push(i);
                    }
                }
            }
            (false, true) => {
                if let Some(ra) = &a.right {
                    for i in ra.chunk_start..ra.chunk_start + ra.chunk_count {
                        changed.push(i);
                    }
                }
            }
            (false, false) => {
                if let (Some(ra), Some(rb)) = (&a.right, &b.right) {
                    Self::diff_walk(ra, rb, changed, comparisons);
                }
            }
        }
    }

    // ── serialization ──

    pub fn to_data(&self) -> MerkleTreeData {
        MerkleTreeData {
            root_hash: self.root.hash_value.clone(),
            num_chunks: self.chunk_hashes.len(),
            chunk_size: self.chunk_size,
            chunk_hashes: self.chunk_hashes.clone(),
        }
    }

    pub fn from_data(data: MerkleTreeData) -> Self {
        Self::new(data.chunk_hashes, data.chunk_size)
    }
}

// ── LayerMerkleTree ────────────────────────────────────────────────────

/// Merkle tree for a model layer (groups parameter trees).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMerkleTree {
    pub layer_name: String,
    pub param_trees: BTreeMap<String, MerkleTreeData>,
    pub layer_root: Option<String>,
    #[serde(skip)]
    built_trees: Option<BTreeMap<String, MerkleTree>>,
}

impl LayerMerkleTree {
    pub fn new(layer_name: &str) -> Self {
        Self {
            layer_name: layer_name.to_string(),
            param_trees: BTreeMap::new(),
            layer_root: None,
            built_trees: None,
        }
    }

    pub fn insert_param(&mut self, param_name: &str, tree: MerkleTree) {
        self.param_trees.insert(param_name.to_string(), tree.to_data());
        self.built_trees.get_or_insert_with(BTreeMap::new)
            .insert(param_name.to_string(), tree);
    }

    pub fn compute_layer_root(&mut self, algorithm: HashAlgorithm) -> String {
        let combined: String = self.param_trees.iter()
            .map(|(_, td)| td.root_hash.as_str())
            .collect();
        let root = compute_hash(combined.as_bytes(), algorithm);
        self.layer_root = Some(root.clone());
        root
    }

    /// Get a built MerkleTree for a parameter (rebuilds from data if needed).
    pub fn get_param_tree(&self, name: &str) -> Option<MerkleTree> {
        if let Some(ref built) = self.built_trees {
            if let Some(t) = built.get(name) { return Some(t.clone()); }
        }
        self.param_trees.get(name).map(|d| MerkleTree::from_data(d.clone()))
    }
}

// ── ModelMerkleTree ────────────────────────────────────────────────────

/// Complete Merkle tree for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMerkleTree {
    pub model_name: String,
    pub layer_trees: BTreeMap<String, LayerMerkleTree>,
    pub model_root: Option<String>,
}

impl ModelMerkleTree {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            layer_trees: BTreeMap::new(),
            model_root: None,
        }
    }

    pub fn compute_model_root(&mut self, algorithm: HashAlgorithm) -> String {
        let combined: String = self.layer_trees.iter()
            .map(|(_, lt)| lt.layer_root.as_deref().unwrap_or(""))
            .collect();
        let root = compute_hash(combined.as_bytes(), algorithm);
        self.model_root = Some(root.clone());
        root
    }

    pub fn verify(&self, algorithm: HashAlgorithm) -> bool {
        // Recompute layer roots
        for (_, lt) in &self.layer_trees {
            let combined: String = lt.param_trees.iter()
                .map(|(_, td)| td.root_hash.as_str())
                .collect();
            let expected = compute_hash(combined.as_bytes(), algorithm);
            if lt.layer_root.as_deref() != Some(&expected) { return false; }
        }
        // Recompute model root
        let combined: String = self.layer_trees.iter()
            .map(|(_, lt)| lt.layer_root.as_deref().unwrap_or(""))
            .collect();
        let expected = compute_hash(combined.as_bytes(), algorithm);
        self.model_root.as_deref() == Some(&expected)
    }

    pub fn get_changed_layers(&self, other: &ModelMerkleTree) -> Vec<String> {
        let all_keys: std::collections::BTreeSet<&String> =
            self.layer_trees.keys().chain(other.layer_trees.keys()).collect();
        let mut changed = Vec::new();
        for k in all_keys {
            let a = self.layer_trees.get(k).and_then(|l| l.layer_root.as_deref());
            let b = other.layer_trees.get(k).and_then(|l| l.layer_root.as_deref());
            if a != b { changed.push(k.clone()); }
        }
        changed
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

// ── Builder ────────────────────────────────────────────────────────────

/// Extract layer name from a parameter name.
pub fn extract_layer_name(param_name: &str) -> String {
    let parts: Vec<&str> = param_name.split('.').collect();

    // GPT-2: transformer.h.N
    if parts.len() >= 3 && parts[0] == "transformer" && parts[1] == "h" {
        return format!("{}.{}.{}", parts[0], parts[1], parts[2]);
    }
    // LLaMA: model.layers.N
    if parts.len() >= 3 && parts[0] == "model" && parts[1] == "layers" {
        return format!("{}.{}.{}", parts[0], parts[1], parts[2]);
    }
    // BERT: encoder.layer.N
    if parts.len() >= 3 && parts[0] == "encoder" && parts[1] == "layer" {
        return format!("{}.{}.{}", parts[0], parts[1], parts[2]);
    }
    // Generic numbered
    for (i, p) in parts.iter().enumerate() {
        if i > 0 && p.parse::<usize>().is_ok() {
            return parts[..=i].join(".");
        }
    }
    parts.first().unwrap_or(&"other").to_string()
}

/// Build a complete ModelMerkleTree from named byte blobs.
///
/// Uses rayon for parallel hashing when `parallel` is true.
pub fn build_model_merkle_tree(
    blobs: &BTreeMap<String, Vec<u8>>,
    chunk_size: usize,
    model_name: &str,
    algorithm: HashAlgorithm,
    parallel: bool,
) -> ModelMerkleTree {
    // Group by layer
    let mut layer_params: BTreeMap<String, Vec<(&str, &[u8])>> = BTreeMap::new();
    for (name, data) in blobs {
        let layer = extract_layer_name(name);
        layer_params.entry(layer).or_default().push((name.as_str(), data.as_slice()));
    }

    let mut model = ModelMerkleTree::new(model_name);

    for (layer_name, params) in &layer_params {
        let mut layer_tree = LayerMerkleTree::new(layer_name);

        if parallel {
            let trees: Vec<(String, MerkleTree)> = params.par_iter().map(|(name, data)| {
                let chunks = chunk_bytes(data, chunk_size);
                let hashes: Vec<String> = chunks.iter()
                    .map(|c| compute_hash(c, algorithm))
                    .collect();
                (name.to_string(), MerkleTree::with_algorithm(hashes, chunk_size, algorithm))
            }).collect();

            for (name, tree) in trees {
                layer_tree.insert_param(&name, tree);
            }
        } else {
            for (name, data) in params {
                let chunks = chunk_bytes(data, chunk_size);
                let hashes: Vec<String> = chunks.iter()
                    .map(|c| compute_hash(c, algorithm))
                    .collect();
                let tree = MerkleTree::with_algorithm(hashes, chunk_size, algorithm);
                layer_tree.insert_param(name, tree);
            }
        }

        layer_tree.compute_layer_root(algorithm);
        model.layer_trees.insert(layer_name.clone(), layer_tree);
    }

    model.compute_model_root(algorithm);
    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::DEFAULT_CHUNK_SIZE;

    fn make_hashes(n: usize) -> Vec<String> {
        (0..n).map(|i| compute_hash(format!("chunk{i}").as_bytes(), HashAlgorithm::Sha256)).collect()
    }

    #[test]
    fn test_single_chunk() {
        let tree = MerkleTree::new(make_hashes(1), 16);
        assert_eq!(tree.num_chunks(), 1);
    }

    #[test]
    fn test_four_chunks() {
        let tree = MerkleTree::new(make_hashes(4), 16);
        assert_eq!(tree.num_chunks(), 4);
        assert_eq!(tree.root_hash().len(), 64);
    }

    #[test]
    fn test_deterministic() {
        let t1 = MerkleTree::new(make_hashes(8), 16);
        let t2 = MerkleTree::new(make_hashes(8), 16);
        assert_eq!(t1.root_hash(), t2.root_hash());
    }

    #[test]
    fn test_proof_verifies() {
        let tree = MerkleTree::new(make_hashes(8), 16);
        for i in 0..8 {
            let proof = tree.get_proof(i);
            assert!(proof.verify(HashAlgorithm::Sha256), "proof failed for chunk {i}");
        }
    }

    #[test]
    fn test_proof_odd_chunks() {
        for n in [3, 5, 7, 9, 11] {
            let tree = MerkleTree::new(make_hashes(n), 16);
            for i in 0..n {
                let proof = tree.get_proof(i);
                assert!(proof.verify(HashAlgorithm::Sha256), "proof failed: chunk {i} in {n}-chunk tree");
            }
        }
    }

    #[test]
    fn test_diff_identical() {
        let t1 = MerkleTree::new(make_hashes(8), 16);
        let t2 = MerkleTree::new(make_hashes(8), 16);
        let (changed, _) = t1.diff_tree(&t2);
        assert!(changed.is_empty());
    }

    #[test]
    fn test_diff_single_change() {
        let mut h = make_hashes(8);
        let t1 = MerkleTree::new(h.clone(), 16);
        h[3] = compute_hash(b"modified", HashAlgorithm::Sha256);
        let t2 = MerkleTree::new(h, 16);
        let (changed, comps) = t1.diff_tree(&t2);
        assert_eq!(changed, vec![3]);
        assert!(comps < 8); // sublinear
    }

    #[test]
    fn test_model_tree_build_and_verify() {
        let mut blobs = BTreeMap::new();
        blobs.insert("layer.0.weight".to_string(), vec![1u8; 4096]);
        blobs.insert("layer.0.bias".to_string(), vec![2u8; 256]);
        blobs.insert("layer.1.weight".to_string(), vec![3u8; 4096]);

        let tree = build_model_merkle_tree(&blobs, 1024, "test", HashAlgorithm::Sha256, false);
        assert!(tree.model_root.is_some());
        assert!(tree.verify(HashAlgorithm::Sha256));
        assert_eq!(tree.layer_trees.len(), 2);
    }

    #[test]
    fn test_model_tree_parallel() {
        let mut blobs = BTreeMap::new();
        for i in 0..10 {
            blobs.insert(format!("layer.{i}.weight"), vec![i as u8; 4096]);
        }
        let serial = build_model_merkle_tree(&blobs, 1024, "s", HashAlgorithm::Sha256, false);
        let par = build_model_merkle_tree(&blobs, 1024, "p", HashAlgorithm::Sha256, true);
        assert_eq!(serial.model_root, par.model_root);
    }

    #[test]
    fn test_extract_layer_name() {
        assert_eq!(extract_layer_name("transformer.h.0.attn.weight"), "transformer.h.0");
        assert_eq!(extract_layer_name("model.layers.3.mlp.weight"), "model.layers.3");
        assert_eq!(extract_layer_name("encoder.layer.11.attention.weight"), "encoder.layer.11");
        assert_eq!(extract_layer_name("fc.weight"), "fc");
    }

    #[test]
    fn test_json_roundtrip() {
        let mut blobs = BTreeMap::new();
        blobs.insert("p.weight".to_string(), vec![0u8; 1024]);
        let tree = build_model_merkle_tree(&blobs, 256, "rt", HashAlgorithm::Sha256, false);
        let json = tree.to_json();
        let restored = ModelMerkleTree::from_json(&json).unwrap();
        assert_eq!(tree.model_root, restored.model_root);
    }

    #[test]
    fn test_streaming_from_reader() {
        let data = vec![42u8; 50_000];
        let mut cursor = std::io::Cursor::new(&data);
        let streamed = MerkleTree::from_reader(&mut cursor, DEFAULT_CHUNK_SIZE, HashAlgorithm::Sha256).unwrap();

        let chunks = chunk_bytes(&data, DEFAULT_CHUNK_SIZE);
        let hashes: Vec<String> = chunks.iter().map(|c| compute_hash(c, HashAlgorithm::Sha256)).collect();
        let in_mem = MerkleTree::new(hashes, DEFAULT_CHUNK_SIZE);

        assert_eq!(streamed.root_hash(), in_mem.root_hash());
    }
}
