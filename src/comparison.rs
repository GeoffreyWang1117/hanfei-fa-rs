use std::collections::BTreeMap;
use serde::{Serialize, Deserialize};

use crate::merkle_tree::{ModelMerkleTree, MerkleTree};

/// Result of comparing two model Merkle trees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub changed_params: Vec<String>,
    pub unchanged_params: Vec<String>,
    pub changed_layers: Vec<String>,
    pub total_changed_bytes: usize,
    pub total_bytes: usize,
    pub change_percentage: f64,
    pub chunk_changes: BTreeMap<String, Vec<usize>>,
    pub hash_comparisons: usize,
}

impl ComparisonResult {
    pub fn summary(&self) -> String {
        format!(
            "Changed: {} params, {} layers, {:.2}% of model ({} hash comparisons)",
            self.changed_params.len(),
            self.changed_layers.len(),
            self.change_percentage,
            self.hash_comparisons,
        )
    }
}

/// Compare two model Merkle trees with 3-level pruning.
pub fn compare_model_trees(
    tree_a: &ModelMerkleTree,
    tree_b: &ModelMerkleTree,
    detailed: bool,
) -> ComparisonResult {
    let mut result = ComparisonResult {
        changed_params: Vec::new(),
        unchanged_params: Vec::new(),
        changed_layers: Vec::new(),
        total_changed_bytes: 0,
        total_bytes: 0,
        change_percentage: 0.0,
        chunk_changes: BTreeMap::new(),
        hash_comparisons: 0,
    };

    let mut total_chunks_a = 0usize;
    let mut total_chunks_b = 0usize;
    let mut changed_chunks = 0usize;

    let all_layers: Vec<&String> = tree_a.layer_trees.keys()
        .chain(tree_b.layer_trees.keys())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter().collect();

    for layer_name in &all_layers {
        let la = tree_a.layer_trees.get(*layer_name);
        let lb = tree_b.layer_trees.get(*layer_name);

        match (la, lb) {
            (None, Some(existing)) | (Some(existing), None) => {
                result.changed_layers.push((*layer_name).clone());
                for (pname, ptdata) in &existing.param_trees {
                    result.changed_params.push(pname.clone());
                    changed_chunks += ptdata.num_chunks;
                    if la.is_some() { total_chunks_a += ptdata.num_chunks; }
                    else { total_chunks_b += ptdata.num_chunks; }
                }
            }
            (Some(la_tree), Some(lb_tree)) => {
                result.hash_comparisons += 1;
                if la_tree.layer_root == lb_tree.layer_root {
                    for (pname, ptd) in &la_tree.param_trees {
                        result.unchanged_params.push(pname.clone());
                        total_chunks_a += ptd.num_chunks;
                    }
                    for (_, ptd) in &lb_tree.param_trees {
                        total_chunks_b += ptd.num_chunks;
                    }
                    continue;
                }

                result.changed_layers.push((*layer_name).clone());

                let all_params: std::collections::BTreeSet<&String> =
                    la_tree.param_trees.keys().chain(lb_tree.param_trees.keys()).collect();

                for pname in all_params {
                    let pa = la_tree.param_trees.get(pname);
                    let pb = lb_tree.param_trees.get(pname);

                    if let Some(p) = pa { total_chunks_a += p.num_chunks; }
                    if let Some(p) = pb { total_chunks_b += p.num_chunks; }

                    match (pa, pb) {
                        (None, _) | (_, None) => {
                            result.changed_params.push(pname.clone());
                            let existing = pa.or(pb).unwrap();
                            changed_chunks += existing.num_chunks;
                        }
                        (Some(pa_data), Some(pb_data)) => {
                            result.hash_comparisons += 1;
                            if pa_data.root_hash == pb_data.root_hash {
                                result.unchanged_params.push(pname.clone());
                                continue;
                            }
                            result.changed_params.push(pname.clone());

                            if detailed {
                                let ta = MerkleTree::from_data(pa_data.clone());
                                let tb = MerkleTree::from_data(pb_data.clone());
                                let (indices, comps) = ta.diff_tree(&tb);
                                result.hash_comparisons += comps;
                                changed_chunks += indices.len();
                                if !indices.is_empty() {
                                    result.chunk_changes.insert(pname.clone(), indices);
                                }
                            } else {
                                changed_chunks += pa_data.num_chunks.max(pb_data.num_chunks);
                            }
                        }
                    }
                }
            }
            (None, None) => {}
        }
    }

    let avg_chunk_size = tree_a.layer_trees.values().next()
        .and_then(|l| l.param_trees.values().next())
        .map(|p| p.chunk_size)
        .unwrap_or(16384);

    result.total_bytes = total_chunks_a.max(total_chunks_b) * avg_chunk_size;
    result.total_changed_bytes = changed_chunks * avg_chunk_size;
    result.change_percentage = if result.total_bytes > 0 {
        (result.total_changed_bytes as f64 / result.total_bytes as f64) * 100.0
    } else { 0.0 };

    result
}

/// Estimate bandwidth savings from incremental sync.
pub fn estimate_sync_savings(
    tree_a: &ModelMerkleTree,
    tree_b: &ModelMerkleTree,
) -> SyncSavings {
    let diff = compare_model_trees(tree_a, tree_b, true);
    let full = diff.total_bytes;
    let incremental = diff.total_changed_bytes;
    let savings = full.saturating_sub(incremental);
    let pct = if full > 0 { (savings as f64 / full as f64) * 100.0 } else { 0.0 };
    SyncSavings { full_sync_bytes: full, incremental_sync_bytes: incremental, savings_bytes: savings, savings_percentage: pct }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSavings {
    pub full_sync_bytes: usize,
    pub incremental_sync_bytes: usize,
    pub savings_bytes: usize,
    pub savings_percentage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hasher::HashAlgorithm;
    use crate::merkle_tree::build_model_merkle_tree;
    use std::collections::BTreeMap;

    #[test]
    fn test_identical_trees() {
        let mut blobs = BTreeMap::new();
        blobs.insert("layer.0.weight".into(), vec![1u8; 4096]);
        blobs.insert("layer.1.weight".into(), vec![2u8; 4096]);

        let t1 = build_model_merkle_tree(&blobs, 1024, "a", HashAlgorithm::Sha256, false);
        let t2 = build_model_merkle_tree(&blobs, 1024, "b", HashAlgorithm::Sha256, false);

        let result = compare_model_trees(&t1, &t2, true);
        assert!(result.changed_params.is_empty());
        assert_eq!(result.change_percentage, 0.0);
    }

    #[test]
    fn test_one_param_changed() {
        let mut blobs1 = BTreeMap::new();
        blobs1.insert("layer.0.weight".into(), vec![1u8; 4096]);
        blobs1.insert("layer.1.weight".into(), vec![2u8; 4096]);

        let mut blobs2 = blobs1.clone();
        blobs2.insert("layer.1.weight".into(), vec![99u8; 4096]);

        let t1 = build_model_merkle_tree(&blobs1, 1024, "a", HashAlgorithm::Sha256, false);
        let t2 = build_model_merkle_tree(&blobs2, 1024, "b", HashAlgorithm::Sha256, false);

        let result = compare_model_trees(&t1, &t2, true);
        assert_eq!(result.changed_params, vec!["layer.1.weight"]);
        assert!(result.change_percentage > 0.0);
    }

    #[test]
    fn test_sync_savings() {
        let mut blobs1 = BTreeMap::new();
        for i in 0..10 {
            blobs1.insert(format!("layer.{i}.weight"), vec![i as u8; 4096]);
        }
        let mut blobs2 = blobs1.clone();
        blobs2.insert("layer.0.weight".into(), vec![255u8; 4096]);

        let t1 = build_model_merkle_tree(&blobs1, 1024, "a", HashAlgorithm::Sha256, false);
        let t2 = build_model_merkle_tree(&blobs2, 1024, "b", HashAlgorithm::Sha256, false);

        let savings = estimate_sync_savings(&t1, &t2);
        assert!(savings.savings_percentage > 50.0);
    }
}
