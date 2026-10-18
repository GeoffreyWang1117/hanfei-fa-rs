//! Build a format v2 manifest for a few tensors and print it as JSON.
//!
//!     cargo run --example v2_manifest > model.merkle.json
//!
//! The output loads in the Python package (`ModelMerkleTree.from_json`).

use hanfei_fa::v2::{ModelTree, TensorMeta};
use hanfei_fa::HashAlgorithm;

fn main() {
    let mut model = ModelTree::new("example", HashAlgorithm::Sha256);
    let weight: Vec<u8> = (0..4096u32).map(|i| (i * 31 % 251) as u8).collect();
    model.insert("model.layers.0.mlp.down_proj.weight",
                 Some(TensorMeta { dtype: "BF16".into(), shape: vec![32, 64] }), &weight, 1024);
    model.insert("model.norm.weight",
                 Some(TensorMeta { dtype: "F32".into(), shape: vec![8] }), &weight[..32], 1024);
    model.insert("extra_blob", None, b"opaque bytes", 1024);

    let proof = model.tensor_proof("model.norm.weight").unwrap();
    assert!(proof.verify(model.algorithm));
    println!("{}", serde_json::to_string_pretty(&model.to_manifest(true)).unwrap());
}
