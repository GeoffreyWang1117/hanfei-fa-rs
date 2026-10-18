//! Format v2 against the vectors shared with the Python package
//! (generated there by an independent reference implementation of the spec).

use hanfei_fa::v2::{parse_algorithm, ChunkTree, ModelTree, TensorMeta};
use hanfei_fa::HashAlgorithm;
use serde_json::Value;

fn load(name: &str) -> Value {
    let path = format!("{}/tests/vectors/{}", env!("CARGO_MANIFEST_DIR"), name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn pattern(seed: u64, n: u64) -> Vec<u8> {
    (0..n).map(|i| ((seed * 131 + i * 7) % 256) as u8).collect()
}

fn build_case(case: &Value, algorithm: HashAlgorithm) -> ModelTree {
    let chunk_size = case["chunk_size"].as_u64().unwrap();
    let mut model = ModelTree::new(case["id"].as_str().unwrap(), algorithm);
    for (name, spec) in case["tensors"].as_object().unwrap() {
        let spec = spec.as_array().unwrap();
        let data = pattern(spec[0].as_u64().unwrap(), spec[1].as_u64().unwrap());
        let meta = spec[2].as_str().map(|dtype| TensorMeta {
            dtype: dtype.to_string(),
            shape: spec[3].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect(),
        });
        model.insert(name, meta, &data, chunk_size);
    }
    model
}

#[test]
fn model_roots_match_vectors_for_every_algorithm() {
    let vectors = load("format_v2.json");
    let mut checked = 0;
    for case in vectors["cases"].as_array().unwrap() {
        for (alg_name, roots) in case["roots"].as_object().unwrap() {
            let algorithm = parse_algorithm(alg_name).unwrap();
            let model = build_case(case, algorithm);
            assert_eq!(model.model_root_hex().unwrap(), roots["model_root"].as_str().unwrap(),
                       "{} / {}", case["id"], alg_name);
            for (name, entry) in &model.tensors {
                assert_eq!(hex::encode(&entry.chunk_root), roots["chunk_roots"][name].as_str().unwrap());
                assert_eq!(hex::encode(entry.tensor_root(name, algorithm)),
                           roots["tensor_roots"][name].as_str().unwrap());
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 6 * 5);
}

#[test]
fn chunk_trees_match_vectors() {
    let vectors = load("format_v2.json");
    for ct in vectors["chunk_trees"].as_array().unwrap() {
        let sl = ct["seed_len"].as_array().unwrap();
        let data = pattern(sl[0].as_u64().unwrap(), sl[1].as_u64().unwrap());
        let tree = ChunkTree::from_bytes(&data, ct["chunk_size"].as_u64().unwrap(), HashAlgorithm::Sha256);
        assert_eq!(tree.root_hex(), ct["root_sha256"].as_str().unwrap());
    }
}

#[test]
fn manifest_written_by_python_loads_and_proves() {
    let manifest = load("manifest_v2_example.json");
    let model = ModelTree::from_manifest(&manifest).unwrap();
    let vectors = load("format_v2.json");
    let case = vectors["cases"].as_array().unwrap().iter().find(|c| c["id"] == "llama-like").unwrap();
    assert_eq!(model.model_root_hex().unwrap(), case["roots"]["sha256"]["model_root"].as_str().unwrap());

    for name in model.tensors.keys() {
        assert!(model.tensor_proof(name).unwrap().verify(model.algorithm), "{name}");
    }
    let name = "model.layers.0.self_attn.q_proj.weight";
    let tree = model.tensors[name].tree.as_ref().unwrap();
    let data = pattern(2, 512);
    let proof = tree.proof(5).unwrap();
    assert!(proof.verify_chunk(&data[5 * 64..6 * 64], model.algorithm));
    assert!(!proof.verify_chunk(&data[4 * 64..5 * 64], model.algorithm));
}

#[test]
fn manifest_roundtrip_compact_and_tampering() {
    let vectors = load("format_v2.json");
    let case = &vectors["cases"][0];
    let model = build_case(case, HashAlgorithm::Blake3);

    let full = model.to_manifest(true);
    assert_eq!(ModelTree::from_manifest(&full).unwrap().model_root(), model.model_root());

    let compact = model.to_manifest(false);
    let loaded = ModelTree::from_manifest(&compact).unwrap();
    assert!(loaded.tensors.values().all(|e| e.tree.is_none()));
    assert!(loaded.tensor_proof("lm_head.weight").unwrap().verify(HashAlgorithm::Blake3));

    let name = "model.norm.weight";
    for (field, value) in [
        ("tensor_root", Value::from("00".repeat(32))),
        ("dtype", Value::from("I8")),
        ("num_chunks", Value::from(99)),
        ("num_bytes", Value::from(-1)),
    ] {
        let mut bad = full.clone();
        bad["tensors"][name][field] = value;
        let e = ModelTree::from_manifest(&bad).unwrap_err();
        assert!(e.0.contains("inconsistent"), "{field}: {e}");
    }
    let mut bad = full.clone();
    bad["model_root"] = Value::from("00".repeat(32));
    assert!(ModelTree::from_manifest(&bad).is_err());
}

#[test]
fn metadata_and_names_are_bound() {
    let base = |name: &str, dtype: &str, shape: Vec<u64>| {
        let mut m = ModelTree::new("", HashAlgorithm::Sha256);
        m.insert(name, Some(TensorMeta { dtype: dtype.into(), shape }), &[0u8; 64], 16);
        m.model_root().unwrap()
    };
    let r = base("layer.0.weight", "F32", vec![16]);
    assert_ne!(r, base("layer.0.zeta", "F32", vec![16]));
    assert_ne!(r, base("layer.0.weight", "I32", vec![16]));
    assert_ne!(r, base("layer.0.weight", "F32", vec![4, 4]));
}
