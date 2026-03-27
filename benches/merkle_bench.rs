use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::collections::BTreeMap;
use hanfei_fa::{
    HashAlgorithm, compute_hash, MerkleTree,
    build_model_merkle_tree, chunk_bytes,
};

fn bench_hash_algorithms(c: &mut Criterion) {
    let data = vec![42u8; 16384];
    let mut group = c.benchmark_group("hash_16kb");
    for algo in [
        HashAlgorithm::Sha256,
        HashAlgorithm::Sha512,
        HashAlgorithm::Sha3_256,
        HashAlgorithm::Blake2b,
        HashAlgorithm::Blake3,
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(algo), &algo, |b, &algo| {
            b.iter(|| compute_hash(&data, algo));
        });
    }
    group.finish();
}

fn bench_tree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_build");
    for size_kb in [64, 1024, 16384] {
        let data = vec![0u8; size_kb * 1024];
        let chunks = chunk_bytes(&data, 16384);
        let hashes: Vec<String> = chunks.iter()
            .map(|c| compute_hash(c, HashAlgorithm::Sha256))
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size_kb}KB")),
            &hashes, |b, h| { b.iter(|| MerkleTree::new(h.clone(), 16384)); },
        );
    }
    group.finish();
}

fn bench_model_tree(c: &mut Criterion) {
    let mut blobs = BTreeMap::new();
    for i in 0..12 {
        blobs.insert(format!("layer.{i}.weight"), vec![i as u8; 4 * 1024 * 1024]);
        blobs.insert(format!("layer.{i}.bias"), vec![i as u8; 3072]);
    }

    c.bench_function("model_tree_48MB_serial", |b| {
        b.iter(|| build_model_merkle_tree(&blobs, 16384, "bench", HashAlgorithm::Sha256, false));
    });

    c.bench_function("model_tree_48MB_parallel", |b| {
        b.iter(|| build_model_merkle_tree(&blobs, 16384, "bench", HashAlgorithm::Sha256, true));
    });

    c.bench_function("model_tree_48MB_blake3", |b| {
        b.iter(|| build_model_merkle_tree(&blobs, 16384, "bench", HashAlgorithm::Blake3, false));
    });
}

criterion_group!(benches, bench_hash_algorithms, bench_tree_build, bench_model_tree);
criterion_main!(benches);
