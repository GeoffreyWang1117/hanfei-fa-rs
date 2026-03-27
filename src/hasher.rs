use digest::Digest;
use sha2::{Sha256, Sha512};
use sha3::Sha3_256;
use blake2::Blake2b512;
use std::fmt;

/// Supported hash algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashAlgorithm {
    Sha256,
    Sha512,
    Sha3_256,
    Blake2b,
    Blake3,
}

impl HashAlgorithm {
    /// Expected hex digest length.
    pub fn hash_length(&self) -> usize {
        match self {
            Self::Sha256 => 64,
            Self::Sha512 => 128,
            Self::Sha3_256 => 64,
            Self::Blake2b => 128,
            Self::Blake3 => 64,
        }
    }
}

impl fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sha256 => write!(f, "sha256"),
            Self::Sha512 => write!(f, "sha512"),
            Self::Sha3_256 => write!(f, "sha3_256"),
            Self::Blake2b => write!(f, "blake2b"),
            Self::Blake3 => write!(f, "blake3"),
        }
    }
}

/// Compute the hex-encoded hash of data.
pub fn compute_hash(data: &[u8], algorithm: HashAlgorithm) -> String {
    match algorithm {
        HashAlgorithm::Sha256 => {
            let mut h = Sha256::new();
            h.update(data);
            hex::encode(h.finalize())
        }
        HashAlgorithm::Sha512 => {
            let mut h = Sha512::new();
            h.update(data);
            hex::encode(h.finalize())
        }
        HashAlgorithm::Sha3_256 => {
            let mut h = Sha3_256::new();
            h.update(data);
            hex::encode(h.finalize())
        }
        HashAlgorithm::Blake2b => {
            let mut h = Blake2b512::new();
            h.update(data);
            hex::encode(h.finalize())
        }
        HashAlgorithm::Blake3 => {
            let h = blake3::hash(data);
            h.to_hex().to_string()
        }
    }
}

/// Hash a pair of child hashes to produce a parent hash (Merkle internal node).
pub fn hash_pair(left: &str, right: &str, algorithm: HashAlgorithm) -> String {
    let combined = format!("{}{}", left, right);
    compute_hash(combined.as_bytes(), algorithm)
}

/// Verify that data matches an expected hash.
pub fn verify_hash(data: &[u8], expected: &str, algorithm: HashAlgorithm) -> bool {
    compute_hash(data, algorithm) == expected
}

/// Detect algorithm from hex digest length.
/// Returns SHA-256 for ambiguous 64-char digests (SHA-256/SHA3-256/BLAKE3).
pub fn detect_algorithm(hash_str: &str) -> Option<HashAlgorithm> {
    match hash_str.len() {
        64 => Some(HashAlgorithm::Sha256),
        128 => Some(HashAlgorithm::Sha512), // or Blake2b — ambiguous
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_deterministic() {
        let h1 = compute_hash(b"hello", HashAlgorithm::Sha256);
        let h2 = compute_hash(b"hello", HashAlgorithm::Sha256);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn test_different_data_different_hash() {
        let h1 = compute_hash(b"a", HashAlgorithm::Sha256);
        let h2 = compute_hash(b"b", HashAlgorithm::Sha256);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_all_algorithms() {
        for algo in [
            HashAlgorithm::Sha256,
            HashAlgorithm::Sha512,
            HashAlgorithm::Sha3_256,
            HashAlgorithm::Blake2b,
            HashAlgorithm::Blake3,
        ] {
            let h = compute_hash(b"test", algo);
            assert_eq!(h.len(), algo.hash_length(), "wrong length for {algo}");
        }
    }

    #[test]
    fn test_hash_pair_order_matters() {
        let a = compute_hash(b"left", HashAlgorithm::Sha256);
        let b = compute_hash(b"right", HashAlgorithm::Sha256);
        assert_ne!(hash_pair(&a, &b, HashAlgorithm::Sha256), hash_pair(&b, &a, HashAlgorithm::Sha256));
    }

    #[test]
    fn test_verify_hash() {
        let h = compute_hash(b"payload", HashAlgorithm::Sha256);
        assert!(verify_hash(b"payload", &h, HashAlgorithm::Sha256));
        assert!(!verify_hash(b"wrong", &h, HashAlgorithm::Sha256));
    }

    #[test]
    fn test_blake3_differs_from_sha256() {
        let b3 = compute_hash(b"hello", HashAlgorithm::Blake3);
        let sha = compute_hash(b"hello", HashAlgorithm::Sha256);
        assert_ne!(b3, sha);
        assert_eq!(b3.len(), 64);
    }
}
