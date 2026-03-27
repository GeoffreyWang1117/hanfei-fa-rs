/// Default chunk size: 16 KB.
pub const DEFAULT_CHUNK_SIZE: usize = 16384;

/// Split bytes into fixed-size chunks.
pub fn chunk_bytes(data: &[u8], chunk_size: usize) -> Vec<&[u8]> {
    data.chunks(chunk_size).collect()
}

/// Iterator version for memory efficiency.
pub fn chunk_bytes_iter(data: &[u8], chunk_size: usize) -> impl Iterator<Item = &[u8]> {
    data.chunks(chunk_size)
}

/// Estimate number of chunks for a given data size.
pub fn estimate_chunk_count(data_size: usize, chunk_size: usize) -> usize {
    (data_size + chunk_size - 1) / chunk_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_multiple() {
        let data = vec![0u8; 64];
        let chunks = chunk_bytes(&data, 16);
        assert_eq!(chunks.len(), 4);
        assert!(chunks.iter().all(|c| c.len() == 16));
    }

    #[test]
    fn test_non_exact() {
        let data = vec![0u8; 50];
        let chunks = chunk_bytes(&data, 16);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[3].len(), 2);
    }

    #[test]
    fn test_smaller_than_chunk() {
        let data = vec![0u8; 5];
        let chunks = chunk_bytes(&data, 16);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_estimate() {
        assert_eq!(estimate_chunk_count(100, 16), 7);
        assert_eq!(estimate_chunk_count(64, 16), 4);
        assert_eq!(estimate_chunk_count(0, 16), 0);
    }
}
