/// Trait for chunk size strategies.
pub trait ChunkStrategy {
    fn chunk_size(&self, data_size: usize) -> usize;
    fn name(&self) -> &str;
}

/// Fixed chunk size for all parameters.
pub struct FixedChunkStrategy {
    pub size: usize,
}

impl Default for FixedChunkStrategy {
    fn default() -> Self { Self { size: crate::chunking::DEFAULT_CHUNK_SIZE } }
}

impl ChunkStrategy for FixedChunkStrategy {
    fn chunk_size(&self, _data_size: usize) -> usize { self.size }
    fn name(&self) -> &str { "fixed" }
}

/// Adaptive: chunk size scales with data size.
pub struct AdaptiveChunkStrategy {
    pub target_chunks: usize,
    pub min_chunk: usize,
    pub max_chunk: usize,
}

impl Default for AdaptiveChunkStrategy {
    fn default() -> Self {
        Self { target_chunks: 64, min_chunk: 1024, max_chunk: 65536 }
    }
}

impl ChunkStrategy for AdaptiveChunkStrategy {
    fn chunk_size(&self, data_size: usize) -> usize {
        if data_size == 0 { return self.min_chunk; }
        let raw = data_size / self.target_chunks;
        let aligned = (raw / 1024) * 1024;
        aligned.max(self.min_chunk).min(self.max_chunk)
    }
    fn name(&self) -> &str { "adaptive" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed() {
        let s = FixedChunkStrategy { size: 4096 };
        assert_eq!(s.chunk_size(1_000_000), 4096);
    }

    #[test]
    fn test_adaptive_small() {
        let s = AdaptiveChunkStrategy::default();
        assert_eq!(s.chunk_size(500), s.min_chunk);
    }

    #[test]
    fn test_adaptive_large() {
        let s = AdaptiveChunkStrategy::default();
        assert_eq!(s.chunk_size(100_000_000), s.max_chunk);
    }

    #[test]
    fn test_adaptive_mid() {
        let s = AdaptiveChunkStrategy::default();
        let cs = s.chunk_size(1_000_000);
        assert!(cs >= s.min_chunk && cs <= s.max_chunk);
    }
}
