#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn test_cvt(x: __m256i) -> __m256d {
    // _mm256_cvtepi64_pd only exists in AVX512DQ usually.
    // Let's try to compile with it.
    _mm256_cvtepi64_pd(x)
}
