// x86/x86_64 specific AES implementation using AES-NI instructions

use crate::ciphers::aes_common::{Steps, InvSteps, sub_bytes};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

macro_rules! aes128_keyround {
    ($ek:tt, $i:tt, $rcon:tt) => {{
        let mut key = $ek[$i - 1];
        let mut gen = _mm_aeskeygenassist_si128(key, $rcon);
        gen = _mm_shuffle_epi32(gen, 255);
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 8));
        $ek[$i] = _mm_xor_si128(key, gen);
    }};
}

macro_rules! aes192_keyround {
    ($temp1:tt, $temp2:tt, $temp3:tt) => {{
        let mut temp4 = _mm_slli_si128($temp1, 0x4);
        $temp2 = _mm_shuffle_epi32($temp2, 0x55);
        $temp1 = _mm_xor_si128($temp1, temp4);
        temp4 = _mm_slli_si128(temp4, 0x4);
        $temp1 = _mm_xor_si128($temp1, temp4);
        temp4 = _mm_slli_si128(temp4, 0x4);
        $temp1 = _mm_xor_si128($temp1, temp4);
        $temp1 = _mm_xor_si128($temp1, $temp2);
        $temp2 = _mm_shuffle_epi32($temp1, 0xff);
        temp4 = _mm_slli_si128($temp3, 0x4);
        $temp3 = _mm_xor_si128($temp3, temp4);
        $temp3 = _mm_xor_si128($temp3, $temp2);
    }};
}

macro_rules! aes256_keyround_1 {
    ($temp1:tt, $temp2:tt) => {{
        let mut temp4 = _mm_slli_si128($temp1, 0x4);
        $temp2 = _mm_shuffle_epi32($temp2, 0xff);
        $temp1 = _mm_xor_si128($temp1, temp4);
        temp4 = _mm_slli_si128(temp4, 0x4);
        $temp1 = _mm_xor_si128($temp1, temp4);
        temp4 = _mm_slli_si128(temp4, 0x4);
        $temp1 = _mm_xor_si128($temp1, temp4);
        $temp1 = _mm_xor_si128($temp1, $temp2);
    }};
}

macro_rules! aes256_keyround_2 {
    ($temp1:tt, $temp3:tt) => {{
        let mut temp4 = _mm_aeskeygenassist_si128($temp1, 0x0);
        let temp2 = _mm_shuffle_epi32(temp4, 0xaa);
        temp4 = _mm_slli_si128($temp3, 0x4);
        $temp3 = _mm_xor_si128($temp3, temp4);
        temp4 = _mm_slli_si128(temp4, 0x4);
        $temp3 = _mm_xor_si128($temp3, temp4);
        temp4 = _mm_slli_si128(temp4, 0x4);
        $temp3 = _mm_xor_si128($temp3, temp4);
        $temp3 = _mm_xor_si128($temp3, temp2);
    }};
}

#[derive(Clone)]
pub struct Aes128 {
    ek: [__m128i; 20],
    zero_k: __m128i,
}

impl Aes128 {
    pub const BLOCK_LEN: usize = 16;
    pub const KEY_LEN: usize = 16;

    #[inline(always)]
    pub fn new(key: &[u8]) -> Self {
        unsafe { Self::new_simd(key) }
    }

    #[target_feature(enable = "sse2,aes")]
    unsafe fn new_simd(key: &[u8]) -> Self {
        assert_eq!(key.len(), Self::KEY_LEN);
        let zero_k = _mm_setzero_si128();
        let mut ek: [__m128i; 20] = core::mem::zeroed();

        ek[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
        aes128_keyround!(ek, 1, 0x01);
        aes128_keyround!(ek, 2, 0x02);
        aes128_keyround!(ek, 3, 0x04);
        aes128_keyround!(ek, 4, 0x08);
        aes128_keyround!(ek, 5, 0x10);
        aes128_keyround!(ek, 6, 0x20);
        aes128_keyround!(ek, 7, 0x40);
        aes128_keyround!(ek, 8, 0x80);
        aes128_keyround!(ek, 9, 0x1b);
        aes128_keyround!(ek, 10, 0x36);

        ek[11] = _mm_aesimc_si128(ek[9]);
        ek[12] = _mm_aesimc_si128(ek[8]);
        ek[13] = _mm_aesimc_si128(ek[7]);
        ek[14] = _mm_aesimc_si128(ek[6]);
        ek[15] = _mm_aesimc_si128(ek[5]);
        ek[16] = _mm_aesimc_si128(ek[4]);
        ek[17] = _mm_aesimc_si128(ek[3]);
        ek[18] = _mm_aesimc_si128(ek[2]);
        ek[19] = _mm_aesimc_si128(ek[1]);

        Self { ek, zero_k }
    }

    #[inline(always)]
    pub fn encrypt(&self, block: &mut [u8], round: usize, step: Steps) {
        unsafe { self.encrypt_simd(block, round, step) }
    }

    #[target_feature(enable = "sse2,aes")]
    unsafe fn encrypt_simd(&self, block: &mut [u8], round: usize, step: Steps) {
        debug_assert_eq!(block.len(), Self::BLOCK_LEN);
        assert!(round <= 10);
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        m = _mm_xor_si128(m, self.ek[0]);
        match round {
            0 => {
                assert_eq!(step, Steps::AddRoundKey);
            }
            1..=10 => {
                for i in 1..round {
                    m = _mm_aesenc_si128(m, self.ek[i]);
                }
            }
            _ => (),
        };
        match step {
            Steps::AddRoundKey => {
                if round == 10 {
                    m = _mm_aesenclast_si128(m, self.ek[10]);
                } else if round == 0 {
                } else {
                    m = _mm_aesenc_si128(m, self.ek[round]);
                }
            }
            Steps::ShiftRow => {
                m = _mm_aesenclast_si128(m, self.zero_k);
            }
            Steps::MixColumns => {
                assert_ne!(round, 10);
                m = _mm_aesenc_si128(m, self.zero_k);
            }
            Steps::Sbox => {}
        }
        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
        if step == Steps::Sbox {
            sub_bytes(block);
        }
    }

    #[inline(always)]
    pub fn decrypt(&self, block: &mut [u8], round: usize, step: InvSteps) {
        unsafe { self.decrypt_simd(block, round, step) }
    }

    #[target_feature(enable = "sse2,aes")]
    unsafe fn decrypt_simd(&self, block: &mut [u8], round: usize, step: InvSteps) {
        debug_assert_eq!(block.len(), Self::BLOCK_LEN);
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        m = _mm_xor_si128(m, self.ek[10]);
        match round {
            0 => {
                assert_eq!(step, InvSteps::InvAddRoundKey);
            }
            1..=10 => {
                for i in 1..round {
                    m = _mm_aesdec_si128(m, self.ek[i + 10]);
                }
            }
            _ => (),
        };
        match step {
            InvSteps::InvAddRoundKey => {
                if round == 10 {
                    m = _mm_aesdeclast_si128(m, self.ek[0]);
                } else if round == 0 {
                } else {
                    m = _mm_aesdec_si128(m, self.ek[round + 10]);
                }
            }
            InvSteps::InvShiftRow | InvSteps::InvSbox => {
                m = _mm_aesdeclast_si128(m, self.zero_k);
            }
            InvSteps::InvMixColumns => {
                assert_ne!(round, 10);
                m = _mm_aesdec_si128(m, self.zero_k);
            }
        }

        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
        if step == InvSteps::InvShiftRow {
            sub_bytes(block);
        }
    }
}

// Similar implementations for Aes192 and Aes256 would follow...
// For brevity, I'll add placeholder structures

#[derive(Clone)]
pub struct Aes192 {
    ek: [__m128i; 24],
    zero_k: __m128i,
}

#[derive(Clone)]
pub struct Aes256 {
    ek: [__m128i; 28],
    zero_k: __m128i,
}

impl Aes192 {
    pub const BLOCK_LEN: usize = 16;
    pub const KEY_LEN: usize = 24;
    
    #[inline(always)]
    pub fn new(key: &[u8]) -> Self {
        unsafe { Self::new_simd(key) }
    }
    
    #[target_feature(enable = "sse2,aes")]
    unsafe fn new_simd(key: &[u8]) -> Self {
        assert_eq!(key.len(), Self::KEY_LEN);
        use core::mem::transmute;
        
        let zero_k = _mm_setzero_si128();
        let mut ek: [__m128i; 24] = core::mem::zeroed();
        
        // Load the 192-bit key (128 bits + 64 bits)
        let mut k2 = [0u8; 16];
        k2[0..8].copy_from_slice(&key[16..24]);
        
        let mut temp1 = _mm_loadu_si128(key.as_ptr() as *const __m128i);
        let mut temp2: __m128i;
        let mut temp3 = _mm_loadu_si128(k2.as_ptr() as *const __m128i);
        
        ek[0] = temp1;
        ek[1] = temp3;
        
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x1);
        aes192_keyround!(temp1, temp2, temp3);
        
        ek[1] = transmute(_mm_shuffle_pd(transmute(ek[1]), transmute(temp1), 0));
        ek[2] = transmute(_mm_shuffle_pd(transmute(temp1), transmute(temp3), 1));
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x2);
        aes192_keyround!(temp1, temp2, temp3);
        
        ek[3] = temp1;
        ek[4] = temp3;
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x4);
        aes192_keyround!(temp1, temp2, temp3);
        
        ek[4] = transmute(_mm_shuffle_pd(transmute(ek[4]), transmute(temp1), 0));
        ek[5] = transmute(_mm_shuffle_pd(transmute(temp1), transmute(temp3), 1));
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x8);
        aes192_keyround!(temp1, temp2, temp3);
        
        ek[6] = temp1;
        ek[7] = temp3;
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x10);
        aes192_keyround!(temp1, temp2, temp3);
        
        ek[7] = transmute(_mm_shuffle_pd(transmute(ek[7]), transmute(temp1), 0));
        ek[8] = transmute(_mm_shuffle_pd(transmute(temp1), transmute(temp3), 1));
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x20);
        aes192_keyround!(temp1, temp2, temp3);
        
        ek[9] = temp1;
        ek[10] = temp3;
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x40);
        aes192_keyround!(temp1, temp2, temp3);
        
        ek[10] = transmute(_mm_shuffle_pd(transmute(ek[10]), transmute(temp1), 0));
        ek[11] = transmute(_mm_shuffle_pd(transmute(temp1), transmute(temp3), 1));
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x80);
        aes192_keyround!(temp1, temp2, temp3);
        
        ek[12] = temp1;
        
        // Prepare inverse keys for decryption (rounds 1-11)
        ek[13] = _mm_aesimc_si128(ek[11]);
        ek[14] = _mm_aesimc_si128(ek[10]);
        ek[15] = _mm_aesimc_si128(ek[9]);
        ek[16] = _mm_aesimc_si128(ek[8]);
        ek[17] = _mm_aesimc_si128(ek[7]);
        ek[18] = _mm_aesimc_si128(ek[6]);
        ek[19] = _mm_aesimc_si128(ek[5]);
        ek[20] = _mm_aesimc_si128(ek[4]);
        ek[21] = _mm_aesimc_si128(ek[3]);
        ek[22] = _mm_aesimc_si128(ek[2]);
        ek[23] = _mm_aesimc_si128(ek[1]);
        
        Self { ek, zero_k }
    }
    
    #[inline(always)]
    pub fn encrypt(&self, block: &mut [u8], round: usize, step: Steps) {
        unsafe { self.encrypt_simd(block, round, step) }
    }
    
    #[target_feature(enable = "sse2,aes")]
    unsafe fn encrypt_simd(&self, block: &mut [u8], round: usize, step: Steps) {
        debug_assert_eq!(block.len(), Self::BLOCK_LEN);
        assert!(round <= 12);
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        m = _mm_xor_si128(m, self.ek[0]);
        match round {
            0 => {
                assert_eq!(step, Steps::AddRoundKey);
            }
            1..=12 => {
                for i in 1..round {
                    m = _mm_aesenc_si128(m, self.ek[i]);
                }
            }
            _ => ()
        };
        match step {
            Steps::AddRoundKey => {
                if round == 12 {
                    m = _mm_aesenclast_si128(m, self.ek[12]);
                } else if round == 0 {
                } else {
                    m = _mm_aesenc_si128(m, self.ek[round]);
                }
            }
            Steps::ShiftRow => {
                m = _mm_aesenclast_si128(m, self.zero_k);
            }
            Steps::MixColumns => {
                assert_ne!(round, 12);
                m = _mm_aesenc_si128(m, self.zero_k);
            }
            Steps::Sbox => {}
        }
        
        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
        if step == Steps::Sbox {
            sub_bytes(block);
        }
    }
    
    #[inline(always)]
    pub fn decrypt(&self, block: &mut [u8], round: usize, step: InvSteps) {
        unsafe { self.decrypt_simd(block, round, step) }
    }
    
    #[target_feature(enable = "sse2,aes")]
    unsafe fn decrypt_simd(&self, block: &mut [u8], round: usize, step: InvSteps) {
        debug_assert_eq!(block.len(), Self::BLOCK_LEN);
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        m = _mm_xor_si128(m, self.ek[12]);
        match round {
            0 => {
                assert_eq!(step, InvSteps::InvAddRoundKey);
            }
            1..=12 => {
                for i in 1..round {
                    m = _mm_aesdec_si128(m, self.ek[i + 12]);
                }
            }
            _ => ()
        };
        match step {
            InvSteps::InvAddRoundKey => {
                if round == 12 {
                    m = _mm_aesdeclast_si128(m, self.ek[0]);
                } else if round == 0 {
                } else {
                    m = _mm_aesdec_si128(m, self.ek[round + 12]);
                }
            }
            InvSteps::InvShiftRow | InvSteps::InvSbox => {
                m = _mm_aesdeclast_si128(m, self.zero_k);
            }
            InvSteps::InvMixColumns => {
                assert_ne!(round, 12);
                m = _mm_aesdec_si128(m, self.zero_k);
            }
        }
        
        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
        if step == InvSteps::InvShiftRow {
            sub_bytes(block);
        }
    }
}

impl Aes256 {
    pub const BLOCK_LEN: usize = 16;
    pub const KEY_LEN: usize = 32;
    
    #[inline(always)]
    pub fn new(key: &[u8]) -> Self {
        unsafe { Self::new_simd(key) }
    }
    
    #[target_feature(enable = "sse2,aes")]
    unsafe fn new_simd(key: &[u8]) -> Self {
        assert_eq!(key.len(), Self::KEY_LEN);
        let zero_k = _mm_setzero_si128();
        let mut ek: [__m128i; 28] = core::mem::zeroed();
        
        // Load the 256-bit key (128 bits + 128 bits)
        let mut temp1 = _mm_loadu_si128(key.as_ptr() as *const __m128i);
        let mut temp2: __m128i;
        let mut temp3 = _mm_loadu_si128(key.as_ptr().offset(16) as *const __m128i);
        
        ek[0] = temp1;
        ek[1] = temp3;
        
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x01);
        aes256_keyround_1!(temp1, temp2);
        ek[2] = temp1;
        aes256_keyround_2!(temp1, temp3);
        ek[3] = temp3;
        
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x02);
        aes256_keyround_1!(temp1, temp2);
        ek[4] = temp1;
        aes256_keyround_2!(temp1, temp3);
        ek[5] = temp3;
        
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x04);
        aes256_keyround_1!(temp1, temp2);
        ek[6] = temp1;
        aes256_keyround_2!(temp1, temp3);
        ek[7] = temp3;
        
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x08);
        aes256_keyround_1!(temp1, temp2);
        ek[8] = temp1;
        aes256_keyround_2!(temp1, temp3);
        ek[9] = temp3;
        
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x10);
        aes256_keyround_1!(temp1, temp2);
        ek[10] = temp1;
        aes256_keyround_2!(temp1, temp3);
        ek[11] = temp3;
        
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x20);
        aes256_keyround_1!(temp1, temp2);
        ek[12] = temp1;
        aes256_keyround_2!(temp1, temp3);
        ek[13] = temp3;
        
        temp2 = _mm_aeskeygenassist_si128(temp3, 0x40);
        aes256_keyround_1!(temp1, temp2);
        ek[14] = temp1;
        
        // Prepare inverse keys for decryption (rounds 1-13)
        ek[15] = _mm_aesimc_si128(ek[13]);
        ek[16] = _mm_aesimc_si128(ek[12]);
        ek[17] = _mm_aesimc_si128(ek[11]);
        ek[18] = _mm_aesimc_si128(ek[10]);
        ek[19] = _mm_aesimc_si128(ek[9]);
        ek[20] = _mm_aesimc_si128(ek[8]);
        ek[21] = _mm_aesimc_si128(ek[7]);
        ek[22] = _mm_aesimc_si128(ek[6]);
        ek[23] = _mm_aesimc_si128(ek[5]);
        ek[24] = _mm_aesimc_si128(ek[4]);
        ek[25] = _mm_aesimc_si128(ek[3]);
        ek[26] = _mm_aesimc_si128(ek[2]);
        ek[27] = _mm_aesimc_si128(ek[1]);
        
        Self { ek, zero_k }
    }
    
    #[inline(always)]
    pub fn encrypt(&self, block: &mut [u8], round: usize, step: Steps) {
        unsafe { self.encrypt_simd(block, round, step) }
    }
    
    #[target_feature(enable = "sse2,aes")]
    unsafe fn encrypt_simd(&self, block: &mut [u8], round: usize, step: Steps) {
        debug_assert_eq!(block.len(), Self::BLOCK_LEN);
        assert!(round <= 14);
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        m = _mm_xor_si128(m, self.ek[0]);
        match round {
            0 => {
                assert_eq!(step, Steps::AddRoundKey);
            }
            1..=14 => {
                for i in 1..round {
                    m = _mm_aesenc_si128(m, self.ek[i]);
                }
            }
            _ => ()
        };
        match step {
            Steps::AddRoundKey => {
                if round == 14 {
                    m = _mm_aesenclast_si128(m, self.ek[14]);
                } else if round == 0 {
                } else {
                    m = _mm_aesenc_si128(m, self.ek[round]);
                }
            }
            Steps::ShiftRow => {
                m = _mm_aesenclast_si128(m, self.zero_k);
            }
            Steps::MixColumns => {
                assert_ne!(round, 14);
                m = _mm_aesenc_si128(m, self.zero_k);
            }
            Steps::Sbox => {}
        }
        
        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
        if step == Steps::Sbox {
            sub_bytes(block);
        }
    }
    
    #[inline(always)]
    pub fn decrypt(&self, block: &mut [u8], round: usize, step: InvSteps) {
        unsafe { self.decrypt_simd(block, round, step) }
    }
    
    #[target_feature(enable = "sse2,aes")]
    unsafe fn decrypt_simd(&self, block: &mut [u8], round: usize, step: InvSteps) {
        debug_assert_eq!(block.len(), Self::BLOCK_LEN);
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        m = _mm_xor_si128(m, self.ek[14]);
        match round {
            0 => {
                assert_eq!(step, InvSteps::InvAddRoundKey);
            }
            1..=14 => {
                for i in 1..round {
                    m = _mm_aesdec_si128(m, self.ek[i + 14]);
                }
            }
            _ => ()
        };
        match step {
            InvSteps::InvAddRoundKey => {
                if round == 14 {
                    m = _mm_aesdeclast_si128(m, self.ek[0]);
                } else if round == 0 {
                } else {
                    m = _mm_aesdec_si128(m, self.ek[round + 14]);
                }
            }
            InvSteps::InvShiftRow | InvSteps::InvSbox => {
                m = _mm_aesdeclast_si128(m, self.zero_k);
            }
            InvSteps::InvMixColumns => {
                assert_ne!(round, 14);
                m = _mm_aesdec_si128(m, self.zero_k);
            }
        }
        
        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
        if step == InvSteps::InvShiftRow {
            sub_bytes(block);
        }
    }
}