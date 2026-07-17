// AArch64 specific AES implementation using NEON and crypto extensions

use crate::ciphers::aes_common::{Steps, InvSteps, sub_bytes, SBOX, SHIFT_ROWS};
use core::arch::aarch64::*;

// The round constant word array.
const RCON: [u32; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

#[inline]
fn sub_word(x: u32) -> u32 {
    let mut bytes = x.to_le_bytes();
    bytes[0] = SBOX[bytes[0] as usize];
    bytes[1] = SBOX[bytes[1] as usize];
    bytes[2] = SBOX[bytes[2] as usize];
    bytes[3] = SBOX[bytes[3] as usize];
    u32::from_le_bytes(bytes)
}

#[inline]
#[allow(dead_code)]
fn shift_rows(block: &mut [u8]) {
    let mut temp = [0u8; 16];
    for i in 0..16 {
        temp[i] = block[SHIFT_ROWS[i]];
    }
    block.copy_from_slice(&temp);
}

#[target_feature(enable = "neon,aes")]
unsafe fn encrypt_simd_generic(
    ek: &[uint8x16_t], 
    block: &mut [u8], 
    round: usize, 
    step: Steps, 
    max_rounds: usize
) {
    debug_assert_eq!(block.len(), 16);
    assert!(round <= max_rounds);
    let mut state: uint8x16_t = vld1q_u8(block.as_ptr());
    
    if round == 0 {
        assert_eq!(step, Steps::AddRoundKey);
        state = veorq_u8(state, ek[0]);
    } else {
        state = vaeseq_u8(state, ek[0]);
        
        for i in 1..round {
            state = vaesmcq_u8(state);
            state = vaeseq_u8(state, ek[i]);
        }
        
        match step {
            Steps::AddRoundKey => {
                if round == max_rounds {
                    state = veorq_u8(state, ek[round]);
                } else {
                    state = vaesmcq_u8(state);
                    state = veorq_u8(state, ek[round]);
                }
            }
            Steps::MixColumns => {
                assert_ne!(round, max_rounds);
                state = vaesmcq_u8(state);
            }
            Steps::ShiftRow => {
                vst1q_u8(block.as_mut_ptr(), state);
                return;
            }
            Steps::Sbox => {
                let temp_block = unsafe { core::mem::transmute::<uint8x16_t, [u8; 16]>(state)};
                for i in 0..16 {
                    block[SHIFT_ROWS[i]] = temp_block[i];
                }
                return;
            }
        }
    }
    vst1q_u8(block.as_mut_ptr(), state);
}

#[target_feature(enable = "neon,aes")]
unsafe fn decrypt_simd_generic(
    ek: &[uint8x16_t], 
    block: &mut [u8], 
    round: usize, 
    step: InvSteps, 
    max_rounds: usize
) {
    debug_assert_eq!(block.len(), 16);
    assert!(round <= max_rounds);
    let mut state: uint8x16_t = vld1q_u8(block.as_ptr());
    
    if round == 0 {
        assert_eq!(step, InvSteps::InvAddRoundKey);
        state = veorq_u8(state, ek[max_rounds]);
    } else {
        // Apply inverse rounds
        state = vaesdq_u8(state, ek[max_rounds]);
        for i in 1..round {
            state = vaesimcq_u8(state);
            state = vaesdq_u8(state, ek[max_rounds + i]);
        }
        
        // Handle the final partial round based on step
        match step {
            InvSteps::InvAddRoundKey => {
                state = veorq_u8(state, ek[max_rounds - round]);
            }
            InvSteps::InvMixColumns => {
                assert_ne!(round, max_rounds);
                state = veorq_u8(state, ek[max_rounds - round]);
                state = vaesimcq_u8(state);
            }
            InvSteps::InvShiftRow => {
                vst1q_u8(block.as_mut_ptr(), state);
                sub_bytes(block);
                return;
            }
            InvSteps::InvSbox => {
                // Already done by vaesdq_u8
            }
        }
    }
    vst1q_u8(block.as_mut_ptr(), state);
}

#[derive(Clone)]
pub struct Aes128 {
    ek: [uint8x16_t; 20],
}

impl Aes128 {
    pub const KEY_LEN: usize = 16;

    #[inline(always)]
    pub fn new(key: &[u8]) -> Self {
        unsafe { Self::new_simd(key) }
    }

    #[target_feature(enable = "neon,aes")]
    unsafe fn new_simd(key: &[u8]) -> Self {
        assert_eq!(key.len(), Self::KEY_LEN);
        
        use core::mem::transmute;
        let mut ek_u32: [u32; 44] = [0u32; 44];
        
        let k1: [u32; 4] = transmute(vld1q_u32(key.as_ptr() as *const u32));
        ek_u32[0] = k1[0];
        ek_u32[1] = k1[1];
        ek_u32[2] = k1[2];
        ek_u32[3] = k1[3];
        
        // Key expansion - explicitly unrolled as in reference
        ek_u32[4] = ek_u32[0] ^ (sub_word(ek_u32[3]).rotate_left(24) ^ RCON[0]);
        ek_u32[5] = ek_u32[1] ^ ek_u32[4];
        ek_u32[6] = ek_u32[2] ^ ek_u32[5];
        ek_u32[7] = ek_u32[3] ^ ek_u32[6];

        ek_u32[8] = ek_u32[4] ^ (sub_word(ek_u32[7]).rotate_left(24) ^ RCON[1]);
        ek_u32[9] = ek_u32[5] ^ ek_u32[8];
        ek_u32[10] = ek_u32[6] ^ ek_u32[9];
        ek_u32[11] = ek_u32[7] ^ ek_u32[10];

        ek_u32[12] = ek_u32[8] ^ (sub_word(ek_u32[11]).rotate_left(24) ^ RCON[2]);
        ek_u32[13] = ek_u32[9] ^ ek_u32[12];
        ek_u32[14] = ek_u32[10] ^ ek_u32[13];
        ek_u32[15] = ek_u32[11] ^ ek_u32[14];

        ek_u32[16] = ek_u32[12] ^ (sub_word(ek_u32[15]).rotate_left(24) ^ RCON[3]);
        ek_u32[17] = ek_u32[13] ^ ek_u32[16];
        ek_u32[18] = ek_u32[14] ^ ek_u32[17];
        ek_u32[19] = ek_u32[15] ^ ek_u32[18];

        ek_u32[20] = ek_u32[16] ^ (sub_word(ek_u32[19]).rotate_left(24) ^ RCON[4]);
        ek_u32[21] = ek_u32[17] ^ ek_u32[20];
        ek_u32[22] = ek_u32[18] ^ ek_u32[21];
        ek_u32[23] = ek_u32[19] ^ ek_u32[22];

        ek_u32[24] = ek_u32[20] ^ (sub_word(ek_u32[23]).rotate_left(24) ^ RCON[5]);
        ek_u32[25] = ek_u32[21] ^ ek_u32[24];
        ek_u32[26] = ek_u32[22] ^ ek_u32[25];
        ek_u32[27] = ek_u32[23] ^ ek_u32[26];

        ek_u32[28] = ek_u32[24] ^ (sub_word(ek_u32[27]).rotate_left(24) ^ RCON[6]);
        ek_u32[29] = ek_u32[25] ^ ek_u32[28];
        ek_u32[30] = ek_u32[26] ^ ek_u32[29];
        ek_u32[31] = ek_u32[27] ^ ek_u32[30];

        ek_u32[32] = ek_u32[28] ^ (sub_word(ek_u32[31]).rotate_left(24) ^ RCON[7]);
        ek_u32[33] = ek_u32[29] ^ ek_u32[32];
        ek_u32[34] = ek_u32[30] ^ ek_u32[33];
        ek_u32[35] = ek_u32[31] ^ ek_u32[34];

        ek_u32[36] = ek_u32[32] ^ (sub_word(ek_u32[35]).rotate_left(24) ^ RCON[8]);
        ek_u32[37] = ek_u32[33] ^ ek_u32[36];
        ek_u32[38] = ek_u32[34] ^ ek_u32[37];
        ek_u32[39] = ek_u32[35] ^ ek_u32[38];

        ek_u32[40] = ek_u32[36] ^ (sub_word(ek_u32[39]).rotate_left(24) ^ RCON[9]);
        ek_u32[41] = ek_u32[37] ^ ek_u32[40];
        ek_u32[42] = ek_u32[38] ^ ek_u32[41];
        ek_u32[43] = ek_u32[39] ^ ek_u32[42];
        
        let ptr = ek_u32.as_ptr();
        let mut ek = [vdupq_n_u8(0); 20];
        
        // Load round keys for encryption
        for i in 0..11 {
            ek[i] = vreinterpretq_u8_u32(vld1q_u32(ptr.add(i * 4)));
        }
        
        // Prepare inverse keys for decryption (rounds 1-9 need InvMixColumns)
        for i in 0..9 {
            ek[11 + i] = vaesimcq_u8(ek[9 - i]);
        }
        
        Self { ek }
    }

    #[inline(always)]
    pub fn encrypt(&self, block: &mut [u8], round: usize, step: Steps) {
        unsafe { self.encrypt_simd(block, round, step) }
    }

    #[target_feature(enable = "neon,aes")]
    unsafe fn encrypt_simd(&self, block: &mut [u8], round: usize, step: Steps) {
        encrypt_simd_generic(&self.ek, block, round, step, 10);
    }

    #[inline(always)]
    pub fn decrypt(&self, block: &mut [u8], round: usize, step: InvSteps) {
        unsafe { self.decrypt_simd(block, round, step) }
    }

    #[target_feature(enable = "neon,aes")]
    unsafe fn decrypt_simd(&self, block: &mut [u8], round: usize, step: InvSteps) {
        decrypt_simd_generic(&self.ek, block, round, step, 10);
    }
}

// Aes192 implementation
#[derive(Clone)]
pub struct Aes192 {
    ek: [uint8x16_t; 24],
}

impl Aes192 {
    pub const KEY_LEN: usize = 24;
    
    #[inline(always)]
    pub fn new(key: &[u8]) -> Self {
        unsafe { Self::new_simd(key) }
    }
    
    #[target_feature(enable = "neon,aes")]
    unsafe fn new_simd(key: &[u8]) -> Self {
        assert_eq!(key.len(), Self::KEY_LEN);
        
        use core::mem::transmute;
        let mut ek_u32: [u32; 52] = [0u32; 52];
        
        let k1: [u32; 4] = transmute(vld1q_u32(key.as_ptr() as *const u32));
        ek_u32[0] = k1[0];
        ek_u32[1] = k1[1];
        ek_u32[2] = k1[2];
        ek_u32[3] = k1[3];
        ek_u32[4] = *((key.as_ptr() as *const u32).add(4));
        ek_u32[5] = *((key.as_ptr() as *const u32).add(5));
        
        // Key expansion - explicitly unrolled as in reference
        ek_u32[6] = ek_u32[0] ^ (sub_word(ek_u32[5]).rotate_left(24) ^ RCON[0]);
        ek_u32[7] = ek_u32[1] ^ ek_u32[6];
        ek_u32[8] = ek_u32[2] ^ ek_u32[7];
        ek_u32[9] = ek_u32[3] ^ ek_u32[8];
        ek_u32[10] = ek_u32[4] ^ ek_u32[9];
        ek_u32[11] = ek_u32[5] ^ ek_u32[10];

        ek_u32[12] = ek_u32[6] ^ (sub_word(ek_u32[11]).rotate_left(24) ^ RCON[1]);
        ek_u32[13] = ek_u32[7] ^ ek_u32[12];
        ek_u32[14] = ek_u32[8] ^ ek_u32[13];
        ek_u32[15] = ek_u32[9] ^ ek_u32[14];
        ek_u32[16] = ek_u32[10] ^ ek_u32[15];
        ek_u32[17] = ek_u32[11] ^ ek_u32[16];

        ek_u32[18] = ek_u32[12] ^ (sub_word(ek_u32[17]).rotate_left(24) ^ RCON[2]);
        ek_u32[19] = ek_u32[13] ^ ek_u32[18];
        ek_u32[20] = ek_u32[14] ^ ek_u32[19];
        ek_u32[21] = ek_u32[15] ^ ek_u32[20];
        ek_u32[22] = ek_u32[16] ^ ek_u32[21];
        ek_u32[23] = ek_u32[17] ^ ek_u32[22];

        ek_u32[24] = ek_u32[18] ^ (sub_word(ek_u32[23]).rotate_left(24) ^ RCON[3]);
        ek_u32[25] = ek_u32[19] ^ ek_u32[24];
        ek_u32[26] = ek_u32[20] ^ ek_u32[25];
        ek_u32[27] = ek_u32[21] ^ ek_u32[26];
        ek_u32[28] = ek_u32[22] ^ ek_u32[27];
        ek_u32[29] = ek_u32[23] ^ ek_u32[28];

        ek_u32[30] = ek_u32[24] ^ (sub_word(ek_u32[29]).rotate_left(24) ^ RCON[4]);
        ek_u32[31] = ek_u32[25] ^ ek_u32[30];
        ek_u32[32] = ek_u32[26] ^ ek_u32[31];
        ek_u32[33] = ek_u32[27] ^ ek_u32[32];
        ek_u32[34] = ek_u32[28] ^ ek_u32[33];
        ek_u32[35] = ek_u32[29] ^ ek_u32[34];

        ek_u32[36] = ek_u32[30] ^ (sub_word(ek_u32[35]).rotate_left(24) ^ RCON[5]);
        ek_u32[37] = ek_u32[31] ^ ek_u32[36];
        ek_u32[38] = ek_u32[32] ^ ek_u32[37];
        ek_u32[39] = ek_u32[33] ^ ek_u32[38];
        ek_u32[40] = ek_u32[34] ^ ek_u32[39];
        ek_u32[41] = ek_u32[35] ^ ek_u32[40];

        ek_u32[42] = ek_u32[36] ^ (sub_word(ek_u32[41]).rotate_left(24) ^ RCON[6]);
        ek_u32[43] = ek_u32[37] ^ ek_u32[42];
        ek_u32[44] = ek_u32[38] ^ ek_u32[43];
        ek_u32[45] = ek_u32[39] ^ ek_u32[44];
        ek_u32[46] = ek_u32[40] ^ ek_u32[45];
        ek_u32[47] = ek_u32[41] ^ ek_u32[46];

        ek_u32[48] = ek_u32[42] ^ (sub_word(ek_u32[47]).rotate_left(24) ^ RCON[7]);
        ek_u32[49] = ek_u32[43] ^ ek_u32[48];
        ek_u32[50] = ek_u32[44] ^ ek_u32[49];
        ek_u32[51] = ek_u32[45] ^ ek_u32[50];
        
        let ptr = ek_u32.as_ptr();
        let mut ek = [vdupq_n_u8(0); 24];
        
        // Load round keys
        for i in 0..13 {
            ek[i] = vreinterpretq_u8_u32(vld1q_u32(ptr.add(i * 4)));
        }
        
        // Prepare inverse keys for decryption
        for i in 0..11 {
            ek[13 + i] = vaesimcq_u8(ek[11 - i]);
        }
        
        Self { ek }
    }
    
    #[inline(always)]
    pub fn encrypt(&self, block: &mut [u8], round: usize, step: Steps) {
        unsafe { self.encrypt_simd(block, round, step) }
    }
    
    #[target_feature(enable = "neon,aes")]
    unsafe fn encrypt_simd(&self, block: &mut [u8], round: usize, step: Steps) {
        encrypt_simd_generic(&self.ek, block, round, step, 12);
    }
    
    #[inline(always)]
    pub fn decrypt(&self, block: &mut [u8], round: usize, step: InvSteps) {
        unsafe { self.decrypt_simd(block, round, step) }
    }
    
    #[target_feature(enable = "neon,aes")]
    unsafe fn decrypt_simd(&self, block: &mut [u8], round: usize, step: InvSteps) {
        decrypt_simd_generic(&self.ek, block, round, step, 12);
    }
}

// Aes256 implementation
#[derive(Clone)]
pub struct Aes256 {
    ek: [uint8x16_t; 28],
}

impl Aes256 {
    pub const KEY_LEN: usize = 32;
    
    #[inline(always)]
    pub fn new(key: &[u8]) -> Self {
        unsafe { Self::new_simd(key) }
    }
    
    #[target_feature(enable = "neon,aes")]
    unsafe fn new_simd(key: &[u8]) -> Self {
        assert_eq!(key.len(), Self::KEY_LEN);
        
        use core::mem::transmute;
        let mut ek_u32: [u32; 60] = [0u32; 60];
        
        let k1: [u32; 4] = transmute(vld1q_u32(key.as_ptr().add(0) as *const u32));
        let k2: [u32; 4] = transmute(vld1q_u32(key.as_ptr().add(16) as *const u32));
        
        ek_u32[0] = k1[0];
        ek_u32[1] = k1[1];
        ek_u32[2] = k1[2];
        ek_u32[3] = k1[3];
        ek_u32[4] = k2[0];
        ek_u32[5] = k2[1];
        ek_u32[6] = k2[2];
        ek_u32[7] = k2[3];
        
        // Key expansion - explicitly unrolled as in reference
        ek_u32[8] = ek_u32[0] ^ (sub_word(ek_u32[7]).rotate_left(24) ^ RCON[0]);
        ek_u32[9] = ek_u32[1] ^ ek_u32[8];
        ek_u32[10] = ek_u32[2] ^ ek_u32[9];
        ek_u32[11] = ek_u32[3] ^ ek_u32[10];
        ek_u32[12] = ek_u32[4] ^ sub_word(ek_u32[11]);
        ek_u32[13] = ek_u32[5] ^ ek_u32[12];
        ek_u32[14] = ek_u32[6] ^ ek_u32[13];
        ek_u32[15] = ek_u32[7] ^ ek_u32[14];

        ek_u32[16] = ek_u32[8] ^ (sub_word(ek_u32[15]).rotate_left(24) ^ RCON[1]);
        ek_u32[17] = ek_u32[9] ^ ek_u32[16];
        ek_u32[18] = ek_u32[10] ^ ek_u32[17];
        ek_u32[19] = ek_u32[11] ^ ek_u32[18];
        ek_u32[20] = ek_u32[12] ^ sub_word(ek_u32[19]);
        ek_u32[21] = ek_u32[13] ^ ek_u32[20];
        ek_u32[22] = ek_u32[14] ^ ek_u32[21];
        ek_u32[23] = ek_u32[15] ^ ek_u32[22];

        ek_u32[24] = ek_u32[16] ^ (sub_word(ek_u32[23]).rotate_left(24) ^ RCON[2]);
        ek_u32[25] = ek_u32[17] ^ ek_u32[24];
        ek_u32[26] = ek_u32[18] ^ ek_u32[25];
        ek_u32[27] = ek_u32[19] ^ ek_u32[26];
        ek_u32[28] = ek_u32[20] ^ sub_word(ek_u32[27]);
        ek_u32[29] = ek_u32[21] ^ ek_u32[28];
        ek_u32[30] = ek_u32[22] ^ ek_u32[29];
        ek_u32[31] = ek_u32[23] ^ ek_u32[30];

        ek_u32[32] = ek_u32[24] ^ (sub_word(ek_u32[31]).rotate_left(24) ^ RCON[3]);
        ek_u32[33] = ek_u32[25] ^ ek_u32[32];
        ek_u32[34] = ek_u32[26] ^ ek_u32[33];
        ek_u32[35] = ek_u32[27] ^ ek_u32[34];
        ek_u32[36] = ek_u32[28] ^ sub_word(ek_u32[35]);
        ek_u32[37] = ek_u32[29] ^ ek_u32[36];
        ek_u32[38] = ek_u32[30] ^ ek_u32[37];
        ek_u32[39] = ek_u32[31] ^ ek_u32[38];

        ek_u32[40] = ek_u32[32] ^ (sub_word(ek_u32[39]).rotate_left(24) ^ RCON[4]);
        ek_u32[41] = ek_u32[33] ^ ek_u32[40];
        ek_u32[42] = ek_u32[34] ^ ek_u32[41];
        ek_u32[43] = ek_u32[35] ^ ek_u32[42];
        ek_u32[44] = ek_u32[36] ^ sub_word(ek_u32[43]);
        ek_u32[45] = ek_u32[37] ^ ek_u32[44];
        ek_u32[46] = ek_u32[38] ^ ek_u32[45];
        ek_u32[47] = ek_u32[39] ^ ek_u32[46];

        ek_u32[48] = ek_u32[40] ^ (sub_word(ek_u32[47]).rotate_left(24) ^ RCON[5]);
        ek_u32[49] = ek_u32[41] ^ ek_u32[48];
        ek_u32[50] = ek_u32[42] ^ ek_u32[49];
        ek_u32[51] = ek_u32[43] ^ ek_u32[50];
        ek_u32[52] = ek_u32[44] ^ sub_word(ek_u32[51]);
        ek_u32[53] = ek_u32[45] ^ ek_u32[52];
        ek_u32[54] = ek_u32[46] ^ ek_u32[53];
        ek_u32[55] = ek_u32[47] ^ ek_u32[54];

        ek_u32[56] = ek_u32[48] ^ (sub_word(ek_u32[55]).rotate_left(24) ^ RCON[6]);
        ek_u32[57] = ek_u32[49] ^ ek_u32[56];
        ek_u32[58] = ek_u32[50] ^ ek_u32[57];
        ek_u32[59] = ek_u32[51] ^ ek_u32[58];
        
        let ptr = ek_u32.as_ptr();
        let mut ek = [vdupq_n_u8(0); 28];
        
        // Load round keys
        for i in 0..15 {
            ek[i] = vreinterpretq_u8_u32(vld1q_u32(ptr.add(i * 4)));
        }
        
        // Prepare inverse keys for decryption
        for i in 0..13 {
            ek[15 + i] = vaesimcq_u8(ek[13 - i]);
        }
        
        Self { ek }
    }
    
    #[inline(always)]
    pub fn encrypt(&self, block: &mut [u8], round: usize, step: Steps) {
        unsafe { self.encrypt_simd(block, round, step) }
    }
    
    #[target_feature(enable = "neon,aes")]
    unsafe fn encrypt_simd(&self, block: &mut [u8], round: usize, step: Steps) {
        encrypt_simd_generic(&self.ek, block, round, step, 14);
    }
    
    #[inline(always)]
    pub fn decrypt(&self, block: &mut [u8], round: usize, step: InvSteps) {
        unsafe { self.decrypt_simd(block, round, step) }
    }
    
    #[target_feature(enable = "neon,aes")]
    unsafe fn decrypt_simd(&self, block: &mut [u8], round: usize, step: InvSteps) {
        decrypt_simd_generic(&self.ek, block, round, step, 14);
    }
}