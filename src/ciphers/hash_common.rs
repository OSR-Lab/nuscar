pub(super) const BLOCK_SIZE: usize = 64;

#[inline]
pub(super) fn bytes_to_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Compute known_bytes sum for byte-by-byte attacks
#[inline]
pub(super) fn compute_known_sum(known_bytes: &Option<Vec<u8>>) -> u32 {
    match known_bytes {
        Some(bytes) => {
            let mut sum = 0u32;
            for (i, &b) in bytes.iter().enumerate() {
                sum = sum.wrapping_add((b as u32) << (i * 8));
            }
            sum
        }
        None => 0,
    }
}

/// Pad plaintext for HMAC inner hash second block.
/// length_in_bits = (BLOCK_SIZE + plaintext_len) * 8
pub(super) fn hmac_pad_plaintext(plaintext: &[u8]) -> Vec<u8> {
    let total_len_bits = ((BLOCK_SIZE + plaintext.len()) * 8) as u64;
    let mut data = Vec::with_capacity(BLOCK_SIZE);
    data.extend_from_slice(plaintext);
    data.push(0x80);
    while (data.len() * 8) % 512 != 448 {
        data.push(0x00);
    }
    data.extend_from_slice(&total_len_bits.to_be_bytes());
    data
}

/// Parse optional IV from Python list to [u32; 8] (shared by SHA-256 and SM3)
#[inline]
pub(super) fn parse_iv_8(iv: &Option<Vec<u32>>) -> Option<[u32; 8]> {
    iv.as_ref().map(|v| {
        let mut arr = [0u32; 8];
        arr.copy_from_slice(v);
        arr
    })
}

/// Normalize HMAC key: hash if > BLOCK_SIZE, then pad to BLOCK_SIZE.
/// Returns (inner_key, outer_key) where inner = k XOR 0x36, outer = k XOR 0x5c.
pub(super) fn hmac_derive_keys(normalized_key: &[u8]) -> ([u8; 64], [u8; 64]) {
    let mut inner_key = [0u8; 64];
    let mut outer_key = [0u8; 64];
    for i in 0..64 {
        inner_key[i] = normalized_key[i] ^ 0x36;
        outer_key[i] = normalized_key[i] ^ 0x5c;
    }
    (inner_key, outer_key)
}
