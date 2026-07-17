// DES module with common definitions: constants and primitive functions.
// This is a pure Rust file with no PyO3 dependencies.
// Ported from nuscar/ciphers/des.py

use num_enum::FromPrimitive;

// ============================================================================
// DES S-BOX constants: 8 S-boxes, each 64 entries (6-bit input -> 4-bit output)
// ============================================================================
pub const SBOXES: [[u8; 64]; 8] = [
    [
        0xE, 0x0, 0x4, 0xF, 0xD, 0x7, 0x1, 0x4, 0x2, 0xE, 0xF, 0x2, 0xB, 0xD, 0x8, 0x1,
        0x3, 0xA, 0xA, 0x6, 0x6, 0xC, 0xC, 0xB, 0x5, 0x9, 0x9, 0x5, 0x0, 0x3, 0x7, 0x8,
        0x4, 0xF, 0x1, 0xC, 0xE, 0x8, 0x8, 0x2, 0xD, 0x4, 0x6, 0x9, 0x2, 0x1, 0xB, 0x7,
        0xF, 0x5, 0xC, 0xB, 0x9, 0x3, 0x7, 0xE, 0x3, 0xA, 0xA, 0x0, 0x5, 0x6, 0x0, 0xD,
    ],
    [
        0xF, 0x3, 0x1, 0xD, 0x8, 0x4, 0xE, 0x7, 0x6, 0xF, 0xB, 0x2, 0x3, 0x8, 0x4, 0xE,
        0x9, 0xC, 0x7, 0x0, 0x2, 0x1, 0xD, 0xA, 0xC, 0x6, 0x0, 0x9, 0x5, 0xB, 0xA, 0x5,
        0x0, 0xD, 0xE, 0x8, 0x7, 0xA, 0xB, 0x1, 0xA, 0x3, 0x4, 0xF, 0xD, 0x4, 0x1, 0x2,
        0x5, 0xB, 0x8, 0x6, 0xC, 0x7, 0x6, 0xC, 0x9, 0x0, 0x3, 0x5, 0x2, 0xE, 0xF, 0x9,
    ],
    [
        0xA, 0xD, 0x0, 0x7, 0x9, 0x0, 0xE, 0x9, 0x6, 0x3, 0x3, 0x4, 0xF, 0x6, 0x5, 0xA,
        0x1, 0x2, 0xD, 0x8, 0xC, 0x5, 0x7, 0xE, 0xB, 0xC, 0x4, 0xB, 0x2, 0xF, 0x8, 0x1,
        0xD, 0x1, 0x6, 0xA, 0x4, 0xD, 0x9, 0x0, 0x8, 0x6, 0xF, 0x9, 0x3, 0x8, 0x0, 0x7,
        0xB, 0x4, 0x1, 0xF, 0x2, 0xE, 0xC, 0x3, 0x5, 0xB, 0xA, 0x5, 0xE, 0x2, 0x7, 0xC,
    ],
    [
        0x7, 0xD, 0xD, 0x8, 0xE, 0xB, 0x3, 0x5, 0x0, 0x6, 0x6, 0xF, 0x9, 0x0, 0xA, 0x3,
        0x1, 0x4, 0x2, 0x7, 0x8, 0x2, 0x5, 0xC, 0xB, 0x1, 0xC, 0xA, 0x4, 0xE, 0xF, 0x9,
        0xA, 0x3, 0x6, 0xF, 0x9, 0x0, 0x0, 0x6, 0xC, 0xA, 0xB, 0x1, 0x7, 0xD, 0xD, 0x8,
        0xF, 0x9, 0x1, 0x4, 0x3, 0x5, 0xE, 0xB, 0x5, 0xC, 0x2, 0x7, 0x8, 0x2, 0x4, 0xE,
    ],
    [
        0x2, 0xE, 0xC, 0xB, 0x4, 0x2, 0x1, 0xC, 0x7, 0x4, 0xA, 0x7, 0xB, 0xD, 0x6, 0x1,
        0x8, 0x5, 0x5, 0x0, 0x3, 0xF, 0xF, 0xA, 0xD, 0x3, 0x0, 0x9, 0xE, 0x8, 0x9, 0x6,
        0x4, 0xB, 0x2, 0x8, 0x1, 0xC, 0xB, 0x7, 0xA, 0x1, 0xD, 0xE, 0x7, 0x2, 0x8, 0xD,
        0xF, 0x6, 0x9, 0xF, 0xC, 0x0, 0x5, 0x9, 0x6, 0xA, 0x3, 0x4, 0x0, 0x5, 0xE, 0x3,
    ],
    [
        0xC, 0xA, 0x1, 0xF, 0xA, 0x4, 0xF, 0x2, 0x9, 0x7, 0x2, 0xC, 0x6, 0x9, 0x8, 0x5,
        0x0, 0x6, 0xD, 0x1, 0x3, 0xD, 0x4, 0xE, 0xE, 0x0, 0x7, 0xB, 0x5, 0x3, 0xB, 0x8,
        0x9, 0x4, 0xE, 0x3, 0xF, 0x2, 0x5, 0xC, 0x2, 0x9, 0x8, 0x5, 0xC, 0xF, 0x3, 0xA,
        0x7, 0xB, 0x0, 0xE, 0x4, 0x1, 0xA, 0x7, 0x1, 0x6, 0xD, 0x0, 0xB, 0x8, 0x6, 0xD,
    ],
    [
        0x4, 0xD, 0xB, 0x0, 0x2, 0xB, 0xE, 0x7, 0xF, 0x4, 0x0, 0x9, 0x8, 0x1, 0xD, 0xA,
        0x3, 0xE, 0xC, 0x3, 0x9, 0x5, 0x7, 0xC, 0x5, 0x2, 0xA, 0xF, 0x6, 0x8, 0x1, 0x6,
        0x1, 0x6, 0x4, 0xB, 0xB, 0xD, 0xD, 0x8, 0xC, 0x1, 0x3, 0x4, 0x7, 0xA, 0xE, 0x7,
        0xA, 0x9, 0xF, 0x5, 0x6, 0x0, 0x8, 0xF, 0x0, 0xE, 0x5, 0x2, 0x9, 0x3, 0x2, 0xC,
    ],
    [
        0xD, 0x1, 0x2, 0xF, 0x8, 0xD, 0x4, 0x8, 0x6, 0xA, 0xF, 0x3, 0xB, 0x7, 0x1, 0x4,
        0xA, 0xC, 0x9, 0x5, 0x3, 0x6, 0xE, 0xB, 0x5, 0x0, 0x0, 0xE, 0xC, 0x9, 0x7, 0x2,
        0x7, 0x2, 0xB, 0x1, 0x4, 0xE, 0x1, 0x7, 0x9, 0x4, 0xC, 0xA, 0xE, 0x8, 0x2, 0xD,
        0x0, 0xF, 0x6, 0xC, 0xA, 0x9, 0xD, 0x0, 0xF, 0x3, 0x3, 0x5, 0x5, 0x6, 0x8, 0xB,
    ],
];

// ============================================================================
// ROUND_KEY_BITS_INDEXES: [16][8][6] array
// ============================================================================
pub const ROUND_KEY_BITS_INDEXES: [[[u8; 6]; 8]; 16] = [
    // Round key 0
    [
        [9, 50, 33, 59, 48, 16],
        [32, 56, 1, 8, 18, 41],
        [2, 34, 25, 24, 43, 57],
        [58, 0, 35, 26, 17, 40],
        [21, 27, 38, 53, 36, 3],
        [46, 29, 4, 52, 22, 28],
        [60, 20, 37, 62, 14, 19],
        [44, 13, 12, 61, 54, 30],
    ],
    // Round key 1
    [
        [1, 42, 25, 51, 40, 8],
        [24, 48, 58, 0, 10, 33],
        [59, 26, 17, 16, 35, 49],
        [50, 57, 56, 18, 9, 32],
        [13, 19, 30, 45, 28, 62],
        [38, 21, 27, 44, 14, 20],
        [52, 12, 29, 54, 6, 11],
        [36, 5, 4, 53, 46, 22],
    ],
    // Round key 2
    [
        [50, 26, 9, 35, 24, 57],
        [8, 32, 42, 49, 59, 17],
        [43, 10, 1, 0, 48, 33],
        [34, 41, 40, 2, 58, 16],
        [60, 3, 14, 29, 12, 46],
        [22, 5, 11, 28, 61, 4],
        [36, 27, 13, 38, 53, 62],
        [20, 52, 19, 37, 30, 6],
    ],
    // Round key 3
    [
        [34, 10, 58, 48, 8, 41],
        [57, 16, 26, 33, 43, 1],
        [56, 59, 50, 49, 32, 17],
        [18, 25, 24, 51, 42, 0],
        [44, 54, 61, 13, 27, 30],
        [6, 52, 62, 12, 45, 19],
        [20, 11, 60, 22, 37, 46],
        [4, 36, 3, 21, 14, 53],
    ],
    // Round key 4
    [
        [18, 59, 42, 32, 57, 25],
        [41, 0, 10, 17, 56, 50],
        [40, 43, 34, 33, 16, 1],
        [2, 9, 8, 35, 26, 49],
        [28, 38, 45, 60, 11, 14],
        [53, 36, 46, 27, 29, 3],
        [4, 62, 44, 6, 21, 30],
        [19, 20, 54, 5, 61, 37],
    ],
    // Round key 5
    [
        [2, 43, 26, 16, 41, 9],
        [25, 49, 59, 1, 40, 34],
        [24, 56, 18, 17, 0, 50],
        [51, 58, 57, 48, 10, 33],
        [12, 22, 29, 44, 62, 61],
        [37, 20, 30, 11, 13, 54],
        [19, 46, 28, 53, 5, 14],
        [3, 4, 38, 52, 45, 21],
    ],
    // Round key 6
    [
        [51, 56, 10, 0, 25, 58],
        [9, 33, 43, 50, 24, 18],
        [8, 40, 2, 1, 49, 34],
        [35, 42, 41, 32, 59, 17],
        [27, 6, 13, 28, 46, 45],
        [21, 4, 14, 62, 60, 38],
        [3, 30, 12, 37, 52, 61],
        [54, 19, 22, 36, 29, 5],
    ],
    // Round key 7
    [
        [35, 40, 59, 49, 9, 42],
        [58, 17, 56, 34, 8, 2],
        [57, 24, 51, 50, 33, 18],
        [48, 26, 25, 16, 43, 1],
        [11, 53, 60, 12, 30, 29],
        [5, 19, 61, 46, 44, 22],
        [54, 14, 27, 21, 36, 45],
        [38, 3, 6, 20, 13, 52],
    ],
    // Round key 8
    [
        [56, 32, 51, 41, 1, 34],
        [50, 9, 48, 26, 0, 59],
        [49, 16, 43, 42, 25, 10],
        [40, 18, 17, 8, 35, 58],
        [3, 45, 52, 4, 22, 21],
        [60, 11, 53, 38, 36, 14],
        [46, 6, 19, 13, 28, 37],
        [30, 62, 61, 12, 5, 44],
    ],
    // Round key 9
    [
        [40, 16, 35, 25, 50, 18],
        [34, 58, 32, 10, 49, 43],
        [33, 0, 56, 26, 9, 59],
        [24, 2, 1, 57, 48, 42],
        [54, 29, 36, 19, 6, 5],
        [44, 62, 37, 22, 20, 61],
        [30, 53, 3, 60, 12, 21],
        [14, 46, 45, 27, 52, 28],
    ],
    // Round key 10
    [
        [24, 0, 48, 9, 34, 2],
        [18, 42, 16, 59, 33, 56],
        [17, 49, 40, 10, 58, 43],
        [8, 51, 50, 41, 32, 26],
        [38, 13, 20, 3, 53, 52],
        [28, 46, 21, 6, 4, 45],
        [14, 37, 54, 44, 27, 5],
        [61, 30, 29, 11, 36, 12],
    ],
    // Round key 11
    [
        [8, 49, 32, 58, 18, 51],
        [2, 26, 0, 43, 17, 40],
        [1, 33, 24, 59, 42, 56],
        [57, 35, 34, 25, 16, 10],
        [22, 60, 4, 54, 37, 36],
        [12, 30, 5, 53, 19, 29],
        [61, 21, 38, 28, 11, 52],
        [45, 14, 13, 62, 20, 27],
    ],
    // Round key 12
    [
        [57, 33, 16, 42, 2, 35],
        [51, 10, 49, 56, 1, 24],
        [50, 17, 8, 43, 26, 40],
        [41, 48, 18, 9, 0, 59],
        [6, 44, 19, 38, 21, 20],
        [27, 14, 52, 37, 3, 13],
        [45, 5, 22, 12, 62, 36],
        [29, 61, 60, 46, 4, 11],
    ],
    // Round key 13
    [
        [41, 17, 0, 26, 51, 48],
        [35, 59, 33, 40, 50, 8],
        [34, 1, 57, 56, 10, 24],
        [25, 32, 2, 58, 49, 43],
        [53, 28, 3, 22, 5, 4],
        [11, 61, 36, 21, 54, 60],
        [29, 52, 6, 27, 46, 20],
        [13, 45, 44, 30, 19, 62],
    ],
    // Round key 14
    [
        [25, 1, 49, 10, 35, 32],
        [48, 43, 17, 24, 34, 57],
        [18, 50, 41, 40, 59, 8],
        [9, 16, 51, 42, 33, 56],
        [37, 12, 54, 6, 52, 19],
        [62, 45, 20, 5, 38, 44],
        [13, 36, 53, 11, 30, 4],
        [60, 29, 28, 14, 3, 46],
    ],
    // Round key 15
    [
        [17, 58, 41, 2, 56, 24],
        [40, 35, 9, 16, 26, 49],
        [10, 42, 33, 32, 51, 0],
        [1, 8, 43, 34, 25, 48],
        [29, 4, 46, 61, 44, 11],
        [54, 37, 12, 60, 30, 36],
        [5, 28, 45, 3, 22, 27],
        [52, 21, 20, 6, 62, 38],
    ],
];

// ============================================================================
// DES Steps enum - matches Python Steps enum values 0-9
// ============================================================================
#[derive(Debug, Eq, PartialEq, FromPrimitive, Clone, Copy)]
#[repr(u8)]
pub enum DesSteps {
    InitialPermutation = 0,
    ExpansivePermutation = 1,
    AddRoundKey = 2,
    Sboxes = 3,
    PermutationP = 4,
    XorWithSavedLeftRight = 5,
    PermuteRightLeft = 6,
    InvPermutationPRight = 7,
    InvPermutationPDeltaRight = 8,
    #[num_enum(default)]
    FinalPermutation = 9,
}

// ============================================================================
// Primitive functions
// ============================================================================

/// Compute DES initial permutation (IP).
/// Input: 8 bytes. Output: 8 bytes (L0R0).
/// Ported from Python initial_permutation (des.py lines 454-487).
pub fn initial_permutation(input: &[u8; 8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    // Python iterates current_byte 0..8, using data[7 - current_byte]
    for current_byte in 0..8u8 {
        let d = input[(7 - current_byte) as usize];
        out[0] <<= 1;
        out[0] += (d >> 6) & 0x01;
        out[1] <<= 1;
        out[1] += (d >> 4) & 0x01;
        out[2] <<= 1;
        out[2] += (d >> 2) & 0x01;
        out[3] <<= 1;
        out[3] += d & 0x01;
        out[4] <<= 1;
        out[4] += (d >> 7) & 0x01;
        out[5] <<= 1;
        out[5] += (d >> 5) & 0x01;
        out[6] <<= 1;
        out[6] += (d >> 3) & 0x01;
        out[7] <<= 1;
        out[7] += (d >> 1) & 0x01;
    }
    out
}

/// Compute DES expansive permutation (EP).
/// Input: 4 bytes (R half). Output: 8 bytes (8x6-bit words).
/// Ported from Python expansive_permutation (des.py lines 490-510).
pub fn expansive_permutation(input: &[u8; 4]) -> [u8; 8] {
    let mut out = [0u8; 8];
    for current_2byte_block in 0..4usize {
        out[2 * current_2byte_block] = (input[(current_2byte_block + 3) % 4] & 0x01) << 5;
        out[2 * current_2byte_block] += (input[current_2byte_block] & 0xF8) >> 3;
        out[2 * current_2byte_block + 1] = (input[current_2byte_block] & 0x1F) << 1;
        out[2 * current_2byte_block + 1] += (input[(current_2byte_block + 1) % 4] & 0x80) >> 7;
    }
    out
}

/// Compute DES SBOXes operation.
/// Input: 8 bytes (8x6-bit words). Output: 8 bytes (8x4-bit words).
/// Ported from Python sboxes (des.py lines 538-555).
pub fn sboxes(input: &[u8; 8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    for current_word in 0..8usize {
        out[current_word] = SBOXES[current_word][input[current_word] as usize];
    }
    out
}

/// Compute DES permutation P (PP).
/// Input: 8 bytes (8x4-bit words). Output: 4 bytes.
/// Ported from Python permutation_p (des.py lines 558-610).
pub fn permutation_p(input: &[u8; 8]) -> [u8; 4] {
    let d = input;
    let mut out = [0u8; 4];

    out[0] = (d[3] & 0x01) << 7;
    out[0] += (d[1] & 0x02) << 5;
    out[0] += (d[4] & 0x01) << 5;
    out[0] += (d[5] & 0x08) << 1;
    out[0] += d[7] & 0x08;
    out[0] += (d[2] & 0x01) << 2;
    out[0] += (d[6] & 0x01) << 1;
    out[0] += (d[4] & 0x08) >> 3;

    out[1] = (d[0] & 0x08) << 4;
    out[1] += (d[3] & 0x02) << 5;
    out[1] += (d[5] & 0x02) << 4;
    out[1] += (d[6] & 0x04) << 2;
    out[1] += d[1] & 0x08;
    out[1] += d[4] & 0x04;
    out[1] += d[7] & 0x02;
    out[1] += (d[2] & 0x04) >> 2;

    out[2] = (d[0] & 0x04) << 5;
    out[2] += (d[1] & 0x01) << 6;
    out[2] += (d[5] & 0x01) << 5;
    out[2] += (d[3] & 0x04) << 2;
    out[2] += (d[7] & 0x01) << 3;
    out[2] += (d[6] & 0x02) << 1;
    out[2] += d[0] & 0x02;
    out[2] += (d[2] & 0x08) >> 3;

    out[3] = (d[4] & 0x02) << 6;
    out[3] += (d[3] & 0x08) << 3;
    out[3] += (d[7] & 0x04) << 3;
    out[3] += (d[1] & 0x04) << 2;
    out[3] += (d[5] & 0x04) << 1;
    out[3] += (d[2] & 0x02) << 1;
    out[3] += (d[0] & 0x01) << 1;
    out[3] += (d[6] & 0x08) >> 3;

    out
}

/// Compute inverse of DES permutation P.
/// Input: 4 bytes. Output: 8 bytes (8x4-bit words).
/// Ported from Python inv_permutation_p (des.py lines 613-668).
pub fn inv_permutation_p(input: &[u8; 4]) -> [u8; 8] {
    let d = input;
    let mut out = [0u8; 8];

    out[0] += (((d[1] & 0x80) != 0) as u8) << 3;
    out[0] += (((d[2] & 0x80) != 0) as u8) << 2;
    out[0] += (((d[2] & 0x02) != 0) as u8) << 1;
    out[0] += ((d[3] & 0x02) != 0) as u8;

    out[1] += (((d[1] & 0x08) != 0) as u8) << 3;
    out[1] += (((d[3] & 0x10) != 0) as u8) << 2;
    out[1] += (((d[0] & 0x40) != 0) as u8) << 1;
    out[1] += ((d[2] & 0x40) != 0) as u8;

    out[2] += (((d[2] & 0x01) != 0) as u8) << 3;
    out[2] += (((d[1] & 0x01) != 0) as u8) << 2;
    out[2] += (((d[3] & 0x04) != 0) as u8) << 1;
    out[2] += ((d[0] & 0x04) != 0) as u8;

    out[3] += (((d[3] & 0x40) != 0) as u8) << 3;
    out[3] += (((d[2] & 0x10) != 0) as u8) << 2;
    out[3] += (((d[1] & 0x40) != 0) as u8) << 1;
    out[3] += ((d[0] & 0x80) != 0) as u8;

    out[4] += (((d[0] & 0x01) != 0) as u8) << 3;
    out[4] += (((d[1] & 0x04) != 0) as u8) << 2;
    out[4] += (((d[3] & 0x80) != 0) as u8) << 1;
    out[4] += ((d[0] & 0x20) != 0) as u8;

    out[5] += (((d[0] & 0x10) != 0) as u8) << 3;
    out[5] += (((d[3] & 0x08) != 0) as u8) << 2;
    out[5] += (((d[1] & 0x20) != 0) as u8) << 1;
    out[5] += ((d[2] & 0x20) != 0) as u8;

    out[6] += (((d[3] & 0x01) != 0) as u8) << 3;
    out[6] += (((d[1] & 0x10) != 0) as u8) << 2;
    out[6] += (((d[2] & 0x04) != 0) as u8) << 1;
    out[6] += ((d[0] & 0x02) != 0) as u8;

    out[7] += (((d[0] & 0x08) != 0) as u8) << 3;
    out[7] += (((d[3] & 0x20) != 0) as u8) << 2;
    out[7] += (((d[1] & 0x02) != 0) as u8) << 1;
    out[7] += ((d[2] & 0x08) != 0) as u8;

    out
}

/// Compute DES final permutation (FP).
/// Input: 8 bytes. Output: 8 bytes (the final ciphertext).
/// Ported from Python final_permutation (des.py lines 671-705).
pub fn final_permutation(input: &[u8; 8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    for current_byte in 0..8u8 {
        // real_current_byte_index = int(current_byte / 2) + 4 * (1 - (current_byte % 2))
        let real_current_byte_index =
            (current_byte / 2) as usize + 4 * (1 - (current_byte % 2) as usize);
        let d = input[real_current_byte_index];
        out[0] <<= 1;
        out[0] += d & 0x01;
        out[1] <<= 1;
        out[1] += (d >> 1) & 0x01;
        out[2] <<= 1;
        out[2] += (d >> 2) & 0x01;
        out[3] <<= 1;
        out[3] += (d >> 3) & 0x01;
        out[4] <<= 1;
        out[4] += (d >> 4) & 0x01;
        out[5] <<= 1;
        out[5] += (d >> 5) & 0x01;
        out[6] <<= 1;
        out[6] += (d >> 6) & 0x01;
        out[7] <<= 1;
        out[7] += (d >> 7) & 0x01;
    }
    out
}

/// Compute DES key schedule.
/// Input: 8-byte key. Output: 16 round keys, each 8 bytes (8x6-bit words).
/// Ported from Python key_schedule (des.py lines 309-359).
pub fn key_schedule(key: &[u8; 8]) -> [[u8; 8]; 16] {
    // Split key into 64 individual bits
    let mut key_bits = [0u8; 64];
    for current_key_byte in 0..8usize {
        key_bits[current_key_byte * 8 + 0] = ((key[current_key_byte] & 0x80) != 0x00) as u8;
        key_bits[current_key_byte * 8 + 1] = ((key[current_key_byte] & 0x40) != 0x00) as u8;
        key_bits[current_key_byte * 8 + 2] = ((key[current_key_byte] & 0x20) != 0x00) as u8;
        key_bits[current_key_byte * 8 + 3] = ((key[current_key_byte] & 0x10) != 0x00) as u8;
        key_bits[current_key_byte * 8 + 4] = ((key[current_key_byte] & 0x08) != 0x00) as u8;
        key_bits[current_key_byte * 8 + 5] = ((key[current_key_byte] & 0x04) != 0x00) as u8;
        key_bits[current_key_byte * 8 + 6] = ((key[current_key_byte] & 0x02) != 0x00) as u8;
        key_bits[current_key_byte * 8 + 7] = ((key[current_key_byte] & 0x01) != 0x00) as u8;
    }

    let mut output = [[0u8; 8]; 16];
    for current_round in 0..16usize {
        for current_word in 0..8usize {
            output[current_round][current_word] = 0x20
                * key_bits[ROUND_KEY_BITS_INDEXES[current_round][current_word][0] as usize]
                + 0x10
                    * key_bits
                        [ROUND_KEY_BITS_INDEXES[current_round][current_word][1] as usize]
                + 0x08
                    * key_bits
                        [ROUND_KEY_BITS_INDEXES[current_round][current_word][2] as usize]
                + 0x04
                    * key_bits
                        [ROUND_KEY_BITS_INDEXES[current_round][current_word][3] as usize]
                + 0x02
                    * key_bits
                        [ROUND_KEY_BITS_INDEXES[current_round][current_word][4] as usize]
                + 0x01
                    * key_bits
                        [ROUND_KEY_BITS_INDEXES[current_round][current_word][5] as usize];
        }
    }
    output
}

// ============================================================================
// DES encrypt/decrypt with stop control
// ============================================================================

/// DES encrypt with stop control at a specific round and step.
///
/// The encrypt function processes:
/// - Round 0: IP -> EP -> AddRK -> SBOX -> PP -> XOR_LR -> PermuteRL
/// - Rounds 1-14: EP -> AddRK -> SBOX -> PP -> XOR_LR -> PermuteRL
/// - Round 15: EP -> AddRK -> SBOX -> PP -> XOR_LR -> FP (no PermuteRL on last round)
///
/// Stop at (at_round, after_step) and return the intermediate state.
pub fn des_encrypt_at(
    plaintext: &[u8; 8],
    expanded_key: &[[u8; 8]; 16],
    at_round: usize,
    after_step: u8,
) -> [u8; 8] {
    des_cipher_at(plaintext, expanded_key, at_round, after_step, false)
}

/// DES decrypt with stop control at a specific round and step.
///
/// Same structure as encrypt but reversed round key order (key[15] used first).
pub fn des_decrypt_at(
    ciphertext: &[u8; 8],
    expanded_key: &[[u8; 8]; 16],
    at_round: usize,
    after_step: u8,
) -> [u8; 8] {
    des_cipher_at(ciphertext, expanded_key, at_round, after_step, true)
}

/// Internal DES cipher with stop control.
/// When decrypt=true, the round keys are used in reverse order.
fn des_cipher_at(
    input: &[u8; 8],
    expanded_key: &[[u8; 8]; 16],
    at_round: usize,
    after_step: u8,
    decrypt: bool,
) -> [u8; 8] {
    debug_assert!(at_round < 16, "at_round must be < 16, got {}", at_round);

    // Build the operations list for each round, mirroring Python's _prepare_des_iterations
    // FIRST_ROUND = [IP, EP, AddRK, SBOX, PP, XOR_LR, PermuteRL, None, None, None]
    // ROUND       = [None, EP, AddRK, SBOX, PP, XOR_LR, PermuteRL, None, None, None]
    // FINAL_ROUND = [None, EP, AddRK, SBOX, PP, XOR_LR, None, None, None, FP]

    let mut state = *input;
    let mut saved_left_right = [0u8; 8];

    // Index mapping: decrypt uses reverse key order, encrypt uses forward
    let key_idx = |round: usize| -> usize {
        if decrypt { 15 - round } else { round }
    };

    for round_number in 0..=at_round {
        let is_first_round = round_number == 0;
        let is_last_round = round_number == 15;
        let is_stop_round = round_number == at_round;

        // Determine operations for this round:
        // For first round: [IP, EP, AddRK, SBOX, PP, XOR_LR, PermuteRL, InvPPR, InvPPDR, FP]
        //   where IP is active, PermuteRL active (unless last round), FP only on last round
        // For middle rounds: [None, EP, AddRK, SBOX, PP, XOR_LR, PermuteRL, InvPPR, InvPPDR, FP]
        //   where PermuteRL active, FP inactive
        // For last round: [None, EP, AddRK, SBOX, PP, XOR_LR, None, InvPPR, InvPPDR, FP]
        //   where PermuteRL inactive, FP active

        // Determine maximum step for this round
        let max_step = if is_stop_round {
            after_step
        } else {
            // If not the stop round, run all relevant steps
            if is_last_round {
                DesSteps::FinalPermutation as u8
            } else {
                DesSteps::PermuteRightLeft as u8
            }
        };

        for step_idx in 0..=max_step {
            let step = DesSteps::from(step_idx);

            match step {
                DesSteps::InitialPermutation => {
                    if is_first_round {
                        state = initial_permutation(&state);
                    }
                    // else: None for non-first rounds - skip
                }
                DesSteps::ExpansivePermutation => {
                    saved_left_right = state;
                    let r_half: [u8; 4] = [state[4], state[5], state[6], state[7]];
                    let ep = expansive_permutation(&r_half);
                    state = ep;
                }
                DesSteps::AddRoundKey => {
                    for i in 0..8 {
                        state[i] ^= expanded_key[key_idx(round_number)][i];
                    }
                }
                DesSteps::Sboxes => {
                    state = sboxes(&state);
                }
                DesSteps::PermutationP => {
                    let pp = permutation_p(&state);
                    // Python: out_state[:, 0:4] = permutation_p(out_state)
                    //         out_state[:, 4:8] = zeros
                    state[0] = pp[0];
                    state[1] = pp[1];
                    state[2] = pp[2];
                    state[3] = pp[3];
                    state[4] = 0;
                    state[5] = 0;
                    state[6] = 0;
                    state[7] = 0;
                }
                DesSteps::XorWithSavedLeftRight => {
                    for i in 0..8 {
                        state[i] ^= saved_left_right[i];
                    }
                }
                DesSteps::PermuteRightLeft => {
                    // In the Python round structure:
                    // FIRST_ROUND and ROUND have PermuteRL at step 6
                    // LAST_ROUND and FINAL_ROUND have None at step 6
                    //
                    // However, _prepare_rounds forces PermuteRL when:
                    // - after_step == PermuteRL (step 6)
                    // - after_step == InvPPR (step 7) - forces step 6 to PermuteRL
                    //
                    // So: execute PermuteRL if:
                    // 1. Not the last round (always active in non-last rounds), OR
                    // 2. Is the last round AND is the stop round AND
                    //    after_step is PermuteRL or InvPPR
                    let should_execute = if !is_last_round {
                        true
                    } else if is_stop_round {
                        after_step == DesSteps::PermuteRightLeft as u8
                            || after_step == DesSteps::InvPermutationPRight as u8
                    } else {
                        false
                    };
                    if should_execute {
                        // np.roll(out_state, shift=4, axis=-1) rotates bytes by 4
                        let tmp = state;
                        state[0] = tmp[4];
                        state[1] = tmp[5];
                        state[2] = tmp[6];
                        state[3] = tmp[7];
                        state[4] = tmp[0];
                        state[5] = tmp[1];
                        state[6] = tmp[2];
                        state[7] = tmp[3];
                    }
                }
                DesSteps::InvPermutationPRight => {
                    // Only execute when this is the target stop step
                    if is_stop_round && after_step == DesSteps::InvPermutationPRight as u8 {
                        let r_half: [u8; 4] = [state[4], state[5], state[6], state[7]];
                        let result = inv_permutation_p(&r_half);
                        return result;
                    }
                }
                DesSteps::InvPermutationPDeltaRight => {
                    // Only execute when this is the target stop step
                    if is_stop_round && after_step == DesSteps::InvPermutationPDeltaRight as u8 {
                        let mut delta = [0u8; 4];
                        delta[0] = state[0] ^ state[4];
                        delta[1] = state[1] ^ state[5];
                        delta[2] = state[2] ^ state[6];
                        delta[3] = state[3] ^ state[7];
                        let result = inv_permutation_p(&delta);
                        return result;
                    }
                }
                DesSteps::FinalPermutation => {
                    if is_last_round {
                        state = final_permutation(&state);
                    }
                }
            }
        }

        // If we stopped at this round, return the current state
        if is_stop_round {
            return state;
        }
    }

    state
}

// ============================================================================
// Unit tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// One DES Feistel round: EP(R) -> XOR(key) -> SBOX -> PP -> XOR(L) -> swap L/R
    fn feistel_round(state: &[u8; 8], round_key: &[u8; 8]) -> [u8; 8] {
        let r_half: [u8; 4] = [state[4], state[5], state[6], state[7]];
        let ep = expansive_permutation(&r_half);
        let mut xored = [0u8; 8];
        for i in 0..8 {
            xored[i] = ep[i] ^ round_key[i];
        }
        let sb = sboxes(&xored);
        let pp = permutation_p(&sb);
        let mut out = [0u8; 8];
        out[0] = state[4];
        out[1] = state[5];
        out[2] = state[6];
        out[3] = state[7];
        out[4] = state[0] ^ pp[0];
        out[5] = state[1] ^ pp[1];
        out[6] = state[2] ^ pp[2];
        out[7] = state[3] ^ pp[3];
        out
    }

    /// Full DES encryption (test-only convenience).
    fn des_encrypt(plaintext: &[u8; 8], key: &[u8; 8]) -> [u8; 8] {
        let expanded_key = key_schedule(key);
        des_encrypt_at(plaintext, &expanded_key, 15, DesSteps::FinalPermutation as u8)
    }

    /// Full DES decryption (test-only convenience).
    fn des_decrypt(ciphertext: &[u8; 8], key: &[u8; 8]) -> [u8; 8] {
        let expanded_key = key_schedule(key);
        des_decrypt_at(ciphertext, &expanded_key, 15, DesSteps::FinalPermutation as u8)
    }

    #[test]
    fn test_sbox_values() {
        // Verify a few SBOX entries from each box
        assert_eq!(SBOXES[0][0], 0xE);
        assert_eq!(SBOXES[0][63], 0xD);
        assert_eq!(SBOXES[1][0], 0xF);
        assert_eq!(SBOXES[1][63], 0x9);
        assert_eq!(SBOXES[7][0], 0xD);
        assert_eq!(SBOXES[7][63], 0xB);
        // Verify all SBOX outputs are 4-bit (0-15)
        for s in 0..8 {
            for i in 0..64 {
                assert!(SBOXES[s][i] <= 0xF, "SBOX[{}][{}] = {} > 15", s, i, SBOXES[s][i]);
            }
        }
    }

    #[test]
    fn test_round_key_bits_indexes() {
        // Verify first round key 0
        assert_eq!(ROUND_KEY_BITS_INDEXES[0][0], [9, 50, 33, 59, 48, 16]);
        // Verify last round key 15
        assert_eq!(ROUND_KEY_BITS_INDEXES[15][7], [52, 21, 20, 6, 62, 38]);
        // All values should be < 64
        for r in 0..16 {
            for w in 0..8 {
                for b in 0..6 {
                    assert!(
                        ROUND_KEY_BITS_INDEXES[r][w][b] < 64,
                        "ROUND_KEY_BITS_INDEXES[{}][{}][{}] = {} >= 64",
                        r, w, b, ROUND_KEY_BITS_INDEXES[r][w][b]
                    );
                }
            }
        }
    }

    #[test]
    fn test_initial_permutation_and_final_permutation_inverse() {
        // IP followed by FP should give back the original for a specific structure
        // Actually IP and FP are inverses of each other
        let input: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let ip = initial_permutation(&input);
        let fp = final_permutation(&ip);
        assert_eq!(fp, input, "FP(IP(x)) should equal x");
    }

    #[test]
    fn test_expansive_permutation() {
        // Test that EP produces 8 bytes from 4 bytes, each <= 63 (6-bit)
        let input: [u8; 4] = [0xF0, 0xAA, 0xCC, 0x55];
        let ep = expansive_permutation(&input);
        for i in 0..8 {
            assert!(ep[i] <= 63, "EP output byte {} = {} > 63", i, ep[i]);
        }
    }

    #[test]
    fn test_permutation_p_and_inv() {
        // Test that inv_permutation_p(permutation_p(x)) == x for 4-bit values
        let input: [u8; 8] = [0x05, 0x0A, 0x03, 0x0F, 0x07, 0x01, 0x0C, 0x08];
        let pp = permutation_p(&input);
        let inv_pp = inv_permutation_p(&pp);
        assert_eq!(inv_pp, input, "inv_PP(PP(x)) should equal x");
    }

    #[test]
    fn test_des_encrypt_known_vector() {
        // Known DES test vector:
        // Key: 0x0123456789ABCDEF
        // PT:  0x4E6F772069732074
        // CT:  0x3FA40E8A984D4815
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let pt: [u8; 8] = [0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74];
        let expected_ct: [u8; 8] = [0x3F, 0xA4, 0x0E, 0x8A, 0x98, 0x4D, 0x48, 0x15];

        let ct = des_encrypt(&pt, &key);
        assert_eq!(
            ct, expected_ct,
            "DES encrypt test vector failed: got {:02X?}, expected {:02X?}",
            ct, expected_ct
        );
    }

    #[test]
    fn test_des_decrypt_reverses_encrypt() {
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let pt: [u8; 8] = [0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74];

        let ct = des_encrypt(&pt, &key);
        let decrypted = des_decrypt(&ct, &key);
        assert_eq!(
            decrypted, pt,
            "DES decrypt(encrypt(pt)) should equal pt: got {:02X?}, expected {:02X?}",
            decrypted, pt
        );
    }

    #[test]
    fn test_des_encrypt_at_full_matches_encrypt() {
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let pt: [u8; 8] = [0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74];
        let expanded_key = key_schedule(&key);

        let ct_full = des_encrypt(&pt, &key);
        let ct_at = des_encrypt_at(&pt, &expanded_key, 15, DesSteps::FinalPermutation as u8);
        assert_eq!(ct_full, ct_at, "des_encrypt_at full should match des_encrypt");
    }

    #[test]
    fn test_des_decrypt_at_full_matches_decrypt() {
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let ct: [u8; 8] = [0x3F, 0xA4, 0x0E, 0x8A, 0x98, 0x4D, 0x48, 0x15];
        let expanded_key = key_schedule(&key);

        let pt_full = des_decrypt(&ct, &key);
        let pt_at = des_decrypt_at(&ct, &expanded_key, 15, DesSteps::FinalPermutation as u8);
        assert_eq!(pt_full, pt_at, "des_decrypt_at full should match des_decrypt");
    }

    #[test]
    fn test_des_steps_enum_values() {
        assert_eq!(DesSteps::InitialPermutation as u8, 0);
        assert_eq!(DesSteps::ExpansivePermutation as u8, 1);
        assert_eq!(DesSteps::AddRoundKey as u8, 2);
        assert_eq!(DesSteps::Sboxes as u8, 3);
        assert_eq!(DesSteps::PermutationP as u8, 4);
        assert_eq!(DesSteps::XorWithSavedLeftRight as u8, 5);
        assert_eq!(DesSteps::PermuteRightLeft as u8, 6);
        assert_eq!(DesSteps::InvPermutationPRight as u8, 7);
        assert_eq!(DesSteps::InvPermutationPDeltaRight as u8, 8);
        assert_eq!(DesSteps::FinalPermutation as u8, 9);
    }

    #[test]
    fn test_key_schedule() {
        // Test that key schedule produces 16 round keys, each with 8 bytes <= 63
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let ks = key_schedule(&key);
        assert_eq!(ks.len(), 16);
        for r in 0..16 {
            for w in 0..8 {
                assert!(
                    ks[r][w] <= 63,
                    "Round key [{}][{}] = {} > 63",
                    r, w, ks[r][w]
                );
            }
        }
    }

    #[test]
    fn test_des_encrypt_another_vector() {
        // Another well-known DES test vector:
        // Key: 0x133457799BBCDFF1
        // PT:  0x0123456789ABCDEF
        // CT:  0x85E813540F0AB405
        let key: [u8; 8] = [0x13, 0x34, 0x57, 0x79, 0x9B, 0xBC, 0xDF, 0xF1];
        let pt: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let expected_ct: [u8; 8] = [0x85, 0xE8, 0x13, 0x54, 0x0F, 0x0A, 0xB4, 0x05];

        let ct = des_encrypt(&pt, &key);
        assert_eq!(
            ct, expected_ct,
            "DES encrypt test vector 2 failed: got {:02X?}, expected {:02X?}",
            ct, expected_ct
        );

        let decrypted = des_decrypt(&ct, &key);
        assert_eq!(decrypted, pt, "Decrypt should reverse encrypt for vector 2");
    }

    #[test]
    fn test_feistel_round() {
        // Test that feistel_round matches doing the steps manually
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let pt: [u8; 8] = [0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74];
        let expanded_key = key_schedule(&key);

        // After IP
        let after_ip = initial_permutation(&pt);

        // Do one feistel round
        let after_feistel = feistel_round(&after_ip, &expanded_key[0]);

        // Verify by doing the steps manually
        let r_half: [u8; 4] = [after_ip[4], after_ip[5], after_ip[6], after_ip[7]];
        let ep = expansive_permutation(&r_half);
        let mut xored = [0u8; 8];
        for i in 0..8 {
            xored[i] = ep[i] ^ expanded_key[0][i];
        }
        let sb = sboxes(&xored);
        let pp = permutation_p(&sb);

        let mut expected = [0u8; 8];
        expected[0] = after_ip[4];
        expected[1] = after_ip[5];
        expected[2] = after_ip[6];
        expected[3] = after_ip[7];
        expected[4] = after_ip[0] ^ pp[0];
        expected[5] = after_ip[1] ^ pp[1];
        expected[6] = after_ip[2] ^ pp[2];
        expected[7] = after_ip[3] ^ pp[3];

        assert_eq!(after_feistel, expected);
    }

    #[test]
    fn test_des_encrypt_at_intermediate_steps() {
        // Test stopping at round 0 after AddRK
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let pt: [u8; 8] = [0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74];
        let expanded_key = key_schedule(&key);

        // Stop at round 0, after EP
        let after_ep = des_encrypt_at(&pt, &expanded_key, 0, DesSteps::ExpansivePermutation as u8);
        assert_eq!(after_ep.len(), 8, "EP output should be 8 bytes");

        // Stop at round 0, after SBOX
        let after_sbox = des_encrypt_at(&pt, &expanded_key, 0, DesSteps::Sboxes as u8);
        assert_eq!(after_sbox.len(), 8, "SBOX output should be 8 bytes");

        // Verify all SBOX outputs are 4-bit
        for i in 0..8 {
            assert!(after_sbox[i] <= 0xF, "SBOX output[{}] = {} > 15", i, after_sbox[i]);
        }
    }

    #[test]
    fn test_des_encrypt_zero_key_zero_pt() {
        // All-zero test
        let key: [u8; 8] = [0x00; 8];
        let pt: [u8; 8] = [0x00; 8];

        let ct = des_encrypt(&pt, &key);
        let decrypted = des_decrypt(&ct, &key);
        assert_eq!(decrypted, pt, "Decrypt(Encrypt(0,0)) should equal 0");
    }

    #[test]
    fn test_des_encrypt_at_inv_pp_right() {
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let pt: [u8; 8] = [0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74];
        let expanded_key = key_schedule(&key);

        // Stop at round 0, after InvPermutationPRight (step 7)
        let result = des_encrypt_at(&pt, &expanded_key, 0, DesSteps::InvPermutationPRight as u8);
        assert_eq!(result.len(), 8, "InvPPR output should be 8 bytes");

        // Verify all values are 4-bit (inv_permutation_p returns 8x4-bit words)
        for i in 0..8 {
            assert!(result[i] <= 0xF, "InvPPR output[{}] = {} > 15", i, result[i]);
        }

        // Cross-check: manually compute IP -> EP -> AddRK -> SBOX -> PP -> XOR_LR -> PermuteRL -> InvPP(R)
        let after_permute_rl = des_encrypt_at(&pt, &expanded_key, 0, DesSteps::PermuteRightLeft as u8);
        let r_half: [u8; 4] = [after_permute_rl[4], after_permute_rl[5], after_permute_rl[6], after_permute_rl[7]];
        let expected = inv_permutation_p(&r_half);
        assert_eq!(result, expected, "InvPPR should equal inv_permutation_p(R) after PermuteRL");
    }

    #[test]
    fn test_des_encrypt_at_inv_pp_delta_right() {
        let key: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let pt: [u8; 8] = [0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74];
        let expanded_key = key_schedule(&key);

        // Stop at round 0, after InvPermutationPDeltaRight (step 8)
        let result = des_encrypt_at(&pt, &expanded_key, 0, DesSteps::InvPermutationPDeltaRight as u8);
        assert_eq!(result.len(), 8, "InvPPDR output should be 8 bytes");

        // Verify all values are 4-bit
        for i in 0..8 {
            assert!(result[i] <= 0xF, "InvPPDR output[{}] = {} > 15", i, result[i]);
        }

        // Cross-check: manually compute delta = L XOR R, then inv_permutation_p(delta)
        let after_permute_rl = des_encrypt_at(&pt, &expanded_key, 0, DesSteps::PermuteRightLeft as u8);
        let mut delta = [0u8; 4];
        delta[0] = after_permute_rl[0] ^ after_permute_rl[4];
        delta[1] = after_permute_rl[1] ^ after_permute_rl[5];
        delta[2] = after_permute_rl[2] ^ after_permute_rl[6];
        delta[3] = after_permute_rl[3] ^ after_permute_rl[7];
        let expected = inv_permutation_p(&delta);
        assert_eq!(result, expected, "InvPPDR should equal inv_permutation_p(L XOR R) after PermuteRL");
    }
}
