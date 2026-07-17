mod distinguisher;
use pyo3::prelude::*;
mod utils;
use distinguisher::*;
use utils::*;
mod ciphers;
use ciphers::*;
mod signal_process;
use signal_process::*;

/// A Python module implemented in Rust.
#[pymodule]
fn nuscar_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ThreadPool>()?;
    m.add_function(wrap_pyfunction!(cpa_update_r, m)?)?;
    m.add_function(wrap_pyfunction!(cpa_final_r, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_update_r, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_final_r, m)?)?;
    m.add_function(wrap_pyfunction!(template_update_train_r, m)?)?;
    m.add_function(wrap_pyfunction!(partition_update_r, m)?)?;

    m.add_function(wrap_pyfunction!(cpa_update_r32, m)?)?;
    m.add_function(wrap_pyfunction!(cpa_final_r32, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_update_r32, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_final_r32, m)?)?;
    m.add_function(wrap_pyfunction!(template_update_train_r32, m)?)?;
    m.add_function(wrap_pyfunction!(partition_update_r32, m)?)?;

    // Begin XOR
    m.add_function(wrap_pyfunction!(xor_attack_bit_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(xor_attack_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(xor_attack_value_with_guess_r, m)?)?;
    // End XOR
    // Begin AES
    m.add_function(wrap_pyfunction!(aes_attack_first_sbox_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_attack_first_sbox_bit_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(
        aes_attack_first_sbox_value_with_guess_r,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aes_attack_last_round_xor_hw_with_guess_r,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aes_attack_last_round_xor_bit_with_guess_r,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(aes_attack_last_sbox_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_attack_last_sbox_bit_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_encrypt_step_fix_key_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_encrypt_step_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_decrypt_step_fix_key_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_decrypt_step_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_sbox_sr_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_mixcolumns_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_inv_mixcolumns_r, m)?)?;
    m.add_function(wrap_pyfunction!(aes_decrypt_step_r, m)?)?;
    // End AES
    
    // Begin DES
    m.add_function(wrap_pyfunction!(des_attack_first_addRk_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_first_addRk_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_first_sbox_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_first_sbox_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_first_round_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_first_round_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_delta_first_round_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_delta_first_round_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_last_addRk_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_last_addRk_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_last_sbox_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_last_sbox_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_last_round_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_last_round_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_delta_last_round_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_attack_delta_last_round_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_encrypt_step_fix_key_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_encrypt_step_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_decrypt_step_fix_key_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_decrypt_step_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_initial_permutation_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_expansive_permutation_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_sboxes_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_permutation_p_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_inv_permutation_p_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_final_permutation_r, m)?)?;
    m.add_function(wrap_pyfunction!(des_key_schedule_r, m)?)?;
    // End DES

    // Begin SM4
    m.add_function(wrap_pyfunction!(
        sm4_attack_first_sbox_value_with_guess_r,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sm4_attack_first_sbox_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm4_attack_first_sbox_bit_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm4_attack_sbox_value_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm4_attack_sbox_hw_with_guess_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm4_attack_sbox_bit_with_guess_r, m)?)?;

    m.add_function(wrap_pyfunction!(sm4_encrypt_fix_key_step_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm4_encrypt_step_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm4_decrypt_fix_key_step_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm4_decrypt_step_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm4_opl_r, m)?)?;
    // End SM4

    // Begin SM3
    m.add_function(wrap_pyfunction!(sm3_hash_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_hash_batch_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_hmac_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_expand_message_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_compute_hi_ho_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_compute_w_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_compute_w_prime_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_attack_delta1_0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_attack_delta2_0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_attack_a0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_attack_b0_rotl9_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_attack_e0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_attack_f0_rotl19_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_attack_c0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sm3_attack_g0_hw_r, m)?)?;
    // End SM3

    // Begin SHA-256
    m.add_function(wrap_pyfunction!(sha256_hash_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_hash_batch_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_expand_message_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_hmac_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_compute_hi_ho_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_compute_w_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_attack_delta0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_attack_t20_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_attack_a0_or_b0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_attack_d0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_attack_e0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_attack_f0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_attack_g0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_attack_c0_hw_r, m)?)?;
    // End SHA-256

    // Begin SHA-1
    m.add_function(wrap_pyfunction!(sha1_hash_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_hash_batch_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_expand_message_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_hmac_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_compute_hi_ho_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_compute_w_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_attack_delta0_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_attack_r1_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_attack_a0_rot30_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_attack_b0_rot30_hw_r, m)?)?;
    m.add_function(wrap_pyfunction!(sha1_attack_c0_hw_r, m)?)?;
    // End SHA-1

    // Begin Signal
    m.add_function(wrap_pyfunction!(moving_mean_r, m)?)?;
    m.add_function(wrap_pyfunction!(moving_var_r, m)?)?;
    m.add_function(wrap_pyfunction!(low_pass_r, m)?)?;
    m.add_function(wrap_pyfunction!(high_pass_r, m)?)?;
    m.add_function(wrap_pyfunction!(band_pass_r, m)?)?;
    m.add_function(wrap_pyfunction!(band_stop_r, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_corr_r, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_dist_r, m)?)?;
    m.add_function(wrap_pyfunction!(combine_product_r, m)?)?;
    m.add_function(wrap_pyfunction!(combine_abs_diff_r, m)?)?;
    m.add_function(wrap_pyfunction!(combine_center_product_r, m)?)?;
    m.add_function(wrap_pyfunction!(combine_diff_r, m)?)?;

    m.add_function(wrap_pyfunction!(moving_mean_r32, m)?)?;
    m.add_function(wrap_pyfunction!(moving_var_r32, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_corr_r32, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_dist_r32, m)?)?;
    m.add_function(wrap_pyfunction!(combine_product_r32, m)?)?;
    m.add_function(wrap_pyfunction!(combine_abs_diff_r32, m)?)?;
    m.add_function(wrap_pyfunction!(combine_center_product_r32, m)?)?;
    m.add_function(wrap_pyfunction!(combine_diff_r32, m)?)?;
    m.add_function(wrap_pyfunction!(elastic_align_batch_r, m)?)?;
    m.add_function(wrap_pyfunction!(fast_dtw_r, m)?)?;
    // End Signal

    m.add_function(wrap_pyfunction!(leak_hamming_weight_row_r, m)?)?;
    Ok(())
}
