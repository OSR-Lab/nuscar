mod dtw;
mod filter;
mod highorder;
mod moving;
mod pattern;

pub use moving::moving_mean_r;
pub use moving::moving_var_r;

pub use filter::band_pass_r;
pub use filter::band_stop_r;
pub use filter::high_pass_r;
pub use filter::low_pass_r;

pub use highorder::combine_abs_diff_r;
pub use highorder::combine_center_product_r;
pub use highorder::combine_diff_r;
pub use highorder::combine_product_r;

pub use pattern::pattern_corr_r;
pub use pattern::pattern_dist_r;
pub use dtw::elastic_align_batch_r;
pub use dtw::fast_dtw_r;


pub use moving::moving_mean_r32;
pub use moving::moving_var_r32;
pub use highorder::combine_abs_diff_r32;
pub use highorder::combine_center_product_r32;
pub use highorder::combine_diff_r32;
pub use highorder::combine_product_r32;
pub use pattern::pattern_corr_r32;
pub use pattern::pattern_dist_r32;
