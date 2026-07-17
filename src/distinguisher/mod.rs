mod cpa;
mod partition;
mod template;
mod ttest;

pub use cpa::cpa_final_r;
pub use cpa::cpa_update_r;
pub use partition::partition_update_r;
pub use template::template_update_train_r;
pub use ttest::ttest_final_r;
pub use ttest::ttest_update_r;

pub use cpa::cpa_final_r32;
pub use cpa::cpa_update_r32;
pub use partition::partition_update_r32;
pub use template::template_update_train_r32;
pub use ttest::ttest_final_r32;
pub use ttest::ttest_update_r32;