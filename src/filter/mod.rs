//! Filtering algorithms are contained here separated into different modules by types. Namely
//! major separations include Kalman and particle filter, while in future modules for RFS
//! filters and filters based on other methodologies are going to be provided.
pub mod traits;
pub mod kalman;
mod kalman_common;

pub use traits::*;

