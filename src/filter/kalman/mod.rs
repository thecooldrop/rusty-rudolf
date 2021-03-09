//! Contains modules representing algorithms based on Kalman filtering methodology. Currently
//! the algorithms are classified into the following modules:
//! - Linear algorithms ; based on regular kalman filter
//! - nonlinear algorithms; based on extended Kalman filter ( linear approximations ),
//!   unscented ( cubature methods ) and others.

pub mod linear;
pub mod nonlinear;
