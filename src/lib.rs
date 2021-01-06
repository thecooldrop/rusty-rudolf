
#![crate_name = "rusty_rudolf"]
//! The `rusty-rudolf` crate contains implementations of filtering algorithms used for state
//! estimation and for solving general filtering problems. Goal of the crate is provide
//! a production-ready, optimized and generic implementations for these algorithms, so that
//! these may be further reused for more general problems, which depend on such algorithms.
//! Further, besides filtering algorithms, their smoothing counterparts shall be included as
//! well.
//!
//! ## Types of algorithms
//! Aim of the crate is to be collection of all algorithms, which may be considered related to
//! filtering problems. This includes, but is not any way limited to Kalman filter, Extended Kalman
//! filter (EKF), Unscented Kalman filter (UKF), cubature Kalman filters, particle filters
//! (including many varieties of it), as well as their smoothing-algorithm counterparts such
//! as Rauch-Tung-Striebel Smoother (RTS-Smoother) and many others.
pub mod kalman;
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
