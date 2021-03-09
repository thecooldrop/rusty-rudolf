//! Traits representing the filtering algorithms

use cauchy::Scalar;
use ndarray::{ArrayBase, Data, Ix2, Ix3};
use ndarray_linalg::Lapack;

/// Filtering algorithm trait
///
/// This trait indicates that implementor is a representation of a filtering algorithm, and
/// that it performs filtering operations on inputs of type `T: Scalar + Lapack`.
/// It can be used to represent general filtering algorithms, which are usually split into
/// prediction and update steps.
pub trait Filter<T: Scalar + Lapack> {
    /// Result of prediction operation executed on states
    type Prediction;
    /// Result of update operation executed on states
    type Update;

    /// Prediction operation executed by filtering algorithm.
    ///
    /// The prediction operation result produces the predicted values for states and their
    /// associated covariances. Usually the result of prediction is a 2-tuple of n-arrays, whose
    /// entries usually represent predicted states and covariances respectfully.
    ///
    /// Note that this method has two parameters, whose meanings are as follows:
    /// * states - A two-dimensional array of numbers, where each row represents a state
    /// * covariances - A three-dimensional array of numbers, where each entry along first axis
    /// represents a covariance matrix
    ///
    /// The generic parameters on this method indicate that it is applicable to any combination of
    /// ndarray arrays, which own their data.
    fn predict<A: Data<Elem = T>, B: Data<Elem = T>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
    ) -> Self::Prediction;

    /// Update operation executed by filtering algorithm.
    ///
    /// The update operation result produces the updates values for states and their
    /// associated covariances. Usually the result of update is a 2-tuple of nd-arrays, whose
    /// entries usually represent updates states and covariances respectfully.
    ///
    /// This method has three parameters:
    /// * states - A two-dimensional array of numbers, where each row represents a state
    /// * covariances - A three-dimensional array of numbers, where each entry along first axis
    /// represents a covariance matrix
    /// * mesurements - A two-dimensional array of numbers, where each row represents a measurement
    /// of a state
    ///
    /// It is expected that number of entries along first axis of states and covariances is equal,
    /// roughly speaking we expect that there is same number of state and covariance matrices given.
    /// The i-th state row has covariance matrix given by i-th entry of covariances matrix.
    ///
    /// This method is expected to update each (state,covariance) pair with all of the measurements
    fn update<A: Data<Elem = T>, B: Data<Elem = T>, C: Data<Elem = T>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> Self::Update;
}
