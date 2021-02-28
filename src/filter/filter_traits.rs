use cauchy::Scalar;
use ndarray_linalg::Lapack;
use nalgebra::Dim;
use nalgebra::constraint::{DimEq, ShapeConstraint, AreMultipliable};
use crate::types::{Covariances,States2};

/// Filtering algorithm trait
///
/// This trait indicates that implementor is a representation of a filtering algorithm, and
/// that it performs filtering operations on inputs of numeric type `T: Scalar + Lapack`, with
/// dimension of input `S:Dim` and dimension of measurements `M:Dim`.
/// It can be used to represent general filtering algorithms, which are usually split into
/// prediction and update steps.
pub trait Filter<T, S, M> where
    T: Scalar + Lapack,
    S: Dim,
    M: Dim
{
    /// Result of prediction operation executed on states
    type Predict;
    /// Result of update operation executed on states
    type Update;

    /// Prediction operation executed by filtering algorithm.
    ///
    /// The prediction operation usually produces the predicted values for states and their
    /// associated covariances.
    ///
    /// Note that this method has two parameters, whose meanings are as follows:
    /// * states - A two-dimensional array of numbers, where each row represents a state
    /// * covariances - A three-dimensional array of numbers, where each entry along first axis
    /// represents a covariance matrix
    ///
    /// If states are D-dimensional, then the method is applicable to D-dimensional states and
    /// DxD covariance matrices.
    fn predict<N>(&self, states: &States2<T, N>, covariances: &Covariances<T, N>) -> Self::Predict
    where
        N: Dim,
        ShapeConstraint: AreMultipliable<S, S, N, N> + AreMultipliable<N, N, S, S> + DimEq<S, N> + DimEq<N,S>;

    /// Update operation executed by filtering algorithm.
    ///
    /// The update operation usually produces the updates values for states and their
    /// associated covariances.
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
    fn update<N, Q>(&self, states: &States2<T, N>, covariances: &Covariances<T, N>, measurements: &States2<T, Q>) -> Self::Update
    where
        N: Dim,
        Q: Dim,
        ShapeConstraint: DimEq<S,N> + DimEq<N, S> + DimEq<Q, M> + DimEq<M,Q>;
}


pub trait Predictable<T, S> where
    T: Scalar + Lapack,
    S: Dim
{
    type Output;
    fn predict(&self, states: &States2<T, S>, covariances: &Covariances<T,S>);
}