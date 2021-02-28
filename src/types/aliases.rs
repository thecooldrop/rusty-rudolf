use super::matrix::Matrix;
use crate::types::matrix::States;

pub type Matrix2<A, N, M> = Matrix<A, N, M, ndarray::Ix2, ndarray::OwnedRepr<A>>;
pub type Matrix3<A, N, M> = Matrix<A, N, M, ndarray::Ix3, ndarray::OwnedRepr<A>>;
pub type States2<A, N> = States<A, N, ndarray::Ix2, ndarray::OwnedRepr<A>>;
pub type States3<A, N> = States<A, N, ndarray::Ix3, ndarray::OwnedRepr<A>>;
pub type StatesView2<'a, A, N> = States<A, N, ndarray::Ix2, ndarray::ViewRepr<&'a A>>;
pub type StatesView3<'a ,A, N> = States<A, N, ndarray::Ix3, ndarray::ViewRepr<&'a A>>;
pub type StatesViewMut2<'a, A, N> = States<A, N, ndarray::Ix2, ndarray::ViewRepr<&'a mut A>>;
pub type StatesViewMut3<'a, A, N> = States<A, N, ndarray::Ix3, ndarray::ViewRepr<&'a mut A>>;
pub type Covariances<A, N> = Matrix3<A, N, N>;
pub type MatrixView2<'a, A, N, M> = Matrix<A, N, M, ndarray::Ix2, ndarray::ViewRepr<&'a A>>;
pub type MatrixView3<'a, A, N, M> =  Matrix<A, N, M, ndarray::Ix3, ndarray::ViewRepr<&'a A>>;
pub type MatrixViewMut2<'a, A, N, M> = Matrix<A, N, M, ndarray::Ix2, ndarray::ViewRepr<&'a mut A>>;
pub type MatrixViewMut3<'a, A, N, M> = Matrix<A, N, M, ndarray::Ix3, ndarray::ViewRepr<&'a mut A>>;
pub type CovariancesView<'a, A, N> = MatrixView3<'a, A, N, N>;