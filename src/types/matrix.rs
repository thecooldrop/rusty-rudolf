use cauchy::Scalar;
use ndarray_linalg::Lapack;
use nalgebra::Dim;
use ndarray::{Dimension, ArrayBase, OwnedRepr, Data};
use std::marker::PhantomData;


pub struct Matrix<A, N, M, D, R> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    D: Dimension,
    R: Data<Elem = A>
{
    pub(in super) row_dim_phantom: PhantomData<N>,
    pub(in super) col_dim_phantom: PhantomData<M>,
    pub(in super) data: ArrayBase<R, D>
}

pub struct States<A, N, D, R> where
    A: Scalar + Lapack,
    N: Dim,
    D: Dimension,
    R: Data<Elem=A>
{
    pub(in super) _dim_phantom: PhantomData<N>,
    pub(in super) data: ArrayBase<R, D>
}
