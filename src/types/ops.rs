use super::Transpose;
use ndarray_linalg::{Lapack, InverseC};
use cauchy::Scalar;
use nalgebra::{Dim, DimAdd, constraint::{ShapeConstraint, AreMultipliable, DimEq, SameDimension, SameNumberOfRows, SameNumberOfColumns}};
use ndarray::{Axis, Data, ViewRepr, Ix2, Ix3, Dimension, Array3, azip, Array2, DataMut, linalg::{Dot, general_mat_mul}};
use crate::types::{MatrixView3, MatrixView2, IntoTranspose, Matrix2, TransposeView, Matrix3, QuadraticForm, DotInplace, States2, States3, matrix::{States, Matrix}};
use std::ops::{AddAssign, SubAssign, Sub};

impl<'a, A, N, M, R> TransposeView<'a> for Matrix<A, N, M, ndarray::Ix2, R> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R: Data<Elem=A>
{
    type Output = MatrixView2<'a, A, M, N>;

    fn t_view(&'a self) -> Self::Output {
        Matrix {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self.data.t()
        }
    }
}

impl<'a, A, N, M, R> TransposeView<'a> for Matrix<A, N, M, ndarray::Ix3, R> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R: Data<Elem=A>
{
    type Output = MatrixView3<'a, A, M, N>;

    fn t_view(&'a self) -> Self::Output {
        let mut self_view = self.data.view();
        self_view.swap_axes(1,2);
        MatrixView3 {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self_view
        }
    }
}


impl<A, N, M> IntoTranspose for Matrix2<A, N, M> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim
{
    type Output = Matrix2<A, M, N>;

    fn into_t(self) -> Self::Output {
        let mut self_data = self.data;
        self_data.swap_axes(0, 1);
        Matrix2 {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self_data
        }
    }
}

impl<A, N, M> IntoTranspose for Matrix3<A, N, M> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim
{
    type Output = Matrix3<A, M, N>;

    fn into_t(self) -> Self::Output {
        let mut self_data = self.data;
        self_data.swap_axes(1,2);
        Matrix3 {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self_data
        }
    }
}

impl<A, N, M, R> Transpose for Matrix<A, N, M, ndarray::Ix2, R> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R: Data<Elem=A>
{
    type Output = Matrix2<A, M, N>;

    fn t(&self) -> Self::Output {
        let mut self_data = self.data.to_owned();
        self_data.swap_axes(0,1);
        Matrix2 {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self_data
        }
    }
}

impl<A, N, M, R> Transpose for Matrix<A, N, M, ndarray::Ix3, R> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R: Data<Elem=A>
{
    type Output = Matrix3<A, M, N>;

    fn t(&self) -> Self::Output {
        let mut self_data = self.data.to_owned();
        self_data.swap_axes(1,2);
        Matrix3 {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self_data
        }
    }
}

impl<A, N, M, P, Q, R1, R2> Dot<Matrix<A, N, M, Ix2, R1>> for Matrix<A, P, Q, Ix2, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    Q: Dim,
    R1: Data<Elem=A>,
    R2: Data<Elem=A>,
    ShapeConstraint: AreMultipliable<P,Q,N,M> + DimEq<Q,N> + DimEq<N,Q>
{
    type Output = Matrix2<A, P, M>;

    fn dot(&self, rhs: &Matrix<A, N, M, Ix2, R1>) -> Self::Output {
        Matrix2 {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self.data.dot(&rhs.data)
        }
    }
}

impl<A, N, M, P, Q, R1, R2> Dot<Matrix<A, N, M, Ix3, R1>> for Matrix<A, P, Q, Ix2, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    Q: Dim,
    R1: Data<Elem=A>,
    R2: Data<Elem=A>,
    ShapeConstraint: AreMultipliable<P,Q,N,M> + DimEq<Q,N> + DimEq<N,Q>
{
    type Output = Matrix3<A, P, M>;

    fn dot(&self, rhs: &Matrix<A, N, M, Ix3, R1>) -> Self::Output {
        let (rhs_entries, _, rhs_cols) = rhs.data.dim();
        let (_, lhs_rows) = self.data.dim();
        let mut result = Array3::zeros([rhs_entries, lhs_rows, rhs_cols]);
        azip!((mut out in result.outer_iter_mut(), rhs_elem in rhs.data.outer_iter()) general_mat_mul(A::one(), &self.data, &rhs_elem, A::zero(), &mut out));
        Matrix3 {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: result
        }
    }
}


impl<A, N, M, P, Q, R1, R2> Dot<Matrix<A, N, M, Ix2, R1>> for Matrix<A, P, Q, Ix3, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    Q: Dim,
    R1: Data<Elem=A>,
    R2: Data<Elem=A>,
    ShapeConstraint: AreMultipliable<P, Q, N, M> + DimEq<Q,N>
{
    type Output = Matrix3<A, P, M>;

    fn dot(&self, rhs: &Matrix<A, N, M, Ix2, R1>) -> Self::Output {
        let (lhs_entries, lhs_rows, _) = self.data.dim();
        let (_, rhs_cols) = rhs.data.dim();
        let mut result = Array3::zeros([lhs_entries, lhs_rows, rhs_cols]);
        azip!((mut out in result.outer_iter_mut(), lhs_elem in self.data.outer_iter()) general_mat_mul(A::one(), &lhs_elem, &rhs.data, A::zero(), &mut out));
        Matrix3 {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: result
        }
    }
}

// TODO: Make Dot impls return Results, because the number of entries can mismatch along first axis
impl<A, N, M, P, Q, R1, R2> Dot<Matrix<A, N, M, Ix3, R1>> for Matrix<A, P, Q, Ix3, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    Q: Dim,
    R1: Data<Elem=A>,
    R2: Data<Elem=A>,
    ShapeConstraint: AreMultipliable<P, Q, N, M> + DimEq<Q, N>
{
    type Output = Matrix3<A, P, M>;

    fn dot(&self, rhs: &Matrix<A, N, M, Ix3, R1>) -> Self::Output {
        let (lhs_entries, lhs_rows, _) = self.data.dim();
        let (_, _, rhs_cols) = rhs.data.dim();
        let mut result = Array3::zeros([lhs_entries, lhs_rows, rhs_cols]);

        azip!((mut out in result.outer_iter_mut(),
               lhs_elem in self.data.outer_iter(),
               rhs_elem in rhs.data.outer_iter())
        general_mat_mul(A::one(), &lhs_elem, &rhs_elem, A::zero(), &mut out));

        Matrix {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: result
        }
    }
}

impl<'a, LHS, MID, RHS> QuadraticForm<'a, LHS, RHS> for MID where
    LHS: Dot<MID>,
    RHS: TransposeView<'a>,
    <LHS as Dot<MID>>::Output : Dot<<RHS as TransposeView<'a>>::Output>
{
    type Output = <<LHS as Dot<MID>>::Output as Dot<<RHS as TransposeView<'a>>::Output>>::Output;

    fn quadratic_form(&self, lhs: &LHS, rhs: &'a RHS) -> Self::Output {
        lhs.dot(self).dot(&rhs.t_view())
    }
}

impl<A, N, M, R> AddAssign for Matrix<A, N, M , Ix2, R> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R: DataMut<Elem=A>
{
    fn add_assign(&mut self, rhs: Self) {
        self.data.add_assign(&rhs.data);
    }
}

impl<'a, A, N, M, P, Q, R1, R2> AddAssign<&'a Matrix<A, N, M, Ix2, R1>> for Matrix<A, P, Q, Ix3, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    Q: Dim,
    R1: Data<Elem=A>,
    R2: DataMut<Elem=A>,
    ShapeConstraint: SameNumberOfRows<N, P> + SameNumberOfColumns<M, Q>
{
    fn add_assign(&mut self, rhs: &'a Matrix<A, N, M, Ix2, R1>) {
        azip!((mut lhs_elem in self.data.outer_iter_mut()) lhs_elem.add_assign(&rhs.data));
    }
}


impl<'a, A, N, M, P, Q, R1, R2> AddAssign<&'a Matrix<A, N, M, Ix3, R1>> for Matrix<A, P, Q, Ix3, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    Q: Dim,
    R1: Data<Elem=A>,
    R2: DataMut<Elem=A>,
    ShapeConstraint: SameNumberOfRows<N, P> + SameNumberOfColumns<M,Q>
{
    fn add_assign(&mut self, rhs: &'a Matrix<A, N, M, Ix3, R1>) {
        azip!((mut lhs_elem in self.data.outer_iter_mut(), rhs_elem in rhs.data.outer_iter()) lhs_elem.add_assign(&rhs_elem));
    }
}


impl<A,N,M,P,R1,R2> Dot<States<A, N, Ix2, R1>> for Matrix<A, M, P, Ix2, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    R1: Data<Elem=A>,
    R2: Data<Elem=A>,
    ShapeConstraint: DimEq<P, N>
{
    type Output = States2<A, M>;

    fn dot(&self, rhs: &States<A, N, Ix2, R1>) -> Self::Output {
        let mut dot_result = self.data.dot(&rhs.data.t());
        dot_result.swap_axes(0,1);
        States {
            _dim_phantom: Default::default(),
            data: dot_result
        }
    }
}

impl<A,N,M,P,R1,R2> Dot<States<A,N,Ix3,R1>> for Matrix<A,M,P,Ix3,R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    R1: Data<Elem=A>,
    R2: Data<Elem=A>,
    ShapeConstraint: DimEq<P,N>
{
    type Output = States3<A,M>;

    fn dot(&self, rhs: &States<A, N, Ix3, R1>) -> Self::Output {
        let (lhs_entries, lhs_rows, lhs_cols) = self.data.dim();
        let (rhs_entries, rhs_rows, rhs_cols) = rhs.data.dim();
        let mut result = Array3::zeros([lhs_entries, lhs_rows, rhs_cols]);
        azip!((mut out in result.outer_iter_mut(), lhs_elem in self.data.outer_iter(), rhs_elem in rhs.data.outer_iter())
         general_mat_mul(A::one(), &lhs_elem, &rhs_elem, A::zero(), &mut out));
        States {
            _dim_phantom: Default::default(),
            data: result
        }
    }
}

impl<A,N,R> States<A, N, Ix2, R> where
    A: Scalar + Lapack,
    N: Dim,
    R: Data<Elem=A>
{
    pub fn pairwise_difference<M,R2>(&self, rhs: &States<A, M, Ix2, R2>) -> States3<A, N>
        where
            M: Dim,
            ShapeConstraint: DimEq<N,M>,
            R2: Data<Elem=A>
    {
        let (lhs_rows, lhs_cols) = self.data.dim();
        let (rhs_rows, _) = self.data.dim();
        let out_dim = [lhs_rows, rhs_rows, lhs_cols];
        let mut out_matrix = self.data
            .view()
            .insert_axis(Axis(1))
            .broadcast(out_dim)
            .unwrap()
            .to_owned();
        azip!((mut out in out_matrix.outer_iter_mut()) out.sub_assign(&rhs.data));
        States {
            _dim_phantom: Default::default(),
            data: out_matrix
        }
    }
}

impl<A,N,M,R> InverseC for Matrix<A,N,M,Ix2,R> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R: Data<Elem=A>,
    ShapeConstraint: DimEq<N,M>
{
    type Output = Matrix2<A,N,N>;

    fn invc(&self) -> ndarray_linalg::error::Result<Self::Output> {
        Ok(Matrix {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self.data.invc()?
        })
    }
}

impl<A,N,M,R> InverseC for Matrix<A,N,M,Ix3,R> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R: Data<Elem=A>,
    ShapeConstraint: DimEq<N,M>
{
    type Output = Matrix3<A,N,N>;

    fn invc(&self) -> ndarray_linalg::error::Result<Self::Output> {
        let mut out_matrix = Array3::zeros(self.data.raw_dim());
        for (mut out, elem) in out_matrix.outer_iter_mut().zip(self.data.outer_iter()) {
            out.assign(&elem.invc()?)
        }
        Ok(Matrix {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: out_matrix
        })
    }
}

impl<'a, A, N, M, R1, R2> SubAssign<&'a States<A, N, Ix2, R1>> for States<A, M, Ix2, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R1: Data<Elem=A>,
    R2: DataMut<Elem=A>,
    ShapeConstraint: DimEq<M, N>
{
    fn sub_assign(&mut self, rhs: &'a States<A, N, Ix2, R1>) {
        self.data.sub_assign(&rhs.data);
    }
}


impl<'a, A, N, M, R1, R2> Sub<&'a States<A, N, Ix3, R1>> for &States<A, M, Ix2, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    R1: Data<Elem=A>,
    R2: Data<Elem=A>,
    ShapeConstraint: DimEq<M,N>
{
    type Output = States3<A, N>;

    fn sub(self, rhs: &'a States<A, N, Ix3, R1>) -> Self::Output {
        let mut out = self.data.view().insert_axis(Axis(0)).to_owned();
        out.sub_assign(&rhs.data);
        States {
            _dim_phantom: Default::default(),
            data: out
        }
    }
}

impl<'a, A, N, M, P, Q, R1, R2> Sub<&'a Matrix<A, N, M, Ix3, R1>> for Matrix<A, P, Q, Ix3, R2> where
    A: Scalar + Lapack,
    N: Dim,
    M: Dim,
    P: Dim,
    Q: Dim,
    R1: Data<Elem=A>,
    R2: Data<Elem=A>,
    ShapeConstraint: DimEq<P, N> + DimEq<Q, M>
{
    type Output = Matrix3<A, P, Q>;

    fn sub(self, rhs: &'a Matrix<A, N, M, Ix3, R1>) -> Self::Output {
        Matrix {
            row_dim_phantom: Default::default(),
            col_dim_phantom: Default::default(),
            data: self.data.to_owned().sub(&rhs.data)
        }
    }
}