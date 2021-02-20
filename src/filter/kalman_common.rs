use std::ops::{SubAssign, Neg};

use cauchy::Scalar;
use ndarray::{Array2, Array3, ArrayBase, Axis, CowArray, Data, Ix2, Ix3, azip};
use ndarray_linalg::InverseC;
use ndarray_linalg::Lapack;
use ndarray::linalg::general_mat_mul;
use ndarray_linalg::error::LinalgError;


pub(in crate) fn broad_dot_ix3_ix2<A, S, S2>(lhs: &ArrayBase<S, Ix3>, rhs: &ArrayBase<S2, Ix2>) -> Array3<A>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (lhs_len, lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_rows, rhs_cols) = rhs.dim();
    if lhs_cols != rhs_rows {
        // TODO: Add code
    }
    let mut output = Array3::zeros([lhs_len, lhs_rows, rhs_cols]);
    azip!((elem in lhs.outer_iter(), mut out in output.outer_iter_mut()) general_mat_mul(A::one(), &elem, &rhs, A::zero(), &mut out));
    output
}

pub(in crate) fn broad_dot_ix3_ix3<A, S, S2>(lhs: &ArrayBase<S, Ix3>, rhs: &ArrayBase<S2, Ix3>) -> Array3<A>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (lhs_len, lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_len, rhs_rows, rhs_cols) = rhs.dim();
    if lhs_cols != rhs_rows {
        // TODO: Write code to throw error
    }

    if lhs_len != rhs_len {
        // TODO: Write code to throw error
    }
    let mut output = Array3::zeros([lhs_len, lhs_rows, rhs_cols]);
    azip!((l in lhs.outer_iter(), r in rhs.outer_iter(), mut out in output.outer_iter_mut())
            general_mat_mul(A::one(), &l, &r, A::zero(), &mut out));
    output
}

pub(in crate) fn quadratic_form_ix3_ix3_ix3<A, S1, S2>(inner: &ArrayBase<S1, Ix3>, outer: &ArrayBase<S2, Ix3>) -> Array3<A>
where
    A: Scalar+Lapack,
    S1: Data<Elem=A>,
    S2: Data<Elem=A>
{
    let (inner_len, inner_rows, inner_cols) = inner.dim();
    let (outer_len, outer_rows, outer_cols) = outer.dim();

    if outer_cols != inner_rows || outer_rows != inner_cols {
        //TODO: Write code to throw exception if someting does not match
    }

    if inner_len != outer_len {
        //TODO: Write code to throw exception if something does not match
    }

    let intermediate = broad_dot_ix3_ix3(outer, inner);
    let mut outer_view = outer.view();
    outer_view.swap_axes(1, 2);
    broad_dot_ix3_ix3(&intermediate, &outer_view)
}

pub(in crate) fn update_states<A: Scalar+Lapack, S1: Data<Elem=A>>(states: &ArrayBase<S1, Ix2>, kalman_gains: &Array3<A>, innovations_negated: &Array3<A>) -> Array3<A> {

    let (state_rows, state_cols) = states.dim();
    let (innovations_len, innovation_rows, innovation_cols) = innovations_negated.dim();
    let (kalman_gain_len, kalman_gain_rows, kalman_gains_cols) = kalman_gains.dim();

    if kalman_gains_cols != innovation_cols {
        //TODO: Write code to throw error
    }

    if kalman_gain_rows != state_cols {
        //TODO: Write code to throw error
    }

    if state_rows != kalman_gain_len || state_rows != innovations_len {
        //TODO: Write code to throw error
    }

    let updated_states_dim = [state_rows, innovation_rows, state_cols];
    let expanded_states_view = states.view().insert_axis(Axis(1));
    let broadcasted_view = expanded_states_view.broadcast(updated_states_dim).unwrap();
    let mut updated_states = CowArray::from(broadcasted_view);
    for ((mut state_updated, kalman_gain), innovation) in updated_states
        .outer_iter_mut()
        .zip(kalman_gains.outer_iter())
        .zip(innovations_negated.outer_iter())
    {
        let result = kalman_gain.dot(&innovation.t()).t().into_owned();
        state_updated.sub_assign(&result);
    }
    updated_states.into_owned()
}

pub(in crate) fn update_covariance<A: Scalar+Lapack, S1: Data<Elem=A>>(covariances: &ArrayBase<S1, Ix3>, kalman_gains: &Array3<A>, l_matrices: &Array3<A>) -> Array3<A>{
    let (kalman_gain_len, kalman_gain_rows, kalman_gain_cols) = kalman_gains.dim();
    let (covariances_len, covariances_rows, covariances_cols) = covariances.dim();
    let (l_matrix_len, l_matrix_rows, l_matrix_cols) = l_matrices.dim();

    if kalman_gain_len != covariances_len || covariances_len != l_matrix_len {
        // TODO: Add error handling
    }

    if kalman_gain_cols != l_matrix_cols {
        // TODO: Add error handling
    }

    if (kalman_gain_rows, l_matrix_rows) != (covariances_rows, covariances_cols) {
        // TODO: Add error handling
    }

    let mut updated_covariances = covariances.to_owned();
    for ((mut elem, kalman_gain), l) in updated_covariances
        .outer_iter_mut()
        .zip(kalman_gains.outer_iter())
        .zip(l_matrices.outer_iter())
    {
        general_mat_mul(A::one().neg(), &kalman_gain, &l.t(), A::one(), &mut elem);
    }
    updated_covariances
}

#[inline(always)]
pub(in crate) fn quadratic_form_ix2_ix3_ix2_add_ix2<A, S1, S2, S3, S4>(
    lhs: &ArrayBase<S1, Ix2>,
    mid: &ArrayBase<S2, Ix3>,
    rhs: &ArrayBase<S3, Ix2>,
    add: &ArrayBase<S4, Ix2>,
) -> Array3<A>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    S3: Data<Elem = A>,
    S4: Data<Elem = A>,
{
    let (lhs_rows, lhs_cols) = lhs.dim();
    let (_, mid_rows, mid_cols) = mid.dim();
    let (rhs_rows, rhs_cols) = rhs.dim();
    let (add_rows, add_cols) = add.dim();

    if lhs_cols != mid_rows || mid_cols != rhs_rows {
        // TODO Add error handling
    }

    if (lhs_rows, rhs_cols) != (add_rows, add_cols) {
        // TODO Add error handling
    }

    let mid_dim = mid.dim();
    let add_dim = add.dim();
    let mut output = add.broadcast([mid_dim.0, add_dim.0, add_dim.1]).unwrap().to_owned();
    for (mut out, elem) in output.outer_iter_mut().zip(mid.outer_iter()) {
        let left_intermediate = lhs.dot(&elem);
        general_mat_mul(A::one(), &left_intermediate, rhs, A::one(), &mut out);
    }
    output
}

pub(in crate) fn innovation_covariances_ix2<A: Scalar + Lapack>(
    observation_matrix: &Array2<A>,
    observation_covariance: &Array2<A>,
    l_matrices: &Array3<A>,
) -> Array3<A> {
    let (observation_matrix_rows, observation_matrix_cols) = observation_matrix.dim();
    let (observation_covariance_rows, observation_covariance_cols) = observation_covariance.dim();
    let (l_matrices_len, l_matrix_rows, l_matrix_cols) = l_matrices.dim();

    if observation_matrix_cols != l_matrix_rows {
        // TODO : Add error handling
    }

    if (observation_matrix_rows, l_matrix_cols) != (observation_covariance_rows, observation_covariance_cols) {
        // TODO : Add error handling
    }

    let output_dim = [l_matrices_len, observation_matrix_rows, observation_matrix_cols];
    let mut output_matrix = observation_covariance.broadcast(output_dim).unwrap().to_owned();
    azip!((mut output in output_matrix.axis_iter_mut(Axis(0)), l_matrix in l_matrices.axis_iter(Axis(0)))
     general_mat_mul(A::one(), observation_matrix, &l_matrix, A::one(), &mut output));
    output_matrix
}

pub(in crate) fn pairwise_difference<A, S1, S2>(
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Array3<A>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_rows, rhs_cols) = rhs.dim();

    if lhs_cols != rhs_cols {
        // TODO : Add error handling
    }

    let out_dim = [lhs_rows, rhs_rows, rhs_cols];
    let expanded_lhs = lhs.to_owned().insert_axis(Axis(1));
    let expanded_lhs_view = expanded_lhs.view();
    let broadcast = expanded_lhs_view.broadcast(out_dim).unwrap();
    let mut output = broadcast.to_owned();
    azip!((mut out in output.outer_iter_mut()) out.sub_assign(&rhs));
    output
}

pub(in crate) fn invc_all_ix3<A, S1>(matrices: &ArrayBase<S1, Ix3>) -> Result<Array3<A>, LinalgError>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
{
    let mut inv_matrices = Array3::zeros(matrices.raw_dim());
    for (mut destination, source) in inv_matrices.outer_iter_mut().zip(matrices.outer_iter()) {
        destination.assign(&source.invc()?);
    }
    Ok(inv_matrices)
}

pub(in crate) enum MatrixOperationError {
    MatrixMultiplicationError {lhs_shape: (usize, usize), rhs_shape: (usize, usize)},
    VectorArithmeticError {lhs_dim: usize, rhs_dim: usize},
    MatrixArithmeticError {lhs_dim: (usize, usize), rhs_dim: (usize, usize)}
}

pub(in crate) enum InputNumberMismatchError {
    UnequalNumberOfInputsError
}

pub(in crate) enum InternalOperationError {
    InvalidArgumentsError(InputNumberMismatchError),
    MatrixOperationError(MatrixOperationError),
    LapackError(LinalgError)
}