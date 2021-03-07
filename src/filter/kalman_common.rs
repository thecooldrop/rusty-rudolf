use cauchy::Scalar;
use ndarray::{
    azip, linalg::general_mat_mul, Array2, Array3, ArrayBase, Axis, Data, Ix2, Ix3, Zip,
};
use ndarray_linalg::{error::LinalgError, Inverse, InverseC, Lapack};
use std::fmt::Debug;
use std::ops::SubAssign;

/// Multiplies each of the lhs matrices with rhs matrix
pub(in crate) fn broad_dot_ix3_ix2<A, S, S2>(
    lhs: &ArrayBase<S, Ix3>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Result<Array3<A>, MatrixOperationError>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (lhs_len, lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_rows, rhs_cols) = rhs.dim();
    if lhs_cols != rhs_rows {
        return Result::Err(MatrixOperationError::MatrixMultiplicationError {
            lhs_shape: (lhs_rows, lhs_cols),
            rhs_shape: (rhs_rows, rhs_cols),
        });
    }
    let mut output = Array3::zeros([lhs_len, lhs_rows, rhs_cols]);
    azip!((elem in lhs.outer_iter(), mut out in output.outer_iter_mut()) general_mat_mul(A::one(), &elem, &rhs, A::zero(), &mut out));
    Result::Ok(output)
}

/// Multiples each of i-th matrix in lhs with i-th matrix in rhs
pub(in crate) fn broad_dot_ix3_ix3<A, S, S2>(
    lhs: &ArrayBase<S, Ix3>,
    rhs: &ArrayBase<S2, Ix3>,
) -> Result<Array3<A>, MatrixOperationError>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (lhs_len, lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_len, rhs_rows, rhs_cols) = rhs.dim();
    if lhs_cols != rhs_rows {
        return Result::Err(MatrixOperationError::MatrixMultiplicationError {
            lhs_shape: (lhs_rows, lhs_cols),
            rhs_shape: (rhs_rows, rhs_cols),
        });
    }

    if lhs_len != rhs_len {
        return Result::Err(MatrixOperationError::UnequalNumberOfInputsError);
    }
    let mut output = Array3::zeros([lhs_len, lhs_rows, rhs_cols]);
    azip!((l in lhs.outer_iter(), r in rhs.outer_iter(), mut out in output.outer_iter_mut())
            general_mat_mul(A::one(), &l, &r, A::zero(), &mut out));
    Result::Ok(output)
}

/// For i-th matrix in inner B and i-th matrix in outer A computes AB(A^T)
pub(in crate) fn quadratic_form_ix3_ix3_ix3<A, S1, S2>(
    inner: &ArrayBase<S1, Ix3>,
    outer: &ArrayBase<S2, Ix3>,
) -> Result<Array3<A>, MatrixOperationError>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (inner_len, inner_rows, inner_cols) = inner.dim();
    let (outer_len, outer_rows, outer_cols) = outer.dim();

    if outer_cols != inner_rows {
        return Result::Err(MatrixOperationError::MatrixMultiplicationError {
            lhs_shape: (outer_rows, outer_cols),
            rhs_shape: (inner_rows, inner_cols),
        });
    }

    if inner_cols != outer_cols {
        return Result::Err(MatrixOperationError::MatrixMultiplicationError {
            lhs_shape: (inner_rows, inner_cols),
            rhs_shape: (outer_cols, outer_rows),
        });
    }

    if inner_len != outer_len {
        return Result::Err(MatrixOperationError::UnequalNumberOfInputsError);
    }

    let intermediate = broad_dot_ix3_ix3(outer, inner)?;
    let mut outer_view = outer.view();
    outer_view.swap_axes(1, 2);
    broad_dot_ix3_ix3(&intermediate, &outer_view)
}

pub(in crate) fn update_states<A: Scalar + Lapack, S1: Data<Elem = A>>(
    states: &ArrayBase<S1, Ix2>,
    kalman_gains: &Array3<A>,
    innovations_negated: &Array3<A>,
) -> Result<Array3<A>, MatrixOperationError> {
    let (state_rows, state_cols) = states.dim();
    let (innovations_len, innovation_rows, innovation_cols) = innovations_negated.dim();
    let (kalman_gain_len, kalman_gain_rows, kalman_gains_cols) = kalman_gains.dim();

    if kalman_gains_cols != innovation_cols {
        return Result::Err(MatrixOperationError::MatrixMultiplicationError {
            lhs_shape: (kalman_gain_rows, kalman_gains_cols),
            rhs_shape: (innovation_cols, innovation_rows),
        });
    }

    if kalman_gain_rows != state_cols {
        return Result::Err(MatrixOperationError::MatrixArithmeticError {
            lhs_dim: (state_rows, state_cols),
            rhs_dim: (innovation_rows, kalman_gain_rows),
        });
    }

    if state_rows != kalman_gain_len || state_rows != innovations_len {
        return Result::Err(MatrixOperationError::UnequalNumberOfInputsError);
    }

    let updated_states_dim = [state_rows, innovation_rows, state_cols];
    let mut updated_states = states
        .view()
        .insert_axis(Axis(1))
        .broadcast(updated_states_dim)
        .unwrap()
        .to_owned();
    Zip::from(updated_states.outer_iter_mut())
        .and(kalman_gains.outer_iter())
        .and(innovations_negated.outer_iter())
        .apply(|mut states, gain, innovation| {
            let mut result = gain.dot(&innovation.t());
            result.swap_axes(1, 0);
            states.sub_assign(&result);
        });
    Result::Ok(updated_states)
}

pub(in crate) fn update_covariance<A: Scalar + Lapack, S1: Data<Elem = A>>(
    covariances: &ArrayBase<S1, Ix3>,
    kalman_gains: &Array3<A>,
    l_matrices: &Array3<A>,
) -> Result<Array3<A>, MatrixOperationError> {
    let (kalman_gain_len, kalman_gain_rows, kalman_gain_cols) = kalman_gains.dim();
    let (covariances_len, covariances_rows, covariances_cols) = covariances.dim();
    let (l_matrix_len, l_matrix_rows, l_matrix_cols) = l_matrices.dim();
    if kalman_gain_len != covariances_len || covariances_len != l_matrix_len {
        return Result::Err(MatrixOperationError::UnequalNumberOfInputsError);
    }

    if kalman_gain_cols != l_matrix_cols {
        return Result::Err(MatrixOperationError::MatrixMultiplicationError {
            lhs_shape: (kalman_gain_rows, kalman_gain_cols),
            rhs_shape: (l_matrix_cols, l_matrix_rows),
        });
    }

    if (kalman_gain_rows, l_matrix_rows) != (covariances_rows, covariances_cols) {
        return Result::Err(MatrixOperationError::MatrixArithmeticError {
            lhs_dim: (covariances_rows, covariances_cols),
            rhs_dim: (kalman_gain_rows, l_matrix_rows),
        });
    }

    let mut updated_covariances = covariances.to_owned();
    Zip::from(updated_covariances.outer_iter_mut())
        .and(kalman_gains.outer_iter())
        .and(l_matrices.outer_iter())
        .apply(|mut cov, gain, l| {
            general_mat_mul(A::one().neg(), &gain, &l.t(), A::one(), &mut cov);
        });
    Result::Ok(updated_covariances)
}

pub(in crate) fn innovation_covariances_ix2<A: Scalar + Lapack>(
    observation_matrix: &Array2<A>,
    observation_covariance: &Array2<A>,
    l_matrices: &Array3<A>,
) -> Result<Array3<A>, MatrixOperationError> {
    let (observation_matrix_rows, observation_matrix_cols) = observation_matrix.dim();
    let (observation_covariance_rows, observation_covariance_cols) = observation_covariance.dim();
    let (l_matrices_len, l_matrix_rows, l_matrix_cols) = l_matrices.dim();

    if observation_matrix_cols != l_matrix_rows {
        return Result::Err(MatrixOperationError::MatrixMultiplicationError {
            lhs_shape: (observation_matrix_rows, observation_covariance_cols),
            rhs_shape: (l_matrix_rows, l_matrix_cols),
        });
    }

    if (observation_matrix_rows, l_matrix_cols)
        != (observation_covariance_rows, observation_covariance_cols)
    {
        return Result::Err(MatrixOperationError::MatrixArithmeticError {
            lhs_dim: (observation_matrix_rows, l_matrix_cols),
            rhs_dim: (observation_covariance_rows, observation_covariance_cols),
        });
    }

    let output_dim = [
        l_matrices_len,
        observation_matrix_rows,
        observation_matrix_cols,
    ];
    let mut output_matrix = observation_covariance
        .broadcast(output_dim)
        .unwrap()
        .to_owned();
    azip!((mut output in output_matrix.axis_iter_mut(Axis(0)), l_matrix in l_matrices.axis_iter(Axis(0)))
     general_mat_mul(A::one(), observation_matrix, &l_matrix, A::one(), &mut output));
    Result::Ok(output_matrix)
}

pub(in crate) fn pairwise_difference<A, S1, S2>(
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Result<Array3<A>, MatrixOperationError>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (lhs_rows, lhs_cols) = lhs.dim();
    let (rhs_rows, rhs_cols) = rhs.dim();

    if lhs_cols != rhs_cols {
        return Result::Err(MatrixOperationError::VectorArithmeticError {
            lhs_dim: lhs_cols,
            rhs_dim: rhs_cols,
        });
    }

    let out_dim = [lhs_rows, rhs_rows, rhs_cols];
    let expanded_lhs = lhs.to_owned().insert_axis(Axis(1));
    let expanded_lhs_view = expanded_lhs.view();
    let broadcast = expanded_lhs_view.broadcast(out_dim).unwrap();
    let mut output = broadcast.to_owned();
    azip!((mut out in output.outer_iter_mut()) out.sub_assign(&rhs));
    Result::Ok(output)
}

pub(in crate) fn invc_all_ix3<A, S1>(
    matrices: &ArrayBase<S1, Ix3>,
) -> Result<Array3<A>, MatrixOperationError>
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

pub(in crate) fn inv_all_ix3<A, S1>(
    matrices: &ArrayBase<S1, Ix3>,
) -> Result<Array3<A>, MatrixOperationError>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
{
    let mut inv_matrices = matrices.to_owned();
    for (mut destination, source) in inv_matrices.outer_iter_mut().zip(matrices.outer_iter()) {
        destination.assign(&source.inv()?);
    }
    Ok(inv_matrices)
}

#[derive(Debug)]
pub(in crate) enum MatrixOperationError {
    MatrixMultiplicationError {
        lhs_shape: (usize, usize),
        rhs_shape: (usize, usize),
    },
    VectorArithmeticError {
        lhs_dim: usize,
        rhs_dim: usize,
    },
    MatrixArithmeticError {
        lhs_dim: (usize, usize),
        rhs_dim: (usize, usize),
    },
    UnequalNumberOfInputsError,
    LapackError(LinalgError),
}

impl From<LinalgError> for MatrixOperationError {
    fn from(err: LinalgError) -> Self {
        MatrixOperationError::LapackError(err)
    }
}
