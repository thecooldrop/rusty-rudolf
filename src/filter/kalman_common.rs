use cauchy::Scalar;
use ndarray::linalg::Dot;
use ndarray::{Array2, Array3, ArrayBase, Axis, Data, Ix2, Ix3};
use ndarray_linalg::InverseC;
use ndarray_linalg::Lapack;
use std::ops::{AddAssign, SubAssign};

pub fn broad_dot_ix3_ix2<A, S, S2>(lhs: &ArrayBase<S, Ix3>, rhs: &ArrayBase<S2, Ix2>) -> Array3<A>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let lhs_dim = lhs.dim();
    let rhs_dim = rhs.dim();
    let mut output = Array3::zeros([lhs_dim.0, lhs_dim.1, rhs_dim.1]);
    for (elem, mut out) in lhs.outer_iter().zip(output.outer_iter_mut()) {
        out.assign(&elem.dot(rhs));
    }
    output
}

pub fn broad_dot_ix2_ix3<A, S, S2>(lhs: &ArrayBase<S, Ix2>, rhs: &ArrayBase<S2, Ix3>) -> Array3<A>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let lhs_dim = lhs.dim();
    let rhs_dim = rhs.dim();
    let mut output = Array3::zeros([rhs_dim.0, lhs_dim.0, rhs_dim.2]);
    for (elem, mut out) in rhs.outer_iter().zip(output.outer_iter_mut()) {
        out.assign(&lhs.dot(&elem));
    }
    output
}

pub fn broad_dot_ix3_ix3<A, S, S2>(lhs: &ArrayBase<S, Ix3>, rhs: &ArrayBase<S2, Ix3>) -> Array3<A>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let self_dim = lhs.dim();
    let rhs_dim = rhs.dim();
    let mut output = Array3::zeros([self_dim.0, self_dim.1, rhs_dim.2]);
    for ((elem, mut out), input) in lhs
        .outer_iter()
        .zip(output.outer_iter_mut())
        .zip(rhs.outer_iter())
    {
        out.assign(&elem.dot(&input));
    }
    output
}

pub fn quadratic_form_ix2_ix3_ix2<A, S1, S2, S3>(
    lhs: &ArrayBase<S1, Ix2>,
    mid: &ArrayBase<S2, Ix3>,
    rhs: &ArrayBase<S3, Ix2>,
) -> Array3<A>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    S3: Data<Elem = A>,
{
    let lhs_dim = lhs.dim();
    let mid_dim = mid.dim();
    let rhs_dim = rhs.dim();
    unsafe {
        let mut output = Array3::uninitialized([mid_dim.0, lhs_dim.0, rhs_dim.1]);
        for (mut elem, input) in output.outer_iter_mut().zip(mid.outer_iter()) {
            elem.assign(&lhs.dot(&input).dot(rhs));
        }
        output
    }
}

#[inline(always)]
pub fn quadratic_form_ix2_ix3_ix2_add_ix2<A, S1, S2, S3, S4>(
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
    let mid_dim = mid.dim();
    let add_dim = add.dim();
    unsafe {
        let mut output = Array3::uninitialized([mid_dim.0, add_dim.0, add_dim.1]);
        for (mut out, elem) in output.outer_iter_mut().zip(mid.outer_iter()) {
            let result = lhs.dot(&elem).dot(rhs) + add;
            out.assign(&result);
        }
        output
    }
}

pub fn innovation_covariances_ix2<A: Scalar + Lapack>(
    observation_matrix: &Array2<A>,
    observation_covariance: &Array2<A>,
    l_matrices: &Array3<A>,
) -> Array3<A> {
    let mut innovation_covariances = broad_dot_ix2_ix3(observation_matrix, l_matrices);
    for mut cov in innovation_covariances.outer_iter_mut() {
        cov.add_assign(observation_covariance);
    }
    innovation_covariances
}

pub fn pairwise_difference<A, S1, S2>(
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Array3<A>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let lhs_dim = lhs.dim();
    let rhs_dim = rhs.dim();
    let out_dim = [lhs_dim.0, rhs_dim.0, rhs_dim.1];
    let mut difference = lhs
        .to_owned()
        .insert_axis(Axis(1))
        .broadcast(out_dim)
        .unwrap()
        .into_owned();
    for mut elem in difference.outer_iter_mut() {
        elem.sub_assign(rhs);
    }
    difference
}

pub fn invc_all_ix3<A, S1>(matrices: &ArrayBase<S1, Ix3>) -> Array3<A>
where
    A: Scalar + Lapack,
    S1: Data<Elem = A>,
{
    let mut inv_matrices = Array3::zeros(matrices.raw_dim());
    for (mut destination, source) in inv_matrices.outer_iter_mut().zip(matrices.outer_iter()) {
        destination.assign(&source.invc().unwrap());
    }
    inv_matrices
}
