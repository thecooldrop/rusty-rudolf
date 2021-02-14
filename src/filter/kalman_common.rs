use cauchy::Scalar;
use ndarray::linalg::Dot;
use ndarray::{Array2, Array3, ArrayBase, Axis, CowArray, Data, Ix2, Ix3};
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
    let mut output = Array3::zeros([mid_dim.0, lhs_dim.0, rhs_dim.1]);
    for (mut elem, input) in output.outer_iter_mut().zip(mid.outer_iter()) {
        elem.assign(&lhs.dot(&input).dot(rhs));
    }
    output
}

pub(in crate) fn quadratic_form_ix3_ix3_ix3<A, S1, S2>(inner: &ArrayBase<S1, Ix3>, outer: &mut ArrayBase<S2, Ix3>) -> Array3<A>
where
    A: Scalar+Lapack,
    S1: Data<Elem=A>,
    S2: Data<Elem=A>
{
    let intermediate = broad_dot_ix3_ix3(outer, inner);
    outer.swap_axes(1, 2);
    let result = broad_dot_ix3_ix3(&intermediate, outer);
    outer.swap_axes(1, 2);
    result
}

pub(in crate) fn update_states<A: Scalar+Lapack, S1: Data<Elem=A>>(states: &ArrayBase<S1, Ix2>, kalman_gains: &Array3<A>, innovations_negated: &Array3<A>) -> Array3<A> {
    let num_measurements = innovations_negated.dim().1;
    let state_dim = states.dim();
    let updated_states_dim = [state_dim.0, num_measurements, state_dim.1];
    let expanded_states_view = states.view().insert_axis(Axis(1));
    let broadcasted_view = expanded_states_view.broadcast(updated_states_dim).unwrap();
    let mut broadcast_states_updated = CowArray::from(broadcasted_view);
    for ((mut state_updated, kalman_gain), innovation) in broadcast_states_updated
        .outer_iter_mut()
        .zip(kalman_gains.outer_iter())
        .zip(innovations_negated.outer_iter())
    {
        let result = kalman_gain.dot(&innovation.t()).t().into_owned();
        state_updated.sub_assign(&result);
    }
    broadcast_states_updated.into_owned()
}

pub(in crate) fn update_covariance<A: Scalar+Lapack, S1: Data<Elem=A>>(covariances: &ArrayBase<S1, Ix3>, kalman_gains: &Array3<A>, l_matrices: &Array3<A>) -> Array3<A>{
    let mut updated_covariances = covariances.to_owned();
    for ((mut elem, kalman_gain), l) in updated_covariances
        .outer_iter_mut()
        .zip(kalman_gains.outer_iter())
        .zip(l_matrices.outer_iter())
    {
        let intermediate = kalman_gain.dot(&l.t());
        elem.sub_assign(&intermediate);
    }
    updated_covariances
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
    let mut output = Array3::zeros([mid_dim.0, add_dim.0, add_dim.1]);
    for (mut out, elem) in output.outer_iter_mut().zip(mid.outer_iter()) {
        let result = lhs.dot(&elem).dot(rhs) + add;
        out.assign(&result);
    }
    output
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
    let expanded_lhs = lhs.view().insert_axis(Axis(1));
    let broadcast_lhs = expanded_lhs.broadcast(out_dim).unwrap();
    let mut difference = CowArray::from(broadcast_lhs);
    for mut elem in difference.outer_iter_mut() {
        elem.sub_assign(rhs);
    }
    difference.into_owned()
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
