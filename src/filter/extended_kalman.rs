use crate::filter::filter_traits::Filter;
use cauchy::Scalar;
use ndarray::{Array2, Array3, ArrayBase, ArrayView, ArrayView1, Data, Dimension, Ix1, Ix2, Ix3, OwnedRepr, ArrayView2, Axis, CowArray};
use ndarray_linalg::Lapack;
use std::marker::PhantomData;
use std::ops::{AddAssign, SubAssign};
use crate::filter::kalman_common::{pairwise_difference, broad_dot_ix3_ix3, invc_all_ix3};

/// Extended Kalman filter with additive Gaussian noise
///
/// This struct represents the extended Kalman filter where transition function is parametrized only
/// by previous state, while the measurement function is solely parametrized by predicted state.
/// Further it is assumed that transition and measurement noise has zero-mean and is Gaussian.
///
/// The struct is generic with following type parameters:
/// - Num type parameter represents the numeric type on which the algorithm is going to work
/// - Dim represents the dimension of inputs for transition and measurement functions. For example
///   if Dim is Ix1, then it is assumed that transition and measurement functions compute the
///   predictions and Jacobians for single row at a time. In case that Dim is Ix2, then it is
///   assumed that the transition and measurement functions are vectorized, and can compute the
///   predictions and Jacobian matrices for many states at once.
/// - Trans represents the type of function for transition and measurement function, while Jacobi
///   represents the type of function for computing the Jacobians of corresponding functions.
pub struct AdditiveExtendedKalmanFilter<Num, Dim, Trans, Jacobi>
where
    Num: Scalar + Lapack,
    Dim: Dimension,
    Trans: Fn(&ArrayView<Num, Dim>) -> ArrayBase<OwnedRepr<Num>, Dim>,
    Jacobi: Fn(&ArrayView<Num, Dim>) -> ArrayBase<OwnedRepr<Num>, Dim::Larger>,
{
    dimension_phantom: PhantomData<Dim>,
    transition_function: Box<Trans>,
    transition_function_jacobian: Box<Jacobi>,
    measurement_function: Box<Trans>,
    measurement_function_jacobi: Box<Jacobi>,
    transition_covariance: Array2<Num>,
    measurement_covariance: Array2<Num>
}

impl<Num, Trans, Jacobi> Filter<Num> for AdditiveExtendedKalmanFilter<Num, Ix1, Trans, Jacobi>
where
    Num: Scalar + Lapack,
    Trans: Fn(&ArrayView1<Num>) -> ArrayBase<OwnedRepr<Num>, Ix1>,
    Jacobi: Fn(&ArrayView1<Num>) -> ArrayBase<OwnedRepr<Num>, Ix2>,
{
    type Prediction = (Array2<Num>, Array3<Num>);
    type Update = (Array3<Num>, Array3<Num>);

    fn predict<A: Data<Elem = Num>, B: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
    ) -> Self::Prediction {
        let mut predicted_states = Array2::zeros(states.raw_dim());
        let states_dim = states.dim();
        let states_count = states_dim.0;
        let states_length = states_dim.1;
        let mut state_jacobis_transposed =
            Array3::zeros([states_count, states_length, states_length]);
        for (mut state, mut jacobi) in predicted_states
            .outer_iter_mut()
            .zip(state_jacobis_transposed.outer_iter_mut())
        {
            let intermediate_state = (self.transition_function)(&state.view());
            let state_jacobi = (self.transition_function_jacobian)(&state.view());
            jacobi.assign(&state_jacobi);
            state.assign(&intermediate_state);
        }

        let mut predicted_covariances = self
            .transition_covariance
            .broadcast(covariances.raw_dim())
            .unwrap()
            .to_owned();
        for (mut cov, jacobi) in predicted_covariances
            .outer_iter_mut()
            .zip(state_jacobis_transposed.outer_iter())
        {
            let predicted_cov = jacobi.dot(&cov).dot(&jacobi.t());
            cov.add_assign(&predicted_cov);
        }

        (predicted_states, predicted_covariances)
    }

    fn update<A: Data<Elem = Num>, B: Data<Elem = Num>, C: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> Self::Update {
        // compute expected measurements
        let measurements_dim = measurements.dim();
        let states_dim = states.dim();
        let expected_meas_dim = [states_dim.0, measurements_dim.1];
        let mut expected_measurements = Array2::zeros(expected_meas_dim);
        for (mut expected_measurement, state) in expected_measurements.outer_iter_mut().zip(states.outer_iter()) {
            expected_measurement.assign(&(self.measurement_function)(&state));
        }
        let innovations = pairwise_difference(&expected_measurements,  measurements);

        // compute measurement jacobi matrices
        let measurement_jacobis_dim = [states_dim.0, measurements_dim.1, measurements_dim.1];
        let mut measurement_jacobis = Array3::zeros(measurement_jacobis_dim);
        for (mut elem, state) in measurement_jacobis.outer_iter_mut().zip(states.outer_iter()) {
            elem.assign(&(self.measurement_function_jacobi)(&state));
        }

        // compute l-matrices
        measurement_jacobis.swap_axes(1, 2);
        let l_matrices = broad_dot_ix3_ix3(covariances, &measurement_jacobis);
        measurement_jacobis.swap_axes(1,2);

        let mut innovation_covariances = broad_dot_ix3_ix3(&measurement_jacobis, &l_matrices);
        for mut elem in innovation_covariances.outer_iter_mut() {
            elem.add_assign(&self.measurement_covariance);
        }
        let innovation_covariances = innovation_covariances;

        let inv_innovation_covariances = invc_all_ix3(&innovation_covariances);
        let kalman_gains = broad_dot_ix3_ix3(&l_matrices, &inv_innovation_covariances);

        let updated_states_dim = [states_dim.0, measurements_dim.0, states_dim.1];
        let expanded_state = states.view().insert_axis(Axis(1));
        let broadcasted_state = expanded_state.broadcast(updated_states_dim).unwrap();
        let mut updated_states = CowArray::from(broadcasted_state);
        for ((mut updated_state, kalman_gain), innovation) in updated_states
            .outer_iter_mut()
            .zip(kalman_gains.outer_iter())
            .zip(innovations.outer_iter())
        {
            let intermediate_result = kalman_gain.dot(&innovation.t()).t().into_owned();
            updated_state.add_assign(&intermediate_result);
        }

        let mut updated_covariances = covariances.to_owned();
        for ((mut elem, kalman_gain), l) in updated_covariances
            .outer_iter_mut()
            .zip(kalman_gains.outer_iter())
            .zip(l_matrices.outer_iter())
        {
            let intermediate = kalman_gain.dot(&l.t());
            elem.sub_assign(&intermediate);
        }

        (updated_states.into_owned(), updated_covariances)
    }
}


impl <Num, Trans, Jacobi> Filter<Num> for AdditiveExtendedKalmanFilter<Num, Ix2, Trans, Jacobi> where
    Num: Scalar + Lapack,
    Trans: Fn(&ArrayView2<Num>) -> ArrayBase<OwnedRepr<Num>, Ix2>,
    Jacobi: Fn(&ArrayView2<Num>) -> ArrayBase<OwnedRepr<Num>, Ix3> {

    type Prediction = (Array2<Num>, Array3<Num>);
    type Update = (Array3<Num>, Array3<Num>);

    fn predict<A: Data<Elem=Num>, B: Data<Elem=Num>>(&self, states: &ArrayBase<A, Ix2>, covariances: &ArrayBase<B, Ix3>) -> Self::Prediction {
        let mut predicted_states = (self.transition_function)(&states.view());
        let state_jacobis = (self.transition_function_jacobian)(&states.view());
        let mut predicted_covariances = self
            .transition_covariance
            .broadcast(covariances.raw_dim())
            .unwrap()
            .to_owned();
        for (mut cov, jacobi) in predicted_covariances
            .outer_iter_mut()
            .zip(state_jacobis.outer_iter())
        {
            let predicted_cov = jacobi.dot(&cov).dot(&jacobi.t());
            cov.add_assign(&predicted_cov);
        }
        (predicted_states, predicted_covariances)
    }

    fn update<A: Data<Elem=Num>, B: Data<Elem=Num>, C: Data<Elem=Num>>(&self, states: &ArrayBase<A, Ix2>, covariances: &ArrayBase<B, Ix3>, measurements: &ArrayBase<C, Ix2>) -> Self::Update {
        let expected_measurements = (self.measurement_function)(&states.view());
        let innovations = pairwise_difference(&expected_measurements, measurements);

        let mut measurement_jacobis = (self.measurement_function_jacobi)(&states.view());
        measurement_jacobis.swap_axes(1,2);
        let l_matrices = broad_dot_ix3_ix3(covariances, &measurement_jacobis);
        measurement_jacobis.swap_axes(1,2);

        let mut innovation_covariances = broad_dot_ix3_ix3(&measurement_jacobis, &l_matrices);
        for mut inno_cov in innovation_covariances.outer_iter_mut() {
            inno_cov.add_assign(&self.measurement_covariance);
        }

        let inv_innovation_covariances = invc_all_ix3(&innovation_covariances);
        let kalman_gains = broad_dot_ix3_ix3(&l_matrices, &inv_innovation_covariances);

        let states_dim = states.dim();
        let measurements_dim = measurements.dim();
        let updated_states_dim = [states_dim.0, measurements_dim.0, states_dim.1];
        let expanded_state = states.view().insert_axis(Axis(1));
        let broadcasted_state = expanded_state.broadcast(updated_states_dim).unwrap();
        let mut updated_states = CowArray::from(broadcasted_state);
        for ((mut updated_state, kalman_gain), innovation) in updated_states
            .outer_iter_mut()
            .zip(kalman_gains.outer_iter())
            .zip(innovations.outer_iter())
        {
            let intermediate_result = kalman_gain.dot(&innovation.t()).t().into_owned();
            updated_state.add_assign(&intermediate_result);
        }

        let mut updated_covariances = covariances.to_owned();
        for ((mut elem, kalman_gain), l) in updated_covariances
            .outer_iter_mut()
            .zip(kalman_gains.outer_iter())
            .zip(l_matrices.outer_iter())
        {
            let intermediate = kalman_gain.dot(&l.t());
            elem.sub_assign(&intermediate);
        }

        (updated_states.into_owned(), updated_covariances)
    }
}