use crate::filter::filter_traits::Filter;
use crate::filter::kalman_common::{broad_dot_ix3_ix3, invc_all_ix3, pairwise_difference, quadratic_form_ix3_ix3_ix3, update_states, update_covariance};
use cauchy::Scalar;
use ndarray::{
    Array1, Array2, Array3, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, CowArray, Data,
    Dimension, Ix1, Ix2, Ix3, OwnedRepr,
};
use ndarray_linalg::Lapack;
use std::marker::PhantomData;
use std::ops::{AddAssign, SubAssign};

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
pub struct AdditiveNoiseExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    transition_function: Box<dyn Fn(&ArrayView2<Num>) -> Array2<Num>>,
    transition_function_jacobian: Box<dyn Fn(&ArrayView2<Num>) -> Array3<Num>>,
    measurement_function: Box<dyn Fn(&ArrayView2<Num>) -> Array2<Num>>,
    measurement_function_jacobi: Box<dyn Fn(&ArrayView2<Num>) -> Array3<Num>>,
    transition_covariance: Array2<Num>,
    measurement_covariance: Array2<Num>,
}

impl<Num> Filter<Num> for AdditiveNoiseExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    type Prediction = (Array2<Num>, Array3<Num>);
    type Update = (Array3<Num>, Array3<Num>);

    fn predict<A: Data<Elem = Num>, B: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
    ) -> Self::Prediction {
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

    fn update<A: Data<Elem = Num>, B: Data<Elem = Num>, C: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> Self::Update {
        let expected_measurements = (self.measurement_function)(&states.view());
        let innovations = pairwise_difference(&expected_measurements, measurements);

        let mut measurement_jacobis = (self.measurement_function_jacobi)(&states.view());
        measurement_jacobis.swap_axes(1, 2);
        let l_matrices = broad_dot_ix3_ix3(covariances, &measurement_jacobis);
        measurement_jacobis.swap_axes(1, 2);

        let mut innovation_covariances = broad_dot_ix3_ix3(&measurement_jacobis, &l_matrices);
        for mut inno_cov in innovation_covariances.outer_iter_mut() {
            inno_cov.add_assign(&self.measurement_covariance);
        }

        let inv_innovation_covariances = invc_all_ix3(&innovation_covariances);
        let kalman_gains = broad_dot_ix3_ix3(&l_matrices, &inv_innovation_covariances);
        let updated_states = update_states(&states, &kalman_gains, &innovations);
        let updated_covariances = update_covariance(covariances, &kalman_gains, &l_matrices);

        (updated_states.into_owned(), updated_covariances)
    }
}

type MatrixStackProducer<Num> = Box<dyn Fn(&ArrayView2<Num>, &ArrayView2<Num>) -> Array3<Num>>;
type RowStackProducer<Num> = Box<dyn Fn(&ArrayView2<Num>, &ArrayView2<Num>) -> Array2<Num>>;

pub struct ExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    transition_function: RowStackProducer<Num>,
    transition_function_jacobi_state: MatrixStackProducer<Num>,
    transition_function_jacobi_noise: MatrixStackProducer<Num>,
    measurement_function: RowStackProducer<Num>,
    measurement_function_jacobi_state: MatrixStackProducer<Num>,
    measurement_function_jacobi_noise: MatrixStackProducer<Num>,
    transition_covariance: Array2<Num>,
    measurement_covariance: Array2<Num>,
}

impl<Num> Filter<Num> for ExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    type Prediction = (Array2<Num>, Array3<Num>);
    type Update = (Array3<Num>, Array3<Num>);

    fn predict<A: Data<Elem = Num>, B: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
    ) -> Self::Prediction {
        let zeros = Array2::zeros(states.raw_dim());
        let predicted_states = (self.transition_function)(&states.view(), &zeros.view());

        let mut state_jacobi = (self.transition_function_jacobi_state)(&states.view(), &zeros.view());
        let mut predicted_covariances = quadratic_form_ix3_ix3_ix3(&covariances, &mut state_jacobi);

        let broadcast_transition_covariance = self.transition_covariance
            .broadcast(covariances.raw_dim())
            .unwrap();
        let mut noise_jacobi = (self.transition_function_jacobi_noise)(&states.view(), &zeros.view());
        let second_summand = quadratic_form_ix3_ix3_ix3(&broadcast_transition_covariance, &mut noise_jacobi);
        predicted_covariances.add_assign(&second_summand);

        (predicted_states, predicted_covariances)
    }

    fn update<A: Data<Elem = Num>, B: Data<Elem = Num>, C: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> Self::Update {
        let states_shape = states.dim();
        let measurements_shape = measurements.dim();
        let zeros = Array2::zeros([states_shape.0, measurements_shape.1]);
        let expected_measurements = (self.measurement_function)(&states.view(), &zeros.view());
        let innovations_negated = pairwise_difference(&expected_measurements, measurements);

        let mut measurement_jacobi = (self.measurement_function_jacobi_state)(&states.view(), &zeros.view());
        measurement_jacobi.swap_axes(1, 2);
        let state_l_matrices = broad_dot_ix3_ix3(&covariances, &measurement_jacobi);
        measurement_jacobi.swap_axes(1, 2);

        let mut noise_jacobi = (self.measurement_function_jacobi_noise)(&states.view(), &zeros.view());
        let second_summand = quadratic_form_ix3_ix3_ix3(&covariances, &mut noise_jacobi);

        let mut innovation_covariances = broad_dot_ix3_ix3(&measurement_jacobi, &state_l_matrices);
        innovation_covariances.add_assign(&second_summand);

        let inv_innovation_covariances = invc_all_ix3(&innovation_covariances);
        let kalman_gains = broad_dot_ix3_ix3(&state_l_matrices, &inv_innovation_covariances);

        let updated_states = update_states(&states, &kalman_gains, &innovations_negated);
        let updated_covariances = update_covariance(covariances, &kalman_gains, &state_l_matrices);

        (updated_states, updated_covariances)
    }
}

impl<Num> ExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    pub fn new<A: Data<Elem = Num>>(
        transition_function: RowStackProducer<Num>,
        transition_function_jacobi_state: MatrixStackProducer<Num>,
        transition_function_jacobi_noise: MatrixStackProducer<Num>,
        measurement_function: RowStackProducer<Num>,
        measurement_function_jacobi_state: MatrixStackProducer<Num>,
        measurement_function_jacobi_noise: MatrixStackProducer<Num>,
        transition_covariance: &ArrayBase<A, Ix2>,
        measurement_covariance: &ArrayBase<A, Ix2>,
    ) -> Self {
        ExtendedKalmanFilter {
            transition_function,
            transition_function_jacobi_state,
            transition_function_jacobi_noise,
            measurement_function,
            measurement_function_jacobi_state,
            measurement_function_jacobi_noise,
            transition_covariance: transition_covariance.to_owned(),
            measurement_covariance: measurement_covariance.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::filter::extended_kalman::ExtendedKalmanFilter;
    use ndarray::{Array2, ArrayView, ArrayView2, Axis};

    #[test]
    fn can_create_extended_kalman_filter() {
        let transition_function = |a: &ArrayView2<f64>, b: &ArrayView2<f64>| a.to_owned();
        let transition_function_jacobi_state =
            |a: &ArrayView2<f64>, b: &ArrayView2<f64>| a.insert_axis(Axis(0)).to_owned();
        let transition_function_jacobi_noise =
            |a: &ArrayView2<f64>, b: &ArrayView2<f64>| a.insert_axis(Axis(0)).to_owned();
        let measurement_function = |a: &ArrayView2<f64>, b: &ArrayView2<f64>| a.to_owned();
        let measurement_function_jacobi_state =
            |a: &ArrayView2<f64>, b: &ArrayView2<f64>| a.insert_axis(Axis(0)).to_owned();
        let measurement_function_jacobi_noise =
            |a: &ArrayView2<f64>, b: &ArrayView2<f64>| a.insert_axis(Axis(0)).to_owned();
        let cova: Array2<f64> = Array2::zeros([8, 8]);
        let covb: Array2<f64> = Array2::zeros([8, 8]);
        let c = ExtendedKalmanFilter::new(
            Box::new(transition_function),
            Box::new(transition_function_jacobi_state),
            Box::new(transition_function_jacobi_noise),
            Box::new(measurement_function),
            Box::new(measurement_function_jacobi_state),
            Box::new(measurement_function_jacobi_noise),
            &cova,
            &covb,
        );
    }
}
