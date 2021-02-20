use std::ops::{AddAssign};

use cauchy::Scalar;
use ndarray::{Array2, Array3, ArrayBase, ArrayView2, Data, Ix2, Ix3,
};
use ndarray_linalg::Lapack;

use crate::filter::filter_traits::Filter;
use crate::filter::kalman_common::{broad_dot_ix3_ix3, invc_all_ix3, pairwise_difference, quadratic_form_ix3_ix3_ix3, update_covariance, update_states};


/// Extended Kalman filter with additive Gaussian noise
///
/// This struct represents the extended Kalman filter where transition function is parametrized only
/// by previous state, while the measurement function is solely parametrized by predicted state.
/// Further it is assumed that transition and measurement noise has zero-mean and is Gaussian.
///
/// The struct is generic with following type parameters:
/// - Num type parameter represents the numeric type on which the algorithm is going to work
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
        let predicted_states = (self.transition_function)(&states.view());
        let state_jacobis = (self.transition_function_jacobian)(&states.view());
        let mut predicted_covariances = self
            .transition_covariance
            .broadcast(covariances.raw_dim())
            .unwrap()
            .to_owned();
        let quadratic_summand = quadratic_form_ix3_ix3_ix3(covariances, &state_jacobis);
        predicted_covariances.add_assign(&quadratic_summand);
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

        let measurement_jacobis = (self.measurement_function_jacobi)(&states.view());
        let mut measurement_jacobis_view = measurement_jacobis.view();
        measurement_jacobis_view.swap_axes(1, 2);

        let l_matrices = broad_dot_ix3_ix3(covariances, &measurement_jacobis_view);

        let mut innovation_covariances = broad_dot_ix3_ix3(&measurement_jacobis, &l_matrices);
        innovation_covariances.add_assign(&self.measurement_covariance);

        let inv_innovation_covariances = invc_all_ix3(&innovation_covariances);
        let kalman_gains = broad_dot_ix3_ix3(&l_matrices, &inv_innovation_covariances);
        let updated_states = update_states(&states, &kalman_gains, &innovations);
        let updated_covariances = update_covariance(covariances, &kalman_gains, &l_matrices);

        (updated_states.into_owned(), updated_covariances)
    }
}

type JacobiMatrixProducer<Num> = Box<dyn Fn(&ArrayView2<Num>, &ArrayView2<Num>) -> Array3<Num>>;
type RowStackProducer<Num> = Box<dyn Fn(&ArrayView2<Num>, &ArrayView2<Num>) -> Array2<Num>>;

pub struct ExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    transition_function: RowStackProducer<Num>,
    transition_function_jacobi_state: JacobiMatrixProducer<Num>,
    transition_function_jacobi_noise: JacobiMatrixProducer<Num>,
    measurement_function: RowStackProducer<Num>,
    measurement_function_jacobi_state: JacobiMatrixProducer<Num>,
    measurement_function_jacobi_noise: JacobiMatrixProducer<Num>,
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

        let state_jacobi = (self.transition_function_jacobi_state)(&states.view(), &zeros.view());
        let mut predicted_covariances = quadratic_form_ix3_ix3_ix3(&covariances, &state_jacobi);

        let broadcast_transition_covariance = self.transition_covariance
            .broadcast(covariances.raw_dim())
            .unwrap();
        let noise_jacobi = (self.transition_function_jacobi_noise)(&states.view(), &zeros.view());
        let second_summand = quadratic_form_ix3_ix3_ix3(&broadcast_transition_covariance, &noise_jacobi);
        predicted_covariances.add_assign(&second_summand);

        (predicted_states, predicted_covariances)
    }

    fn update<A: Data<Elem = Num>, B: Data<Elem = Num>, C: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> Self::Update {
        let states_number = states.dim().0;
        let measurement_dim = measurements.dim().1;

        let zeros = Array2::zeros([states_number, measurement_dim]);
        let expected_measurements = (self.measurement_function)(&states.view(), &zeros.view());
        let innovations_negated = pairwise_difference(&expected_measurements, measurements);

        let measurement_jacobi = (self.measurement_function_jacobi_state)(&states.view(), &zeros.view());
        let mut measurement_jacobi_view = measurement_jacobi.view();
        measurement_jacobi_view.swap_axes(1, 2);
        let state_l_matrices = broad_dot_ix3_ix3(&covariances, &measurement_jacobi_view);

        let noise_jacobi = (self.measurement_function_jacobi_noise)(&states.view(), &zeros.view());
        let measurement_covariance_shape = self.measurement_covariance.dim();
        let target_broadcast_shape = [noise_jacobi.dim().0, measurement_covariance_shape.0, measurement_covariance_shape.1];
        let broadcast_measurement_covariances = self.measurement_covariance.broadcast(target_broadcast_shape).unwrap();
        let second_summand = quadratic_form_ix3_ix3_ix3(&broadcast_measurement_covariances, &noise_jacobi);

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
        transition_function_jacobi_state: JacobiMatrixProducer<Num>,
        transition_function_jacobi_noise: JacobiMatrixProducer<Num>,
        measurement_function: RowStackProducer<Num>,
        measurement_function_jacobi_state: JacobiMatrixProducer<Num>,
        measurement_function_jacobi_noise: JacobiMatrixProducer<Num>,
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
    use ndarray::{Array2, ArrayView, ArrayView2, Axis};

    use crate::filter::extended_kalman::ExtendedKalmanFilter;

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
