use ndarray::{Array2, Array3, Axis};
use ndarray_linalg::InverseC;
use std::ops::AddAssign;
use std::ops::SubAssign;

pub struct KalmanFilterWithoutControl {
    pub transition_matrix: Array2<f64>,
    pub observation_matrix: Array2<f64>,
    pub transition_covariance: Array2<f64>,
    pub observation_covariance: Array2<f64>,
}

impl KalmanFilterWithoutControl {

    pub fn predict(
        &self,
        states: &Array2<f64>,
        covariances: &Array3<f64>,
    ) -> (Array2<f64>, Array3<f64>) {
        let predicted_states = self.transition_matrix.dot(&states.t());
        unsafe {
            let mut predicted_covariances = Array3::uninitialized(covariances.raw_dim());
            let predicted_covariances_iter = predicted_covariances.outer_iter_mut();
            let input_covariances_iter = covariances.outer_iter();
            for (mut output_view, input) in predicted_covariances_iter.zip(input_covariances_iter) {
                output_view.assign(
                    &(self
                        .transition_matrix
                        .dot(&input)
                        .dot(&self.transition_matrix.t())
                        + &self.transition_covariance),
                );
            }
            (predicted_states, predicted_covariances)
        }
    }

    pub fn update(
        &self,
        states: &Array2<f64>,
        covariances: &Array3<f64>,
        measurements: &Array2<f64>,
    ) -> (Array3<f64>, Array3<f64>) {
        let expected_measurements = self.observation_matrix.dot(&states.t()).t().into_owned();
        let innovations = self.innovations(measurements, &expected_measurements);
        let innovation_covariances = self.innovation_covariances(covariances);
        let mut inv_inno_covs = innovation_covariances.clone();
        for mut elem in inv_inno_covs.outer_iter_mut() {
            elem.assign(&elem.invc().unwrap());
        }
        let kalman_gains = self.kalman_gains(covariances, &inv_inno_covs);
        let updated_states = self.update_states(states, &kalman_gains, &innovations); 
        let updated_covs = self.update_covariances(covariances, &kalman_gains);
        return (updated_states, updated_covs);
    }

    fn innovations(&self, measurements: &Array2<f64>, expected_measurements: &Array2<f64>) -> Array3<f64> {
        let meas_dim = measurements.dim();
        let state_count = expected_measurements.dim().0;
        let innovation_dim = [meas_dim.0, state_count, meas_dim.1];
        let mut innovations: Array3<f64> = measurements.clone().insert_axis(Axis(1)).broadcast(innovation_dim).unwrap().to_owned();
        for mut inno_view in innovations.outer_iter_mut()
        {
            inno_view.sub_assign(expected_measurements);
        }
        innovations
    }

    fn innovation_covariances(&self, covariances: &Array3<f64>) -> Array3<f64> {

        let innovation_covariance_dim = self.observation_matrix.dim().0;
        let mut innovation_covariances: Array3<f64> =
            Array3::zeros([covariances.dim().0, innovation_covariance_dim, innovation_covariance_dim]);
        for (mut inno_mat_view, cov_view) in innovation_covariances
            .outer_iter_mut()
            .zip(covariances.outer_iter())
        {
            inno_mat_view.assign(
                &(self
                    .observation_matrix
                    .dot(&cov_view)
                    .dot(&self.observation_matrix.t())
                    + &self.observation_covariance),
            );
        }
        innovation_covariances
    }

    fn kalman_gains(&self, covariances: &Array3<f64>, inv_innovation_covariances: &Array3<f64>) -> Array3<f64> {
        let cov_dim = covariances.dim();
        let meas_dim = self.observation_matrix.dim();
        let kalman_gains_dimension = [cov_dim.0, cov_dim.1, meas_dim.0];
        let mut kalman_gains = Array3::zeros(kalman_gains_dimension);
        for ((mut kg_view, cov), inv_icov) in kalman_gains
            .outer_iter_mut()
            .zip(covariances.outer_iter())
            .zip(inv_innovation_covariances.outer_iter())
        {
            kg_view.assign(&(cov.dot(&self.observation_matrix.t()).dot(&inv_icov)))
        }
        kalman_gains
    }

    fn update_states(&self, states: &Array2<f64>, kalman_gains: &Array3<f64>, innovations: &Array3<f64>) -> Array3<f64> {
        let num_measurements = innovations.dim().0;
        let state_dim = states.dim();
        let updated_states_dim = [num_measurements, state_dim.0, state_dim.1];
        let mut broadcast_states_updated = states
            .broadcast(updated_states_dim)
            .unwrap()
            .into_owned();
        for ((mut state_updated, kalman_gain), innovation) in broadcast_states_updated
            .outer_iter_mut()
            .zip(kalman_gains.outer_iter())
            .zip(innovations.outer_iter())
        {
            let result = kalman_gain.dot(&innovation.t()).t().into_owned();
            state_updated.add_assign(&result);
        }
        broadcast_states_updated
    }

    fn update_covariances(&self, covariances: &Array3<f64> ,kalman_gains: &Array3<f64>) -> Array3<f64> {
        let mut updated_covariances = covariances.clone();
        let identity = Array2::eye(self.observation_matrix.dim().0);
        for (mut covariance, kalman_gain) in updated_covariances.outer_iter_mut().zip(kalman_gains.outer_iter()) {
            let result = (&identity - &kalman_gain.dot(&self.observation_matrix)).dot(&covariance);
            covariance.assign(&result);
        }
        updated_covariances
    }
}
