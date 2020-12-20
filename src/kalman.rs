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
        let l_matrices = self.l_matrices(covariances);
        let u_matrices = self.u_matrices(&l_matrices);
        let mut u_matrices_inv = u_matrices.clone();
        for mut elem in u_matrices_inv.outer_iter_mut() {
            elem.assign(&elem.invc().unwrap());
        }
        let kalman_gains = self.kalman_gains(&l_matrices, &u_matrices_inv);
        let updated_states = self.update_states(states, &kalman_gains, &innovations); 
        let updated_covs = self.update_covariances(covariances, &kalman_gains, &l_matrices);
        return (updated_states, updated_covs);
    }

    fn l_matrices(&self, covariances: &Array3<f64>) -> Array3<f64> {
        let cov_dims = covariances.dim();
        let l_matrix_dim = [cov_dims.0, cov_dims.1, self.observation_matrix.dim().1];
        let mut l_matrices = Array3::zeros(l_matrix_dim);
        let observation_matrix_transpose = self.observation_matrix.t();
        for (mut elem, cov) in l_matrices.outer_iter_mut().zip(covariances.outer_iter()) {
            elem.assign(&cov.dot(&observation_matrix_transpose))
        }
        l_matrices
    }

    fn u_matrices(&self, l_matrices: &Array3<f64>) -> Array3<f64> {
        let l_dim = l_matrices.dim();
        let r_dim = self.observation_covariance.dim();
        let u_dim = [l_dim.0, r_dim.0, r_dim.1];
        let mut u_matrices = self.observation_covariance.clone().broadcast(u_dim).unwrap().to_owned();
        for (mut u, l) in u_matrices.outer_iter_mut().zip(l_matrices.outer_iter()) {
            u.add_assign(&self.observation_matrix.dot(&l));
        }
        u_matrices
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

    fn kalman_gains(&self, l_matrices: &Array3<f64>, u_matrices_inv: &Array3<f64>) -> Array3<f64> {
        let l_dim = l_matrices.dim();
        let u_dim = u_matrices_inv.dim();
        let kalman_gain_dim = [l_dim.0, l_dim.1, u_dim.1];
        let mut kalman_gains = Array3::zeros(kalman_gain_dim);
        for ((mut elem, l), u) in kalman_gains.outer_iter_mut().zip(l_matrices.outer_iter()).zip(u_matrices_inv.outer_iter()) {
            elem.assign(&l.dot(&u));
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

    fn update_covariances(&self, covariances: &Array3<f64> ,kalman_gains: &Array3<f64>, l_matrices: &Array3<f64>) -> Array3<f64> {
        let mut updated_covariances = covariances.clone();
        for ((mut elem, kalman_gain), l) in updated_covariances.outer_iter_mut().zip(kalman_gains.outer_iter()).zip(l_matrices.outer_iter()) {
            let intermediate = &kalman_gain.dot(&l.t());
            elem.sub_assign(intermediate);
        }
        updated_covariances
    }
}
