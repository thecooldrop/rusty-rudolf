pub mod kalman {
    use ndarray::{Array2, Array3, Axis, ArrayViewMut3, ArrayViewMut2};
    use ndarray_linalg::Inverse;
    use std::ops::AddAssign;

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
                let mut predicted_covariances_iter = predicted_covariances.outer_iter_mut();
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
        ) -> (Array2<f64>, Array3<f64>) {

            let state_dim = states.dim();
            let cov_dim = covariances.dim();
            let meas_dim = measurements.dim();
            let inno_broad_dim = [meas_dim.0, state_dim.0, meas_dim.1];
            let expected_measurements = self.observation_matrix.dot(&states.t()).t().into_owned();

            let mut innovations : Array3<f64> = Array3::zeros(inno_broad_dim);
            for (mut inno_view, meas_view) in innovations.outer_iter_mut().zip(measurements.outer_iter()) {
                inno_view.assign(&(meas_view.into_owned().insert_axis(Axis(0)).broadcast([state_dim.0, meas_dim.1]).unwrap().into_owned() - &expected_measurements));
            }

            let mut innovation_covariances : Array3<f64>= Array3::zeros([cov_dim.0, meas_dim.1, meas_dim.1]);
            for (mut inno_mat_view, cov_view) in innovation_covariances.outer_iter_mut().zip(covariances.outer_iter()) {
                inno_mat_view.assign(&(self.observation_matrix.dot(&cov_view).dot(&self.observation_matrix.t()) + &self.observation_covariance));
            }
            
            let mut inv_inno_covs = innovation_covariances.clone();
            for mut elem in inv_inno_covs.outer_iter_mut() {
                elem.assign(&elem.inv().unwrap());
            }

            let mut kalman_gains = Array3::zeros([cov_dim.0, cov_dim.1, meas_dim.1]);
            for ((mut kg_view, cov), inv_icov) in kalman_gains.outer_iter_mut().zip(covariances.outer_iter()).zip(inv_inno_covs.outer_iter()) {
                kg_view.assign(&(cov.dot(&self.observation_matrix.t()).dot(&inv_icov)))
            }

            let mut broadcast_states_updated = states.to_owned().insert_axis(Axis(0)).broadcast([meas_dim.0, state_dim.0, state_dim.1]).unwrap().into_owned();
            for ((mut state_updated, kalman_gain), innovation) in broadcast_states_updated
                .outer_iter_mut()
                .zip(kalman_gains.outer_iter())
                .zip(innovations.outer_iter())
            {
                let result = kalman_gain.dot(&innovation.t()).t().into_owned();
                state_updated.add_assign(&result);
            }

            let mut broad_updated_covs = covariances.to_owned().insert_axis(Axis(0)).broadcast([meas_dim.0, cov_dim.0, cov_dim.1, cov_dim.2]).unwrap().to_owned();
            let identity : Array2<f64>= Array2::eye(meas_dim.1);
            for ((mut upd_cov, kg), in_cov) in broad_updated_covs
                .outer_iter_mut()
                .zip(kalman_gains.outer_iter())
                .zip(covariances.outer_iter())
            {
                let outer_result = (&identity - &kg.dot(&self.observation_matrix));
                for mut inn_upd_cov in upd_cov.outer_iter_mut() {
                    let result = outer_result.dot(&in_cov);
                    inn_upd_cov.assign(&result);
                }
            }

            return (Array2::zeros([1,1]), Array3::zeros([1,1,1]));
        }
    }

    mod tests {

        use ndarray::{Array2, Array3};
        use super::KalmanFilterWithoutControl;

        #[test]
        fn bench_kalman_predict() {
            let states = Array2::zeros([50,8]);
            let covariances = Array3::zeros([50, 8, 8]);
            let kf = KalmanFilterWithoutControl {
                transition_matrix: Array2::eye(8),
                transition_covariance: Array2::eye(8),
                observation_matrix: Array2::eye(8),
                observation_covariance: Array2::eye(8)
            };
            kf.predict(&states, &covariances);
        }

        #[test]
        fn bench_kalman_update() {
            let states: Array2<f64> = Array2::zeros([50,8]);
            let covariances: Array3<f64> = Array3::zeros([50, 8, 8]);
            let measurements: Array2<f64> = Array2::zeros([50,8]);
            let kf = KalmanFilterWithoutControl {
                transition_matrix: Array2::eye(8),
                transition_covariance: Array2::eye(8),
                observation_matrix: Array2::eye(8),
                observation_covariance: Array2::eye(8)
            };
            kf.update(&states, &covariances, &measurements);
        }
    }
}