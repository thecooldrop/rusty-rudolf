//! This module contains the implementation of Kalman filtering algorithms, as well as traits
//! which are used to encapsulate the functionality of filtering algorithms.
use super::filter_traits::Filter;
use super::kalman_common::*;
use cauchy::Scalar;
use ndarray::{Array2, Array3, ArrayBase, Axis, Data, ErrorKind, Ix2, Ix3, ShapeError};
use ndarray_linalg::lapack::Lapack;
use ndarray_linalg::InverseC;
use std::ops::{AddAssign, SubAssign};

/// Basic linear Kalman filtering algorithm
///
/// This type encapsulates basic linear Kalman filtering algorithm. This implementation depends
/// on four arrays, which represent the operational parameters necessary for this algorithm.
/// As some assumption have to hold with respect to dimensions of arrays, please use the associated
/// function `KalmanFilter::new` to initialize an instance of Kalman filter.
///
/// Type parameter `T: Scalar + Lapack` is used to indicate that Kalman filter can contain any
/// matrices, which are considered to contain numbers ( i.e real or complex numbers ).
pub struct KalmanFilter<T: Scalar + Lapack> {
    transition_matrix: Array2<T>,
    observation_matrix: Array2<T>,
    transition_covariance: Array2<T>,
    observation_covariance: Array2<T>,
}

/// Implementation of filtering methods for Kalman filter
impl<T: Scalar + Lapack> Filter<T> for KalmanFilter<T> {
    /// Kalman filter produces predictions, which are 2-tuples. First element of the tuple is
    /// an array representing predicted states, while second element represents the predicted
    /// an array of matrices representing predicted covariances.
    type Prediction = (Array2<T>, Array3<T>);

    /// Updated values produced by Kalman filter are represented by two three-dimensional arrays.
    /// The first array in tuple represents the updated states. The (i,j)-th row in first matrix in
    /// tuple represents j-th state row updated with i-th measurement row. The j-th matrix in
    /// second matrix in tuple represents the covariance matrices for all updated values of j-th
    /// states ( or pedantically speaking the j-th matrix in second matrix in tuple represents
    /// the covariance matrix of (i,j)-th rows of updated states, for all i.
    type Update = (Array3<T>, Array3<T>);

    fn predict<A: Data<Elem = T>, B: Data<Elem = T>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
    ) -> Self::Prediction {
        let predicted_states = self.transition_matrix.dot(&states.t()).t().to_owned();
        let predicted_covariances = quadratic_form_ix2_ix3_ix2_add_ix2(
            &self.transition_matrix,
            covariances,
            &self.transition_matrix.t(),
            &self.transition_covariance,
        );
        (predicted_states, predicted_covariances)
    }

    fn update<A: Data<Elem = T>, B: Data<Elem = T>, C: Data<Elem = T>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> (Array3<T>, Array3<T>) {
        let expected_measurements = self.observation_matrix.dot(&states.t()).t().into_owned();
        let innovations = pairwise_difference(measurements, &expected_measurements);
        let l_matrices = broad_dot_ix3_ix2(&covariances, &self.observation_matrix.t());
        let u_matrices = innovation_covariances_ix2(
            &self.observation_matrix,
            &self.observation_covariance,
            &l_matrices,
        );
        let mut u_matrices_inv = invc_all_ix3(&u_matrices);
        let kalman_gains = broad_dot_ix3_ix3(&l_matrices, &u_matrices_inv);
        let updated_states = self.update_states(states, &kalman_gains, &innovations);
        let updated_covs = self.update_covariances(covariances, &kalman_gains, &l_matrices);
        (updated_states, updated_covs)
    }
}

impl<T: Scalar + Lapack> KalmanFilter<T> {
    /// Creates new Kalman filter with given matrices
    ///
    /// This constructor expects following conditions to hold:
    /// * covariance matrices should be square
    /// * outer dimensions of observation and transition matrix are equal. This ensures that
    /// given observation matrix can be used as left factor in multiplication with states.
    /// * inner dimensions of observation matrix and observation covariance are equal
    ///
    /// If any of the conditions is not upheld, then the return value is the error variant,
    /// otherwise a well-formed Kalman filter is returned.
    pub fn new<A: Data<Elem = T>>(
        transition_matrix: &ArrayBase<A, Ix2>,
        observation_matrix: &ArrayBase<A, Ix2>,
        transition_covariance: &ArrayBase<A, Ix2>,
        observation_covariance: &ArrayBase<A, Ix2>,
    ) -> Result<KalmanFilter<T>, ShapeError> {
        KalmanFilter::check_dimension_compatibilities(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
        )?;

        let kalman_filter = KalmanFilter {
            transition_matrix: transition_matrix.to_owned(),
            observation_matrix: observation_matrix.to_owned(),
            transition_covariance: transition_covariance.to_owned(),
            observation_covariance: observation_covariance.to_owned(),
        };

        Result::Ok(kalman_filter)
    }

    fn check_dimension_compatibilities<A: Data<Elem = T>>(
        transition_matrix: &ArrayBase<A, Ix2>,
        observation_matrix: &ArrayBase<A, Ix2>,
        transition_covariance: &ArrayBase<A, Ix2>,
        observation_covariance: &ArrayBase<A, Ix2>,
    ) -> Result<(), ShapeError> {
        let transition_matrix_dim = transition_matrix.dim();
        let observation_matrix_dim = observation_matrix.dim();
        let observation_covariance_dim = observation_covariance.dim();

        if observation_matrix_dim.1 != transition_matrix_dim.1 {
            return Result::Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        if observation_matrix_dim.0 != observation_covariance_dim.0 {
            return Result::Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        Self::is_not_square(transition_matrix)?;
        Self::is_not_square(transition_covariance)?;
        Self::is_not_square(observation_covariance)?;

        Result::Ok(())
    }

    fn is_not_square<A: Data<Elem = T>>(arr: &ArrayBase<A, Ix2>) -> Result<(), ShapeError> {
        let arr_dim = arr.dim();
        if arr_dim.0 != arr_dim.1 {
            return Result::Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        Ok(())
    }

    fn update_states<A: Data<Elem = T>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        kalman_gains: &Array3<T>,
        innovations: &Array3<T>,
    ) -> Array3<T> {
        let num_measurements = innovations.dim().0;
        let state_dim = states.dim();
        let updated_states_dim = [num_measurements, state_dim.0, state_dim.1];
        let mut broadcast_states_updated =
            states.broadcast(updated_states_dim).unwrap().into_owned();
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

    fn update_covariances<A: Data<Elem = T>>(
        &self,
        covariances: &ArrayBase<A, Ix3>,
        kalman_gains: &Array3<T>,
        l_matrices: &Array3<T>,
    ) -> Array3<T> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_covariance_matrix_has_to_be_square() -> Result<(), String> {
        let kf = KalmanFilter::<f64>::new(
            &Array2::eye(8),
            &Array2::eye(8),
            &Array2::ones([8, 7]),
            &Array2::eye(8),
        );
        match kf {
            Result::Err(_) => return Result::Ok(()),
            _ => return Result::Err(
                "Kalman filter can not accept non-square matrix as transition covariance matrix"
                    .to_string(),
            ),
        };
    }

    #[test]
    fn observation_covariance_matrix_has_to_be_square() -> Result<(), String> {
        let kf = KalmanFilter::<f64>::new(
            &Array2::eye(8),
            &Array2::eye(8),
            &Array2::eye(8),
            &Array2::ones([8, 7]),
        );
        match kf {
            Result::Err(_) => return Result::Ok(()),
            _ => return Result::Err(
                "Kalman filter can not accept non-square matrix as observation covariance matrix"
                    .to_string(),
            ),
        };
    }

    #[test]
    fn transition_matrix_has_to_be_square() -> Result<(), String> {
        let kf = KalmanFilter::<f64>::new(
            &Array2::ones([7, 8]),
            &Array2::eye(8),
            &Array2::eye(8),
            &Array2::eye(8),
        );
        match kf {
            Result::Err(_) => return Result::Ok(()),
            _ => {
                return Result::Err(
                    "Kalman filter can not accept non-square matrix as transition matrix"
                        .to_string(),
                )
            }
        };
    }

    #[test]
    fn outer_dimensions_of_transition_and_observation_matrix_have_to_match() -> Result<(), String> {
        let eye8 = &Array2::eye(8);
        let eye7 = &Array2::eye(7);
        let kf = KalmanFilter::<f64>::new(eye8, eye7, eye8, eye8);
        match kf {
            Result::Err(_) => return Result::Ok(()),
            _ => {
                return Result::Err(
                    "Outer dimensions of transition and observation matrix should have to be equal"
                        .to_string(),
                )
            }
        }
    }

    #[test]
    fn inner_dimensions_of_observation_matrix_and_observation_covariance_have_to_match(
    ) -> Result<(), String> {
        let eye8 = &Array2::eye(8);
        let eye7 = &Array2::eye(7);
        let kf = KalmanFilter::<f64>::new(eye8, eye8, eye8, eye7);
        match kf {
            Result::Err(_) => return Result::Ok(()),
            _ => return Result::Err("Inner dimensions of observation covariance matrix and observation matrix should have to be equal".to_string())
        }
    }

    #[test]
    fn dimension_of_returned_prediction_matches_dimensions_of_input() -> Result<(), String> {
        let eye8 = &Array2::eye(8);
        let kf = KalmanFilter::<f64>::new(eye8, eye8, eye8, eye8).unwrap();
        let states = Array2::ones([100, 8]);
        let covariances = Array3::ones([100, 8, 8]);
        let (predicted_states, predicted_covariances) = kf.predict(&states, &covariances);
        if states.dim().eq(&predicted_states.dim())
            && covariances.dim().eq(&predicted_covariances.dim())
        {
            Ok(())
        } else {
            Err(
                "Predicted states and covariances do not have same dimensions as inputs"
                    .to_string(),
            )
        }
    }

    #[test]
    fn dimensions_of_updates_are_multipled_by_number_of_measurements() -> Result<(), String> {
        let eye8 = &Array2::eye(8);
        let kf = KalmanFilter::<f64>::new(eye8, eye8, eye8, eye8).unwrap();
        let states = Array2::ones([100, 8]);
        let covariances = Array3::ones([100, 8, 8]);
        let measurements = Array2::ones([10, 8]);
        let (updated_states, updated_covs) = kf.update(&states, &covariances, &measurements);

        let expected_dimension_state = (10, 100, 8);
        let expected_dimension_covs = (100, 8, 8);

        if updated_states.dim().eq(&expected_dimension_state)
            && updated_covs.dim().eq(&expected_dimension_covs)
        {
            Ok(())
        } else {
            Err(
                "The dimension of updated states and covariances does not match the expectation"
                    .to_string(),
            )
        }
    }
}
