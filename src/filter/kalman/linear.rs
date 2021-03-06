//! This module contains the implementation of Kalman filtering algorithms, as well as traits
//! which are used to encapsulate the functionality of filtering algorithms.

use cauchy::Scalar;
use ndarray::{Array2, Array3, ArrayBase, Axis, Data, ErrorKind, Ix2, Ix3, ShapeError, Zip};
use ndarray_linalg::lapack::Lapack;
use ndarray::linalg::general_mat_mul;
use crate::filter::kalman_common::{pairwise_difference, broad_dot_ix3_ix2, innovation_covariances_ix2, invc_all_ix3, broad_dot_ix3_ix3, update_states, update_covariance};
use crate::filter::Filter;

/// Basic linear Kalman filtering algorithm
///
/// This type encapsulates basic linear Kalman filtering algorithm. This implementation depends
/// on four arrays, which represent the operational parameters necessary for this algorithm.
/// As some assumption have to hold with respect to dimensions of arrays, please use the associated
/// function `KalmanFilter::new` to initialize an instance of Kalman filter.
///
/// Type parameter `T: Scalar + Lapack` is used to indicate that Kalman filter can contain any
/// matrices, which are considered to contain numbers ( i.e real or complex numbers ).
/// ```
/// use ndarray::{Array2, Array3, Axis};
/// use rusty_rudolf::filter::traits::Filter;
/// use rusty_rudolf::filter::kalman::linear::KalmanFilter;
/// let identity = Array2::<f64>::eye(8);
/// let kalman_filter = KalmanFilter::new(&identity, &identity, &identity, &identity).unwrap();
/// let states = Array2::eye(8);
/// let covariances = Array2::eye(8)
///     .insert_axis(Axis(0))
///     .broadcast([8,8,8])
///     .unwrap()
///     .to_owned();
/// let (predicted_states, predicted_covariances) = kalman_filter.predict(&states, &covariances);
/// assert_eq!(predicted_states.dim(), (8,8));
/// assert_eq!(predicted_covariances.dim(), (8,8,8));
///
/// let measurements = Array2::ones([5,8]);
/// let (updated_states, updated_covariances) = kalman_filter.update(&predicted_states, &predicted_covariances, &measurements);
/// assert_eq!(updated_states.dim(), (8,5,8));
/// assert_eq!(updated_covariances.dim(), (8,8,8));
/// ```
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
    /// tuple represents i-th state row updated with j-th measurement row. The i-th matrix in
    /// second matrix in tuple represents the covariance matrices for all updated values of i-th
    /// states ( or pedantically speaking the i-th matrix in second matrix in tuple represents
    /// the covariance matrix of (i,j)-th rows of updated states, for all j ).
    type Update = (Array3<T>, Array3<T>);

    /// This function computes prediction of given states and covariances according to regular
    /// Kalman filtering algorithm.
    /// ```
    /// use ndarray::{Array2, Axis};
    /// use rusty_rudolf::filter::traits::Filter;
    /// use rusty_rudolf::filter::kalman::linear::KalmanFilter;
    /// let eye = Array2::eye(8);
    /// let kf = KalmanFilter::<f64>::new(&eye, &eye, &eye, &eye).unwrap();
    /// let covariances = eye.view().insert_axis(Axis(0)).broadcast((8,8,8)).unwrap().to_owned();
    /// let (updated_states, updated_covariances) = kf.predict(&eye, &covariances);
    /// ```
    fn predict<A: Data<Elem = T>, B: Data<Elem = T>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
    ) -> Self::Prediction {
        let predicted_states = self.transition_matrix.dot(&states.t()).t().to_owned();
        let mut predicted_covariances = self
            .transition_covariance
            .view()
            .insert_axis(Axis(0))
            .broadcast(covariances.raw_dim())
            .unwrap()
            .to_owned();
        Zip::from(predicted_covariances.outer_iter_mut())
            .and(covariances.outer_iter())
            .apply(|mut out, cov| {
                let left_multiple = self.transition_matrix.dot(&cov);
                general_mat_mul(
                    T::one(),
                    &left_multiple,
                    &self.transition_matrix.t(),
                    T::one(),
                    &mut out,
                );
            });
        (predicted_states, predicted_covariances)
    }

    /// Updates given states and covariances with given measurements according to regular Kalman
    /// filtering algorithm. Note that this method computes the update of every state with every
    /// measurement, and is highly optimized to avoid unncessary computations.
    ///
    /// Method gives two outputs, where both are of type `Array3<T>`. The first output represents
    /// prediction of states, while second represents covariances. Note that covariances do not
    /// depend on measurement and thus need only be computed once per state. Here is an example.
    ///
    /// ```
    /// use ndarray::{Array2, Axis};
    /// use rusty_rudolf::filter::traits::Filter;
    /// use rusty_rudolf::filter::kalman::linear::KalmanFilter;
    /// let eye = Array2::eye(8);
    /// let kf = KalmanFilter::<f64>::new(&eye, &eye, &eye, &eye).unwrap();
    /// let states = Array2::eye(8);
    /// let covariances = Array2::eye(8).insert_axis(Axis(0)).broadcast([8,8,8]).unwrap().to_owned();
    /// let measurements = Array2::ones([10,8]);
    ///
    /// let (updated_states, update_covariances) = kf.update(&states, &covariances, &measurements);
    /// // First dimension indexed by state, second by measurement, and rows represent outputs
    /// // thus for each of input states ( of which there are 8 ), there are 10 updates ( one for
    /// // each of the measurements )
    /// assert_eq!(updated_states.dim(), (8, 10, 8));
    /// // First dimension indexed by state. Thus first matrix [i, :, :] represents the covariance
    /// // for all of the states stored in [i, :, :] updated states.
    /// assert_eq!(update_covariances.dim(), (8, 8, 8));
    /// for due_to_state in updated_states.outer_iter() {
    ///     // Here we are iterating over resulting updated states which resulted from states
    ///     for due_to_measurement in due_to_state.outer_iter() {
    ///         // Here we are iterating over updated states due to measurements
    ///         continue;
    ///     }
    /// }
    /// ```
    fn update<A: Data<Elem = T>, B: Data<Elem = T>, C: Data<Elem = T>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> (Array3<T>, Array3<T>) {
        let expected_measurements = {
            let mut measurements = self.observation_matrix.dot(&states.t());
            measurements.swap_axes(1, 0);
            measurements
        };
        let innovations_negated =
            pairwise_difference(&expected_measurements, measurements).unwrap();
        let l_matrices = broad_dot_ix3_ix2(&covariances, &self.observation_matrix.t()).unwrap();
        let u_matrices = innovation_covariances_ix2(
            &self.observation_matrix,
            &self.observation_covariance,
            &l_matrices,
        )
        .unwrap();
        let u_matrices_inv = invc_all_ix3(&u_matrices).unwrap();
        let kalman_gains = broad_dot_ix3_ix3(&l_matrices, &u_matrices_inv).unwrap();
        let updated_states = update_states(&states, &kalman_gains, &innovations_negated).unwrap();
        let updated_covs = update_covariance(covariances, &kalman_gains, &l_matrices).unwrap();
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn transition_covariance_matrix_has_to_be_square() -> Result<(), String> {
        let kf = KalmanFilter::<f64>::new(
            &Array2::eye(8),
            &Array2::eye(8),
            &Array2::ones([8, 7]),
            &Array2::eye(8),
        );
        return match kf {
            Result::Err(_) => Result::Ok(()),
            _ => Result::Err(
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
        return match kf {
            Result::Err(_) => Result::Ok(()),
            _ => Result::Err(
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
        return match kf {
            Result::Err(_) => Result::Ok(()),
            _ => Result::Err(
                "Kalman filter can not accept non-square matrix as transition matrix".to_string(),
            ),
        };
    }

    #[test]
    fn outer_dimensions_of_transition_and_observation_matrix_have_to_match() -> Result<(), String> {
        let eye8 = &Array2::eye(8);
        let eye7 = &Array2::eye(7);
        let kf = KalmanFilter::<f64>::new(eye8, eye7, eye8, eye8);
        return match kf {
            Result::Err(_) => Result::Ok(()),
            _ => Result::Err(
                "Outer dimensions of transition and observation matrix should have to be equal"
                    .to_string(),
            ),
        };
    }

    #[test]
    fn inner_dimensions_of_observation_matrix_and_observation_covariance_have_to_match(
    ) -> Result<(), String> {
        let eye8 = &Array2::eye(8);
        let eye7 = &Array2::eye(7);
        let kf = KalmanFilter::<f64>::new(eye8, eye8, eye8, eye7);
        return match kf {
            Result::Err(_) => Result::Ok(()),
            _ => Result::Err("Inner dimensions of observation covariance matrix and observation matrix should have to be equal".to_string())
        };
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

        let expected_dimension_state = (100, 10, 8);
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

    #[test]
    fn prediction_has_correct_output() -> Result<(), String> {
        let eye4 = Array2::eye(4);
        let transition_matrix = 2.0 * &eye4;
        let kf = KalmanFilter::<f64>::new(&transition_matrix, &eye4, &eye4, &eye4).unwrap();
        let states = Array2::ones([2, 4]);
        let covariances = Array2::eye(4).broadcast([2, 4, 4]).unwrap().to_owned();
        let (predicted_states, predicted_covariances) = kf.predict(&states, &covariances);

        let expected_states = arr2(&[[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]);
        let expected_covariances = arr3(&[
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
            ],
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
            ],
        ]);
        assert_eq!(predicted_states, expected_states);
        assert_eq!(predicted_covariances, expected_covariances);
        Ok(())
    }
}
