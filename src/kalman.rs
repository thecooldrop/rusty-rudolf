//! This module contains the implementation of Kalman filtering algorithms, as well as traits
//! which are used to encapsulate the functionality of filtering algorithms.
use std::ops::{AddAssign, SubAssign, Deref};
use cauchy::Scalar;
use ndarray::{Array2, Array3, ArrayBase, Axis, Data, ErrorKind, Ix2, Ix3, ShapeError, Array4};
use ndarray_linalg::InverseC;
use ndarray_linalg::lapack::Lapack;


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

/// Filtering algorithm trait
///
/// This trait indicates that implementor is a representation of a filtering algorithm, and
/// that it performs filtering operations on inputs of type `T: Scalar + Lapack`.
/// It can be used to represent general filtering algorithms, which are usually split into
/// prediction and update steps.
pub trait Filter<T: Scalar + Lapack> {
    /// Result of prediction operation executed on states
    type Prediction;
    /// Result of update operation executed on states
    type Update;

    /// Prediction operation executed by filtering algorithm.
    ///
    /// The prediction operation result produces the predicted values for states and their
    /// associated covariances. Usually the result of prediction is a 2-tuple of n-arrays, whose
    /// entries usually represent predicted states and covariances respectfully.
    ///
    /// Note that this method has two parameters, whose meanings are as follows:
    /// * states - A two-dimensional array of numbers, where each row represents a state
    /// * covariances - A three-dimensional array of numbers, where each entry along first axis
    /// represents a covariance matrix
    ///
    /// The generic parameters on this method indicate that it is applicable to any combination of
    /// ndarray arrays, which own their data.
    fn predict<A: Data<Elem=T>, B: Data<Elem=T>>(&self, states: &ArrayBase<A, Ix2>, covariances: &ArrayBase<B, Ix3>) -> Self::Prediction;

    /// Update operation executed by filtering algorithm.
    ///
    /// The update operation result produces the updates values for states and their
    /// associated covariances. Usually the result of update is a 2-tuple of nd-arrays, whose
    /// entries usually represent updates states and covariances respectfully.
    ///
    /// This method has three parameters:
    /// * states - A two-dimensional array of numbers, where each row represents a state
    /// * covariances - A three-dimensional array of numbers, where each entry along first axis
    /// represents a covariance matrix
    /// * mesurements - A two-dimensional array of numbers, where each row represents a measurement
    /// of a state
    ///
    /// It is expected that number of entries along first axis of states and covariances is equal,
    /// roughly speaking we expect that there is same number of state and covariance matrices given.
    /// The i-th state row has covariance matrix given by i-th entry of covariances matrix.
    ///
    /// This method is expected to update each (state,covariance) pair with all of the measurements
    fn update<A: Data<Elem=T>, B: Data<Elem=T>, C: Data<Elem=T>>(&self, states: &ArrayBase<A, Ix2>, covariances: &ArrayBase<B, Ix3>, measurements: &ArrayBase<C, Ix2>) -> Self::Update;
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

    fn predict<A: Data<Elem=T>, B: Data<Elem=T>>(&self, states: &ArrayBase<A, Ix2>, covariances: &ArrayBase<B, Ix3>) -> Self::Prediction {
        let predicted_states = self.transition_matrix.dot(&states.t()).t().to_owned();
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

    fn update<A: Data<Elem=T>, B: Data<Elem=T>, C: Data<Elem=T>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> (Array3<T>, Array3<T>) {
        let expected_measurements = self.observation_matrix.dot(&states.t()).t().into_owned();
        let innovations = self.innovations(measurements, &expected_measurements);
        let l_matrices = self.l_matrices(covariances);
        let u_matrices = self.u_matrices(&l_matrices);
        let mut u_matrices_inv = u_matrices.to_owned();
        for mut elem in u_matrices_inv.outer_iter_mut() {
            elem.assign(&elem.invc().unwrap());
        }
        let kalman_gains = self.kalman_gains(&l_matrices, &u_matrices_inv);
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
    pub fn new<A: Data<Elem=T>>(transition_matrix: &ArrayBase<A, Ix2>,
                                observation_matrix: &ArrayBase<A, Ix2>,
                                transition_covariance: &ArrayBase<A, Ix2>,
                                observation_covariance: &ArrayBase<A, Ix2>) -> Result<KalmanFilter<T>, ShapeError>
    {
        KalmanFilter::check_dimension_compatibilities(transition_matrix,
                                                      observation_matrix,
                                                      transition_covariance,
                                                      observation_covariance)?;

        let kalman_filter = KalmanFilter {
            transition_matrix: transition_matrix.to_owned(),
            observation_matrix: observation_matrix.to_owned(),
            transition_covariance: transition_covariance.to_owned(),
            observation_covariance: observation_covariance.to_owned(),
        };

        Result::Ok(kalman_filter)
    }

    fn check_dimension_compatibilities<A: Data<Elem=T>>(transition_matrix: &ArrayBase<A, Ix2>,
                                                        observation_matrix: &ArrayBase<A, Ix2>,
                                                        transition_covariance: &ArrayBase<A, Ix2>,
                                                        observation_covariance: &ArrayBase<A, Ix2>) -> Result<(), ShapeError>
    {
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

    fn is_not_square<A: Data<Elem=T>>(arr: &ArrayBase<A, Ix2>) -> Result<(), ShapeError> {
        let arr_dim = arr.dim();
        if arr_dim.0 != arr_dim.1 {
            return Result::Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        Ok(())
    }

    fn l_matrices<A: Data<Elem=T>>(&self, covariances: &ArrayBase<A, Ix3>) -> Array3<T> {
        let cov_dims = covariances.dim();
        let l_matrix_dim = [cov_dims.0, cov_dims.1, self.observation_matrix.dim().1];
        let mut l_matrices = Array3::zeros(l_matrix_dim);
        let observation_matrix_transpose = self.observation_matrix.t();
        for (mut elem, cov) in l_matrices.outer_iter_mut().zip(covariances.outer_iter()) {
            elem.assign(&cov.dot(&observation_matrix_transpose))
        }
        l_matrices
    }

    fn u_matrices(&self, l_matrices: &Array3<T>) -> Array3<T> {
        let l_dim = l_matrices.dim();
        let r_dim = self.observation_covariance.dim();
        let u_dim = [l_dim.0, r_dim.0, r_dim.1];
        let mut u_matrices = self.observation_covariance.to_owned().broadcast(u_dim).unwrap().to_owned();
        for (mut u, l) in u_matrices.outer_iter_mut().zip(l_matrices.outer_iter()) {
            u.add_assign(&self.observation_matrix.dot(&l));
        }
        u_matrices
    }

    fn innovations<A: Data<Elem=T>>(&self, measurements: &ArrayBase<A, Ix2>, expected_measurements: &Array2<T>) -> Array3<T> {
        let meas_dim = measurements.dim();
        let state_count = expected_measurements.dim().0;
        let innovation_dim = [meas_dim.0, state_count, meas_dim.1];
        let mut innovations: Array3<T> = measurements.to_owned().insert_axis(Axis(1)).broadcast(innovation_dim).unwrap().into_owned();
        for mut inno_view in innovations.outer_iter_mut()
        {
            inno_view.sub_assign(expected_measurements);
        }
        innovations
    }

    fn kalman_gains(&self, l_matrices: &Array3<T>, u_matrices_inv: &Array3<T>) -> Array3<T> {
        let l_dim = l_matrices.dim();
        let u_dim = u_matrices_inv.dim();
        let kalman_gain_dim = [l_dim.0, l_dim.1, u_dim.1];
        let mut kalman_gains = Array3::zeros(kalman_gain_dim);
        for ((mut elem, l), u) in kalman_gains.outer_iter_mut().zip(l_matrices.outer_iter()).zip(u_matrices_inv.outer_iter()) {
            elem.assign(&l.dot(&u));
        }
        kalman_gains
    }

    fn update_states<A: Data<Elem=T>>(&self, states: &ArrayBase<A, Ix2>, kalman_gains: &Array3<T>, innovations: &Array3<T>) -> Array3<T> {
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

    fn update_covariances<A: Data<Elem=T>>(&self, covariances: &ArrayBase<A, Ix3>, kalman_gains: &Array3<T>, l_matrices: &Array3<T>) -> Array3<T> {
        let mut updated_covariances = covariances.to_owned();
        for ((mut elem, kalman_gain), l) in updated_covariances.outer_iter_mut()
            .zip(kalman_gains.outer_iter())
            .zip(l_matrices.outer_iter()) {
            let intermediate = &kalman_gain.dot(&l.t());
            elem.sub_assign(&intermediate);
        }
        updated_covariances
    }
}
