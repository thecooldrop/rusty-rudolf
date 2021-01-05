use std::ops::{AddAssign, SubAssign};
use cauchy::Scalar;
use ndarray::{Array2, Array3, ArrayBase, Axis, Data, ErrorKind, Ix2, Ix3, ShapeError};
use ndarray_linalg::InverseC;
use ndarray_linalg::lapack::Lapack;

pub struct KalmanFilter<T: Scalar + Lapack> {
    transition_matrix: Array2<T>,
    observation_matrix: Array2<T>,
    transition_covariance: Array2<T>,
    observation_covariance: Array2<T>,
}

pub trait Filter<T: Scalar + Lapack> {
    type Prediction;
    type Update;
    fn predict<A: Data<Elem=T>, B: Data<Elem=T>>(&self, states: &ArrayBase<A, Ix2>, covariances: &ArrayBase<B, Ix3>) -> Self::Prediction;
    fn update<A: Data<Elem=T>, B: Data<Elem=T>, C: Data<Elem=T>>(&self, states: &ArrayBase<A, Ix2>, covariances: &ArrayBase<B, Ix3>, measurements: &ArrayBase<C, Ix2>) -> Self::Update;
}

impl<T: Scalar + Lapack> Filter<T> for KalmanFilter<T> {
    type Prediction = (Array2<T>, Array3<T>);
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
