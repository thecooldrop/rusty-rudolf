//! Here series-based approximations to optimal Kalman filter are considered. These filters
//! are usually known as Kalman filters.

use std::ops::AddAssign;
use cauchy::Scalar;
use ndarray::{Array2, Array3, ArrayBase, ArrayView2, Data, Ix2, Ix3};
use ndarray_linalg::Lapack;

use crate::filter::traits::Filter;
use crate::filter::kalman_common::{
    broad_dot_ix3_ix3, inv_all_ix3, invc_all_ix3, pairwise_difference, quadratic_form_ix3_ix3_ix3,
    update_covariance, update_states,
};

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

impl<Num> AdditiveNoiseExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    /// Creates a new AdditiveNoiseExtendedKalmanFilter. The Kalman filter constructed here
    /// requires four functions as input, where each of the functions plays the role of non-linear
    /// models, as required by standard extended Kalman filtering algorithm.
    ///
    /// An example of how this would be used is as follows:
    ///
    /// ```
    /// // Here we construct a Kalman filter, whose states are given by positions and velocities
    /// // in x and y dimension, while measurements are obtained by radar-like sensor, and provide
    /// // x, y and radial velocity measurements. We assume that transition function is simple
    /// // linear transition function:
    /// use ndarray::{arr2, Axis, Array2, Zip, Array3, ArrayView2};
    /// use rusty_rudolf::filter::kalman::nonlinear::extended::AdditiveNoiseExtendedKalmanFilter;
    /// use rusty_rudolf::filter::traits::Filter;
    /// let transition_matrix = arr2(&[[1.0, 0.0, 1.0, 0.0],
    ///                                [0.0, 1.0, 0.0, 1.0],
    ///                                [0.0, 0.0, 1.0, 0.0],
    ///                                [0.0, 0.0, 0.0, 1.0]]);
    /// let transition_function = {
    ///    let inner_transition_matrix = transition_matrix.clone();
    ///    move |states: &ArrayView2<f64>| {
    ///        let output_states = inner_transition_matrix.dot(&states.t()).t().to_owned();
    ///        output_states
    ///    }
    /// };
    ///
    /// let transition_jacobis = {
    ///    let inner_transition_matrix = transition_matrix.clone();
    ///    move |states: &ArrayView2<f64>| {
    ///        let (states_rows, states_cols) = states.dim();
    ///        inner_transition_matrix.view()
    ///            .insert_axis(Axis(0))
    ///            .broadcast([states_rows, states_cols, states_cols])
    ///            .unwrap()
    ///            .to_owned()
    ///    }
    /// };
    ///
    ///let measurement_function = |states: &ArrayView2<f64>| {
    ///    let (state_rows, state_cols) = states.dim();
    ///    let mut measurements = Array2::zeros([state_rows, 3]);
    ///    let x_states_view = states.slice(ndarray::s![.., 0]);
    ///    let y_states_view = states.slice(ndarray::s![.., 2]);
    ///    let velx_states = states.slice(ndarray::s![.., 1]);
    ///    let vely_states = states.slice(ndarray::s![.., 3]);
    ///
    ///    measurements.slice_mut(ndarray::s![.., 0]).assign(&x_states_view);
    ///    measurements.slice_mut(ndarray::s![.., 1]).assign(&y_states_view);
    ///    let mut vel_r_outputs_view = measurements.slice_mut(ndarray::s![.., 2]);
    ///    Zip::from(&x_states_view)
    ///        .and(&y_states_view)
    ///        .and(&velx_states)
    ///        .and(&vely_states)
    ///        .apply_assign_into(vel_r_outputs_view, |x, y, vel_x, vel_y| {
    ///            let range = f64::sqrt(x*x + y*y);
    ///            let numerator = x*vel_x + y*vel_y;
    ///            numerator / range
    ///    });
    ///    measurements
    /// };
    /// let measurement_jacobis = |states: &ArrayView2<f64>| {
    ///    let x_states_view = states.slice(ndarray::s![.., 0]);
    ///    let y_states_view = states.slice(ndarray::s![.., 2]);
    ///    let x_vel_states_view = states.slice(ndarray::s![.., 1]);
    ///    let y_vel_states_view = states.slice(ndarray::s![.., 3]);
    ///    let (state_rows, state_cols) = states.dim();
    ///    let mut jacobis = Array3::<f64>::zeros([state_rows, 3, state_cols]);
    ///    Zip::from(jacobis.outer_iter_mut())
    ///        .and(x_states_view)
    ///        .and(y_states_view)
    ///        .and(x_vel_states_view)
    ///        .and(y_vel_states_view)
    ///        .apply(|mut jacobi, x, y, velx, vely| {
    ///            let range_sq = x*x + y*y;
    ///            let range = f64::sqrt(range_sq);
    ///            let numerator = x*velx + y*vely;
    ///
    ///            let radial_vel_derivative_x = (velx / range) - x * numerator;
    ///            let radial_vel_derivative_y = (vely / range) - y * numerator;
    ///            let radial_vel_derivative_velx = x / range;
    ///            let radial_vel_derivative_vely = y / range;
    ///
    ///            jacobi[(0,0)] = 1.0;
    ///            jacobi[(1,1)] = 1.0;
    ///            jacobi[(2,0)] = radial_vel_derivative_x;
    ///            jacobi[(2,1)] = radial_vel_derivative_vely;
    ///            jacobi[(2,2)] = radial_vel_derivative_velx;
    ///            jacobi[(2,3)] = radial_vel_derivative_y;
    ///        });
    ///    jacobis
    /// };
    /// let transition_covariance = Array2::<f64>::eye(4);
    /// let measurement_covariance = Array2::<f64>::eye(3);
    /// let ekf = AdditiveNoiseExtendedKalmanFilter::new(Box::new(transition_function),
    ///                                                  Box::new(transition_jacobis),
    ///                                                  Box::new(measurement_function),
    ///                                                  Box::new(measurement_jacobis),
    ///                                                  &transition_covariance,
    ///                                                  &measurement_covariance);
    /// ```
    ///
    pub fn new(
        transition_function: Box<dyn Fn(&ArrayView2<Num>) -> Array2<Num>>,
        transition_function_jacobian: Box<dyn Fn(&ArrayView2<Num>) -> Array3<Num>>,
        measurement_function: Box<dyn Fn(&ArrayView2<Num>) -> Array2<Num>>,
        measurement_function_jacobi: Box<dyn Fn(&ArrayView2<Num>) -> Array3<Num>>,
        transition_covariance: &ArrayBase<impl Data<Elem = Num>, Ix2>,
        measurement_covariance: &ArrayBase<impl Data<Elem = Num>, Ix2>,
    ) -> Self
where {
        AdditiveNoiseExtendedKalmanFilter {
            transition_function,
            transition_function_jacobian,
            measurement_function,
            measurement_function_jacobi,
            transition_covariance: transition_covariance.to_owned(),
            measurement_covariance: measurement_covariance.to_owned(),
        }
    }
}

impl<Num> Filter<Num> for AdditiveNoiseExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    /// Represents the type of output produced by prediction method. The first component of the
    /// tuple represents the predicted states. The i-th row represents the prediction of the
    /// of the i-th input state. The analogous holds for second component, where i-th entry along
    /// first axis represents the prediction of the i-th covariance matrix input.
    type Prediction = (Array2<Num>, Array3<Num>);

    /// Type of output produced by update method. The first component represents a data structure
    /// containing the update of each of input states by each of the input measurements.
    ///
    /// Assume that we have following matrix shapes are given as input states=5x8, covariances=5x8x8
    /// and measurements=10x8. Then the first component of output would be 5x10x8 and second component
    /// would be 5x8x8. Thus for example the slice [1,2,:] of first component would represent
    /// the second state updated by third input measurement, whike [1,:,:] of second component
    /// would represent the covariance matices of all states in slice [1,:,:] in first component.
    type Update = (Array3<Num>, Array3<Num>);


    /// Computes the predicted states and covariances as given by regular extended Kalman filter
    /// equations.
    ///
    /// Assume that AdditiveNoiseExtendedKalmanFilter is constructed as described in documentation
    /// of `AdditiveNoiseExtendedKalmanFilter::new`, then this method can be used as follows
    ///
    /// ```
    /// # use ndarray::{arr2, Axis, Array2, Zip, Array3, ArrayView2};
    /// # use rusty_rudolf::filter::kalman::nonlinear::extended::AdditiveNoiseExtendedKalmanFilter;
    /// # use rusty_rudolf::filter::traits::Filter;
    /// # let transition_matrix = arr2(&[[1.0, 0.0, 1.0, 0.0],
    /// #                               [0.0, 1.0, 0.0, 1.0],
    /// #                               [0.0, 0.0, 1.0, 0.0],
    /// #                               [0.0, 0.0, 0.0, 1.0]]);
    /// # let transition_function = {
    /// #   let inner_transition_matrix = transition_matrix.clone();
    /// #   move |states: &ArrayView2<f64>| {
    /// #       let output_states = inner_transition_matrix.dot(&states.t()).t().to_owned();
    /// #       output_states
    /// #   }
    /// # };
    /// #
    /// # let transition_jacobis = {
    /// #   let inner_transition_matrix = transition_matrix.clone();
    /// #   move |states: &ArrayView2<f64>| {
    /// #       let (states_rows, states_cols) = states.dim();
    /// #       inner_transition_matrix.view()
    /// #           .insert_axis(Axis(0))
    /// #           .broadcast([states_rows, states_cols, states_cols])
    /// #           .unwrap()
    /// #           .to_owned()
    /// #   }
    /// #  };
    ///
    /// # let measurement_function = |states: &ArrayView2<f64>| {
    /// #   let (state_rows, state_cols) = states.dim();
    /// #   let mut measurements = Array2::zeros([state_rows, 3]);
    /// #   let x_states_view = states.slice(ndarray::s![.., 0]);
    /// #   let y_states_view = states.slice(ndarray::s![.., 2]);
    /// #   let velx_states = states.slice(ndarray::s![.., 1]);
    /// #   let vely_states = states.slice(ndarray::s![.., 3]);
    ///
    /// #   measurements.slice_mut(ndarray::s![.., 0]).assign(&x_states_view);
    /// #   measurements.slice_mut(ndarray::s![.., 1]).assign(&y_states_view);
    /// #   let mut vel_r_outputs_view = measurements.slice_mut(ndarray::s![.., 2]);
    /// #   Zip::from(&x_states_view)
    /// #       .and(&y_states_view)
    /// #       .and(&velx_states)
    /// #       .and(&vely_states)
    /// #       .apply_assign_into(vel_r_outputs_view, |x, y, vel_x, vel_y| {
    /// #           let range = f64::sqrt(x*x + y*y);
    /// #           let numerator = x*vel_x + y*vel_y;
    /// #           numerator / range
    /// #   });
    /// #   measurements
    /// # };
    /// # let measurement_jacobis = |states: &ArrayView2<f64>| {
    /// #   let x_states_view = states.slice(ndarray::s![.., 0]);
    /// #   let y_states_view = states.slice(ndarray::s![.., 2]);
    /// #   let x_vel_states_view = states.slice(ndarray::s![.., 1]);
    /// #   let y_vel_states_view = states.slice(ndarray::s![.., 3]);
    /// #   let (state_rows, state_cols) = states.dim();
    /// #   let mut jacobis = Array3::<f64>::zeros([state_rows, 3, state_cols]);
    /// #   Zip::from(jacobis.outer_iter_mut())
    /// #       .and(x_states_view)
    /// #       .and(y_states_view)
    /// #       .and(x_vel_states_view)
    /// #       .and(y_vel_states_view)
    /// #       .apply(|mut jacobi, x, y, velx, vely| {
    /// #           let range_sq = x*x + y*y;
    /// #           let range = f64::sqrt(range_sq);
    /// #           let numerator = x*velx + y*vely;
    /// #
    /// #           let radial_vel_derivative_x = (velx / range) - x * numerator;
    /// #           let radial_vel_derivative_y = (vely / range) - y * numerator;
    /// #           let radial_vel_derivative_velx = x / range;
    /// #           let radial_vel_derivative_vely = y / range;
    /// #
    /// #          jacobi[(0,0)] = 1.0;
    /// #           jacobi[(1,1)] = 1.0;
    /// #           jacobi[(2,0)] = radial_vel_derivative_x;
    /// #           jacobi[(2,1)] = radial_vel_derivative_vely;
    /// #           jacobi[(2,2)] = radial_vel_derivative_velx;
    /// #           jacobi[(2,3)] = radial_vel_derivative_y;
    /// #       });
    /// #   jacobis
    /// # };
    /// # let transition_covariance = Array2::<f64>::eye(4);
    /// # let measurement_covariance = Array2::<f64>::eye(3);
    /// let ekf = AdditiveNoiseExtendedKalmanFilter::new(Box::new(transition_function),
    ///                                                  Box::new(transition_jacobis),
    ///                                                  Box::new(measurement_function),
    ///                                                  Box::new(measurement_jacobis),
    ///                                                  &transition_covariance,
    ///                                                 &measurement_covariance);
    ///
    /// let states = Array2::ones([4,4]);
    /// let covariances = Array2::eye(4)
    ///    .view()
    ///    .insert_axis(Axis(0))
    ///    .broadcast([4,4,4])
    ///    .unwrap()
    ///    .to_owned();
    ///
    /// let (predicted_states, predicted_covariances) = ekf.predict(&states, &covariances);
    /// let measurements = Array2::ones([10,3]);
    /// let (updated_states, updated_covariances) = ekf.update(&predicted_states, &predicted_covariances, &measurements);
    /// ```
    ///
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
        let quadratic_summand = quadratic_form_ix3_ix3_ix3(covariances, &state_jacobis).unwrap();
        predicted_covariances.add_assign(&quadratic_summand);
        (predicted_states, predicted_covariances)
    }

    /// Computes the update of each of the states and covariances with each of the input
    /// measurements as described by regular extended Kalman filter equations.
    ///
    /// Assuming that the filter has been initialized as described in
    /// `AdditiveNoiseExtendedKalmanFilter::new`, then the update can be computed as follows:
    /// ```
    /// # use ndarray::{arr2, Axis, Array2, Zip, Array3, ArrayView2};
    /// # use rusty_rudolf::filter::kalman::nonlinear::extended::AdditiveNoiseExtendedKalmanFilter;
    /// # use rusty_rudolf::filter::traits::Filter;
    /// # let transition_matrix = arr2(&[[1.0, 0.0, 1.0, 0.0],
    /// #                               [0.0, 1.0, 0.0, 1.0],
    /// #                               [0.0, 0.0, 1.0, 0.0],
    /// #                               [0.0, 0.0, 0.0, 1.0]]);
    /// # let transition_function = {
    /// #   let inner_transition_matrix = transition_matrix.clone();
    /// #   move |states: &ArrayView2<f64>| {
    /// #       let output_states = inner_transition_matrix.dot(&states.t()).t().to_owned();
    /// #       output_states
    /// #   }
    /// # };
    /// #
    /// # let transition_jacobis = {
    /// #   let inner_transition_matrix = transition_matrix.clone();
    /// #   move |states: &ArrayView2<f64>| {
    /// #       let (states_rows, states_cols) = states.dim();
    /// #       inner_transition_matrix.view()
    /// #           .insert_axis(Axis(0))
    /// #           .broadcast([states_rows, states_cols, states_cols])
    /// #           .unwrap()
    /// #           .to_owned()
    /// #   }
    /// #  };
    ///
    /// # let measurement_function = |states: &ArrayView2<f64>| {
    /// #   let (state_rows, state_cols) = states.dim();
    /// #   let mut measurements = Array2::zeros([state_rows, 3]);
    /// #   let x_states_view = states.slice(ndarray::s![.., 0]);
    /// #   let y_states_view = states.slice(ndarray::s![.., 2]);
    /// #   let velx_states = states.slice(ndarray::s![.., 1]);
    /// #   let vely_states = states.slice(ndarray::s![.., 3]);
    ///
    /// #   measurements.slice_mut(ndarray::s![.., 0]).assign(&x_states_view);
    /// #   measurements.slice_mut(ndarray::s![.., 1]).assign(&y_states_view);
    /// #   let mut vel_r_outputs_view = measurements.slice_mut(ndarray::s![.., 2]);
    /// #   Zip::from(&x_states_view)
    /// #       .and(&y_states_view)
    /// #       .and(&velx_states)
    /// #       .and(&vely_states)
    /// #       .apply_assign_into(vel_r_outputs_view, |x, y, vel_x, vel_y| {
    /// #           let range = f64::sqrt(x*x + y*y);
    /// #           let numerator = x*vel_x + y*vel_y;
    /// #           numerator / range
    /// #   });
    /// #   measurements
    /// # };
    /// # let measurement_jacobis = |states: &ArrayView2<f64>| {
    /// #   let x_states_view = states.slice(ndarray::s![.., 0]);
    /// #   let y_states_view = states.slice(ndarray::s![.., 2]);
    /// #   let x_vel_states_view = states.slice(ndarray::s![.., 1]);
    /// #   let y_vel_states_view = states.slice(ndarray::s![.., 3]);
    /// #   let (state_rows, state_cols) = states.dim();
    /// #   let mut jacobis = Array3::<f64>::zeros([state_rows, 3, state_cols]);
    /// #   Zip::from(jacobis.outer_iter_mut())
    /// #       .and(x_states_view)
    /// #       .and(y_states_view)
    /// #       .and(x_vel_states_view)
    /// #       .and(y_vel_states_view)
    /// #       .apply(|mut jacobi, x, y, velx, vely| {
    /// #           let range_sq = x*x + y*y;
    /// #           let range = f64::sqrt(range_sq);
    /// #           let numerator = x*velx + y*vely;
    /// #
    /// #           let radial_vel_derivative_x = (velx / range) - x * numerator;
    /// #           let radial_vel_derivative_y = (vely / range) - y * numerator;
    /// #           let radial_vel_derivative_velx = x / range;
    /// #           let radial_vel_derivative_vely = y / range;
    /// #
    /// #          jacobi[(0,0)] = 1.0;
    /// #           jacobi[(1,1)] = 1.0;
    /// #           jacobi[(2,0)] = radial_vel_derivative_x;
    /// #           jacobi[(2,1)] = radial_vel_derivative_vely;
    /// #           jacobi[(2,2)] = radial_vel_derivative_velx;
    /// #           jacobi[(2,3)] = radial_vel_derivative_y;
    /// #       });
    /// #   jacobis
    /// # };
    /// # let transition_covariance = Array2::<f64>::eye(4);
    /// # let measurement_covariance = Array2::<f64>::eye(3);
    /// let ekf = AdditiveNoiseExtendedKalmanFilter::new(Box::new(transition_function),
    ///                                                  Box::new(transition_jacobis),
    ///                                                  Box::new(measurement_function),
    ///                                                  Box::new(measurement_jacobis),
    ///                                                  &transition_covariance,
    ///                                                 &measurement_covariance);
    ///
    /// let states = Array2::ones([4,4]);
    /// let covariances = Array2::eye(4)
    ///    .view()
    ///    .insert_axis(Axis(0))
    ///    .broadcast([4,4,4])
    ///    .unwrap()
    ///    .to_owned();
    ///
    /// let (predicted_states, predicted_covariances) = ekf.predict(&states, &covariances);
    /// let measurements = Array2::ones([10,3]);
    /// let (updated_states, updated_covariances) = ekf.update(&predicted_states, &predicted_covariances, &measurements);
    ///
    /// // Access the second state, updated by third measurement:
    /// updated_states.slice(ndarray::s![1,2,..]);
    ///
    /// // Access the all updated values for second input state, and covariance matrix which
    /// // valid for all of the outputs assigned to second output states:
    /// let second_states = updated_states.slice(ndarray::s![1,..,..]);
    /// let second_covariances = updated_covariances.slice(ndarray::s![1,..,..]);
    /// ```
    fn update<A: Data<Elem = Num>, B: Data<Elem = Num>, C: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
        measurements: &ArrayBase<C, Ix2>,
    ) -> Self::Update {
        let expected_measurements = (self.measurement_function)(&states.view());
        let innovations_negated =
            pairwise_difference(&expected_measurements, measurements).unwrap();

        let measurement_jacobis = (self.measurement_function_jacobi)(&states.view());
        let mut measurement_jacobis_view = measurement_jacobis.view();
        measurement_jacobis_view.swap_axes(1, 2);

        let l_matrices = broad_dot_ix3_ix3(covariances, &measurement_jacobis_view).unwrap();

        let mut innovation_covariances =
            broad_dot_ix3_ix3(&measurement_jacobis, &l_matrices).unwrap();
        innovation_covariances.add_assign(&self.measurement_covariance);
        let inv_innovation_covariances = inv_all_ix3(&innovation_covariances).unwrap();
        let kalman_gains = broad_dot_ix3_ix3(&l_matrices, &inv_innovation_covariances).unwrap();
        let updated_states = update_states(&states, &kalman_gains, &innovations_negated).unwrap();
        let updated_covariances =
            update_covariance(covariances, &kalman_gains, &l_matrices).unwrap();

        (updated_states.into_owned(), updated_covariances)
    }
}

pub type JacobiMatrixProducer<Num> = Box<dyn Fn(&ArrayView2<Num>, &ArrayView2<Num>) -> Array3<Num>>;
pub type RowStackProducer<Num> = Box<dyn Fn(&ArrayView2<Num>, &ArrayView2<Num>) -> Array2<Num>>;


/// General extended Kalman for transition and measurement functions parametrized both by state and
/// noise
///
/// As described in `Simo Särkka - Bayesian Filtering and Smoothing` this filter is linear
/// approximations for following types of dynamic and measurement models. Assuming that
/// **X ~ N(m,P)** is PDF of state or measuremnt and **q ~ N(0,Q)** is PDF of noise in transition
/// or measurement process, then this filter can be used for  transition/measurement
/// models given by functions of form **y = g(x,q)**, where **y** is then the new state or
/// measurement
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
    /// Represents the type of output produced by prediction method. The first component of the
    /// tuple represents the predicted states. The i-th row represents the prediction of the
    /// of the i-th input state. The analogous holds for second component, where i-th entry along
    /// first axis represents the prediction of the i-th covariance matrix input.
    type Prediction = (Array2<Num>, Array3<Num>);

    /// Type of output produced by update method. The first component represents a data structure
    /// containing the update of each of input states by each of the input measurements.
    ///
    /// Assume that we have following matrix shapes are given as input states=5x8, covariances=5x8x8
    /// and measurements=10x8. Then the first component of output would be 5x10x8 and second component
    /// would be 5x8x8. Thus for example the slice [1,2,:] of first component would represent
    /// the second state updated by third input measurement, whike [1,:,:] of second component
    /// would represent the covariance matices of all states in slice [1,:,:] in first component.
    type Update = (Array3<Num>, Array3<Num>);

    /// Computes the predicition of states and covariances for general extended Kalman filter
    ///
    /// Exact equations implemented here can be found in
    /// `Simo Särkka - Bayesian Filtering and Smoothing` in under **Algorithm 5.5**. Example usage
    /// would be ( note this code will not compile, since currently no model is known for use
    /// with this type of filter. Feel free to contribute one ) :
    /// ```ignore
    /// // Asumme filter is initialized properly as described in ExtendedKalmanFilter::new
    /// let ekf: AdditiveExtendedKalmanFilter<f64> = ... // Some initialization of filter
    /// let states = Array2::<f64>::ones([4,4]);
    /// let covariances = Array2::<f64>::eye(4).insert_axis(Axis(0)).broadcast([4,4,4]).unwrap().into_owned();
    /// let (predicted_states, predicted_covariances) = ekf.predict(&states, &covariances);
    /// ```
    fn predict<A: Data<Elem = Num>, B: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
    ) -> Self::Prediction {
        let zeros = Array2::zeros(states.raw_dim());
        let predicted_states = (self.transition_function)(&states.view(), &zeros.view());

        let state_jacobi = (self.transition_function_jacobi_state)(&states.view(), &zeros.view());
        let mut predicted_covariances =
            quadratic_form_ix3_ix3_ix3(&covariances, &state_jacobi).unwrap();

        let broadcast_transition_covariance = self
            .transition_covariance
            .broadcast(covariances.raw_dim())
            .unwrap();
        let noise_jacobi = (self.transition_function_jacobi_noise)(&states.view(), &zeros.view());
        let second_summand =
            quadratic_form_ix3_ix3_ix3(&broadcast_transition_covariance, &noise_jacobi).unwrap();
        predicted_covariances.add_assign(&second_summand);

        (predicted_states, predicted_covariances)
    }

    /// Computes the predicition of states and covariances for general extended Kalman filter
    ///
    /// Exact equations implemented here can be found in
    /// `Simo Särkka - Bayesian Filtering and Smoothing` in under **Algorithm 5.5**. Example usage
    /// would be ( note this code will not compile, since currently no model is known for use
    /// with this type of filter. Feel free to contribute one ) :
    /// ```ignore
    /// // Asumme filter is initialized properly as described in ExtendedKalmanFilter::new
    /// let ekf: AdditiveExtendedKalmanFilter<f64> = ... // Some initialization of filter
    /// let states = Array2::<f64>::ones([4,4]);
    /// let covariances = Array2::<f64>::eye(4).insert_axis(Axis(0)).broadcast([4,4,4]).unwrap().into_owned();
    /// let measurements = Array3::<f64>::ones([10,4,4]);
    /// let (updated_states, updated_covariances) = ekf.update(&states, &covariances, &measurements);
    /// ```
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
        let innovations_negated =
            pairwise_difference(&expected_measurements, measurements).unwrap();

        let measurement_jacobi =
            (self.measurement_function_jacobi_state)(&states.view(), &zeros.view());
        let mut measurement_jacobi_view = measurement_jacobi.view();
        measurement_jacobi_view.swap_axes(1, 2);
        let state_l_matrices = broad_dot_ix3_ix3(&covariances, &measurement_jacobi_view).unwrap();

        let noise_jacobi = (self.measurement_function_jacobi_noise)(&states.view(), &zeros.view());
        let measurement_covariance_shape = self.measurement_covariance.dim();
        let target_broadcast_shape = [
            noise_jacobi.dim().0,
            measurement_covariance_shape.0,
            measurement_covariance_shape.1,
        ];
        let broadcast_measurement_covariances = self
            .measurement_covariance
            .broadcast(target_broadcast_shape)
            .unwrap();
        let second_summand =
            quadratic_form_ix3_ix3_ix3(&broadcast_measurement_covariances, &noise_jacobi).unwrap();

        let mut innovation_covariances =
            broad_dot_ix3_ix3(&measurement_jacobi, &state_l_matrices).unwrap();
        innovation_covariances.add_assign(&second_summand);

        let inv_innovation_covariances = invc_all_ix3(&innovation_covariances).unwrap();
        let kalman_gains =
            broad_dot_ix3_ix3(&state_l_matrices, &inv_innovation_covariances).unwrap();

        let updated_states = update_states(&states, &kalman_gains, &innovations_negated).unwrap();
        let updated_covariances =
            update_covariance(covariances, &kalman_gains, &state_l_matrices).unwrap();

        (updated_states, updated_covariances)
    }
}

impl<Num> ExtendedKalmanFilter<Num>
where
    Num: Scalar + Lapack,
{
    /// Creates a new instance of ExtendedKalmanFilter
    ///
    /// Assuming that following facts hold for your model:
    /// - **x~N(m,P)** is state PDF
    /// - **q~N(0,Q)** is state transition noise
    /// - **r~N(0,R)** is measurement noise
    /// - **y=g(x,q)** is the state random variable after transition
    /// - **z=h(y,r)** is the measurement random variable after transition
    ///
    /// then the filter may be initialized as follows ( note that this code is no running example
    /// since the author currently knows no applications of this approximation ):
    /// ```ignore
    /// use rusty_rudolf::filter::kalman::nonlinear::extended::ExtendedKalmanFilter;
    /// let transition_function = {
    ///     // here provide the implementation of function g(x,q) as defined above
    /// };
    ///
    /// let transition_function_state_jacobi = {
    ///     // here provide the implementation of function which returns the part of Jacobian of g
    ///     // matrix, derived with respect to components of first input parameter.
    ///     // Basically if state has n dimensions, then this function should obtain first n columns
    ///     // of Jacobian of g
    /// };
    ///
    /// let transition_function_noise_jacobi = {
    ///     // here provide the implementation of function which returns the part of Jacobian of g
    ///     // matrix, derived with respect to components of second input parameter.
    ///     // Basically if noise has n dimensions, then this function should obtain first last n
    ///     // columns of Jacobian of g
    /// };
    ///
    /// let measurement_function = {
    ///     // Same as for transition_function, but this time do it for function h
    /// };
    ///
    /// let measurement_function_state_jacobi = {
    ///     // Same as for transition_function_state_jacobi, but this time for function h
    /// };
    ///
    /// let transition_function_noise_jacobi = {
    ///     // Same as for transition_function_noise_jacobi, but this time for function h
    /// };
    ///
    /// let transition_covariance = .. // Some 2D matrix representing the transition covariance
    /// let measurement_covariance = .. // Some 2D matrix representing the measurement covariances
    /// let ekf: AdditiveExtendedKalmanFilter<f64> = ExtendedKalmanFilter::new(Box::new(transition_function),
    ///     Box::new(transition_function_state_jacobi),
    ///     Box::new(transition_function_noise_jacobi),
    ///     Box::new(measurement_function),
    ///     Box::new(measurement_function_state_jacobi),
    ///     Box::new(measurement_function_noise_jacobi),
    ///     &transition_covariance,
    ///     &measurement_covariance);
    /// ```
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
    use ndarray::{Array2, ArrayView2, Axis};

    use crate::filter::kalman::nonlinear::extended::ExtendedKalmanFilter;

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
