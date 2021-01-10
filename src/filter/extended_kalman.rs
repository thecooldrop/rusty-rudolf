use crate::filter::filter_traits::Filter;
use cauchy::Scalar;
use ndarray::{
    Array2, Array3, ArrayBase, ArrayView, ArrayView1, Data, Dimension, Ix1, Ix2, Ix3, OwnedRepr,
};
use ndarray_linalg::Lapack;
use std::marker::PhantomData;
use std::ops::AddAssign;

pub struct AdditiveExtendedKalmanFilter<Num, Dim, Trans, Jacobi>
where
    Num: Scalar + Lapack,
    Dim: Dimension,
    Trans: Fn(&ArrayView<Num, Dim>) -> ArrayBase<OwnedRepr<Num>, Dim>,
    Jacobi: Fn(&ArrayView<Num, Dim>) -> ArrayBase<OwnedRepr<Num>, Dim::Larger>,
{
    dimension_phantom: PhantomData<Dim>,
    transition_function: Trans,
    transition_function_jacobian: Jacobi,
    transition_covariance: Array2<Num>,
}

impl<Num, Trans, Jacobi> Filter<Num> for AdditiveExtendedKalmanFilter<Num, Ix1, Trans, Jacobi>
where
    Num: Scalar + Lapack,
    Trans: Fn(&ArrayView1<Num>) -> ArrayBase<OwnedRepr<Num>, Ix1>,
    Jacobi: Fn(&ArrayView1<Num>) -> ArrayBase<OwnedRepr<Num>, Ix2>,
{
    type Prediction = (Array2<Num>, Array3<Num>);
    type Update = (Array3<Num>, Array3<Num>);

    fn predict<A: Data<Elem = Num>, B: Data<Elem = Num>>(
        &self,
        states: &ArrayBase<A, Ix2>,
        covariances: &ArrayBase<B, Ix3>,
    ) -> Self::Prediction {
        let mut predicted_states = Array2::zeros(states.raw_dim());
        let states_dim = states.dim();
        let states_count = states_dim.0;
        let states_length = states_dim.1;
        let mut state_jacobis = Array3::zeros([states_count, states_length, states_length]);
        for (mut state, mut jacobi) in predicted_states
            .outer_iter_mut()
            .zip(state_jacobis.outer_iter_mut())
        {
            let intermediate_state = (self.transition_function)(&state.view());
            let state_jacobi = (self.transition_function_jacobian)(&state.view());
            jacobi.assign(&state_jacobi);
            state.assign(&intermediate_state);
        }

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
        _states: &ArrayBase<A, Ix2>,
        _covariances: &ArrayBase<B, Ix3>,
        _measurements: &ArrayBase<C, Ix2>,
    ) -> Self::Update {
        unimplemented!()
    }
}
