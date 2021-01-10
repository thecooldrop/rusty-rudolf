use ndarray::{Ix1, Ix2, Ix3, Dimension, ArrayBase, Data, OwnedRepr, Array2, Array3, ArrayViewMut1, ArrayViewMut};
use cauchy::Scalar;
use ndarray_linalg::Lapack;
use crate::filter::filter_traits::Filter;
use std::ops::AddAssign;
use std::marker::PhantomData;


pub struct AdditiveExtendedKalmanFilter<Num, Dim, Trans, Jacobi> where
    Num: Scalar + Lapack,
    Dim: Dimension,
    Trans: Fn(&ArrayViewMut<Num, Dim>) -> ArrayBase<OwnedRepr<Num>, Dim>,
    Jacobi: Fn(&ArrayViewMut<Num, Dim>) -> ArrayBase<OwnedRepr<Num>, Dim::Larger>
{
    dimension_phantom: PhantomData<Dim>,
    transition_function: Trans,
    transition_function_jacobian: Jacobi,
    transition_covariance: Array2<Num>
}

impl<Num, Trans, Jacobi> Filter<Num> for AdditiveExtendedKalmanFilter<Num, Ix1, Trans, Jacobi> where
    Num: Scalar + Lapack,
    Trans: Fn(&ArrayViewMut1<Num>) -> ArrayBase<OwnedRepr<Num>, Ix1>,
    Jacobi: Fn(&ArrayViewMut1<Num>) -> ArrayBase<OwnedRepr<Num>, Ix2>
{
    type Prediction = (Array2<Num>, Array3<Num>);
    type Update = (Array3<Num>, Array3<Num>);

    fn predict<A: Data<Elem=Num>, B: Data<Elem=Num>>(&self, states: &ArrayBase<A, Ix2>, covariances: &ArrayBase<B, Ix3>) -> Self::Prediction {
        let mut predicted_states = Array2::zeros(states.raw_dim());
        let mut state_jacobis : Vec<Array2<Num>> = Vec::new();
        for mut elem in predicted_states.outer_iter_mut() {
            let intermediate_state = (self.transition_function)(&elem);
            let state_jacobi = (self.transition_function_jacobian)(&elem);
            state_jacobis.push(state_jacobi);
            elem.assign(&intermediate_state);
        }

        let mut predicted_covariances = self.transition_covariance.broadcast(covariances.raw_dim()).unwrap().to_owned();
        for (mut cov, jacobi) in predicted_covariances.outer_iter_mut().zip(state_jacobis.iter()) {
            let predicted_cov = jacobi.dot(&cov).dot(&jacobi.t());
            cov.add_assign(&predicted_cov);
        }

        (predicted_states, predicted_covariances)
    }

    fn update<A: Data<Elem=Num>, B: Data<Elem=Num>, C: Data<Elem=Num>>(&self, _states: &ArrayBase<A, Ix2>, _covariances: &ArrayBase<B, Ix3>, _measurements: &ArrayBase<C, Ix2>) -> Self::Update {
        unimplemented!()
    }
}