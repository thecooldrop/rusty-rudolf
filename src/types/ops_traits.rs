pub trait Transpose {
    type Output;
    fn t(&self) -> Self::Output;
}

pub trait IntoTranspose {
    type Output;
    fn into_t(self) -> Self::Output;
}

pub trait TransposeView<'a> {
    type Output;
    fn t_view(&'a self) -> Self::Output;
}

pub trait QuadraticForm<'a, LHS, RHS> {
    type Output;
    fn quadratic_form(&self, lhs: &LHS, rhs: &'a RHS) -> Self::Output;
}

pub trait DotInplace<RHS> {
    type Output;
    fn dot_inplace(&mut self, rhs: &RHS);
}