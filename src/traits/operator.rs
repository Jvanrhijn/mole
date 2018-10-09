use traits::function::Function;

pub trait Operator {
    fn act_on<T: Function<T>>(wf: &T) -> T;
}