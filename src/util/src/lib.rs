#[macro_export]
macro_rules! hash_map {
    ($($key:expr => $value:expr),*) => {
        {
            let mut map = HashMap::new();
            $(map.insert($key.to_string(), $value);)*
            map
        }
    }
}

#[macro_export]
macro_rules! operators {
    ($($key:expr => $value:expr),*) => {
        {
            use operator::LocalOperator;
            let mut map = HashMap::new();
            $(map.insert($key.to_string(), Box::new($value) as Box<dyn LocalOperator<_>>);)*
            map
        }
    }
}
