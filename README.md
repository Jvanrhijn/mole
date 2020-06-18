# Mole

[![Build Status](https://travis-ci.com/Jvanrhijn/mole.svg?branch=master)](https://travis-ci.com/Jvanrhijn/mole)
[![codecov](https://codecov.io/gh/Jvanrhijn/mole/branch/master/graph/badge.svg)](https://codecov.io/gh/Jvanrhijn/mole)

Mole is a set of quantum Monte Carlo libraries written in rust.

### Building

Mole requires Rust nightly, cargo, and a version of intel-mkl. Multiple BLAS backends will be
supported in the future.

```
$ git clone [url] && cd Mole
```

Running unit tests:

```
$ cargo test --all
``` 

Some simple examples are included; run as usual with cargo:

```
$ cargo run --example hydrogen_atom
```

### Usage

Currently, Mole supports VMC optimization of all-electron wave functions. No wave functions
are provided, so they must be implemented by the user; Mole simply provides the
tools required to optimize the wave functions and compute expectation values in
a VMC framework. Wave function implementations may be provided in a separate
crate in the future.

Operators can be added by implementing the `Operator<T>` trait; see
`examples/custom_operator.rs` for an example. Some observables are
provided, including a local energy operator as well as certain
molecular potentials.

To implement a wave function, one should implement

* `Function`,
* `WaveFunction`.

If one wishes to compute energies, the trait

* `Differentiate`

should also be implemented, as it provides a `laplacian` function.
Finally, VMC optimization additionally requires an implementation of

* `Optimize`,

which requires a function to compute parameter gradients.

See the examples folder for detailed example usage.

### Example result

The figure below shows the result of optimizing a Jastrow-Slater wave function with two
Jastrow e-e parameters and STO-populated Slater determinant, on a hydrogen molecule (H2)
Hamiltonian. The figure compares a vanilla steepest descent optimization with
the stochastic reconfiguration algorithm by Sorella. See `examples/hydrogen_molecule`.

[![optimization demo](https://i.imgur.com/vxJlrNS.png)](https://i.imgur.com/vxJlrNS.png)

### Goals

As a software project, Mole aims to provide a set of simple and transparent quantum monte carlo
simulation tools. The project aims to allow a high degree of customization to the user. As such,
we also export some of the traits used internally by the program. This allows a user to easily construct
their own wave functions or observables, for instance. This is the advantage of using a modular library
design, as compared to a monolithic program structure.

As a secondary goal, Mole aims to explore the use of Rust in scientific computing. Currently,
most quantum chemistry programs are written in either C++ or Fortran. Both of these languages are
tricky to write quality, maintainable code in; Fortran is easy to program, but hard to keep maintainable,
whereas C++ requires careful programming to write reliable and efficient code. 
Rust provides helpful safeguards to prevent common mistakes in low-level code, in principle without performance cost as
compared to C++ and Fortran. The main cost is the relative lack of library support, since
Rust is such a young language.

### Planned Features

* DMC capabilities

### Dependencies

* ndarray 0.12.0
* rand 0.5.0
* ndarray-rand 0.8.0
* ndarray-linalg 0.10.0
* num-traits 0.2.0
* itertools 0.7.0
