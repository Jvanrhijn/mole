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

Currently, Mole supports VMC optimization of all-electron wave functions. Provided wave functions
are Jastrow-Slater wave functions with two-body correlation Jastrow factor, and a single
Slater determinant. The Slater determinant can be populated with either STO or Gaussian
orbitals. Optimization is at this point in time only supported for the Jastrow parameters.

VMC optimization is very crude and must essentially be done manually. A suitable
abstraction will be implemented soon.

Adding new operators and wave functions is simply a matter of creating a new type and
implementing the relevant traits. In order to create a new operator, one should implement
the `Operator<T>` trait (see `src/operator`). For a new wave function, implement **at least**

* `Function` (see `src/wavefunction/src/traits`)
* `Cache` (see `src/wavefuction/src/traits`)

Other traits that are likely required are `Optimize` and `Differentiate`, but this depends
on the type of computation being done.

See the examples folder for detailed example usage.

### Example result

The figure below shows the result of optimizing a Jastrow-Slater wave function with two
Jastrow e-e parameters and STO-populated Slater determinant, on a hydrogen molecule (H2)
Hamiltonian. See `examples/hydrogen_molecule`.

[![optimization demo](https://i.imgur.com/YtLT54M.png)](https://i.imgur.com/YtLT54M.png)

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

* Variational and diffusion QMC capabilities
* Jastrow-Slater wave functions
* Various optimization schemes
* Concurrency
* Serde integration for storing optimized wave functions
* 

### Dependencies

* ndarray 0.12.0
* rand 0.5.0
* ndarray-rand 0.8.0
* ndarray-linalg 0.10.0
* num-traits 0.2.0
* itertools 0.7.0
* intel-mkl