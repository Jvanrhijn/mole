# Mole

[![Build Status](https://travis-ci.com/Jvanrhijn/Mole.svg?branch=master)](https://travis-ci.com/Jvanrhijn/Mole)
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

Currently, only Monte Carlo integration is supported with simple Slater determinant wave function
ansätze. A few basis functions are provided. In the future it will be possible to provide basis
functions obtained from a previous calculation, for instance from DFT or SCF computations.

See the examples folder for detailed example usage.

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

* Python frontend
* Variational and diffusion QMC capabilities
* Jastrow-Slater wave functions
* Various optimization schemes
* Concurrency

### Dependencies

* ndarray 0.12.0
* rand 0.5.0
* ndarray-rand 0.8.0
* ndarray-linalg 0.10.0
* num-traits 0.2.0
* itertools 0.7.0
* intel-mkl
