# Mole

[![Build Status](https://travis-ci.com/Jvanrhijn/Mole.svg?branch=master)](https://travis-ci.com/Jvanrhijn/Mole)
[![codecov](https://codecov.io/gh/Jvanrhijn/mole/branch/master/graph/badge.svg)](https://codecov.io/gh/Jvanrhijn/mole)

Mole is a quantum monte carlo program written in Rust.

### Building

Mole requires Rust nightly, cargo, and a version of intel-mkl. Multiple BLAS backends will be supported in the future.

```
$ git clone [url] && cd Mole
```

Running unit tests:

```
$ cargo test
``` 

Running the main executable (currently only containing a demo for an H2 molecule):

```
$ cargo run --release
```

This will output an estimate of the ground state energy of H2, with an ansatz wave function consisting of a 1s orbital centered
on each ion.

### Goals

As a software project, Mole aims to provide a simple quantum monte carlo simulation 
program, useful for different goals and fields: molecular science and materials
science for starters. The program will focus on real-space wave function representations,
with different choices of wave functions possible.

As a side goal, Mole aims to explore the use of Rust in scientific computing. Currently,
most quantum chemistry programs are written in either C++ or Fortran. Both of these languages are
tricky to write quality, maintainable code in; Fortran is easy to program, but hard to keep maintainable,
whereas C++ requires careful programming to write reliable and efficient code. 
Rust provides helpful safeguards to prevent common mistakes in low-level code, in principle without performance cost as
compared to C++ and Fortran. The main cost is the relative lack of library support, since
Rust is such a young language.

### Planned Features

* Variational and diffusion QMC capabilities
* Easy configuration file format (INI, YAML or TOML most likely)
* Jastrow-Slater wave functions
* Various optimization schemes (gradient descent for starters)
* Python fontend
* Concurrency

### Dependencies

* ndarray 0.12.0
* rand 0.5.0
* ndarray-rand 0.8.0
* ndarray-linalg 0.10.0
* intel-mkl
