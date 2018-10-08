# Mole

Mole is an open-source quantum monte carlo program, written entirely in the Rust
programming language.

### Goals

As a software project, Mole aims to provide a simple quantum monte carlo simulation 
program, useful for different goals and fields: molecular science and materials
science for starters. The program will focus on real-space wave function representations,
with different choices of wave functions possible.

As a side goal, Mole aims to show the use of Rust in scientific computation. Currently,
most quantum chemistry programs are written in either C++ or Fortran. Both of these languages are
tricky to write quality, maintainable code in. Rust, as a more modern language, provides helpful
safeguards to prevent common mistakes in low-level code, in principle without performance cost as
compared to C++ and Fortran. The main cost is the relative lack of library support, since
Rust is such a young language. Hopefully, this project will contribute to more adoption
of Rust in the physical sciences.

Planned features for Mole:

* Variational QMC capabilities
* Easy configuration file format (INI most likely)
* Different wave functions possible, Jastrow-Slater for starters
* Capable of taking advantege of concurrency
* Various optimization schemes possible (gradient descent and stochastic reconfiguration, for starters)

### Dependencies
* ndarray 0.12.0
* rand 0.5.5
* ndarray-rand 0.8.0
* ndarray-linalg 0.10.0
* assert 0.7.4
* libopenblas
