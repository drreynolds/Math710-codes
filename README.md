# Math710-codes

Codes for in-class collaboration for the course: Math 710, Special Topics in Applied Math (Numerical Solution of ODEs) at the University of Maryland Baltimore County, for the Fall 2026 semester.

These codes require a modern Python or Matlab installation.

   *Note: there is an older `c++` branch that includes implementations of a subset of these solvers, but in C++, and that use the "Armadillo" C++ library (http://arma.sourceforge.net) for vectors, matrices, and linear solvers.*

Codes are grouped according to type:

* `initial_demo` -- simple demonstration scripts showing the use of Python for mathematical calculations and plotting.
* `shared` -- reusable `ImplicitSolver` class, to be used by implicit ODE methods.
* `newton` -- test driver to show use of `ImplicitSolver`.
* `forward_euler` -- simple IVP "evolution" routine, based on the simplest IVP solver.  Basic approach for timestep adaptivity.  Contains two classes, `ForwardEuler` (fixed-step evolution) and `AdaptEuler` (adaptive-step evolution).
* `simple_implicit` -- simple implicit ODE solver classes, `BackwardEuler` and `Trapezoidal`, showing use of the `ImplicitSolver` class for implicit ODE methods.
* `explicit_one_step` -- higher-order explicit, one-step, ODE integration methods, containing the `Taylor2` and `ERK` classes.
* `implicit_one_step` -- higher-order implicit, one-step, ODE integration methods, containing the `DIRK` and `IRK` classes.
* `linear_multistep` -- higher-order explicit and implicit multi-step ODE integration methods, containing the classes `ExplicitLMM` and `ImplicitLMM`.
* `bvp` -- two-point boundary-value problem solvers.

## Installing Python dependencies

To install the Python packages that are used by the codes in this repository, run the following from the Linux/MacOS command line, or the Anaconda Prompt/Terminal in Windows, from the folder containing this repository:

```bash
pip install -r python_requirements.txt
```

## Branch workflow for teaching

This repository is intended to support separate language-specific teaching branches:

- `main` keeps the Python implementation.
- `matlab` is used for an in-place MATLAB implementation using the same folder structure.

For the MATLAB branch, the existing section folders are reused directly, and Python files are ported section-by-section to MATLAB scripts/functions in place. This keeps the course organization consistent across languages while allowing the branches to remain fully independent.

See the migration checklist in `MATLAB_PORT_CHECKLIST.md` for the phased conversion plan.

Daniel R. Reynolds  
Mathematics and Statistics @ UMBC  
