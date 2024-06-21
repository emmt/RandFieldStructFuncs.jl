# Structure functions of random fields in Julia

[![Build Status](https://github.com/emmt/RandFieldStructFuncs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/RandFieldStructFuncs.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/RandFieldStructFuncs.jl?svg=true)](https://ci.appveyor.com/project/emmt/RandFieldStructFuncs-jl) [![Coverage](https://codecov.io/gh/emmt/RandFieldStructFuncs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/RandFieldStructFuncs.jl)

This package is to deal with structure functions of random fields with stationary
increments in Julia.

The structure function of a random field `φ` is defined by:

``` julia
Dᵩ(Δr) = ⟨[φ(r + Δr) - φ(r)]²⟩
```

where `⟨…⟩` denotes expectation while `r` and `Δr` are absolute and relative positions. If
`φ` has stationary increments, its structure function does not depend on `r`, only on `Δr`.

This package let you build structure function objects, generate the associated covariance
matrix, and simulate random fields having a given structure function `Dᵩ` and piston
standard deviation `σ`. If `σ > 0` holds, the covariance of `φ` is invertible.

This [LaTeX file](notes/structure-functions.tex) provides some mathematical background.

## Usage

To measure empirically a structure function:

``` julia
# Create structure function object with support S.
A = EmpiricalStructFunc(S)
for ϕₜ in Φ
    # Integrate structure function.
    push!(A, ϕₜ)
end
T = nobs(A);     # get the number of observations
D_ϕ = values(A); # compute the structure function
```

At any moment:
``` julia
a = A.den; # get the denominator: `a = S ⊗ S`
fft_b =  A.num; # get the numerator in the frequency domain
```

It is also possible to define theoretical structure functions. For example, Kolmogorov
structure function for Fried's parameter `r0` is built by:

``` julia
D = KolmogorovStructFunc(r0);
```

which yields a callable object `D` such that `D(r)` returns the value of the structure
function at position `r`. Note that `r` and `r0` are assumed to have the same units if
`r0` has no units; otherwise; `r` must have the same unit dimensions as `r0`.
