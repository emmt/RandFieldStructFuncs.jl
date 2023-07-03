# Structure functions in Julia

[![Build Status](https://github.com/emmt/StructureFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/StructureFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/StructureFunctions.jl?svg=true)](https://ci.appveyor.com/project/emmt/StructureFunctions-jl) [![Coverage](https://codecov.io/gh/emmt/StructureFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/StructureFunctions.jl)

This package is to deal with structure functions in Julia.

The structure function of a random field `φ` is defined by:

``` julia
Dᵩ(Δr) = ⟨[φ(r + Δr) - φ(r)]²⟩
```

where `⟨…⟩` denotes expectation while `r` and `Δr` are absolute and relative
positions.

This package let you build structure function objects, generate the associated
covariance matrix, and simulate random fields having a given structure function
`Dᵩ` and piston standard deviation `σ`. If `σ > 0` holds, the covariance of `φ`
is invertible.

This [LaTeX file](notes/structure-functions.tex) provides some background.
