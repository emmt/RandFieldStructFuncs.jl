# Structure functions in Julia

[![Build Status](https://github.com/emmt/StructureFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/StructureFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/StructureFunctions.jl?svg=true)](https://ci.appveyor.com/project/emmt/StructureFunctions-jl) [![Coverage](https://codecov.io/gh/emmt/StructureFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/StructureFunctions.jl)

This package is to deal with structure functions in Julia.

The structure function of a random field `ϕ` is defined by:

``` julia
f(Δr) = ⟨[ϕ(r + Δr) - ϕ(r)]^2⟩
```

where `⟨…⟩` denotes expectation while `r` and `Δr` are absolute and relative
positions.

This package let you build structure function objects, generate the associated
covariance matrix, and simulate random fields having a given structure function.
