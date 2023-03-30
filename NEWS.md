# User visible changes in `StructureFunctions`

# Version 0.1.1

- `SampledStructureFunction` renamed as `EmpiricalStructureFunction` and
  `ShrinkedLazyCovariance` renamed as `PackedLazyCovariance`.

- Lazy covariance and empirical structure function object have (read-only)
  public properties to favor using the `A.f` syntax.

- Empirical structure function object is an abstract array (with offsets).

- A demonstration is provided in `test/demo.jl`.
