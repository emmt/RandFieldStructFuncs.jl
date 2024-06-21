# User visible changes in `RandFieldStructFuncs`

# Version 0.3.0

- Package renamed `RandFieldStructFuncs` as `StructureFunctions` already exists in General
  Julia's registry.

# Version 0.2.1

- Fried's parameter and argument `r` of `KolmogorovStructFunc` can have units.

# Version 0.2.0

- `StructureFunction` and `EmpiricalStructureFunction` respectively renamed as
  `AbstractStructFunc` and `EmpiricalStructFunc`.

- By default, empirical structure functions are computed by means of FFTs which
  is much faster (except perhaps for very small arrays).

- The API for empirical structure functions has changed. The denominator is
  computed immediately when the object is created and is left untouched since
  then. The numerator is updated at every `push!`. As a result, empirical
  structure function objects are no longer abstract arrays: the `values` method
  must be called to compute (not just retrieve) the values of the empirical
  structure function which amounts to dividing the numerator by the denominator
  after having, if FFTs are used, inverse Fourier transformed the numerator.

# Version 0.1.2

- `SampledStructureFunction` renamed as `EmpiricalStructureFunction` and
  `ShrinkedLazyCovariance` renamed as `PackedLazyCovariance`.

- Lazy covariance and empirical structure function object have (read-only)
  public properties to favor using the `A.f` syntax.

- Empirical structure function object is an abstract array (with offsets).

- A demonstration is provided in `test/demo.jl`.
