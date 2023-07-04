module StructureFunctions

export
    AbstractStructFunc,
    EmpiricalStructFunc,
    KolmogorovStructFunc,
    cov, var, diag, nobs, weights

using TypeUtils, ArrayTools, OffsetArrays
using AbstractFFTs
using AbstractFFTs: Plan
using StatsBase, Statistics, LinearAlgebra
import Statistics: cov, var
import LinearAlgebra: diag

include("dft.jl")

"""
    AbstractStructFunc{T}

is the abstract type of structure functions using floating-point type `T` for
their computations.

A structure function `Dᵩ` of a random field `φ` is a callable object such that:

    Dᵩ(Δr) = ⟨[φ(r + Δr) - φ(r)]^2⟩

where `⟨…⟩` denotes expectation while `r` and `Δr` are Cartesian coordinates in
units of the grid sampling step.

For sub-types of `AbstractStructFunc{T}`, it is only necessary to implement
calling the structure function object with a tuple of coordinates.

"""
abstract type AbstractStructFunc{T<:AbstractFloat} <: Function; end

# Convert Cartesian index.
# NOTE: Make abstract type callable only works for Julia ≥ 1.3
(Dᵩ::AbstractStructFunc)(r::CartesianIndex) = Dᵩ(Tuple(r))

Base.convert(::Type{S}, Dᵩ::S) where {S<:AbstractStructFunc} = Dᵩ

"""
    KolmogorovStructFunc{T}(r0)

yields a Kolmogorov structure function for Fried's parameter `r0` (in units of
the grid sampling step).

"""
struct KolmogorovStructFunc{T<:AbstractFloat} <: AbstractStructFunc{T}
    r0::T
    q::T  # to store precompted value of 1/r0^2
    KolmogorovStructFunc{T}(r0::T) where {T<:AbstractFloat} =
        new{T}(r0, one(T)/r0^2)
end

# Compute structure function.
@inline (Dᵩ::KolmogorovStructFunc{T})(r::NTuple{2,Real}) where {T<:AbstractFloat} = Dᵩ(map(as(T), r))

(Dᵩ::KolmogorovStructFunc{T})(r::NTuple{2,T}) where {T<:AbstractFloat} =
    as(T,6.88)*(Dᵩ.q*(r[1]^2 + r[2]^2))^as(T,5/6)

# Outer constructors.
KolmogorovStructFunc{T}(r0::Real) where {T<:AbstractFloat} = KolmogorovStructFunc{T}(as(T, r0))
KolmogorovStructFunc(r0::Real) = KolmogorovStructFunc{float(typeof(r0))}(r0)

# Conversions.
KolmogorovStructFunc(Dᵩ::KolmogorovStructFunc) = Dᵩ
KolmogorovStructFunc{T}(Dᵩ::KolmogorovStructFunc{T}) where {T} = Dᵩ
KolmogorovStructFunc{T}(Dᵩ::KolmogorovStructFunc) where {T} = KolmogorovStructFunc{T}(Dᵩ.r0)
for type in (:AbstractStructFunc, :KolmogorovStructFunc)
    @eval begin
        Base.convert(::Type{$type{T}}, Dᵩ::KolmogorovStructFunc{T}) where {T} = Dᵩ
        Base.convert(::Type{$type{T}}, Dᵩ::KolmogorovStructFunc) where {T} =
            KolmogorovStructFunc{T}(Dᵩ)
    end
end

"""
    StructureFunctions.normalize_support(T<:AbstractFloat, S)

yields a normalized support function with elements of floating-point type `T`
given the sampled support function `S` not necessarily normalized and throwing
an error if `S` has invalid (e.g., negative) values. The result is an array of
nonnegative values of type `T` and whose sum is equal to 1.

"""
function normalize_support(::Type{T}, S::AbstractArray{<:Real}) where {T<:AbstractFloat}
    # Build the normalized support function.
    q = one(T)/as(T, check_support(S))
    R = similar(S, T)
    if eltype(S) === Bool
        @inbounds @simd for i in eachindex(R, S)
            R[i] = ifelse(S[i], q, zero(T))
        end
    else
        @inbounds @simd for i in eachindex(R, S)
            R[i] = q*as(T, S[i])
        end
    end
    return R
end

"""
    StructureFunctions.check_support(S)

yields the sum of values in `S` throwing an exception if `S` is not valid to
specify a support.

"""
function check_support(S::AbstractArray{X}) where {X<:Real}
    if X <: Bool
        s = countnz(S)
    else
        T = promote_type(Float64, X) # use at least double-precision
        s = zero(T)
        flag = true
        @inbounds @simd for i in eachindex(S)
            Sᵢ = S[i]
            flag &= (Sᵢ ≥ zero(Sᵢ))
            s += oftype(s, Sᵢ)
        end
        flag || throw(ArgumentError("support function must be nonnegative everywhere"))
    end
    s > zero(s) || throw(ArgumentError("support function must have some nonzeros"))
    return s
end

"""
    StructureFunctions.countnz(S)

yields the number of non-zeros in array `S`.

"""
function countnz(S::AbstractArray)
    nnz = 0 # to count number of non-zeros
    @inbounds @simd for i in eachindex(S)
        nnz += iszero(S[i]) ? 0 : 1
    end
    return nnz
end

"""
    var(Dᵩ, S, σ=0) -> Vᵩ

yields the non-uniform variance `Vᵩ` of a random field having a structure
function `Dᵩ` on a support `S` and a piston mode with standard deviation `σ`.
The covariance between two nodes of Cartesian coordinates `r` and `r′` is then
given by:

    Cov(r,r′) = (Vᵩ[r] + Vᵩ[r′] - Dᵩ(r - r′))/2

if `S[r]` and `S[r′]` are both non-zero, the covariance being zero if any of
`r` or `r′` is outside the support.

"""
function var(Dᵩ::AbstractStructFunc{T},
             S::AbstractArray{X},
             σ::Real = zero(T)) where {T<:AbstractFloat,X<:Real}
    # Check piston variance.
    σ ≥ zero(σ) || throw(ArgumentError("piston variance must be nonnegative"))
    σ² = as(T, σ)^2

    # Check support and compute normalization factor.
    q = as(T, check_support(S))

    # Pre-compute Kᵩ(r) = ∫Dᵩ(r - r′)⋅S(r′)⋅dr′
    R = CartesianIndices(S)
    Kᵩ = similar(S, T)
    @inbounds for r ∈ R
        S_r = S[r]
        if iszero(S_r)
            Kᵩ[r] = zero(T)
        else
            s = zero(T)
            if X <: Bool
                for r′ ∈ R
                    S_r′ = S[r′]
                    iszero(S_r′) && continue
                    s += oftype(s, Dᵩ(r - r′))
                end
            else
                for r′ ∈ R
                    S_r′ = S[r′]
                    iszero(S_r′) && continue
                    s += oftype(s, Dᵩ(r - r′))*oftype(s, S_r′)
                end
            end
            Kᵩ[r] = s/q
        end
    end

    # Compute c0 = σ^2 - (1/2)⋅∫Kᵩ(r)⋅S(r)⋅dr
    s = zero(T)
    if eltype(S) === Bool
        @inbounds @simd for i in eachindex(Kᵩ, S)
            s += Kᵩ[i]
        end
    else
        @inbounds @simd for i in eachindex(Kᵩ, S)
            s += Kᵩ[i]*oftype(s, S[i])
        end
    end
    c0 = σ² - s/2q

    # Overwrite Kᵩ with the variance.
    @inbounds @simd for i in eachindex(Kᵩ)
        Kᵩ[i] = ifelse(iszero(Kᵩ[i]), zero(T), Kᵩ[i] + c0)
    end

    return Kᵩ
end

"""
    cov(Dᵩ::AbstractStructFunc, S, σ=0; pack=false) -> Cᵩ

yields the covariance of a random field whose structure function is `Dᵩ` over
an support defined by `S` and whose piston mode has a standard deviation of
`σ`.

The range of indices to consider for the random field is the same as that of
`S` unless keyword `pack` is true, in which case only the indices inside the
support `S` are considered.

The result is a flattened `n×n` covariance matrix with `n = length(S)` if
`pack` is false, or `n` the number of non-zeros in the support `S` if `pack` is
true.

The implemented method is described in the notes accompanying this package.

"""
function cov(Dᵩ::AbstractStructFunc{T},
             S::AbstractArray{<:Real},
             σ::Real = zero(T);
             pack::Bool = false) where {T<:AbstractFloat}
    # Compute non-uniform variance.
    Vᵩ = var(Dᵩ, S, σ)

    # Compute covariance.
    R = CartesianIndices(S)
    if pack
        nnz = countnz(S)
        Cᵩ = Array{T}(undef, (nnz, nnz))
        i = 0
        @inbounds for r ∈ R
            iszero(S[r]) && continue
            i += 1
            i′ = 0
            for r′ ∈ R
                iszero(S[r′]) && continue
                i′ += 1
                if i ≤ i′
                    # Instantiate covariance.
                    Cᵩ[i′,i] = ((Vᵩ[r] + Vᵩ[r′]) - Dᵩ(r - r′))/2
                else
                    # Avoid computations as Cᵩ is symmetric.
                    Cᵩ[i′,i] = Cᵩ[i,i′]
                end
            end
        end
    else
        n = length(S)
        Cᵩ = zeros(T, (n, n))
        @inbounds for (i,r) ∈ enumerate(R)
            iszero(S[r]) && continue
            for (i′,r′) ∈ enumerate(R)
                iszero(S[r′]) && continue
                if i ≤ i′
                    # Instantiate covariance.
                    Cᵩ[i′,i] = ((Vᵩ[r] + Vᵩ[r′]) - Dᵩ(r - r′))/2
                else
                    # Avoid computations as Cᵩ is symmetric.
                    Cᵩ[i′,i] = Cᵩ[i,i′]
                end
            end
        end
    end
    return Cᵩ
end

"""
    StructureFunctions.LazyCovariance(Dᵩ, S, σ) -> Cᵩ

yields an object that can be used as:

    Cᵩ[i, j]

to compute *on the fly* the covariance of a random field whose structure
function is `Dᵩ` over a support defined by `S` and whose piston mode has a
standard deviation of `σ`. Indices `i` and `j` can be linear indices or
Cartesian indices that must be valid to index `S`.

The fields of a lazy covariance object `Cᵩ` may be retrieved by the `Cᵩ.key`
syntax or via getters:

    Cᵩ.func    # yields the structure function
    Cᵩ.support # yields the normalized support
    Cᵩ.diag    # yields the diagonal entries, also non-uniform variances
    diag(Cᵩ)   # idem.
    var(Cᵩ)    # idem.

"""
struct LazyCovariance{T<:AbstractFloat,N,
                      F<:AbstractStructFunc{T},
                      S<:AbstractArray{T,N},
                      D<:AbstractArray{T,N}} <: AbstractMatrix{T}
    func::F     # structure function
    support::S  # normalized support
    diag::D     # diagonal entries, non-uniform variance

    # The inner constructor is to ensure that arrays standard linear indexing
    # and the the same axes.
    function LazyCovariance(Dᵩ::F, sup::S, diag::D) where {T<:AbstractFloat,N,
                                                           F<:AbstractStructFunc{T},
                                                           S<:AbstractArray{T,N},
                                                           D<:AbstractArray{T,N}}
        has_standard_linear_indexing(sup) || throw(ArgumentError(
            "support array must have standard linear indexing"))
        has_standard_linear_indexing(diag) || throw(ArgumentError(
            "array of diagonal entries must have standard linear indexing"))
        axes(sup) == axes(diag) || throw(DimensionMismatch(
            "support and variance arrays have incompatible dimensions/indices"))
        return new{T,N,F,S,D}(Dᵩ, sup, diag)
    end
end

# Getters.
AbstractStructFunc(A::LazyCovariance) = A.func
diag(A::LazyCovariance) = A.diag
var(A::LazyCovariance) = diag(A)

# Constructor.
function LazyCovariance(Dᵩ::AbstractStructFunc{T},
                        S::AbstractArray{<:Real,N},
                        σ::Real = zero(T)) where {T<:AbstractFloat,N}
    sup = normalize_support(T, S)
    diag = var(Dᵩ, S, σ)
    return LazyCovariance(Dᵩ, sup, diag)
end

# Implement abstract array API.
Base.length(A::LazyCovariance) = begin
    n = length(A.diag)
    return n^2
end
Base.size(A::LazyCovariance) = begin
    n = length(A.diag)
    return (n, n)
end
Base.axes(A::LazyCovariance) = begin
    r = Base.OneTo(length(A.diag))
    return (r, r)
end
Base.IndexStyle(::LazyCovariance) = IndexCartesian()

@inline function Base.getindex(A::LazyCovariance{T}, i::Int, j::Int) where {T}
    S = A.support
    @boundscheck (checkbounds(Bool, S, i) & checkbounds(Bool, S, j)) ||
        throw(BoundsError(A, (i, j)))
    @inbounds begin
        if iszero(S[i]) | iszero(S[j])
            zero(T)
        else
            R, D, var = CartesianIndices(S), A.func, A.diag
            Δr = R[i] - R[j]
            ((var[i] + var[j]) - D(Δr))/2
        end
    end
end

@inline function Base.getindex(A::LazyCovariance{T,N},
                               i::CartesianIndex{N},
                               j::CartesianIndex{N}) where {T,N}
    S = A.support
    @boundscheck (checkbounds(Bool, S, i) & checkbounds(Bool, S, j)) ||
        throw(BoundsError(A, (i,j)))
    @inbounds begin
        if iszero(S[i]) | iszero(S[j])
            zero(T)
        else
            D, var = A.func, A.diag
            Δr = i - j
            ((var[i] + var[j]) - D(Δr))/2
        end
    end
end

"""
    StructureFunctions.PackedLazyCovariance(Dᵩ, S, σ) -> Cᵩ

yields an object that can be used as:

    Cᵩ[i,j]

to compute *on the fly* the covariance of a random field whose structure
function is `Dᵩ` over a support defined by `S` and whose piston mode has a
standard deviation of `σ`. Indices `i` and `j` are linear indices in the range
`1:nnz` with `nnz` the number of non-zeros in the support `S`.

The fields of a packed lazy covariance object `Cᵩ` may be retrieved by the
`Cᵩ.key` syntax or via getters:

    Cᵩ.func    # yields the structure function
    Cᵩ.support # yields the normalized support
    Cᵩ.mask    # yields the boolean support
    Cᵩ.indices # yields the Cartesian indices of the diagonal entries
    Cᵩ.diag    # yields the diagonal entries, also non-uniform variances
    diag(Cᵩ)   # idem.
    var(Cᵩ)    # idem.

 """
struct PackedLazyCovariance{T<:AbstractFloat,N,
                            F<:AbstractStructFunc{T},
                            S<:AbstractArray{T,N},
                            M<:AbstractArray{Bool,N},
                            I<:AbstractVector{<:CartesianIndex{N}},
                            D<:AbstractVector{T}} <: AbstractMatrix{T}
    func::F     # structure function
    support::S  # normalized support
    mask::M     # boolean support
    indices::I  # linear index in variance -> Cartesian index in support
    diag::D     # diagonal entries, also non-uniform variances
    function PackedLazyCovariance(Dᵩ::F, sup::S, mask::M, inds::I,
                                  diag::D) where {T<:AbstractFloat,N,
                                                  F<:AbstractStructFunc{T},
                                                  I<:AbstractVector{<:CartesianIndex{N}},
                                                  S<:AbstractArray{T,N},
                                                  M<:AbstractArray{Bool,N},
                                                  D<:AbstractVector{T}}
        check_struct(PackedLazyCovariance, Dᵩ, sup, mask, inds, diag)
        return new{T,N,F,S,M,I,D}(Dᵩ, sup, mask, inds, diag)
    end
end

check_struct(A::PackedLazyCovariance) = check_struct(
    PackedLazyCovariance, A.func, A.support, A.mask, A.indices, A.diag)

function check_struct(::Type{<:PackedLazyCovariance},
                      Dᵩ::F, sup::S, mask::M, inds::I,
                      diag::D) where {T<:AbstractFloat,N,
                                      F<:AbstractStructFunc{T},
                                      I<:AbstractVector{<:CartesianIndex{N}},
                                      S<:AbstractArray{T,N},
                                      M<:AbstractArray{Bool,N},
                                      D<:AbstractVector{T}}
    has_standard_linear_indexing(inds) || throw(ArgumentError(
        "vector of indices must have standard linear indexing"))
    has_standard_linear_indexing(diag) || throw(ArgumentError(
        "vector of diagonal entries must have standard linear indexing"))
    axes(inds) == axes(diag) || throw(DimensionMismatch(
        "vectors of indices and diagonal entries have incompatible dimensions/indices"))
    axes(sup) == axes(mask) || throw(DimensionMismatch(
        "support and mask arrays have incompatible dimensions/indices"))
    j = 0
    R = CartesianIndices(sup)
    @inbounds for i in eachindex(sup, mask, R)
        sup[i] ≥ zero(T)  || throw(ArgumentError(
            "support must have nonnegative values"))
        mask[i] === !iszero(sup[i]) || throw(ArgumentError(
            "support and mask arrays disagree"))
        if mask[i]
            j += 1
            checkbounds(Bool, inds, j) || throw(DimensionMismatch(
                "indices have invalid number of entries"))
            inds[j] == R[i] || throw(ArgumentError(
                "indices have invalid values"))
        end
    end
    j > 0 || throw(ArgumentError("support must have some non-zeros"))
    length(inds) == j || throw(DimensionMismatch(
        "indices have invalid number of entries"))
    nothing
end

AbstractStructFunc(A::PackedLazyCovariance) = A.func
diag(A::PackedLazyCovariance) = A.diag
var(A::PackedLazyCovariance) = diag(A)

function PackedLazyCovariance(Dᵩ::AbstractStructFunc{T},
                              S::AbstractArray{<:Real,N},
                              σ::Real = zero(T)) where {T<:AbstractFloat,N}
    return PackedLazyCovariance(LazyCovariance(Dᵩ, S, σ))
end

function PackedLazyCovariance(A::LazyCovariance{T,N})  where {T<:AbstractFloat,N}
    Dᵩ, S = A.func, A.support
    nnz = countnz(S)
    inds = Vector{CartesianIndex{N}}(undef, nnz)
    mask = similar(S, Bool)
    @assert axes(mask) == axes(S)
    diag = Vector{T}(undef, nnz)
    i = 0
    @inbounds for r in CartesianIndices(S)
        inside = !iszero(S[r])
        mask[r] = inside
        if inside
            i += 1
            inds[i] = r
            diag[i] = A.diag[r]
        end
    end
    return PackedLazyCovariance(Dᵩ, S, mask, inds, diag)
end

# Implement abstract array API.
Base.length(A::PackedLazyCovariance) = begin
    n = length(A.indices)
    return n^2
end
Base.size(A::PackedLazyCovariance) = begin
    n = length(A.indices)
    return (n, n)
end
Base.axes(A::PackedLazyCovariance) = begin
    r = axes(A.indices)
    return (r..., r...)
end
Base.IndexStyle(::PackedLazyCovariance) = IndexCartesian()

@inline function Base.getindex(A::PackedLazyCovariance, i::Int, j::Int)
    var = A.diag
    @boundscheck (checkbounds(Bool, var, i) & checkbounds(Bool, var, j)) ||
        throw(BoundsError(A, (i, j)))
    @inbounds begin
        R, D = A.indices, A.func
        Δr = R[i] - R[j]
        cov = ((var[i] + var[j]) - D(Δr))/2
    end
    return cov
end

"""
    A = EmpiricalStructFunc{T}(S[, plan])

yields an (empty) empirical structure function with values of floating-point
type `T` and for a support `S`. Parameter `T` may be omitted to determine the
floating-point type from the arguments.

If optional argument `plan` is not specified, the empirical structure function
is updated by means of fast Fourier transforms (FFTs) and keywords `flags` and
`timelimit` may be specified to build the FFT plan (defaults are
`flags=FFTW.ESTIMATE` and `timelimit=Inf`). Otherwise, `plan` maybe a
precomputed suitable FFT plan or `nothing` to use a **slow** update method.

The base method `push!` can be used to *update* the data into the empirical
structure function object:

    push!(A, φ)

where `φ` is a random sample which can be an array of the same size as `S`. If
`plan` is `nothing`, `φ` may also be a vector whose length is the number of
non-zeros in the support `S`.

An empirical structure function object `A` has the following properties:

    A.support # support
    A.num     # cumulated numerator
    A.den     # denominator
    A.nobs    # number of observations

The following methods are applicable:

    nobs(A)         # number of observations
    Dᵩ = values(A)  # compute the empirical structure function
    valtype(A)      # element type of `Dᵩ`

The empirical structure function `Dᵩ` computed by `values(A)` is an
`OffsetArray` instance such that:

    Dᵩ[Δr]

yields the value of the empirical structure function for a displacement `Δr`
which may be specified as a Cartesian index.

"""
mutable struct EmpiricalStructFunc{T<:AbstractFloat,N,P,
                                   S<:AbstractArray{<:Union{Bool,T},N},
                                   A<:AbstractArray{T,N},
                                   B<:AbstractArray{T,N},
                                   W₀}
    plan::P
    support::S # support
    den::A
    num::B
    w₀::W₀
    nobs::Int  # number of observations
end

StatsBase.nobs(A::EmpiricalStructFunc) = getfield(A, :nobs)

# Extend some base methods.
Base.length(A::EmpiricalStructFunc) = length(A.den)
Base.size(A::EmpiricalStructFunc) = size(A.den)
Base.axes(A::EmpiricalStructFunc) = axes(A.den)
Base.valtype(A::EmpiricalStructFunc) = valtype(typeof(A))
Base.valtype(::Type{<:EmpiricalStructFunc{T}}) where {T} = T

# Make properties read-only.
@inline Base.getproperty(A::EmpiricalStructFunc, f::Symbol) = getfield(A, f)
Base.setproperty!(A::EmpiricalStructFunc, f::Symbol) = error(
    "attempt to ", (f ∈ propertynames(A) ? "modify read-only" : "access non-existing"),
    " property `$f`")

#------------------------------------------------------------------------------
# Fast empirical structure function.

EmpiricalStructFunc(S::AbstractArray{T,N}; kwds...) where {T,N} =
    EmpiricalStructFunc{float(T)}(S; kwds...)

EmpiricalStructFunc(S::AbstractArray{<:Any,N}, plan::Plan{Complex{T}}; kwds...) where {T,N} =
    EmpiricalStructFunc{T}(S, plan; kwds...)

function EmpiricalStructFunc{T}(S::AbstractArray{<:Real,N};
                                threshold::Union{Nothing,Real}= nothing,
                                kwds...) where {T<:AbstractFloat,N}
    dims = goodfftdims(map(d -> 2d - 1, size(S)))
    plan = plan_fft(Array{Complex{T}}(undef, dims); kwds...)
    return EmpiricalStructFunc{T}(S, plan; threshold)
end

function EmpiricalStructFunc{T}(S::AbstractArray{<:Real,N},
                                plan::Plan{Complex{T}};
                                threshold::Union{Nothing,Real} = nothing) where {T<:AbstractFloat,N}
    # Check/fix support.
    check_support(S)
    if !(eltype(S) <: Union{Bool,T})
        S = convert(AbstractArray{T,N}, S)
    end

    # Compute denominator (in frequency domain).
    dims = size(plan)
    inds = fftshiftaxes(dims)
    w = Array{Complex{T}}(undef, dims) # workspace
    w₀ = plan*unsafe_zeropad_map(pow_0, w, S) # fft(zeropad(S, dims))

    # Compute the denominator and threshold it.
    a = OffsetArray(fftshift(real.(inv(plan)*abs2.(w₀))), inds)
    τ = determine_threshold(T, S, threshold)
    @inbounds @simd for i in eachindex(a)
        aᵢ = a[i]
        a[i] = ifelse(aᵢ > τ, aᵢ, zero(aᵢ))
    end

    # Allocate numerator (in frequency domain).
    b = zeros(T, size(plan))
    return EmpiricalStructFunc{
        T,N,typeof(plan),typeof(S),typeof(a),typeof(b),typeof(w₀)}(
        plan, S, a, b, w₀, 0)
end

function Base.push!(A::EmpiricalStructFunc{T,N,<:Plan},
                    φ::AbstractArray{<:Real,N}) where {T,N}
    S = A.support
    @assert_same_axes S φ

    # Compute FFT terms needed for the numerator.
    plan = A.plan # plan to compute FFT
    w = Array{Complex{T}}(undef, size(plan)) # workspace for zero-padding
    w₀ = A.w₀                                    # fft(zeropad(S,       dims))
    w₁ = plan*unsafe_zeropad_map(pow_1, w, S, φ) # fft(zeropad(S.*φ,    dims))
    w₂ = plan*unsafe_zeropad_map(pow_2, w, S, φ) # fft(zeropad(S.*φ.^2, dims))

    # Update the numerator (in the frequency domain).
    b = A.num
    @inbounds @simd for k in eachindex(b, w₀, w₁, w₂)
        b[k] += real(conj(w₀[k])*w₂[k]) - abs2(w₁[k])
    end

    # Update the number of observations and return.
    setfield!(A, :nobs, nobs(A) + 1)
    return A
end

function Base.values(A::EmpiricalStructFunc{T,N,<:Plan}) where {T,N}
    # Compute the current value of the numerator by inverse FFT and overwrite
    # the resulting array with the structure function.
    a = A.den
    b = OffsetArray(fftshift(real.(inv(A.plan)*A.num)), axes(a))
    @assert eltype(b) === T
    if A.nobs ≥ 1
        β = as(T, 2//A.nobs) # scaling factor
        @inbounds @simd for i in eachindex(a, b)
            aᵢ = as(T, a[i])
            bᵢ = as(T, b[i])
            b[i] = ifelse(aᵢ > zero(aᵢ), β*bᵢ/aᵢ, zero(T))
        end
    end
    return b
end

determine_threshold(::Type{T}, S::AbstractArray, τ::Real) where {T} =
    τ ≥ zero(τ) ? as(T, τ) : throw(ArgumentError(
        "threshold must be non-negative"))

function determine_threshold(::Type{T}, S::AbstractArray, τ::Nothing) where {T}
    # Find the least positive value in S.
    Sₘᵢₙ = typemax(eltype(S))
    flag = false
    @inbounds @simd for i in eachindex(S)
        Sᵢ = S[i]
        flag |= (Sᵢ > zero(Sᵢ))
        Sₘᵢₙ = ifelse(zero(Sᵢ) < Sᵢ < Sₘᵢₙ, Sᵢ, Sₘᵢₙ)
    end
    return flag ? as(T, Sₘᵢₙ)^2/2 : sqrt(eps(T))
end

pow_0(::Type{T}, S::Bool) where {T} = ifelse(S, one(T), zero(T))
pow_1(::Type{T}, S::Bool, φ::Real) where {T} = ifelse(S, as(T, φ), zero(T))
pow_2(::Type{T}, S::Bool, φ::Real) where {T} = ifelse(S, as(T, φ*φ), zero(T))

pow_0(::Type{T}, S::Real) where {T} = as(T, S)
pow_1(::Type{T}, S::Real, φ::Real) where {T} = as(T, S*φ)
pow_2(::Type{T}, S::Real, φ::Real) where {T} = as(T, S*(φ*φ))

function unsafe_zeropad_map(f::Function,
                            w::AbstractArray{T,N},
                            S::AbstractArray{<:Any,N}) where {T,N}
    fill!(w, zero(T))
    @inbounds @simd for i ∈ @range CartesianIndices(w) ∩ CartesianIndices(S)
        w[i] = f(T, S[i])
    end
    return w
end

function unsafe_zeropad_map(f::Function,
                            w::AbstractArray{T,N},
                            S::AbstractArray{<:Any,N},
                            φ::AbstractArray{<:Any,N}) where {T,N}
    fill!(w, zero(T))
    @inbounds @simd for i ∈ @range CartesianIndices(w) ∩ CartesianIndices(S)
        w[i] = f(T, S[i], φ[i])
    end
    return w
end

#------------------------------------------------------------------------------
# Slow empirical structure function.

EmpiricalStructFunc(S::AbstractArray{T,N}, plan::Nothing) where {T,N} =
    EmpiricalStructFunc{float(T)}(S, plan)

function EmpiricalStructFunc{T}(S::AbstractArray{<:Real,N},
                                plan::Nothing) where {T<:AbstractFloat,N}
    # Check/fix support.
    check_support(S)
    if !(eltype(S) <: Union{Bool,T})
        S = convert(AbstractArray{T,N}, S)
    end

    # Dimensions.
    inds = map(r -> (first(r) - last(r)):(last(r) - first(r)), axes(S))
    dims = map(d -> 2d - 1, size(S))

    # Allocate numerator and denominator arrays.
    a = OffsetArray(zeros(T, dims), inds)
    b = OffsetArray(zeros(T, dims), inds)

    # Initialize denominator.
    R = CartesianIndices(S)
    @inbounds for i in R
        Sᵢ = S[i]
        iszero(Sᵢ) && continue
        for j in R
            Sⱼ = S[j]
            iszero(Sⱼ) && continue
            a[i - j] += as(T, Sᵢ*Sⱼ)
        end
    end

    # Build structure.
    return EmpiricalStructFunc{
        T,N,typeof(plan),typeof(S),typeof(a),typeof(b),Nothing}(
        plan, S, a, b, nothing, 0)
end

function Base.push!(A::EmpiricalStructFunc{T,N,Nothing},
                    φ::Union{AbstractArray{<:Real,N},
                             AbstractVector{<:Real}}) where {T,N}
    mul(::Type{T}, w::Bool, x::Real) = ifelse(w, as(T, x), zero(T))
    mul(::Type{T}, w::Real, x::Real) = as(T, w)*as(T, x)

    b = A.num
    S = A.support
    R = CartesianIndices(S)
    if φ isa AbstractArray{<:Real,N} && axes(φ) == axes(S)
        # Assume φ is for all the nodes, inside and outside the support.
        @inbounds for I in R
            Sᵢ = S[I]
            iszero(Sᵢ) && continue
            φᵢ = φ[I]
            for J in R
                Sⱼ = S[J]
                iszero(Sⱼ) && continue
                φⱼ = φ[J]
                b[I - J] += mul(T, Sᵢ*Sⱼ, abs2(φᵢ - φⱼ))
            end
        end
    elseif φ isa AbstractVector{<:Real} && length(φ) == countnz(S)
        # Assume φ is only for the nodes inside the support.
        i₀ = firstindex(φ) - 1
        i = i₀ # linear index in φ
        @inbounds for I in R
            Sᵢ = S[I]
            iszero(Sᵢ) && continue
            φᵢ = φ[i += 1]
            j = i₀ # linear index in φ
            for J in R
                Sⱼ = S[J]
                iszero(Sⱼ) && continue
                φⱼ = φ[j += 1]
                b[I - J] += mul(T, Sᵢ*Sⱼ, abs2(φᵢ - φⱼ))
            end
        end
    else
        throw(DimensionMismatch("incompatible dimensions/indices"))
    end
    setfield!(A, :nobs, nobs(A) + 1)
    return A
end

function Base.values(A::EmpiricalStructFunc{T,N,Nothing}) where {T,N}
    a = A.den
    b = A.num
    Dᵩ = similar(b)
    if A.nobs < 1
        fill!(Dᵩ, zero(eltype(Dᵩ)))
    else
        β = as(T, 1//A.nobs) # scaling factor
        @inbounds @simd for i in eachindex(Dᵩ, a, b)
            aᵢ = as(T, a[i])
            bᵢ = as(T, b[i])
            Dᵩ[i] = ifelse(aᵢ > zero(aᵢ), β*bᵢ/aᵢ, zero(T))
        end
    end
    return Dᵩ
end

# Provide list of (public) properties.
for T in (LazyCovariance, PackedLazyCovariance, EmpiricalStructFunc)
    @eval Base.propertynames(::$T) = $(Tuple(fieldnames(T)))
end

# Check that an array can be quickly indexed by 1-based linear indices.
has_standard_linear_indexing(A::AbstractArray) =
    IndexStyle(A) === IndexLinear() && firstindex(A) === 1

end # module
