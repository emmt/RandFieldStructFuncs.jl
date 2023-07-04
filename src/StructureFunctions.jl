module StructureFunctions

export
    AbstractStructFunc,
    EmpiricalStructFunc,
    KolmogorovStructFunc,
    cov, var, diag, nobs, weights

using TypeUtils, OffsetArrays

using StatsBase, Statistics, LinearAlgebra
import Statistics: cov, var
import LinearAlgebra: diag

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
        T = promote_type(Float64, X)
        s = zero(T)
        flag = true
        @inbounds @simd for i in eachindex(S)
            S_i = S[i]
            flag &= (S_i ≥ zero(S_i))
            s += oftype(s, S_i)
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
    A = EmpiricalStructFunc{T}(S)

yields an (empty) empirical structure function with values of floating-point
type `T` and for a support `S`. An empirical structure function `A` behaves
like an array. For example:

    A[Δr]

yields the value of the empirical structure function for a displacement `Δr`
which may be specified as a Cartesian index.

The base method `push!` can be used to *integrate* data into the empirical
structure function object:

    push!(A, x)

where `x` is a random sample which can be an array of the same size as `S` or a
vector whose length is the number of non-zeros in the support `S`.

An empirical structure function object `A` has the following properties:

    A.support # normalized support
    A.values  # weighted average of values
    A.weights # cumulated weigts
    A.nobs    # number of observations

Some methods are extended to retrieve these properties:

    nobs(A)    # number of observations
    values(A)  # weighted average of values
    weights(A) # weighs of values (a.k.a. precision)
    valtype(A) # element type of values

"""
mutable struct EmpiricalStructFunc{T<:AbstractFloat,N,
                                   S<:AbstractArray{T,N},
                                   A<:OffsetArray{T,N}} <: AbstractArray{T,N}
    support::S # normalized support
    values::A  # weighted average of values
    weights::A # cumulated weights
    nobs::Int  # number of observations
end

StatsBase.nobs(A::EmpiricalStructFunc) = getfield(A, :nobs)
StatsBase.weights(A::EmpiricalStructFunc) = getfield(A, :weights)

Base.valtype(A::EmpiricalStructFunc) = valtype(typeof(A))
Base.valtype(::Type{<:EmpiricalStructFunc{T}}) where {T} = T
Base.values(A::EmpiricalStructFunc) = getfield(A, :values)

# Make properties read-only.
@inline Base.getproperty(A::EmpiricalStructFunc, f::Symbol) = getfield(A, f)
Base.setproperty!(A::EmpiricalStructFunc, f::Symbol) = error(
    "attempt to ", (f ∈ propertynames(A) ? "modify read-only" : "access non-existing"),
    " property `$f`")

EmpiricalStructFunc(S::AbstractArray{T,N}) where {T,N} =
    EmpiricalStructFunc{float(T)}(S)

function EmpiricalStructFunc{T}(S::AbstractArray{<:Any,N}) where {T<:AbstractFloat,N}
    S = normalize_support(T, S)
    inds = map(r -> (first(r) - last(r)):(last(r) - first(r)), axes(S))
    dims = map(d -> 2d - 1, size(S))
    vals = OffsetArray(zeros(T, dims), inds)
    wgts = OffsetArray(zeros(T, dims), inds)
    return EmpiricalStructFunc{T,N,typeof(S),typeof(vals)}(S, vals, wgts, 0)
end

# Implement abstract array API.
Base.length(A::EmpiricalStructFunc) = length(A.values)
Base.size(A::EmpiricalStructFunc) = size(A.values)
Base.axes(A::EmpiricalStructFunc) = axes(A.values)
Base.IndexStyle(::EmpiricalStructFunc{T,N,S,A}) where {T,N,S,A} = IndexStyle(A)

@inline function Base.getindex(A::EmpiricalStructFunc, I::Vararg{Int})
    vals = A.values
    @boundscheck checkbounds(Bool, vals, I...) || throw(BoundsError(A, I...))
    @inbounds vals[I...]
end

@inline function Base.setindex!(A::EmpiricalStructFunc, x, I::Vararg{Int})
    vals = A.values
    @boundscheck checkbounds(Bool, vals, I...) || throw(BoundsError(A, I...))
    @inbounds vals[I...] = x
    return A
end

function Base.push!(A::EmpiricalStructFunc{T,N},
                    x::Union{AbstractArray{<:Real,N},
                             AbstractVector{<:Real}}) where {T,N}
    S = A.support
    if x isa AbstractArray{<:Real,N} && axes(x) == axes(S)
        # Assume x is for all the nodes, inside and outside the support.
        unsafe_update!(Val(:full), A, x)
    elseif x isa AbstractVector{<:Real} && axes(x) == (Base.OneTo(countnz(S)),)
        # Assume x is only for the nodes inside the support.
        unsafe_update!(Val(:sparse), A, x)
    else
        throw(DimensionMismatch("incompatible dimensions/indices"))
    end
    A.nobs += 1
    return A
end

function unsafe_update!(::Val{:full},
                        A::EmpiricalStructFunc{T,N},
                        x::AbstractArray{<:Real,N}) where {T,N}
    S = A.support
    R = CartesianIndices(S)
    @inbounds for r in R
        S_r = S[r]
        iszero(S_r) && continue
        x_r = as(T, x[r])
        for r′ in R
            S_r′ = S[r′]
            iszero(S_r′) && continue
            x_r′ = as(T, x[r′])
            unsafe_update!(A, r - r′, S_r*S_r′, abs2(x_r - x_r′))
        end
    end
    nothing
end

function unsafe_update!(::Val{:sparse},
                        A::EmpiricalStructFunc{T,N},
                        x::AbstractVector{<:Real}) where {T,N}
    S = A.support
    R = CartesianIndices(S)
    i = 0
    @inbounds for r in R
        S_r = S[r]
        iszero(S_r) && continue
        x_r = as(T, x[i += 1])
        i′ = 0
        for r′ in R
            S_r′ = S[r′]
            iszero(S_r′) && continue
            x_r′ = as(T, x[i′ += 1])
            unsafe_update!(A, r - r′, S_r*S_r′, abs2(x_r - x_r′))
        end
    end
    nothing
end

@inline function unsafe_update!(A::EmpiricalStructFunc{T,N},
                                Δr::CartesianIndex{N},
                                wgt::T,
                                val::T) where {T, N}
    wgts = A.weights
    vals = A.values
    vals[Δr] = (wgts[Δr]*vals[Δr] + wgt*val)/(wgts[Δr] + wgt)
    wgts[Δr] += wgt
end

# Provide list of (public) properties.
for T in (LazyCovariance, PackedLazyCovariance, EmpiricalStructFunc)
    @eval Base.propertynames(::$T) = $(Tuple(fieldnames(T)))
end

# Check that an array can be quickly indexed by 1-based linear indices.
has_standard_linear_indexing(A::AbstractArray) =
    IndexStyle(A) === IndexLinear() && firstindex(A) === 1

end # module
