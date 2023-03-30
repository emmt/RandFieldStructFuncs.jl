module StructureFunctions

export
    EmpiricalStructureFunction,
    KolmogorovStructFunc,
    StructureFunction,
    cov, var, diag

using AsType, OffsetArrays

import Statistics: cov, var
import LinearAlgebra: diag

"""
    StructureFunction{T}

is the abstract type of structure functions using floating-point type `T` for
their computations.

A structure function `f` of a random field `ϕ` is a callable object such that:

    f(Δr) = ⟨[ϕ(r + Δr) - ϕ(r)]^2⟩

where `⟨…⟩` denotes expectation while `r` and `Δr` are Cartesian coordinates in
units of the grid sampling step.

For sub-types of `StructureFunction{T}`, it is only necessary to implement
calling the structure function object with a tuple of coordinates.

"""
abstract type StructureFunction{T<:AbstractFloat} <: Function; end

# convert Cartesian index. NOTE: Only works for Julia ≥ 1.3
(f::StructureFunction)(r::CartesianIndex) = f(Tuple(r))

Base.convert(::Type{S}, f::S) where {S<:StructureFunction} = f

"""
    KolmogorovStructFunc{T}(r0)

yields a Kolmogorov structure function for Fried's parameter `r0` (in units of
the grid sampling step).

"""
struct KolmogorovStructFunc{T<:AbstractFloat} <: StructureFunction{T}
    r0::T
    q::T  # to store precompte value of 1/r0^2
    KolmogorovStructFunc{T}(r0::T) where {T<:AbstractFloat} =
        new{T}(r0, one(T)/r0^2)
end

# Compute structure function.
@inline (f::KolmogorovStructFunc{T})(r::NTuple{2,Real}) where {T<:AbstractFloat} = f(map(as(T), r))

(f::KolmogorovStructFunc{T})(r::NTuple{2,T}) where {T<:AbstractFloat} =
    as(T,6.88)*(f.q*(r[1]^2 + r[2]^2))^as(T,5/6)

# Outer constructors.
KolmogorovStructFunc{T}(r0::Real) where {T<:AbstractFloat} = KolmogorovStructFunc{T}(as(T, r0))
KolmogorovStructFunc(r0::T) where {T<:AbstractFloat} = KolmogorovStructFunc{T}(r0)

# Conversions.
KolmogorovStructFunc(f::KolmogorovStructFunc) = f
KolmogorovStructFunc{T}(f::KolmogorovStructFunc{T}) where {T} = f
KolmogorovStructFunc{T}(f::KolmogorovStructFunc) where {T} = KolmogorovStructFunc{T}(f.r0)
for type in (:StructureFunction, :KolmogorovStructFunc)
    @eval begin
        Base.convert(::Type{$type{T}}, f::KolmogorovStructFunc{T}) where {T} = f
        Base.convert(::Type{$type{T}}, f::KolmogorovStructFunc) where {T} =
            KolmogorovStructFunc{T}(f)
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

yields the sum of values in `S` throwing an exception is `S` is not valid to
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
    var(f, S, σ=0) -> V

yields the non-uniform variance `V` of a random field having a structure
function `f` on a support `S` and a piston mode with standard deviation `σ`.
The covariance between two nodes of Cartesian coordinates `r` and `r′` is then
given by:

    Cov(r,r′) = (V[r] + V[r′] - f(r - r′))/2

if `S[r]` and `S[r′]` are both non-zero, the covariance being zero otherwise.

"""
function var(f::StructureFunction{T},
             S::AbstractArray{X},
             σ::Real = zero(T)) where {T<:AbstractFloat,X<:Real}
    # Check piston variance.
    σ ≥ zero(σ) || throw(ArgumentError("piston variance must be nonnegative"))
    σ² = as(T, σ)^2

    # Check support and compute normalization factor.
    q = as(T, check_support(S))

    # Pre-compute K(r) = ∫f(r - r′)⋅S(r′)⋅dr′
    R = CartesianIndices(S)
    K = similar(S, T)
    @inbounds for r ∈ R
        S_r = S[r]
        if iszero(S_r)
            K[r] = zero(T)
        else
            s = zero(T)
            if X <: Bool
                for r′ ∈ R
                    S_r′ = S[r′]
                    iszero(S_r′) && continue
                    s += oftype(s, f(r - r′))
                end
            else
                for r′ ∈ R
                    S_r′ = S[r′]
                    iszero(S_r′) && continue
                    s += oftype(s, f(r - r′))*oftype(s, S_r′)
                end
            end
            K[r] = s/q
        end
    end

    # Compute c0 = σ^2 - (1/2)⋅∫K(r)⋅S(r)⋅dr
    s = zero(T)
    if eltype(S) === Bool
        @inbounds @simd for i in eachindex(K, S)
            s += K[i]
        end
    else
        @inbounds @simd for i in eachindex(K, S)
            s += K[i]*oftype(s, S[i])
        end
    end
    c0 = σ² - s/2q

    # Overwrite K with the variance.
    @inbounds @simd for i in eachindex(K)
        K[i] = ifelse(iszero(K[i]), zero(T), K[i] + c0)
    end

    return K
end

"""
    cov(f::StructureFunction, S, σ=0; shrink=false) -> C

yields the covariance of a random field whose structure function is `f` over an
support defined by `S` and whose piston mode has a standard deviation of `σ`.

The range of indices to consider for the random field is the same as that of
`S` unless keyword `shrink` is true, in which case only the indices inside the
support `S` are considered.

The result is a flattened `n×n` covariance matrix with `n = length(S)` if
`shrink` is false, or `n` the number of non-zeros in the support `S` if
`shrink` is true.

The implemented method is described in the notes accompanying this package.

"""
function cov(f::StructureFunction{T},
             S::AbstractArray{<:Real},
             σ::Real = zero(T);
             shrink::Bool = false) where {T<:AbstractFloat}
    # Compute non-uniform variance.
    V = var(f, S, σ)

    # Compute covariance.
    R = CartesianIndices(S)
    if shrink
        nnz = countnz(S)
        C = Array{T}(undef, (nnz, nnz))
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
                    C[i′,i] = ((V[r] + V[r′]) - f(r - r′))/2
                else
                    # Avoid computations as C is symmetric.
                    C[i′,i] = C[i,i′]
                end
            end
        end
    else
        n = length(S)
        C = zeros(T, (n, n))
        @inbounds for (i,r) ∈ enumerate(R)
            iszero(S[r]) && continue
            for (i′,r′) ∈ enumerate(R)
                iszero(S[r′]) && continue
                if i ≤ i′
                    # Instantiate covariance.
                    C[i′,i] = ((V[r] + V[r′]) - f(r - r′))/2
                else
                    # Avoid computations as C is symmetric.
                    C[i′,i] = C[i,i′]
                end
            end
        end
    end
    return C
end

"""
    StructureFunctions.LazyCovariance(f, S, σ) -> Cov

yields an object that can be used as:

    Cov[i, j]

to compute *on the fly* the covariance of a random field whose structure
function is `f` over a support defined by `S` and whose piston mode has a
standard deviation of `σ`. Indices `i` and `j` can be linear indices or
Cartesian indices that must be valid to index `S`.

"""
struct LazyCovariance{T<:AbstractFloat,N,
                      F<:StructureFunction{T},
                      S<:AbstractArray{T,N},
                      D<:AbstractArray{T,N}} <: AbstractMatrix{T}
    sf::F   # structure function
    sup::S  # normalized support
    diag::D # diagonal entries, non-uniform variance

    # The inner constructor is to ensure that arrays standard linear indexing
    # and the the same axes.
    function LazyCovariance(sf::F, sup::S, diag::D) where {T<:AbstractFloat,N,
                                                           F<:StructureFunction{T},
                                                           S<:AbstractArray{T,N},
                                                           D<:AbstractArray{T,N}}
        has_standard_linear_indexing(sup) || throw(ArgumentError(
            "support array must have standard linear indexing"))
        has_standard_linear_indexing(diag) || throw(ArgumentError(
            "array of diagonal entries must have standard linear indexing"))
        axes(sup) == axes(diag) || throw(DimensionMismatch(
            "support and variance arrays have incompatible dimensions/indices"))
        return new{T,N,F,S,D}(sf, sup, diag)
    end
end

# Getters.
StructureFunction(A::LazyCovariance) = A.sf
support(A::LazyCovariance) = A.sup
diag(A::LazyCovariance) = A.diag
var(A::LazyCovariance) = diag(A)

# Constructor.
function LazyCovariance(f::StructureFunction{T},
                        S::AbstractArray{<:Real,N},
                        σ::Real = zero(T)) where {T<:AbstractFloat,N}
    sup = normalize_support(T, S)
    diag = var(f, S, σ)
    return LazyCovariance(f, sup, diag)
end

# Implement abstract array API.
Base.length(A::LazyCovariance) = begin
    n = length(diag(A))
    return n^2
end
Base.size(A::LazyCovariance) = begin
    n = length(diag(A))
    return (n, n)
end
Base.axes(A::LazyCovariance) = begin
    r = Base.OneTo(length(diag(A)))
    return (r, r)
end
Base.IndexStyle(::LazyCovariance) = IndexCartesian()

@inline function Base.getindex(A::LazyCovariance{T}, i::Int, j::Int) where {T}
    S = support(A)
    @boundscheck (checkbounds(Bool, S, i) & checkbounds(Bool, S, j)) ||
        throw(BoundsError(A, (i, j)))
    @inbounds begin
        if iszero(S[i]) | iszero(S[j])
            zero(T)
        else
            R, D, var = CartesianIndices(S), StructureFunction(A), diag(A)
            Δr = R[i] - R[j]
            ((var[i] + var[j]) - D(Δr))/2
        end
    end
end

@inline function Base.getindex(A::LazyCovariance{T,N},
                               i::CartesianIndex{N},
                               j::CartesianIndex{N}) where {T,N}
    S = support(A)
    @boundscheck (checkbounds(Bool, S, i) & checkbounds(Bool, S, j)) ||
        throw(BoundsError(A, (i,j)))
    @inbounds begin
        if iszero(S[i]) | iszero(S[j])
            zero(T)
        else
            Δr = i - j
            D = StructureFunction(A)
            var = diag(A)
            ((var[i] + var[j]) - D(Δr))/2
        end
    end
end

"""
    StructureFunctions.ShrinkedLazyCovariance(f, S, σ) -> Cov

yields an object that can be used as:

    Cov[i,j]

to compute *on the fly* the covariance of a random field whose structure
function is `f` over a support defined by `S` and whose piston mode has a
standard deviation of `σ`. Indices `i` and `j` are linear indices in the range
`1:nnz` with `nnz` the number of non-zeros in the support `S`.

"""
struct ShrinkedLazyCovariance{T<:AbstractFloat,N,
                              F<:StructureFunction{T},
                              S<:AbstractArray{T,N},
                              M<:AbstractArray{Bool,N},
                              I<:AbstractVector{<:CartesianIndex{N}},
                              D<:AbstractVector{T}} <: AbstractMatrix{T}
    sf::F   # structure function
    sup::S  # normalized support
    mask::M # boolean support
    inds::I # linear index in variance -> Cartesian index in support
    diag::D # diagonal entries, also non-uniform variances
    function ShrinkedLazyCovariance(sf::F, sup::S, mask::M, inds::I,
                                    diag::D) where {T<:AbstractFloat,N,
                                                    F<:StructureFunction{T},
                                                    I<:AbstractVector{<:CartesianIndex{N}},
                                                    S<:AbstractArray{T,N},
                                                    M<:AbstractArray{Bool,N},
                                                    D<:AbstractVector{T}}
        check_struct(ShrinkedLazyCovariance, sf, sup, mask, inds, diag)
        return new{T,N,F,S,M,I,D}(sf, sup, mask, inds, diag)
    end
end

check_struct(A::ShrinkedLazyCovariance) = check_struct(
    ShrinkedLazyCovariance, StructureFunction(A), support(A), mask(A), indices(A), diag(A))

function check_struct(::Type{<:ShrinkedLazyCovariance},
                      sf::F, sup::S, mask::M, inds::I,
                      diag::D) where {T<:AbstractFloat,N,
                                      F<:StructureFunction{T},
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

StructureFunction(A::ShrinkedLazyCovariance) = A.sf
support(A::ShrinkedLazyCovariance) = A.sup
mask(A::ShrinkedLazyCovariance) = A.mask
indices(A::ShrinkedLazyCovariance) = A.inds
diag(A::ShrinkedLazyCovariance) = A.diag
var(A::ShrinkedLazyCovariance) = diag(A)

function ShrinkedLazyCovariance(f::StructureFunction{T},
                                S::AbstractArray{<:Real,N},
                                σ::Real = zero(T)) where {T<:AbstractFloat,N}
    return ShrinkedLazyCovariance(LazyCovariance(f, S, σ))
end

function ShrinkedLazyCovariance(A::LazyCovariance{T,N})  where {T<:AbstractFloat,N}
    f, S = StructureFunction(A), support(A)
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
            diag[i] = var(A)[r]
        end
    end
    return ShrinkedLazyCovariance(f, S, mask, inds, diag)
end

# Implement abstract array API.
Base.length(A::ShrinkedLazyCovariance) = begin
    n = length(indices(A))
    return n^2
end
Base.size(A::ShrinkedLazyCovariance) = begin
    n = length(indices(A))
    return (n, n)
end
Base.axes(A::ShrinkedLazyCovariance) = begin
    r = axes(indices(A))
    return (r..., r...)
end
Base.IndexStyle(::ShrinkedLazyCovariance) = IndexCartesian()

@inline function Base.getindex(A::ShrinkedLazyCovariance, i::Int, j::Int)
    var = diag(A)
    @boundscheck (checkbounds(Bool, var, i) & checkbounds(Bool, var, j)) ||
        throw(BoundsError(A, (i, j)))
    @inbounds begin
        R, D = indices(A), StructureFunction(A)
        Δr = R[i] - R[j]
        cov = ((var[i] + var[j]) - D(Δr))/2
    end
    return cov
end

"""
    A = EmpiricalStructureFunction{T}(S)

yields an (empty) empirical structure function with values of floating-point
type `T` and for a support `S`.

The base method `push!` can be used to *integrate* data into the sampled
structure function object:

    push!(A, x)

where `x` is a random sample which can be an array of the same size as `S` or a
vector whose length is the number of non-zeros in the support `S`.

An empirical structure function object `A` has the following properties:

    A.support # normalized support
    A.values  # weighted average of values
    A.weights # cumulated weigts
    A.nobs    # number of observations

Base methods `values(A)` and `valtype(A)` yield the integrated values and their
type for the sampled structure function `A`.

Unexported methods `StructureFunctions.support(A)`,
`StructureFunctions.weights(A)`, and `StructureFunctions.nobs(A)` yield the
support, the integrated weights, and the number of observations for the sampled
structure function `A`.

"""
mutable struct EmpiricalStructureFunction{T<:AbstractFloat,N,
                                          S<:AbstractArray{T,N},
                                          A<:OffsetArray{T,N}}
    support::S # normalized support
    values::A  # weighted average of values
    weights::A # cumulated weights
    nobs::Int  # number of observations
end

Base.valtype(A::EmpiricalStructureFunction) = valtype(typeof(A))
Base.valtype(::Type{<:EmpiricalStructureFunction{T}}) where {T} = T
Base.values(A::EmpiricalStructureFunction) = A.values

support(A::EmpiricalStructureFunction) = A.support
weights(A::EmpiricalStructureFunction) = A.weights
nobs(A::EmpiricalStructureFunction) = A.nobs

EmpiricalStructureFunction(S::AbstractArray{T,N}) where {T,N} =
    EmpiricalStructureFunction{float(T)}(S)

function EmpiricalStructureFunction{T}(S::AbstractArray{<:Any,N}) where {T<:AbstractFloat,N}
    S = normalize_support(T, S)
    inds = map(r -> (first(r) - last(r)):(last(r) - first(r)), axes(S))
    dims = map(d -> 2d - 1, size(S))
    vals = OffsetArray(zeros(T, dims), inds)
    wgts = OffsetArray(zeros(T, dims), inds)
    return EmpiricalStructureFunction{T,N,typeof(S),typeof(vals)}(S, vals, wgts, 0)
end

function Base.push!(A::EmpiricalStructureFunction{T,N},
                    x::Union{AbstractArray{<:Real,N},
                             AbstractVector{<:Real}}) where {T,N}
    S = support(A)
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
                        A::EmpiricalStructureFunction{T,N},
                        x::AbstractArray{<:Real,N}) where {T,N}
    S = support(A)
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
                        A::EmpiricalStructureFunction{T,N},
                        x::AbstractVector{<:Real}) where {T,N}
    S = support(A)
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

@inline function unsafe_update!(A::EmpiricalStructureFunction{T,N},
                                Δr::CartesianIndex{N},
                                wgt::T,
                                val::T) where {T, N}
    wgts = weights(A)
    vals = values(A)
    vals[Δr] = (wgts[Δr]*vals[Δr] + wgt*val)/(wgts[Δr] + wgt)
    wgts[Δr] += wgt
end

# Check that an array can be quickly indexed by 1-based linear indices.
has_standard_linear_indexing(A::AbstractArray) =
    IndexStyle(A) === IndexLinear() && firstindex(A) === 1

end # module
