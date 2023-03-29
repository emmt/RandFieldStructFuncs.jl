module StructureFunctions

export
    KolmogorovStructFunc,
    StructureFunction,
    cov

using Statistics, AsType

import Statistics: cov

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

# convert Cartesian index.
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
    normalize_support(T<:AbstractFloat, S)

yields a normalized support function with elements of floating-point type `T`
given the sampled support function `S` not necessarily normalized and throwing
an error if `S` has invalid (e.g., negative) values. The result is an array of
nonnegative values of type `T` and whose sum is equal to 1.

"""
function normalize_support(::Type{T}, S::AbstractMatrix) where {T<:AbstractFloat}
    # Check support function and compute normalization factor q = 1/sum(S).
    s = zero(T)
    flag = true
    @inbounds @simd for i in eachindex(S)
        S_i = as(T, S[i])
        flag &= (S_i ≥ zero(S_i))
        s += S_i
    end
    flag || throw(ArgumentError("support function must be nonnegative everywhere"))
    s > zero(s) || throw(ArgumentError("support function must have some nonzeros"))
    q = one(s)/s

    # Build the normalized support function.
    R = similar(S, T)
    @inbounds @simd for i in eachindex(R, S)
        R[i] = q*as(T, S[i])
    end
    return R
end

"""
    cov(f::StructureFunction, S, σ=0; shrink=false) -> C

yields the covariance of a random field whose structure function is `f` over an
support defined `S`. The range of indices to consider for the random field is
the same as that of `S` unless keyword `shrink` is true, in which case only the
indices inside the support `S` are considered.

Optional argument `σ` is the standard deviation of an additional independent
random piston.

The result is a flattened `n×n` covariance matrix with `n = length(S)` if
`shrink` is false, or `n` the number of non-zeros in the support `S` if
`shrink` is true.

The implemented method is described in the notes accompanying this package.

"""
function cov(f::StructureFunction{T},
             S::AbstractArray,
             σ::Real = zero(T);
             shrink::Bool = false) where {T<:AbstractFloat}
    # Check support function and normalize it so that sum(S) = 1.
    S = normalize_support(T, S)
    σ² = as(T, σ)^2

    # Pre-compute K(r) = ∫f(r - r′)⋅S(r′)⋅dr′
    R = CartesianIndices(S)
    K = similar(S, T)
    @inbounds for r ∈ R
        s = zero(T)
        for r′ ∈ R
            S_r′ = S[r′]
            iszero(S_r′) && continue
            s += oftype(s, f(r - r′)*S_r′)
        end
        K[r] = s
    end

    # Compute c0 = σ^2 - (1/2)⋅∫K(r)⋅S(r)⋅dr
    s = zero(T)
    @inbounds @simd for i in eachindex(K, S)
        s += oftype(s, K[i]*S[i])
    end
    c0 = σ² - s/2

    # Compute covariance.
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
                    C[i′,i] = (K[r] + K[r′] - f(r - r′))/2 + c0
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
                    C[i′,i] = (K[r] + K[r′] - f(r - r′))/2 + c0
                else
                    # Avoid computations as C is symmetric.
                    C[i′,i] = C[i,i′]
                end
            end
        end
    end
    return C
end

function countnz(S::AbstractArray{T}) where {T}
    nnz = 0 # to count number of non-zeros
    @inbounds @simd for i in eachindex(S)
        nnz += iszero(S[i]) ? 0 : 1
    end
    return nnz
end

end
