using AbstractFFTs, FFTW
using EasyRanges, TypeUtils

"""
    fftshiftaxis(dim) -> rng

yields the range of indices along the dimension of length `dim` of an FFT
transformed array (after having applied `fftshift`).

"""
function fftshiftaxis(dim::Integer)
    dim = as(Int, dim)
    n = div(dim, 2) # Nyquist frequency
    return -n : dim - 1 - n
end

"""
    fftshiftaxes(dims...) -> rngs

yields the ranges of indices along the dimensions `dims...` of an FFT
transformed array (after having applied `fftshift`).

"""
fftshiftaxes(dims::Integer...) = fftshiftaxes(dims)
fftshiftaxes(dims::Tuple{Vararg{Integer}}) = map(fftshiftaxis, dims)

"""
    goodfftdim(len)

yields the smallest integer which is greater or equal `len` and which is a
multiple of powers of 2, 3 and/or 5. If argument is an array dimesion list
(i.e. a tuple of integers), a tuple of good FFT dimensions is returned.

Also see: [`goodfftdims`](@ref), [`rfftdims`](@ref), [`FFTOperator`](@ref).

"""
goodfftdim(len::Int) = nextprod(goodfftdim_multiples, as(Int, len))
const goodfftdim_multiples = VERSION ≥ v"1.6" ? (2,3,5) : [2,3,5]

"""
    goodfftdims(dims)

yields a list of dimensions suitable for computing the FFT of arrays whose
dimensions are `dims` (a tuple or a vector of integers).

Also see: [`goodfftdim`](@ref), [`rfftdims`](@ref), [`FFTOperator`](@ref).

"""
goodfftdims(dims::Integer...) = goodfftdims(dims)
goodfftdims(dims::Union{AbstractVector{<:Integer},Tuple{Vararg{Integer}}}) =
    map(goodfftdim, dims)

"""
    rfftdims(dims)

yields the dimensions of the complex array produced by a real-complex FFT of a
real array of size `dims`.

Also see: [`goodfftdim`](@ref), [`FFTOperator`](@ref).

"""
rfftdims(dims::Integer...) = rfftdims(dims)
rfftdims(dims::NTuple{N,Integer}) where {N} =
    ntuple(d -> (d == 1 ? (as(Int, dims[d]) >>> 1) + 1 : as(Int, dims[d])), Val(N))
# NOTE: The above version is equivalent but much faster than
#     ((dims[1] >>> 1) + 1, dims[2:end]...)
# which is not optimized out by the compiler.

"""
    k = fftfreq([T,] dim)

yields a vector of `dim` Discrete Fourier Transform (DFT) frequency indices:

    k = [0, 1, 2, ..., n-1, -n, ..., -2, -1]   if dim = 2*n
    k = [0, 1, 2, ..., n,   -n, ..., -2, -1]   if dim = 2*n + 1

depending whether `dim` is even or odd. These rules are compatible to the
assumptions made by `fftshift` (which to see) in the sense that:

    fftshift(fftfreq(dim)) = [-n, ..., -2, -1, 0, 1, 2, ...]

Optional argument `T` is to specify the element type of the result.

"""
fftfreq(dim::Integer) = fftfreq(Int, dim)
function fftfreq(::Type{T}, dim::Integer) where {T}
    dim = as(Int, dim)
    n = div(dim, 2)
    f = Array{T}(undef, dim)
    @inbounds begin
        for k in 1:dim-n
            f[k] = k - 1
        end
        for k in dim-n+1:dim
            f[k] = k - (1 + dim)
        end
    end
    return f
end

"""
    f = fftfreq([T,] dim, step) -> fftfreq(dim)./(dim*step)

yields a vector of `dim` Discrete Fourier Transform (DFT) frequencies with
`step` the sample spacing in the direct space. The result is a floating-point
vector with `dim` elements set with the frequency bin centers in cycles per
unit of the sample spacing (with zero at the start). For example, if the sample
spacing is in seconds, then the frequency unit is cycles/second.

Optional argument `T` is to specify the floating-point type of the result.

"""
fftfreq(dim::Integer, step::Number) = fftfreq(float(typeof(step)), dim, step)
function fftfreq(::Type{T}, dim::Integer, step::Number) where {T<:AbstractFloat}
    dim = as(Int, dim)
    step = as(T, step)
    n = div(dim, 2)
    scl = inv(dim*step) # 1/(dim*step)
    f = Array{T}(undef, dim)
    @inbounds begin
        for k in 1:dim-n
            f[k] = (k - 1)*scl
        end
        for k in dim-n+1:dim
            f[k] = (k - (1 + dim))*scl
        end
    end
    return f
end

function zeropad(A::AbstractArray{T,N}, dims::Dims{N}) where {T,N}
    return zeropad!(similar(A, dims), A)
end

function zeropad!(dst::AbstractArray{<:Any,N}, src::AbstractArray{<:Any,N}) where {N}
    fill!(dst, zero(eltype(dst)))
    @inbounds @simd for i ∈ @range CartesianIndices(dst) ∩ CartesianIndices(src)
        dst[i] = src[i]
    end
    return dst
end
