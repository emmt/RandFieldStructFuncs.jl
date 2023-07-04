module TestingStructureFunctions

using Test, Statistics, LinearAlgebra, EasyRanges
using StructureFunctions
using StructureFunctions: LazyCovariance, PackedLazyCovariance

function max_abs_dif(A::AbstractArray, B::AbstractArray)
    v = zero(promote_type(eltype(A), eltype(B)))
    for i in @range CartesianIndices(A) ∩ CartesianIndices(B)
        v = max(v, abs(A[i] - B[i]))
    end
    return v
end

@testset "StructureFunctions.jl" begin
    f = @inferred KolmogorovStructFunc(1.2)
    let f32 = KolmogorovStructFunc{Float32}(f)
        z1 = CartesianIndex(0,0)
        z2 = (zero(BigFloat), zero(Int))
        @test f isa KolmogorovStructFunc{Float64}
        @test f32 isa KolmogorovStructFunc{Float32}
        @test f(z1) === zero(Float64)
        @test f(z2) === zero(Float64)
        @test f32(z1) === zero(Float32)
        @test f32(z2) === zero(Float32)
    end
    S = [sqrt(x^2 + y^2) < 4.5 for x in -6:6, y in -6:6]
    n = length(S)
    nnz = count(x -> x > zero(x), S)
    @test sum(StructureFunctions.normalize_support(Float64, S)) ≈ 1

    # Full covariance matrices.
    C = cov(f, S)
    @test size(C) == (n, n)
    @test C == C' # is matrix symmetric?
    σ = 0.1
    C = @inferred cov(f, S, σ)
    @test size(C) == (n, n)
    @test C == C' # is matrix symmetric?
    LC = @inferred LazyCovariance(f, S, σ)
    @test size(LC) == (n,n)
    @test axes(LC) === map(Base.OneTo, size(LC))
    @test LC ≈ C
    @test diag(LC) === var(LC)
    R = CartesianIndices(S)
    flag = true # to test different kinds of indices
    for (i,r) in enumerate(R)
        for (i′,r′) in enumerate(R)
            flag &= LC[i,i′] == LC[r,r′]
        end
    end
    @test flag

    # Packed covariance matrices.
    C = @inferred cov(f, S, σ; pack=true)
    @test size(C) == (nnz, nnz)
    @test C == C' # is matrix symmetric?
    LC = @inferred PackedLazyCovariance(f, S, σ)
    @test size(LC) == (nnz,nnz)
    @test axes(LC) === map(Base.OneTo, size(LC))
    @test LC ≈ C
    @test diag(LC) === var(LC)

    # Empirical structure function.
    A = @inferred EmpiricalStructFunc(S, nothing) # slow
    B = @inferred EmpiricalStructFunc(S) # fast
    let countnz = StructureFunctions.countnz
        @test countnz(A.support) == countnz(S)
        @test minimum(A.support) ≥ 0
        @test A.support == B.support
        let A_vals = values(A), B_vals = values(B)
            @test extrema(A_vals) == (0, 0)
            @test valtype(A) === eltype(A_vals)
            @test extrema(B_vals) == (0, 0)
            @test valtype(B) === eltype(B_vals)
        end
        @test max_abs_dif(A.den, B.den) ≤ 1e-10
        @test extrema(A.num) == (0, 0)
        @test extrema(B.num) == (0, 0)
        @test nobs(A) === A.nobs
        @test A.nobs == 0
        @test nobs(B) === B.nobs
        @test B.nobs == 0
        for i in 1:3
            x = randn(valtype(A), size(A.support))
            push!(A, x)
            push!(B, x)
            @test A.nobs == i
            @test B.nobs == i
            let A_vals = values(A), B_vals = values(B)
                @test max_abs_dif(A_vals, B_vals) ≤ 1e-10
            end
        end
        n = A.nobs
        nnz = countnz(A.support)
        for i in 1:2
            x = randn(valtype(A), nnz)
            push!(A, x)
            @test A.nobs == i+n
            @test nobs(A) === A.nobs
        end
    end

end

end # module

nothing
