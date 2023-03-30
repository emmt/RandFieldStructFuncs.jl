module TestingStructureFunctions

using Test, Statistics, LinearAlgebra
using StructureFunctions
using StructureFunctions: LazyCovariance, PackedLazyCovariance

@testset "StructureFunctions.jl" begin
    f = KolmogorovStructFunc(1.2)
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
    C = cov(f, S, σ)
    @test size(C) == (n, n)
    @test C == C' # is matrix symmetric?
    LC = LazyCovariance(f, S, σ)
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
    C = cov(f, S, σ; pack=true)
    @test size(C) == (nnz, nnz)
    @test C == C' # is matrix symmetric?
    LC = PackedLazyCovariance(f, S, σ)
    @test size(LC) == (nnz,nnz)
    @test axes(LC) === map(Base.OneTo, size(LC))
    @test LC ≈ C
    @test diag(LC) === var(LC)

    # Empirical Structure function.
    A =  EmpiricalStructureFunction(S)
    let countnz = StructureFunctions.countnz
        @test values(A) === A.values
        @test valtype(A) === eltype(values(A))
        @test A.nobs == 0
        @test countnz(A.support) == countnz(S)
        @test extrema(A.values) == (0, 0)
        @test extrema(A.weights) == (0, 0)
        for i in 1:43
            x = randn(valtype(A), size(A.support))
            push!(A, x)
            @test A.nobs == i
            wmin, wmax = extrema(A.weights)
            @test wmin ≥ 0
            @test wmax > 0
        end
        n = A.nobs
        nnz = countnz(A.support)
        for i in 1:2
            x = randn(valtype(A), nnz)
            push!(A, x)
            @test A.nobs == i+n
            wmin, wmax = extrema(A.weights)
            @test wmin ≥ 0
            @test wmax > 0
        end
    end
    @test all(A .== A.values)
    for i in eachindex(A)
        A[i] = zero(eltype(A))
    end
    @test all(iszero, A.values)
    for i in eachindex(A)
        A[i] = one(eltype(A))
    end
    @test all(isone, A.values)

end

end # module

nothing
