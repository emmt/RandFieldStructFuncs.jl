module TestingStructureFunctions

using StructureFunctions
using Test, Statistics

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

    # Covariance matrix.
    C = cov(f, S)
    @test size(C) == (n, n)
    @test C == C'
    C = cov(f, S, 0.1)
    @test size(C) == (n, n)
    @test C == C'
    C = cov(f, S; shrink=true)
    @test size(C) == (nnz, nnz)
    @test C == C'

    # Sampled Structure function.
    A =  SampledStructureFunction(S)
    let nobs = StructureFunctions.nobs,
        weights = StructureFunctions.weights,
        support = StructureFunctions.support,
        countnz = StructureFunctions.countnz
        @test valtype(A) === eltype(values(A))
        @test nobs(A) == 0
        @test countnz(support(A)) == countnz(S)
        @test extrema(values(A)) == (0, 0)
        @test extrema(weights(A)) == (0, 0)
        for i in 1:43
            x = randn(valtype(A), size(support(A)))
            push!(A, x)
            @test nobs(A) == i
            wmin, wmax = extrema(weights(A))
            @test wmin ≥ 0
            @test wmax > 0
        end
        n = nobs(A)
        nnz = countnz(support(A))
        for i in 1:2
            x = randn(valtype(A), nnz)
            push!(A, x)
            @test nobs(A) == i+n
            wmin, wmax = extrema(weights(A))
            @test wmin ≥ 0
            @test wmax > 0
        end
    end

end

end # module

nothing
