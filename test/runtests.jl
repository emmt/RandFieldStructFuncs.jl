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
    C = cov(f, S)
    @test size(C) == (n, n)
    @test C == C'
    C = cov(f, S, 0.1)
    @test size(C) == (n, n)
    @test C == C'
    C = cov(f, S; shrink=true)
    @test size(C) == (nnz, nnz)
    @test C == C'
end

end # module

nothing
