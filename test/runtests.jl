# TODO: more tests!

using ChainRules, Test, FDM, LinearAlgebra, Random
using ChainRules: extern, accumulate, accumulate!, store!, @scalar_rule,
    Wirtinger, wirtinger_primal, wirtinger_conjugate, add_wirtinger, mul_wirtinger,
    Zero, add_zero, mul_zero, One, add_one, mul_one, Casted, cast, add_casted, mul_casted,
    DNE, Thunk, Casted
using Base.Broadcast: broadcastable

include("test_util.jl")

@testset "ChainRules" begin
    include("differentials.jl")
    include("rules.jl")
    @testset "rules" begin
        include(joinpath("rules", "base.jl"))
        include(joinpath("rules", "array.jl"))
        @testset "linalg" begin
            include(joinpath("rules", "linalg", "dense.jl"))
            include(joinpath("rules", "linalg", "diagonal.jl"))
            include(joinpath("rules", "linalg", "symmetric.jl"))
            include(joinpath("rules", "linalg", "factorization.jl"))
        end
        include(joinpath("rules", "broadcast.jl"))
        include(joinpath("rules", "blas.jl"))
        include(joinpath("rules", "nanmath.jl"))
        include(joinpath("rules", "specialfunctions.jl"))
    end
end
