using LinearAlgebra.BLAS: gemm, gemv

@testset "BLAS" begin
    @testset "gemm" begin
        rng = MersenneTwister(1)
        dims = 3:5
        for m in dims, n in dims, p in dims, tA in ('N', 'T'), tB in ('N', 'T')
            α = randn(rng)
            A = randn(rng, tA === 'N' ? (m, n) : (n, m))
            B = randn(rng, tB === 'N' ? (n, p) : (p, n))
            C = gemm(tA, tB, α, A, B)
            fAB, (dtA, dtB, dα, dA, dB) = rrule(gemm, tA, tB, α, A, B)
            @test C ≈ fAB
            @test dtA isa ChainRules.DNERule
            @test dtB isa ChainRules.DNERule
            for (f, x, dx) in [(X->gemm(tA, tB, X, A, B), α, dα),
                               (X->gemm(tA, tB, α, X, B), A, dA),
                               (X->gemm(tA, tB, α, A, X), B, dB)]
                ȳ = randn(rng, size(C)...)
                x̄_ad = dx(ȳ)
                x̄_fd = j′vp(central_fdm(5, 1), f, ȳ, x)
                @test x̄_ad ≈ x̄_fd rtol=1e-9 atol=1e-9
                if size(x) != ()  # A and B
                    @test dx isa Rule{<:Function,<:Function}
                    x̄ = zeros(size(x)...)
                    ChainRules.accumulate!(x̄, dx, ȳ)
                    @test x̄ ≈ x̄_ad rtol=1e-9 atol=1e-9
                else  # α
                    @test dx isa Rule{<:Function,Nothing}
                end
            end
        end
    end
    @testset "gemv" begin
        rng = MersenneTwister(2)
        for n in 3:5, m in 3:5, t in ('N', 'T')
            α = randn(rng)
            A = randn(rng, m, n)
            x = randn(rng, t === 'N' ? n : m)
            y, (dt, dα, dA, dx) = rrule(gemv, t, α, A, x)
            @test y ≈ α * (t === 'N' ? A : A') * x
            @test dt isa ChainRules.DNERule
            for (f, z, dz) in [(z->gemv(t, z, A, x), α, dα),
                               (z->gemv(t, α, z, x), A, dA),
                               (z->gemv(t, α, A, z), x, dx)]
                ȳ = randn(rng, size(y)...)
                z̄_ad = dz(ȳ)
                z̄_fd = j′vp(central_fdm(5, 1), f, ȳ, z)
                @test z̄_ad ≈ z̄_fd atol=1e-9 rtol=1e-9
                if size(z) != ()  # A and x
                    @test dz isa Rule{<:Function,<:Function}
                    z̄ = zeros(size(z)...)
                    ChainRules.accumulate!(z̄, dz, ȳ)
                    @test z̄ ≈ z̄_ad atol=1e-9 rtol=1e-9
                else  # α
                    @test dz isa Rule{<:Function,Nothing}
                end
            end
        end
    end
end
