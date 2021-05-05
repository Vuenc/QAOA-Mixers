using Test
using TestSetExtensions
using LinearAlgebra
using Qaintessent
using QAOAMixers

##==----------------------------------------------------------------------------------------------------------------------


# adapted from https://github.com/FluxML/Zygote.jl/blob/master/test/gradcheck.jl
function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
        if eltype(x) <: Complex
            # derivative with respect to imaginary part
            x[i] = tmp - im*δ/2
            y1 = f(xs...)
            x[i] = tmp + im*δ/2
            y2 = f(xs...)
            x[i] = tmp
            Δ[i] += im*(y2-y1)/δ
        end
    end
    return grads
end


##==----------------------------------------------------------------------------------------------------------------------

@testset ExtendedTestSet "gate gradients" begin
    @testset "MaxKColGates gates" begin
    # fictitious gradients of cost function with respect to quantum gate
    Δ = randn(ComplexF64, 16, 16)
    κ = 2
    graph = QAOAMixers.Graph(2, [(1,2)])
    g  = MaxKColSubgraphPhaseSeparationGate
    f(θ) = 2*real(sum(Δ .* Qaintessent.sparse_matrix(g(θ[], κ, graph))))
    θ = 2π*rand()
    ngrad = ngradient(f, [θ])
    dg = Qaintessent.backward(g(θ, κ, graph), conj(Δ))
    println(dg.γ)
    println(ngrad[1])
    @test isapprox(dg.γ, ngrad[1], rtol=1e-5, atol=1e-5)
    end

    
    @testset "RNearbyValuesMixerGate gates" begin
        Δ = randn(ComplexF64, 4, 4)
        κ = 2
        g = RNearbyValuesMixerGate
        f(θ) = 2*real(sum(Δ .* Qaintessent.sparse_matrix(g(θ[], 1, 2))))
        θ = 2π*rand()
        ngrad = ngradient(f, [θ])
        dg = Qaintessent.backward(g(θ, 1, 2), conj(Δ))
        @test isapprox(dg.β, ngrad[1], rtol=1e-5, atol=1e-5)
    end
end