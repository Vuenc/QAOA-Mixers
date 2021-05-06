using Test
using TestSetExtensions
using LinearAlgebra
using Qaintessent
using Random
using QAOAMixers

graphs = [
    QAOAMixers.Graph(3, [(1,2), (2,3)]),
    QAOAMixers.Graph(5, [(1,2), (2,3), (3,4), (4,5), (5,1), (1,3)]),
    QAOAMixers.Graph(4, [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)])
]

# Test that `MaxKColSubgraphPhaseSeparationGate` has the correct adjoint.
@testset "adjoint MaxKColSubgraphPhaseSeparationGate" begin
    γs = rand(length(graphs)) * 2π
    κs = rand(2:3, length(graphs))

    gates = QAOAMixers.MaxKColSubgraphPhaseSeparationGate.(γs, κs, graphs)
    for i ∈ 1:length(gates)
        @test Qaintessent.matrix(gates[i]) * Qaintessent.matrix(Base.adjoint(gates[i])) ≈ I
    end
end

# Test that `ParityRingMixerGate` has the correct adjoint.
@testset "adjoint ParityRingMixerGate" begin
    N = 3 # number of test cases
    βs = rand(N) * 2π
    ds = rand(2:3, N)

    gates = QAOAMixers.ParityRingMixerGate.(βs, ds)
    for i ∈ 1:length(gates)
        @test Qaintessent.matrix(gates[i]) * Qaintessent.matrix(Base.adjoint(gates[i])) ≈ I
    end
end

# Test that `RNearbyValuesMixerGate` has the correct adjoint.
@testset "adjoint RNearbyValuesMixerGate" begin
    N = 3 # number of test cases
    βs = rand(N) * 2π
    ds = rand(2:3, N)

    gates = QAOAMixers.RNearbyValuesMixerGate.(βs, 1, ds)
    for i ∈ 1:length(gates)
        @test Qaintessent.matrix(gates[i]) * Qaintessent.matrix(Base.adjoint(gates[i])) ≈ I
    end
end
