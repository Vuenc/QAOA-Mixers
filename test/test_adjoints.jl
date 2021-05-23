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
    ds = rand(2:9, N)

    gates = QAOAMixers.ParityRingMixerGate.(βs, ds)
    for i ∈ 1:length(gates)
        @test Qaintessent.matrix(gates[i]) * Qaintessent.matrix(Base.adjoint(gates[i])) ≈ I
    end
end

# Test that `RNearbyValuesMixerGate` has the correct adjoint.
@testset "adjoint RNearbyValuesMixerGate" begin
    βs = rand(4) * 2π
    ds = [2, 3, 6, 9]
    rs = [1, 2, 1, 5]

    gates = QAOAMixers.RNearbyValuesMixerGate.(βs, rs, ds)
    for i ∈ 1:length(gates)
        @test Qaintessent.matrix(gates[i]) * Qaintessent.matrix(Base.adjoint(gates[i])) ≈ I
    end
end

# Test that `PartitionMixerGate` has the correct adjoint.
@testset "adjoint PartitionMixerGate" begin
    ds = [2, 3, 6, 9]
    nums_partition_layers = [2, 3, 4, 5]

    for (d, num_partition_layers) ∈ zip(ds, nums_partition_layers)
        # create num_partition_layers partition parts, each of
        # random length and with random entries
        partition = [
            # create h x 2 matrix with random height 2 from a random permutation of the indices...
            (reshape(randperm(d)[1:(2*rand(1:div(d, 2)))], (:, 2))
            # ... then map the rows to tuples
                |> eachrow .|> Tuple)
            for _ ∈ 1:num_partition_layers
        ]

        β = rand() * 10 + 0.1
        gate = QAOAMixers.PartitionMixerGate(β, d, partition)
        @test Qaintessent.matrix(gate) * Qaintessent.matrix(Base.adjoint(gate)) ≈ I
    end
end