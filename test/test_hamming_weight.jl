using Test
using TestSetExtensions
using LinearAlgebra
using Qaintessent
using Random
using QAOAMixers

# Utility function that returns the Hamming weights of all basis states of ψ
hamming_weights(ψ) = (QAOAMixers.wavefunction_distribution(ψ)
 |> (distr -> filter(d -> !(d[2] ≈ 0), distr))
 .|> (d -> sum(d[1])))


# Test that the RNearbyValuesMixerGate preserves the Hamming weight of coloring
# states (i.e. maps states of Hamming weight one to a superposition of states
# with Hamming weight one)
@testset "Hamming weight RNearbyValuesMixerGate" begin
    ds = [2, 5, 9, 10] # different numbers of colors
    rs = [1, 2, 5]
    nums_samples = [2, 5, 5, 5]
    
    for (d, num_samples) ∈ zip(ds, nums_samples)
        for r ∈ rs[rs .< d] # only test valid rs
            gate = QAOAMixers.RNearbyValuesMixerGate(rand() * 10 + 0.1, r, d)
            U = Qaintessent.matrix(gate)

            for color ∈ randperm(d)[1:num_samples]
                ψ = QAOAMixers.ψ_from_coloring(1, d, [color])
                ψ_out = U * ψ

                # All basis states should have Hamming weight 1
                out_hamming_weights = hamming_weights(ψ_out)
                @test hamming_weights(ψ_out) == ones(Int64, length(out_hamming_weights))
            end
        end
    end
end

# Test that the ParityRingMixerGate preserves the Hamming weight of coloring
# states (i.e. maps states of Hamming weight one to a superposition of states
# with Hamming weight one), if d == 2
@testset "Hamming weight ParityRingMixerGate (d = 2)" begin
    ds = [2] # different numbers of colors
    
    for d ∈ ds
        gate = QAOAMixers.ParityRingMixerGate(rand() * 10 + 0.1, d)
        U = Qaintessent.matrix(gate)

        for color ∈ 1:d
            ψ = QAOAMixers.ψ_from_coloring(1, d, [color])
            ψ_out = U * ψ

            # All basis states should have Hamming weight 1
            out_hamming_weights = hamming_weights(ψ_out)
            @test hamming_weights(ψ_out) == ones(Int64, length(out_hamming_weights))
        end
    end
end

# Test that the ParityRingMixerGate preserves the Hamming weight of coloring
# states (i.e. maps states of Hamming weight one to a superposition of states
# with Hamming weight one), if d ≥ 3
@testset "Hamming weight ParityRingMixerGate (d ≥ 3)" begin
    ds = [5, 8, 9] # different numbers of colors
    num_samples = 5
    
    for d ∈ ds
        gate = QAOAMixers.ParityRingMixerGate(rand() * 10 + 0.1, d)
        U = Qaintessent.matrix(gate)

        for color ∈ randperm(d)[1:num_samples]
            ψ = QAOAMixers.ψ_from_coloring(1, d, [color])
            ψ_out = U * ψ

            # All basis states should have Hamming weight 1
            out_hamming_weights = hamming_weights(ψ_out)
            @test hamming_weights(ψ_out) == ones(Int64, length(out_hamming_weights))
        end
    end
end

# Test that the PartitionMixerGate preserves the Hamming weight of coloring
# states (i.e. maps states of Hamming weight one to a superposition of states
# with Hamming weight one)
@testset "Hamming weight PartitionMixerGate" begin
    ds = [5, 8, 9] # different numbers of colors
    num_samples = 5
    nums_partition_layers = [3, 2, 7]

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

        gate = QAOAMixers.PartitionMixerGate(rand() * 10 + 0.1, d, partition)
        U = Qaintessent.matrix(gate)

        for color ∈ randperm(d)[1:num_samples]
            ψ = QAOAMixers.ψ_from_coloring(1, d, [color])
            ψ_out = U * ψ

            # All basis states should have Hamming weight 1
            out_hamming_weights = hamming_weights(ψ_out)
            @test hamming_weights(ψ_out) == ones(Int64, length(out_hamming_weights))
        end
    end
end