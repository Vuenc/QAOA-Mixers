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
    rs = [1] # currently, only r=1 supported for one-hot
    nums_samples = [2, 5, 5, 5]
    
    for (d, num_samples) ∈ zip(ds, nums_samples)
        for r ∈ rs
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