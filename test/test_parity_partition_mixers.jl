using Test
using TestSetExtensions
using Qaintessent
using Random
using QAOAMixers

# Test that the parity ring mixer is a special case of the partition mixer
@testset "PartitionMixerGate generalizes ParityRingMixerGate" begin
    ds = [2, 5, 9, 10]

    for d ∈ ds
        β = rand() * 10 + 0.1

        # construct the parity mixer
        parity_mixer = QAOAMixers.ParityRingMixerGate(β, d)

        # construct the equivalent partition mixer
        odd_part = [(a, a+1) for a ∈ 1:2:(d-1)]
        even_part = [(a, (a % d) + 1) for a ∈ 2:2:d]
        last_part = isodd(d) ? [(d, 1)] : nothing
        partition = isodd(d) ? [odd_part, even_part, last_part] : [odd_part, even_part]
        partition_mixer = QAOAMixers.PartitionMixerGate(β, d, partition)

        @test Qaintessent.matrix(parity_mixer) ≈ Qaintessent.matrix(partition_mixer)
    end
end
