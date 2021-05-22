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

# Test that the phase separation Hamiltonian used in MaxKColSubgraphPhaseSeparationGate
# implements the objective function correctly.
@testset "Hamiltonian MaxKColSubgraphPhaseSeparationGate" begin
    κs = [4, 2, 3]

    hamiltonians = QAOAMixers.phase_separation_hamiltonian.(graphs, κs)
    colorings = [rand(1:κ, graph.n) for (κ, graph) ∈ zip(κs, graphs)]

    for (H, graph, κ, coloring) ∈ zip(hamiltonians, graphs, κs, colorings)
        f_val = QAOAMixers.properly_colored_edges(graph, coloring)
        ψ_coloring = QAOAMixers.ψ_from_coloring(graph.n, κ, coloring)
        scaling = κ * length(graph.edges) - 4 * f_val # ψ_coloring should be scaled by this factor
        @test H * ψ_coloring ≈ ψ_coloring * scaling
    end
end

# Utility functions to perform circular shift on Vectors (that represent bitstrings)
left_circ_shift(l::Vector) = [l[2:length(l)]; l[1:1]]
right_circ_shift(l::Vector) = [l[length(l):length(l)]; l[1:length(l)-1]]

# Shorthand for repeated function application
# Taken from https://stackoverflow.com/questions/39895672/apply-function-repeatedly-a-specific-number-of-times
(^)(f::Function, i::Int) = i==1 ? f : x->(f^(i-1))(f(x))

# Test that the Hamiltonian which the RNearbyValuesMixerGate is based on
# maps coloring states to a superposition of their neighboring states
@testset "Hamiltonian RNearbyValuesMixerGate" begin
    ds = [2, 5, 9, 12] # different numbers of colors
    rs = [1, 2, 5]
    nums_samples = [2, 5, 5, 5]

    for (d, num_samples) ∈ zip(ds, nums_samples)
        for r ∈ rs[rs .< d] # only test valid rs
            # compute the Hamiltonian
            gate = QAOAMixers.RNearbyValuesMixerGate(0., r, d)
            hamiltonian = QAOAMixers.r_nearby_values_hamiltonian_onehot(gate)

            # for each color (i.e. |1000..00>, |0100..00>, ..., |0000..01>):
            for color ∈ randperm(d)[1:num_samples]
                # compute superposition state bitstrings after applying Hamiltonian
                ψ = QAOAMixers.ψ_from_coloring(1, d, [color])
                ψ_bits = QAOAMixers.wavefunction_distribution(ψ)[1][1]
                out_vec = hamiltonian * ψ # not a wavefunction (we apply a Hamiltonian, not a gate)
                superposition_states = (QAOAMixers.wavefunction_distribution(out_vec)
                    |> (distr -> filter(d -> !(d[2] ≈ 0), distr))
                    .|> (d -> d[1]))

                # compute bitstrings of nearby colors by shifting
                near_values = [
                    [(left_circ_shift ^ k)(ψ_bits) for k ∈ 1:r];
                    [(right_circ_shift ^ k)(ψ_bits) for k ∈ 1:r]
                ] # (^) is repeated function application defined above

                # r-NV Mixer should map state to superposition of its neighbor colors
                @test Set(superposition_states) == Set(near_values)
            end
        end
    end
end
