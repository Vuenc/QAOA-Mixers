using Test
using TestSetExtensions
using LinearAlgebra
using Qaintessent
using Random
using QAOAMixers


# ##==----------------------------------------------------------------------------------------------------------------------


# isunitary(g::AbstractGate) = Qaintessent.matrix(g) * Qaintessent.matrix(Base.adjoint(g)) ≈ I


# ##==----------------------------------------------------------------------------------------------------------------------

graphs = [
    QAOAMixers.Graph(3, [(1,2), (2,3)]),
    QAOAMixers.Graph(5, [(1,2), (2,3), (3,4), (4,5), (5,1), (1,3)]),
    QAOAMixers.Graph(4, [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)])
]

# Test that the phase separation hamiltonian used in MaxKColSubgraphPhaseSeparationGate
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