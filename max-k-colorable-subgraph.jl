include("Qaintessent.jl/src/Qaintessent.jl") # To be replaced by a proper package import at some point
import .Qaintessent.AbstractGate
using LinearAlgebra

# Simple struct that represents a graph via its edges
struct Graph
    n::UInt # number of vertices (indices 1,...,n)
    edges::Set{Set{UInt}} # edges, represented as sets of vertices

    function Graph(n::UInt, edges::Set{Set{UInt}})
        all(edge -> edge ⊆ 1:n, edges) || throw(ArgumentError("Some edges have invalid endpoints"))
        new(n, edges)
    end
end

"""
    Phase separation gate for Max-κ-colorable subgraph QAOA mapping.
    Represents the objective function which counts the number of invalid
    edges (i.e. between vertices of the same color).
    Implemented the for one-hot encoding.

``U_{P}(\\gamma) = e^{-i \\gamma H_{P}}``
``H_{P} = \\sum_{\\{u, v\\} \\in E} \\sum_{a=1}^{\\kappa} Z_{u, a} Z_{v, a}``

Reference:\n
    Stuart Hadfield, Zhihui Wang, Bryan O'Gorman, Eleanor G. Rieffel,Davide Venturelli and Rupak Biswas\n
    From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz\n
    Algorithms 12.2 (2019), p.34
"""
struct MaxKColSubgraphPhaseSeparationGate <: AbstractGate 
    # TODO possibly rename struct when this is in some module?
    # use a reference type (array with 1 entry) for compatibility with Flux
    γ::Vector{Float64} 
    κ::Vector{UInt} # the number of possible colors
    graph::Graph # the underlying graph which should be colored

    function MaxKColSubgraphPhaseSeparationGate(γ::Float64, κ::Int, graph::Graph)
        κ > 0 || throw(ArgumentError("Parameter κ must be a positive integer!"))
        new([γ], [κ], graph)
    end
end

function matrix(g::MaxKColSubgraphPhaseSeparationGate)
    Z = [1 0; 0 -1]

    # Implementation of Eq. (17)
    # one-hot encoding: n * κ vector. Index (a-1)*n + (b-1) corresponds to vertex a, color b.
    H_P_enc = sum(
        kron((color == a && vertex ∈ edge ? Z : I(2) for vertex ∈ 1:g.graph.n for color ∈ 1:g.κ[])...) # Z_{u,a} Z_{v,a}
        for a ∈ 1:g.κ[] for edge in g.graph.edges # Σ_{(u,v) = edge ∈ E} Σ_{a=1..κ}
    )

    # Implementation of one-hot phase seperator, A.3.1, p. 34
    U_P = exp(-im * g.γ[] * H_P_enc)

    U_P
end

sparse_matrix(g::MaxKColSubgraphPhaseSeparationGate) = sparse(matrix(g))

# wires
num_wires(g::MaxKColSubgraphPhaseSeparationGate)::Int = g.κ[] * g.graph.n