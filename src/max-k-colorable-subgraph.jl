using Qaintessent
import Qaintessent: AbstractGate, Circuit, CircuitGate, MeasurementOperator
using LinearAlgebra

# Simple struct that represents a graph via its edges
struct Graph
    n::Int # number of vertices (indices 1,...,n)
    edges::Set{Set{Int}} # edges, represented as sets of vertices

    function Graph(n::Integer, edges::Vector{Tuple{T, T}}) where T <: Integer
        n >= 1 || throw(DomainError("n must be a positive integer"))

        # Turn into set of sets
        edge_set = Set(map(Set, edges))

        # Verify that all edges are valid
        all(edge -> edge ⊆ 1:n, edge_set) || throw(ArgumentError("Some edges have invalid endpoints"))
        new(n, edge_set)
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
    Stuart Hadfield, Zhihui Wang, Bryan O'Gorman, Eleanor G. Rieffel, Davide Venturelli and Rupak Biswas\n
    From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz\n
    Algorithms 12.2 (2019), p.34
"""
struct MaxKColSubgraphPhaseSeparationGate <: AbstractGate 
    # TODO possibly rename struct when this is in some module?
    # use a reference type (array with 1 entry) for compatibility with Flux
    γ::Vector{Float64} 
    κ::Vector{Int} # the number of possible colors
    graph::Graph # the underlying graph which should be colored

    function MaxKColSubgraphPhaseSeparationGate(γ::Float64, κ::Integer, graph::Graph)
        κ > 0 || throw(ArgumentError("Parameter κ must be a positive integer!"))
        new([γ], [κ], graph)
    end
end

function Qaintessent.matrix(g::MaxKColSubgraphPhaseSeparationGate)
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

Qaintessent.sparse_matrix(g::MaxKColSubgraphPhaseSeparationGate) = sparse(matrix(g))

# wires
Qaintessent.num_wires(g::MaxKColSubgraphPhaseSeparationGate)::Int = g.κ[] * g.graph.n

function max_κ_colorable_subgraph_circuit(γs::Vector{Float64}, βs::Vector{Float64}, 
        graph::Graph, κ::Integer)
    length(γs) == length(βs) || throw(ArgumentError("γs and βs must have same length!"))
    N = graph.n * κ

    # Create the circuit gates (multiple stages of phase separation gate and mixer gates)
    gates::Vector{CircuitGate} = []
    for (γ, β) ∈ zip(γs, βs)
        # Add the phase separation gate
        push!(gates, CircuitGate(Tuple(1:N), MaxKColSubgraphPhaseSeparationGate(γ, κ, graph)))

        # Add the mixer, consisting of a partial mixer gate for each vertex
        for vertex in 1:graph.n
            push!(gates, CircuitGate(Tuple(((vertex - 1) * κ + 1):(vertex * κ)), ParityRingMixerGate(β, κ)))
        end
    end

    # One-hot encoding: one qubit for each node/color combination
    Circuit{N}(gates) #, [MeasurementOperator(Matrix{Float64}(I, 2^N, 2^N), Tuple(1:N))])
end

function ψ_initial(n::Integer, κ::Integer)::Vector{ComplexF64}
    (n > 0 && κ > 0) || throw(DomainError("Parameters n and κ must be positive integers"))

    # Create the initial state (all vertices are assigned the first color)
    ψ = kron((color == 1 ? [0.0im, 1] : [1, 0.0im] for vertex ∈ 1:n for color ∈ 1:κ)...)
    ψ
end
