using Qaintessent
# import Qaintessent: AbstractGate, Circuit, CircuitGate, MeasurementOperator
using Qaintellect
using Flux
using LinearAlgebra
using IterTools: ncycle


# Simple struct that represents a graph via its edges
struct Graph
    n::Int # number of vertices (indices 1,...,n)
    edges::Set{Set{Int}} # edges, represented as sets of vertices

    function Graph(n::Integer, edges::Vector{Tuple{T, T}}) where T <: Integer
        n >= 1 || throw(DomainError("n must be a positive integer"))

        # Turn into set of sets
        edge_set = Set(Set.(edges))

        # Verify that all edges are valid
        all(edge -> edge ⊆ 1:n && length(edge) == 2, edge_set) || throw(ArgumentError("Some edges have invalid endpoints"))
        new(n, edge_set)
    end
end

# Utility function to count the properly colored edges in a graph coloring (i.e. endpoints have different colors)
function properly_colored_edges(graph::Graph, coloring::Vector{Int})
    length(coloring) == graph.n || throw(ArgumentError("Length of coloring must equal the number of vertices."))
    count(coloring[a] != coloring[b] for (a, b) in graph.edges)
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
    κ::Int # the number of possible colors
    graph::Graph # the underlying graph which should be colored

    function MaxKColSubgraphPhaseSeparationGate(γ::Float64, κ::Integer, graph::Graph)
        κ > 0 || throw(ArgumentError("Parameter κ must be a positive integer!"))
        new([γ], κ, graph)
    end
end

function phase_separation_hamiltonian(graph::Graph, κ::Int)
    Z = [1 0; 0 -1]
    
    # Implementation of Eq. (17)
    # one-hot encoding: n * κ vector. Index (a-1)*n + (b-1) corresponds to vertex a, color b.
    H_P_enc = sum(
        kron((color == a && vertex ∈ edge ? Z : I(2) for vertex ∈ 1:graph.n for color ∈ 1:κ)...) # Z_{u,a} Z_{v,a}
        for a ∈ 1:κ for edge ∈ graph.edges # Σ_{(u,v) = edge ∈ E} Σ_{a=1..κ}
    )
    H_P_enc
end

function Qaintessent.matrix(g::MaxKColSubgraphPhaseSeparationGate)
    # Calculate the hamiltonian (Eq. 17)
    H_P_enc = phase_separation_hamiltonian(g.graph, g.κ)

    # Implementation of one-hot phase seperator, A.3.1, p. 34
    U_P = exp(-im * g.γ[] * H_P_enc)

    U_P
end

# TODO make sure this really is the correct adjoint!
Qaintessent.adjoint(g::MaxKColSubgraphPhaseSeparationGate) = MaxKColSubgraphPhaseSeparationGate(-g.γ[], g.κ, g.graph)

# TODO make sure this is the right way to do it
function Qaintessent.backward(g::MaxKColSubgraphPhaseSeparationGate, Δ::AbstractMatrix)
    H_P_enc = phase_separation_hamiltonian(g.graph, g.κ)

    # we can exploit that H_P_enc is diagonal
    # TODO make sure to handle Δ correctly
    return MatrixGate(-im * H_P_enc * exp(-im * g.γ[] * H_P_enc) * Δ)
end

Qaintessent.sparse_matrix(g::MaxKColSubgraphPhaseSeparationGate) = sparse(matrix(g))

# wires
Qaintessent.num_wires(g::MaxKColSubgraphPhaseSeparationGate)::Int = g.κ * g.graph.n

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
        for vertex ∈ 1:graph.n
            push!(gates, CircuitGate(Tuple(((vertex - 1) * κ + 1):(vertex * κ)), ParityRingMixerGate(β, κ)))
        end
    end

    # One-hot encoding: one qubit for each node/color combination
    Circuit{N}(gates) #, [MeasurementOperator(Matrix{Float64}(I, 2^N, 2^N), Tuple(1:N))])
end

function ψ_initial(n::Integer, κ::Integer)::Vector{ComplexF64}
    (n > 0 && κ > 0) || throw(DomainError("Parameters n and κ must be positive integers"))

    # Create the initial state (all vertices are assigned the first color)
    ψ_from_coloring(n, κ, repeat([1], n))
end

# Create a state ψ that corresponds to a given coloring
function ψ_from_coloring(n::Int, κ::Int, colors::Vector{Int})::Vector{ComplexF64}
    (n > 0 && κ > 0) || throw(DomainError("Parameters n and κ must be positive integers."))
    colors ⊆ 1:κ || throw(ArgumentError("Parameter `colors` may only contain colors in the range 1:$(κ)."))

    # Create ψ with |1> entries in the indices corresponding to the given colors, |0> elsewhere
    ψ = kron((color == colors[vertex] ? [0.0im, 1] : [1, 0.0im] for vertex ∈ 1:n for color ∈ 1:κ)...)
    ψ
end

# Utility function to compute the probabilities of the outcomes represented by a wavefunction ψ, sorted by descending probability
function output_distribution(ψ_out::Vector{ComplexF64})::Vector{Tuple{Int, Real}}
    distribution = [(i-1, abs(amplitude)^2) for (i, amplitude) ∈ enumerate(ψ_out)]
    return sort(distribution, by=(t -> -t[2]))
end

# Decodes the coloring represented by a single computational basis state, represented as integer
function decode_basis_state(basis_state::Int, n::Int, κ::Int)::Dict{Int, Vector{Int}}
    (n > 0 && κ > 0) || throw(DomainError("Parameters n and κ must be positive integers."))

    N = n * κ
    bits = digits(basis_state, base=2, pad=N) |> reverse
    vertex_bits = vertex -> bits[((vertex - 1) * κ + 1):(vertex * κ)]
    colors_by_vertex = Dict([(v, findall(!iszero, vertex_bits(v))) for v ∈ 1:n])

    all(!isnothing, colors_by_vertex) || throw(ArgumentError("The state `basis_state` has at least one vertex without a color."))
    
    return colors_by_vertex
end

function optimize_qaoa(graph::Graph, κ::Int, p::Int, training_rounds::Int=10, learning_rate::Real=0.1)
    (κ > 0 && p > 0 && training_rounds > 0) || 
        throw(DomainError("Parameters `κ`, `p` and `training_rounds` must be positive integers."))

    # Initialize circuit and wavefunction
    (initial_γs, initial_βs) = (randn(p), randn(p))
    println("Initial γs: $(initial_γs)")
    println("Initial βs: $(initial_βs)")
    circ = max_κ_colorable_subgraph_circuit(initial_γs, initial_βs, graph, κ)
    ψ = ψ_initial(graph.n, κ)
    H_P = phase_separation_hamiltonian(graph, κ)
    # TODO convert H_P output to number of edges via inverse of f(x) |-> κm - 4 f(x)

    # Set up optimization with Flux
    params = Flux.params(circ)
    # data = repeat([()], training_rounds) # empty input data for `training_rounds` rounds of training
    data = ncycle([()], training_rounds)
    optimizer = Descent(learning_rate)
    expectation() = begin # evaluate expectation <f> to be minimized
        ψ_out = apply(ψ, circ.moments)
        real(ψ_out' * H_P * ψ_out)
    end
    print_expectation() = println("Training: <f> = $(expectation())")

    println("Before training: <f> = $(expectation())")

    # Perform training
    Flux.train!(expectation, params, data, optimizer, cb=Flux.throttle(print_expectation, 1))

    circ
end

# Make trainable params available to Flux
Flux.@functor MaxKColSubgraphPhaseSeparationGate