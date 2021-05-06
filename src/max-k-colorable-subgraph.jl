using Qaintessent
using Qaintellect
using Flux
using LinearAlgebra
using SparseArrays: sparse


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
    return count(coloring[a] != coloring[b] for (a, b) ∈ graph.edges)
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
        κ > 0 || throw(ArgumentError("Parameter `κ` must be a positive integer."))
        length(graph.edges) > 0 || throw(ArgumentError("Graph `graph` must have at least one edge."))
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
    U_deriv = conj(-im * H_P_enc * exp(-im * g.γ[] * H_P_enc))
    
    return MaxKColSubgraphPhaseSeparationGate(2 * sum(real(U_deriv .* Δ)), g.κ[], g.graph)
end

Qaintessent.sparse_matrix(g::MaxKColSubgraphPhaseSeparationGate) = sparse(matrix(g))

# wires
Qaintessent.num_wires(g::MaxKColSubgraphPhaseSeparationGate)::Int = g.κ * g.graph.n

function max_κ_colorable_subgraph_circuit(γs::Vector{Float64}, βs::Matrix{Float64},
        graph::Graph, κ::Integer)
    size(βs) == (graph.n, length(γs)) || throw(ArgumentError("γs, βs have incorrect dimensions."))
    N = graph.n * κ

    # Create the circuit gates (multiple stages of phase separation gate and mixer gates)
    gates::Vector{CircuitGate} = []
    for (γ, βs_column) ∈ zip(γs, eachcol(βs))
        # Add the phase separation gate
        push!(gates, CircuitGate(Tuple(1:N), MaxKColSubgraphPhaseSeparationGate(γ, κ, graph)))

        # Add the mixer, consisting of a partial mixer gate for each vertex
        for (vertex, β) ∈ zip(1:graph.n, βs_column)
            # push!(gates, CircuitGate(Tuple(((vertex - 1) * κ + 1):(vertex * κ)), ParityRingMixerGate(β, κ)))
            push!(gates, CircuitGate(Tuple(((vertex - 1) * κ + 1):(vertex * κ)), RNearbyValuesMixerGate(β, 1, κ)))
        end
    end

    # One-hot encoding: one qubit for each node/color combination
    Circuit{N}(gates) #, [MeasurementOperator(Matrix{Float64}(I, 2^N, 2^N), Tuple(1:N))])
end

function ψ_initial(n::Integer, κ::Integer)::Vector{ComplexF64}
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
function wavefunction_distribution(ψ::Vector{ComplexF64}; as_bitstrings::Bool = true,
        include_zero = false)::Union{Vector{Tuple{Int, Float64}}, Vector{Tuple{Vector{Int}, Float64}}}
    distribution = [(i-1, abs(amplitude)^2) for (i, amplitude) ∈ enumerate(ψ) if abs(amplitude) > 0 || include_zero]
    if as_bitstrings
        N = Int(log2(length(ψ)))
        distribution = [(digits(i, base=2, pad=N) |> reverse, p) for (i, p) ∈ distribution]
    end
    
    return sort(distribution, by=(t -> -t[2]))
end

# Utility function to compute the probabilities of the outcome wavefunction of a circuit applied to ψ, sorted by descending probability
function output_distribution(circ::Circuit, ψ::Vector{ComplexF64}; as_bitstrings::Bool = true,
        include_zero = false)::Union{Vector{Tuple{Int, Float64}}, Vector{Tuple{Vector{Int}, Float64}}}
    ψ_out = apply(ψ, circ.moments)
    return wavefunction_distribution(ψ_out, as_bitstrings=as_bitstrings, include_zero=include_zero)
end

# Utility function to compute the probabilities of the output colorings of a circuit applied to ψ, sorted by descending probability
function output_colorings_distribution(circ::Circuit{N}, n::Int;
        include_zero = false)::Vector{Tuple{Dict{Int, Vector{Int}}, Float64}} where {N}
    κ = Int(N / n)
    ψ = ψ_initial(n, κ)
    ψ_out = apply(ψ, circ.moments)
    distribution = wavefunction_distribution(ψ_out, as_bitstrings=false, include_zero=include_zero)

    return [(decode_basis_state(state, n, κ), p) for (state, p) ∈ distribution]
end

# Utility function to compute the probabilities of the output colorings of a circuit applied to ψ, sorted by descending probability
function output_colorings_distribution_scored(circ::Circuit{N}, graph::Graph;
        include_zero = false)::Vector{Tuple{Vector{Int}, Int, Float64}} where {N}
    dist = output_colorings_distribution(circ, graph.n, include_zero = include_zero)
    
    coloring_dict_to_list(dict) = [dict[i][1] for i ∈ 1:graph.n]

    return [(coloring_dict_to_list(coloring), properly_colored_edges(graph, coloring_dict_to_list(coloring)), p) for (coloring, p) ∈ dist]
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

function optimize_qaoa(graph::Graph, κ::Int; p::Union{Int, Nothing}=nothing, training_rounds::Int=10,
        learning_rate::Real=0.005, circ_in::Union{Circuit{N}, Nothing}=nothing, init_stddev=0.1) where {N}

    (isnothing(circ_in) ⊻ isnothing(p)) ||
        throw(ArgumentError("Must specify exactly one of the parameters `circ_in` and `p`."))

    (κ > 0 && (isnothing(p) || p > 0) && training_rounds > 0) ||
        throw(DomainError("Parameters `κ`, `p` and `training_rounds` must be positive integers."))

    if isnothing(circ_in)
        # Initialize circuit and wavefunction
        (initial_γs, initial_βs) = (randn(p) * init_stddev, randn((graph.n, p)) * init_stddev)
        
        println("Initial γs: $(initial_γs)")
        println("Initial βs: $(initial_βs)")
        circ = max_κ_colorable_subgraph_circuit(initial_γs, initial_βs, graph, κ)
    else
        N == κ * graph.n || throw(ArgumentError("Circuit `circ_in` has wrong dimensions."))
        circ = circ_in
    end
    ψ = ψ_initial(graph.n, κ)
    H_P = phase_separation_hamiltonian(graph, κ)
    
    # Undo the transform of HP to the objective function: f(x) |-> κm - 4 f(x). See text after Eq. (17).
    objective_transform(x) = (κ*length(graph.edges) - x)/4

    # Set up optimization with Flux
    params = Flux.params(circ)
    data = repeat([()], training_rounds) # empty input data for `training_rounds` rounds of training
    optimizer = ADAM(learning_rate)
    round = 1
    expectation() = begin # evaluate expectation <f> to be minimized and print it
        ψ_out = apply(ψ, circ.moments)
        objective = objective_transform(real(ψ_out' * H_P * ψ_out))
        println("Training, round $(round): average objective = $(objective)")
        round += 1
        return -objective
    end

    # Perform training
    Flux.train!(expectation, params, data, optimizer) #, cb=Flux.throttle(print_expectation, 1))
    return circ
end

# Make trainable params available to Flux
Flux.@functor MaxKColSubgraphPhaseSeparationGate