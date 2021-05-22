using Qaintessent
using LinearAlgebra: I
using Flux
using SparseArrays: sparse
using Memoize

# Pauli X and Y matrices
X = [0 1; 1 0]
Y = [0 -im; im 0]

# Utility function for a XY mixer
# Implements X_a X_{a+1} + Y_a Y_{a+1}, or more generally (⊗_{i ∈ xy_indices} X_i) + (⊗_{i ∈ xy_indices} Y_i)
function XY_sum(xy_indices::Vector{Int64}, d::Int64)::Matrix{ComplexF64}
    # passing generator into kron via varargs syntax
    return kron((i ∈ xy_indices ? X : I(2) for i ∈ 1:d)...
        ) + kron((i ∈ xy_indices ? Y : I(2) for i ∈ 1:d)...)
end

"""
    r-nearby values single-qudit mixer gate, which acts on a single qudit

``U_{r\\text{-NV}}(\\beta) = e^{-i \\beta H_{r\\text{-NV}}}``
``H_{r\\text{-NV}} = \\sum_{i=1}^r \\left(\\breve{X}^i + \\left(\\breve{X}^\\dagger\\right)^i\\right)``

Reference:\n
    Stuart Hadfield, Zhihui Wang, Bryan O'Gorman, Eleanor G. Rieffel,Davide Venturelli and Rupak Biswas\n
    From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz\n
    Algorithms 12.2 (2019), p.34
"""
struct RNearbyValuesMixerGate <: AbstractGate
    # use a reference type (array with 1 entry) for compatibility with Flux
    β::Vector{Float64}
    r::Integer # this is the r in r-Nearby-values
    d::Integer # d (= κ) = number of colors

    function RNearbyValuesMixerGate(β::Float64, r::Integer, d::Integer)
        (r > 0 && d > 0) || throw(ArgumentError("Parameters r and d must be positive integers."))
        new([β], r, d)
    end
end

@memoize function r_nearby_values_hamiltonian_onehot(g::RNearbyValuesMixerGate)
    g.r == 1 || throw(DomainError("Hamiltonian for one-hot encoding only implemented for r=1."))

    X = [0 1; 1 0]
    Y = [0 -im; im 0]

    H_ring_enc = sum(
        kron((i ∈ [a, (a+1) % g.d] ? X : I(2) for i ∈ 0:g.d-1)...) # passing iterator into kron via varargs syntax
        + kron((i ∈ [a, (a+1) % g.d] ? Y : I(2) for i ∈ 0:g.d-1)...)
        for a ∈ 0:g.d-1
    )
    return H_ring_enc
end

# Implementation of Eq. (6)
function matrix_onehot(g::RNearbyValuesMixerGate)
    g.r == 1 || throw(DomainError("Matrix for one-hot encoding only implemented for r=1."))

    H_ring_enc = r_nearby_values_hamiltonian_onehot(g)
    U_ring_enc = exp(-im * g.β[] * H_ring_enc)

    return U_ring_enc
end

function matrix_qudit(g::RNearbyValuesMixerGate)
    # Only for reference; the simulator does not work with qudits

    # Implementation of Eq. (A3)
    genX = I(g.d)[:,vcat(2:g.d, 1)] # move first columnn of identity matrix to back to obtain d-dim. generalized Pauli-X
    genX_T = transpose(genX)

    # Implementation of H_{r-NV} qudit definition, r-NV paragraph, page 10
    # could be implemented more efficiently (avoid exponentation for each i), but is only for reference anyway
    H_rNV = sum(genX^i + genX_T^i for i ∈ 1:g.r)
    U_rNV = exp(-im * g.β[] * H_rNV)

    U_rNV
end

function Qaintessent.backward(g::RNearbyValuesMixerGate, Δ::AbstractMatrix)
    delta = 1e-10
  
    # uses conjugated gradient matrix
    U_rnv1 = Qaintessent.matrix(RNearbyValuesMixerGate(-(g.β[] - delta/2), g.r, g.d))
    U_rnv2 = Qaintessent.matrix(RNearbyValuesMixerGate(-(g.β[] + delta/2), g.r, g.d))

    U_deriv = (U_rnv2 - U_rnv1) / delta
  
    return RNearbyValuesMixerGate(2 * sum(real(U_deriv .* Δ)), g.r, g.d)
end

function Qaintessent.matrix(g::RNearbyValuesMixerGate)
    matrix_onehot(g)
end

Qaintessent.adjoint(g::RNearbyValuesMixerGate) = RNearbyValuesMixerGate(-g.β[], g.r, g.d)

Qaintessent.sparse_matrix(g::RNearbyValuesMixerGate) = sparse(matrix(g))

# wires (= g.d since we use the one-hot encoding canonically)
Qaintessent.num_wires(g::RNearbyValuesMixerGate)::Int = g.d

"""
    Parity single-qudit ring mixer gate, which acts on a single qudit

``U_{\\text{parity}}(\\beta) = U_{\\text{last}}(\\beta) U_{\\text{even}}(\\beta) U_{\\text{odd}}(\\beta)``
``U_{\\text{odd}}(\\beta) = \\prod_{a~\\text{odd}, a \\neq d} e^{-i \\beta (X_a X_{a+1} + Y_a Y_{a+1})}``
``U_{\\text{even}}(\\beta) = \\prod_{a~\\text{even}} e^{-i \\beta (X_a X_{a+1} + Y_a Y_{a+1})}``
``U_{\\text{last}}(\\beta) = e^{-i \\beta (X_d X_1 + Y_d Y_1)} ~\\text{if}~ d ~\\text{is odd,}~ I ~\\text{otherwise.}``

The formulas require some interpretation. Assumptions for this implementation are:
- the indices a start at 1 (not at zero like elsewhere in the paper)
- X_{a+1} and Y_{a+1} actually means X_1 and Y_1 if a = d

Reference:\n
    Stuart Hadfield, Zhihui Wang, Bryan O'Gorman, Eleanor G. Rieffel, Davide Venturelli and Rupak Biswas\n
    From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz\n
    Algorithms 12.2 (2019), equations (7) - (10), p. 11
"""
struct ParityRingMixerGate <: AbstractGate
    β::Vector{Float64}
    d::Integer # d (= κ) = number of colors

    function ParityRingMixerGate(β::Float64, d::Integer)
        d > 0 || throw(ArgumentError("Parameter d must be a positive integer."))
        new([β], d)
    end
end

function Qaintessent.matrix(g::ParityRingMixerGate)
    # Implements Eq. (8)
    # assumption: by a ≠ n, the paper actually means a ≠ d.
    # assumption: by X_a for a = d+1, the paper means X_1.
    U_odd = prod([exp(-im * g.β[] * XY_sum([a, a+1], g.d)) for a ∈ 1:2:(g.d - 1)], init=I)
    U_even = prod([exp(-im * g.β[] * XY_sum([a, a < g.d ? (a+1) : 1], g.d)) for a ∈ 2:2:g.d], init=I)

    # Implements Eq. (9)
    U_last =  isodd(g.d) ? exp(-im * g.β[] * XY_sum([g.d, 1], g.d)) : I

    # Implements Eq. (7)
    U_parity = U_last * U_even * U_odd
    return U_parity
end

Qaintessent.adjoint(g::ParityRingMixerGate) = ParityRingMixerGate(-g.β[], g.d)

function Qaintessent.backward(g::ParityRingMixerGate, Δ::AbstractMatrix)
    delta = 1e-10

    U_parity1 = Qaintessent.matrix(ParityRingMixerGate(-(g.β[] - delta/2), g.d))
    U_parity2 = Qaintessent.matrix(ParityRingMixerGate(-(g.β[] + delta/2), g.d))

    U_deriv = (U_parity2 - U_parity1) / delta

    return ParityRingMixerGate(2 * sum(real(U_deriv .* Δ)), g.d)
end

Qaintessent.sparse_matrix(g::ParityRingMixerGate) = sparse(matrix(g))

# wires (= g.d since we use the one-hot encoding canonically)
Qaintessent.num_wires(g::ParityRingMixerGate)::Int = g.d

"""
    Partition single-qudit mixer gate (implemented not for qudits, but the one-hot encoding)

``U_{\\mathcal{P}-r-\\text{NV}}(\\beta) = U_{P_p-\\text{XY}}(\\beta) \\dots U_{P_1-\\text{XY}}(\\beta)``
``U_{P-\\text{XY}}(\\beta) = \\prod_{\\{a, b\\} \\in P} e^{-i \\beta (\\ket{a}\\bra{b} + \\ket{b}\\bra{a})}``

Note: the time evolution term is effectively implemented with an additional factor of two in the exponent:
``e^{-i \\beta 2 (\\ket{a}\\bra{b} + \\ket{b}\\bra{a})}``
to be consistent with the XY gates used for the other mixers.

Reference:\n
    Stuart Hadfield, Zhihui Wang, Bryan O'Gorman, Eleanor G. Rieffel, Davide Venturelli and Rupak Biswas\n
    From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz\n
    Algorithms 12.2 (2019), equations (11) - (12), p. 12
"""
struct PartitionMixerGate <: AbstractGate
    β::Vector{Float64}
    d::Int64
    partition::Vector{Vector{Tuple{Int, Int}}} # the Tuple stops Flux.@functor from misinterpreting these as params

    function PartitionMixerGate(β::Float64, d::Int64, partition::Vector{Vector{Tuple{Int, Int}}})
        d > 0 || throw(ArgumentError("Parameter d must be a positive integer."))

        # check that no duplicate indices occur in a part (s.t. the XY mixers within one part commute)
        for partition_part ∈ partition
            part_indices = union(reduce(vcat, collect.(partition_part)))
            part_indices ⊆ 1:d || throw("Indices in partition must be between 1 and d.")
            length(part_indices) == 2 * length(partition_part) ||
                throw("No index must occur more than once within each partition part in `partition`.")
        end

        new([β], d, partition)
    end
end

@memoize function partition_mixer_hamiltonians(g::PartitionMixerGate)::Vector{Matrix{ComplexF64}}
    hamiltonians = Matrix{ComplexF64}[]

    # Implements Eqs. (11), (12)
    # iterate through partition (in reverse to have matrix application from right to left)
    for partition_part ∈ reverse(g.partition)
        for (a, b) ∈ partition_part
            push!(hamiltonians, XY_sum([a, b], g.d))
        end
    end

    return hamiltonians
end

# Defining this extra function allows memoizing the hamiltonians for the backward pass
function partition_mixer_gate_matrix(g::PartitionMixerGate, β::Float64)
    hamiltonians = partition_mixer_hamiltonians(g)

    Us = exp.(-im * β * hamiltonians)
    return prod(Us)
end

Qaintessent.matrix(g::PartitionMixerGate) = partition_mixer_gate_matrix(g, g.β[])

Qaintessent.adjoint(g::PartitionMixerGate) = PartitionMixerGate(-g.β[], g.d, g.partition)

function Qaintessent.backward(g::PartitionMixerGate, Δ::AbstractMatrix)
    delta = 1e-10

    U_partition1 = partition_mixer_gate_matrix(g, -(g.β[] - delta/2))
    U_partition2 = partition_mixer_gate_matrix(g, -(g.β[] + delta/2))

    U_deriv = (U_partition2 - U_partition1) / delta

    return PartitionMixerGate(2 * sum(real(U_deriv .* Δ)), g.d, g.partition)
end

Qaintessent.sparse_matrix(g::PartitionMixerGate) = sparse(matrix(g))

# wires (= g.d since we use the one-hot encoding canonically)
Qaintessent.num_wires(g::PartitionMixerGate)::Int = g.d


# Make trainable params available to Flux
Flux.@functor ParityRingMixerGate
Flux.@functor RNearbyValuesMixerGate
Flux.@functor PartitionMixerGate
