using Qaintessent
import Qaintessent.AbstractGate
import LinearAlgebra: I

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

    function RNearbyValuesMixerGate(β::Real, r::Integer, d::Integer)
        (r > 0 && d > 0) || throw(ArgumentError("Parameters r and d must be positive integers!"))
        new([β], r, d)
    end
end

# Implementation of Eq. (6)
function matrix_onehot(g::RNearbyValuesMixerGate)
    g.r == 1 || throw(DomainError("Matrix for one-hot encoding only implemented for r=1."))
    
    X = [0 1; 1 0]
    Y = [0 -im; im 0]
    H_ring_enc = sum( # question: what does the paper mean by X_a for a = 0? Here, X_0 is just ignored
        kron((i ∈ [a, a+1] ? X : I(2) for i ∈ 1:g.d)...) # passing iterator into kron via varargs syntax
        + kron((i ∈ [a, a+1] ? Y : I(2) for i ∈ 1:g.d)...)
        for a ∈ 0:g.d-1
    )
    U_ring_enc = exp(-im * g.β[] * H_ring_enc)
end

function matrix_qudit(g::RNearbyValuesMixerGate)
    # question: will this simulator work with qudits?

    # Implementation of Eq. (A3)
    genX = I(g.d)[:,vcat(2:g.d, 1)] # move first columnn of identity matrix to back to obtain d-dim. generalized Pauli-X
    genX_T = transpose(genX)

    H_rNV = sum(genX^i + genX_T^i for i ∈ 1:g.r) # TODO implement more efficiently? (avoid exponentation for each i)
    U_rNV = exp(-im * g.β[] * H_rNV)

    U_rNV
end

function Qaintessent.matrix(g::RNearbyValuesMixerGate)
    matrix_onehot(g)
end

Qaintessent.sparse_matrix(g::RNearbyValuesMixerGate) = sparse(matrix(g))

# wires
# TODO: should be d in the one-hot case, 1 for qudit case
Qaintessent.num_wires(g::RNearbyValuesMixerGate)::Int = g.d

"""
    Parity single-qudit ring mixer gate, which acts on a single qudit

``U_{\\text{parity}}(\\beta) = U_{\\text{last}}(\\beta) U_{\\text{even}}(\\beta) U_{\\text{odd}}(\\beta)``
``U_{\\text{odd}}(\\beta) = \\prod_{a~\\text{odd}, a \\neq d} e^{-i \\beta (X_a X_{a+1} + Y_a Y_{a+1})}``
``U_{\\text{even}}(\\beta) = \\prod_{a~\\text{even}} e^{-i \\beta (X_a X_{a+1} + Y_a Y_{a+1})}``
``U_{\\text{last}}(\\beta) = e^{-i \\beta (X_d X_1 + Y_d Y_1)} ~\\text{if}~ d ~\\text{is odd,}~ I ~\\text{otherwise.}``

Reference:\n
    Stuart Hadfield, Zhihui Wang, Bryan O'Gorman, Eleanor G. Rieffel, Davide Venturelli and Rupak Biswas\n
    From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz\n
    Algorithms 12.2 (2019), equations (7) - (10), p. 11
"""
struct ParityRingMixerGate <: AbstractGate
    β::Vector{Float64}
    d::Vector{Integer} # d (= κ) = number of colors

    function ParityRingMixerGate(β::Real, d::Integer)
        d > 0 || throw(ArgumentError("Parameter d must be a positive integer!"))
        new([β], [d])
    end

end

function Qaintessent.matrix(g::ParityRingMixerGate)
    X = [0 1; 1 0]
    Y = [0 -im; im 0]

    # Implements X_a X_{a+1} + Y_a Y_{a+1}, or more generally (⊗_{i ∈ xy_indices} X_i) + (⊗_{i ∈ xy_indices} Y_i)
    # question: what does the paper mean by X_a for a = d+1? Here, X_{d+1} is just ignored. Also, do they use 1-based indexing or 0-based? (cf. matrix_onehot above)
    XY_sum = xy_indices -> begin
        kron((i ∈ xy_indices ? X : I(2) for i ∈ 1:g.d)...) # passing iterator into kron via varargs syntax
        + kron((i ∈ xy_indices ? Y : I(2) for i ∈ 1:g.d)...)
    end

    # Implements Eq. (8)
    # question: by a ≠ n, do they actually mean a ≠ d? I assume so.
    H_odd = sum(XY_sum([a, a+1]) for a ∈ 1:2:(g.d - 1))
    H_even = sum(XY_sum([a, a+1]) for a ∈ 2:2:g.d)
    U_odd = exp(-im * g.β[] * H_odd)
    U_even = exp(-im * g.β[] * H_even)

    # Implements Eq. (9)
    U_last = isodd(g.d) ? exp(-im * g.β[] * XY_sum([g.d, 1])) : I

    # Implements Eq. (7)
    U_parity = U_last * U_even * U_odd
    U_parity
end

Qaintessent.sparse_matrix(g::ParityRingMixerGate) = sparse(matrix(g))

# wires
Qaintessent.num_wires(g::ParityRingMixerGate)::Int = g.d
