include("Qaintessent.jl/src/Qaintessent.jl") # To be replaced by a proper package import at some point
import .Qaintessent.AbstractGate

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
    r::Vector{UInt}
    d::Vector{UInt}

    function RNearbyValuesMixerGate(β::Real, r::UInt, d::UInt)
        (r > 0 && d > 0) || throw(ArgumentError("Parameters r and d must be positive integers!"))
        new([β], [r], [d])
    end
end

# Implementation of Eq. (6)
function matrix_onehot(g::RNearbyValuesMixerGate)
    g.r[] == 1 || throw(DomainError("Matrix for one-hot encoding only implemented for r=1."))
    
    X = [0 1; 1 0]
    Y = [0 -im; im 0]
    H_ring_enc = sum( # question: what does the paper mean by X_a for a = 0? Here, X_0 is just ignored
        kron((i ∈ [a, a+1] ? X : I(2) for i ∈ 1:d)...) # passing iterator into kron via varargs syntax
        + kron((i ∈ [a, a+1] ? Y : I(2) for i ∈ 1:d)...)
        for a ∈ 0:g.d[]-1
    )
    U_ring_enc = exp(-im * g.β[] * H_ring_enc)
end

function matrix_qudit(g::RNearbyValuesMixerGate)
    # question: will this simulator work with qudits?

    # Implementation of Eq. (A3)
    genX = I(g.d[])[:,vcat(2:g.d[], 1)] # move first columnn of identity matrix to back to obtain d-dim. generalized Pauli-X
    genX_T = transpose(genX)

    H_rNV = sum(genX^i + genX_T^i for i ∈ 1:g.r[]) # TODO implement more efficiently? (avoid exponentation for each i)
    U_rNV = exp(-im * g.β[] * H_rNV)

    U_rNV
end

function matrix(g::RNearbyValuesMixerGate)
    matrix_qudit(g)
end

sparse_matrix(g::RNearbyValuesMixerGate) = sparse(matrix(g))

# wires
# TODO: should be d in the one-hot case
num_wires(::RNearbyValuesMixerGate)::Int = 1
