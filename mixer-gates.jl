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
    r::Vector{Float64}

    function RNearbyValuesMixerGate(β::Real, r::Int)
        r > 0 || throw(ArgumentError("Parameter r must be a positive integer!"))
        new([β], [r])
    end
end

function matrix(g::RNearbyValuesMixerGate)
    # question: will this simulator work with qudits? 
    d = 2
    genX = I(d)[:,vcat(2:d, 1)] # move first columnn of identity matrix to back to obtain d-dim. generalized Pauli-X
    genX_T = transpose(genX)
    H_rNV = sum(genX^i + genX_T^i for i ∈ 1:g.r[])
    U_rNV = exp(-im * g.β[] * H_rNV)

    U_rNV
end

sparse_matrix(g::RNearbyValuesMixerGate) = sparse(matrix(g))

# wires
num_wires(::RNearbyValuesMixerGate)::Int = 1
