module QAOAMixers

using Qaintessent
import Qaintessent: AbstractGate, Circuit, CircuitGate, MeasurementOperator
using LinearAlgebra


include("max-k-colorable-subgraph.jl")
export
    MaxKColSubgraphPhaseSeparationGate,
    max_κ_colorable_subgraph_circuit,
    ψ_initial

include("mixer-gates.jl")
export 
    RNearbyValuesMixerGate,
    ParityRingMixerGate

end