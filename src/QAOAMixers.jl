module QAOAMixers

using Qaintessent
import Qaintessent: AbstractGate, Circuit, CircuitGate, MeasurementOperator
using LinearAlgebra


include("mixer-gates.jl")
export 
    RNearbyValuesMixerGate,
    ParityRingMixerGate,
    PartitionMixerGate

include("max-k-colorable-subgraph.jl")
export
    MaxKColSubgraphPhaseSeparationGate,
    max_κ_colorable_subgraph_circuit,
    ψ_initial,
    optimize_qaoa

include("experiment-runner.jl")
export 
    run_experiments_from_file,
    QAOALogger

end