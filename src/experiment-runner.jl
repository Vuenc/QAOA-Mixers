using CSV
using JSON
using Serialization: serialize
using Zygote
using Dates

# Runs a list of experiments specified in a CSV file and saves the results
function run_experiments_from_file(filename::String)
    # load experiment specifications from CSV file
    specifications = CSV.File(filename)

    # run "export JULIA_NUM_THREADS=N" before starting julia to enable multi-threading
    if Threads.nthreads() > 1
        println("Detected `Threads.nthreads()` == $(Threads.nthreads()): using multi-threaded mode.")
        println("Warning: Terminal outputs of different repeat runs will be interleaved.")
        println()
    end

    # each row in the CSV corresponds to one experiment setup (repeated ≥ 1 times)
    for experiment_spec in specifications
        # load the graph: parse edges from "a-b b-c c-d" format to [(a, b), (b, c), (c, d)] format
        edges = experiment_spec.edges |> split .|> (x -> split(x, "-") .|> (y -> parse(Int, y))) .|> Tuple
        graph = Graph(experiment_spec.numberOfNodes, edges)

        # recognize the mixer type, load arguments (like r for RNearbyValuesMixer)
        mixer_args = split(experiment_spec.mixerType)
        if mixer_args[1] == "RNearbyValuesMixer"
            length(mixer_args) == 2 || throw(ArgumentError("""RNearbyValuesMixer
                requires r argument, e.g. "RNearbyValuesMixer 2"."""))
            mixer_type = RNearbyValuesMixerGate
            mixer_params = [parse(Int, mixer_args[2])]
        elseif mixer_args[1] == "ParityRingMixer"
            mixer_type = ParityRingMixerGate
            mixer_params = []
        elseif mixer_args[1] == "PartitionMixer"
            length(mixer_args) == 2 || throw(ArgumentError("""PartitionMixer requires
                partition argument, e.g. "PartitionMixer 1-2,3-4;2-3,4-1"."""))
            mixer_type = PartitionMixerGate
            partition = (split(mixer_args[2], ";") 
                .|> (partition_part -> split(partition_part, ",")
                    .|> (t -> split(t, "-") 
                        .|> (index -> parse(Int, index))
                    |> Tuple)))
            mixer_params = [partition]
        else
            throw(DomainError("""Mixer type "$(experiment_spec.mixerType)" not recognized."""))
        end

        any_run_aborted = false
        # repeat the experiment the specified number of times
        Threads.@threads for i in experiment_spec.startFromRepeat:experiment_spec.repeats
            if !any_run_aborted
                not_aborted = run_single_experiment_repeat(
                    experiment_spec, graph, mixer_type, mixer_params, i)
                if !not_aborted
                    any_run_aborted = true
                end
            end
        end
        if any_run_aborted
            println("Aborting all experiments.")
            return
        end
        println("Experiment $(experiment_spec.experimentId) finished.")
        println()
    end
end

function run_single_experiment_repeat(experiment_spec, graph, mixer_type, mixer_params, i)::Bool
    aborted = false
    logger::Union{Nothing, QAOALogger} = nothing
    circ_out = nothing
    try
        println("Running experiment $(experiment_spec.experimentId), run $(i)...")
        # initialize the logger for this run
        logger = QAOALogger(experiment_spec.numberOfNodes, experiment_spec.edges, experiment_spec.numberOfColors,
            experiment_spec.numberOfLayers, experiment_spec.mixerType, experiment_spec.trainingRounds,
            experiment_spec.learningRate, experiment_spec.initStdDev, experiment_spec.repeats,
            Threads.nthreads() > 1)

        # run the optimization
        circ_out = optimize_qaoa(graph, experiment_spec.numberOfColors, p=experiment_spec.numberOfLayers,
            training_rounds=experiment_spec.trainingRounds, learning_rate=experiment_spec.learningRate,
            init_stddev=experiment_spec.initStdDev, mixer_type=mixer_type, mixer_params=mixer_params,
            logger=logger)
    catch e
        aborted = true
        # gracefully exit when interrupted with Ctrl+C in the REPL
        if e isa InterruptException
            println()
            println("Experiment $(experiment_spec.experimentId), run $(i) interrupted!")
        else
            # Have to catch other exceptions like this, because Julia
            # supresses the error message with a return in a finally block
            println()
            println("Fatal error occurred!")
            println(e)
            rethrow()
        end
        println("Trying to save partial log...")
    finally                
        # save the results (partial results at least if experiment was aborted)
        save_log(logger, "qaoa-experiment-#$(experiment_spec.experimentId)-run-$(i)"
            * (aborted ? "-aborted" : ""), aborted = aborted)
        
        if !aborted
            # serialize the output circuit if the run was not aborted
            serialize("qaoa-experiment-#$(experiment_spec.experimentId)-run-$(i)-circuit.sobj", circ_out)
            println("Experiment $(experiment_spec.experimentId), run $(i) finished.")
            println()
            return true
        else
            println("Partial log of aborted experiment saved.")
            return false
        end
    end
end

# Struct that saves logging data
struct QAOALogger
    logging_dict::Dict{String}
    start_time::DateTime

    # `QAOALogger` is initialized with the experiment params
    function QAOALogger(n::Int, edges::String, κ::Int, p::Int, mixer_type:: String, 
            training_rounds::Int, learning_rate::Float64, init_std_dev::Float64, repeats::Int,
            multi_threaded::Bool)
        experiment_params = Dict("numberOfNodes" => n, "edges" => edges, "numberOfColors" => κ,
            "numberOfLayers" => p, "mixerType" => mixer_type, "trainingRounds" => training_rounds,
            "learningRate" => learning_rate, "initStdDev" => init_std_dev, "repeats" => repeats,
            "multiThreaded" => multi_threaded)
        
        return new(Dict("experimentParams" => experiment_params, "coloringsByRounds" => [],
            "objectiveFunctionByRounds" => [], "gateParamsByRounds" => []),
            Dates.now())
    end
end

# Logs the optimization progress. @nograd necessary to prevent Zygote errors.
Zygote.@nograd function log_qaoa(logger::QAOALogger, round::Int,
        ψ_out::Vector{ComplexF64}, objective::Float64, params::Zygote.Params) 
    colorings = wavefunction_distribution(ψ_out)
    col_vecdict = map(c -> Dict("coloring" => join(c[1]), "p" => c[2]), colorings)
    push!(logger.logging_dict["coloringsByRounds"], Dict("round" => round, "colorings" => col_vecdict))
    push!(logger.logging_dict["objectiveFunctionByRounds"], Dict("round" => round, "objective" => objective))
    push!(logger.logging_dict["gateParamsByRounds"], Dict("round" => round, 
        "gateParams" => deepcopy(params.order.data)))
end

function save_log(logger::QAOALogger, filename::String; serialize_log::Bool=true, aborted=false)
    stop_time = Dates.now()
    time_ms = convert(Dates.Millisecond, stop_time - logger.start_time)
    logger.logging_dict["executionTimeMs"] = time_ms

    if aborted
        logger.logging_dict["aborted"] = true
    end

    open(filename * ".json", "w") do f
        JSON.print(f, logger.logging_dict)
    end
    if serialize_log
        serialize(filename * ".sobj", logger.logging_dict)
    end
end
