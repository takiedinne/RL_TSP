module RL_TSP

using DelimitedFiles
using Plots
using Random

# Set the seed value
seed_value = 123

# Set the seed
Random.seed!(seed_value)

include("TSP.jl")
include("explorationMethods.jl")
include("learningRateMethods.jl")
include("QLearning.jl")
include("experiments.jl")

export
    TSP, 
    Qlearner,
    get_all_states,
    set_final_reward,

    simple_exploration_method,
    ϵ_greedy_exploration_method,
    roulette_wheel,
    # learning learningRateMethods
    constant_learning_rate,
    exponetial_decreasing_learning_rate,
    linear_decreasing_learning_rate,
    #experiments
    solve,
    alpha_influence,
    fixed_epsilon_influence,
    epsilon_greedy_phase_nbr_Influence,
    compare_exploration_methods
end 
