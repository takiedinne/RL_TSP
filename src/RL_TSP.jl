module RL_TSP

using DelimitedFiles

include("TSP.jl")
include("RL_methods.jl")

tsp_id = 4

lr::learning_rate=constant_learning_rate(0.9)
ef::RLExplorationMethod=ϵ_greedy_exploration_method()
iterenary = solve(tsp_id, exploration_function=ef, α=lr, max_episodes=1000000)

println(iterenary)

path = copy(iterenary[1])
sort!(path)
println(path)
end # module RL_TSP
