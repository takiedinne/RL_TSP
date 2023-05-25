import Base: isequal, hash, in
global final_reward = 10
function set_final_reward(fr)
    global final_reward = fr
end
"""
    To Do: describe the TSP problem here
"""
mutable struct TSP_state
    current_city::Int64
    visited_cities::Set{Int64}
    #total_distance::Float64
end
# Define the isequal function
function isequal(x::TSP_state, y::TSP_state)
    return x.current_city == y.current_city && x.visited_cities == y.visited_cities
end

# Define the hash function useful to compare element inside a Dict
function Base.hash(x::TSP_state, h::UInt)
    hash(x.current_city, hash(x.visited_cities, h))
end


"""
    TSP is general struct for the TSP instances
"""
mutable struct TSP
    name::String # name of the instance
    num_cities::Int64 # number of cities
    intercities_dists::Array{Float64,2} # distance matrix
    initial_state::TSP_state
    total_distance::Float64
    #constructor 
    TSP(name::String, num_cities::Int64, intercities_dists::Array{Float64,2}, initial_state::TSP_state) = 
        new(name, num_cities, intercities_dists, initial_state, 0.0)

end
"""
    TSP(inst_id::Int)

Create a TSP instance from data folder which it order is `inst_id`.
"""
function TSP(inst_id::Int)
    # read the TSP_instnaces folder
    instance_name = readdir("Data/TSP_instances")[inst_id]
    tsp_instance_file = joinpath("Data/TSP_instances",instance_name, "intercities_dists.txt")
    intercities_dists = readdlm(tsp_instance_file)
    
    #initial state 
    initial_state = TSP_state(1, Set())
   
    TSP(instance_name, size(intercities_dists,1), intercities_dists, initial_state)
end

function actions(tsp::TSP, state::TSP_state)
    if is_terminal_state(tsp, state) 
        return Set{Nothing}([nothing])
    else
        actions_as_list = filter(x -> !(x in union(state.visited_cities, state.current_city)), collect(1:tsp.num_cities))
        if isempty(actions_as_list)
            push!(actions_as_list, 1)
        end
        Set(actions_as_list)
    end
end

function is_terminal_state(tsp::TSP, state::TSP_state)
    return (state.current_city == 1 && state.visited_cities == Set(1:tsp.num_cities))
end

function reward(tsp::TSP, state::TSP_state)
    global final_reward
    if is_terminal_state(tsp, state)
        return final_reward - tsp.total_distance
    else
        return  -1 * tsp.total_distance
    end
end
"""
    take_single_action(tsp::TSP, s, a)
Given the TSP, a state `s` and a desired action `a`, obtain the new state `s'`. 
This is not random.
but we can add some randomness to it, to simulate the the mistakes of the driver.
"""
function take_single_action(tsp::TSP, state::TSP_state, action)
    new_state = deepcopy(state)
    #add the current city to the visited cities
    push!(new_state.visited_cities, new_state.current_city)
    #increement the total total_distance
    tsp.total_distance += tsp.intercities_dists[state.current_city, action]
    #update the current city
    new_state.current_city = action
    #return
    new_state
end

function init!(tsp::TSP) 
    tsp.total_distance = 0.0
end
""" 
    get_all_states(tsp::TSP)
Return a list of all possible states of the TSP instance `tsp`. becarful it can be very large.
"""
function get_all_states(tsp::TSP)
    states = []
    states_queue = [tsp.initial_state]
    while !isempty(states_queue)
        s = popfirst!(states_queue)
        push!(states, s)
        for a in actions(tsp, s)
            if isnothing(a)
                continue
            end
            sˡ = take_single_action(tsp, s, a)
            seen = any(x -> isequal(x, sˡ), states) || any(x -> isequal(x, sˡ), states_queue)
            !seen && push!(states_queue, sˡ)
        end
    end
    states
end
