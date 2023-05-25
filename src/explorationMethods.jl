# ----------------------------------------------------------- #
#                    The  Exploration functions               #
# ----------------------------------------------------------- #

"""
    learning exploration methods:
N.P: every exploration method have to implement get_next_action function and update_exploration_method!
"""
abstract type RLExplorationMethod end

"""
    simple_exploration_method <: RLExplorationMethod

    simple Exploration methods which makes sure that every action is tried at least N_e times
    parameters:
    - N_e: try (state, action) at least N_e times
    - R⁺: the optimistic estimate best possible reward
"""
mutable struct simple_exploration_method <: RLExplorationMethod
    "try (state, action) at least N_e times"
    N_e::Int64
    "the optimistic estimate best possible reward"
    R⁺::Float64
    "counts of seeing action a in state s"
    N_sa::Dict

    simple_exploration_method(N_e::Int64=10, R⁺::Float64=10.0^10) = new(N_e, R⁺, Dict())
end
"""
    get_next_action(explorer::simple_exploration_method, Q::Dict, state, actions)

    it returns the next action according the simple exploration method policy. 
    For each action it checks if it has been tried at least N_e times, 
    if not it assigne to her the optimistic estimate best possible reward R⁺, otherwise it assigne to her the Q value.
    and then return the action with the highest value.

    parameters:
    - explorer: the simple exploration method
    - Q: the Q values
    - state: the current state
    - actions: the possible actions
"""
function get_next_action(explorer::simple_exploration_method, Q::Dict, state, actions_list)
    #explorer, Q, state, actions_list = f, rlm.Q, sᶥ, actions(tsp, sᶥ)
    #get the estimates of the Q values
    action = argmax(action -> get!(explorer.N_sa, (state, action), 0) < explorer.N_e ?
                              explorer.R⁺ : get!(Q, (state, action), 0.0),
                actions_list)
    
    #increment the count of the action
    explorer.N_sa[(state, action)] += 1

    return action
end
update_exploration_method!(explorer::simple_exploration_method) = nothing
update_exploration_method!(explorer::simple_exploration_method, n, n_max) = nothing
"""
    ϵ_greedy_exploration_method <: RLExplorationMethod

    ϵ-greedy Exploration methods tries to balance between the greedy and random exploration methods.

    parameters:
    - ϵ: between [0, 1] represents the probability of choosing a random action.
    - R⁺: the optimistic estimate best possible reward
"""
mutable struct ϵ_greedy_exploration_method <: RLExplorationMethod
    ϵ::Float64
    R⁺::Float64
    phase_nbr::Int64
    max_episodes::Int64
    ϵ_initial::Float64
    ϵ_final::Float64
    n::Int64
    ϵ_greedy_exploration_method(ϵ::Float64=1.; R⁺::Float64=10.0^10, phase_nbr = 5, max_episodes = 100000000, ϵ_final=0.1) = 
            new(ϵ, R⁺, phase_nbr, max_episodes, ϵ, ϵ_final, 0)
end
"""
    get_next_action(explorer::ϵ_greedy_exploration_method, Q::Dict, state, actions).
    
    it returns the next action according the ϵ_greedy policy. 
    For each action it checks if it has been tried at least N_e times, 
    if not it assigne to her the optimistic estimate best possible reward R⁺, otherwise it assigne to her the Q value.
    and then return the action with the highest value.
    parameters:
    - explorer: the simple exploration method
    - Q: the Q values
    - state: the current state
    - actions: the possible actions
"""
function get_next_action(explorer::ϵ_greedy_exploration_method, Q::Dict, state, actions)
   
    if rand() < explorer.ϵ
        return rand(actions)
    else
        return argmax(action -> get!(Q, (state, action), 0.), actions)
    end
end
function update_exploration_method!(explorer::ϵ_greedy_exploration_method)
    explorer.n += 1
    phase_length = explorer.max_episodes / explorer.phase_nbr
    if explorer.n % phase_length == 0
        phase = explorer.n / (explorer.max_episodes / explorer.phase_nbr)
        explorer.ϵ = explorer.ϵ_initial - (explorer.ϵ_initial - explorer.ϵ_final) / (explorer.phase_nbr-1) * phase
        
    end
end

mutable struct roulette_wheel <: RLExplorationMethod 
    roulette_wheel() = new()
end
function get_next_action(explorer::roulette_wheel, Q::Dict, state, actions)
    f = [get!(Q, (state, action),0.0) for action in actions]
    f .+= abs(minimum(f))
    r = rand() 
    sum_f = sum(f)
    for i in 1:length(f)
        if r <= sum(f[1:i])/sum_f
            return actions[i]
        end
    end
end

update_exploration_method!(explorer::roulette_wheel) = nothing
