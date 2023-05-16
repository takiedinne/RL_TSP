"""
    This code it based on the project published in https://github.com/B4rtDC/DS425-PS
Thank you Bart!
"""
# ----------------------------------------------------------- #
#                    The  Exploration functions               #
# ----------------------------------------------------------- #

"""
    learning exploration methods:
N.P: every exploration method have to implement get_next_action function
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

    ϵ_greedy_exploration_method(ϵ::Float64=0.3, R⁺::Float64=10.0^10) = new(ϵ, R⁺)
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

# ----------------------------------------------------------- #
#                     The alpha functions                     #
# ----------------------------------------------------------- #
abstract type learning_rate end

mutable struct constant_learning_rate <: learning_rate
    α::Float64

    constant_learning_rate(α::Float64=0.1) = new(α)
end
function get_alpha(lr::constant_learning_rate, n)
    return lr.α
end

mutable struct exponetial_decreasing_learning_rate <: learning_rate
    α_initial::Float64
    decay_rate::Float64

    exponential_decreasing_learning_rate(α_initial::Float64=1., decay_rate::Float64=0.99) =
        new(α_initial, decay_rate)
end
function get_alpha(lr::exponetial_decreasing_learning_rate, n)
    return lr.α_initial * lr.decay_rate^n
end

mutable struct linear_decreasing_learning_rate <: learning_rate
    linear_decreasing_learning_rate() = new()
end
function get_alpha(lr::linear_decreasing_learning_rate, n)
    return 1 / (n + 1)
end


# ----------------------------------------------------------- #
#             The  Q reinforcement Learning method            #
# ----------------------------------------------------------- #
mutable struct Qlearner
    "current state"
    state
    "current action"
    action
    "current reward"
    reward
    "possible actions"
    actions::Set
    "terminal states"
    terminal_states::Set
    "Q values"
    Q::Dict
    "γ value"
    γ::Float64
    "learning rate α"
    α::learning_rate
    "exploration method"
    exploration_method::RLExplorationMethod
    "nbr_episodes"
    n::Int64

    Qlearner(state, action, reward, actions, terminal_states, Q, γ, α, exploration_method) =
        new(state, action, reward, actions, terminal_states, Q, γ, α, exploration_method, 0)
end

function Qlearner(tsp::TSP; γ=1.0,
    α::learning_rate=constant_learning_rate(), exploration_method::RLExplorationMethod=simple_exploration_method())

    return Qlearner(nothing, nothing, nothing, actions(tsp,tsp.initial_state), Set(), Dict(), γ, α, exploration_method)
end

"""
    learn!(rlm::Qlearner, sᶥ, rᶥ) 

Update the Q-values by using the state `s` and the associated reward `r`
"""
function learn!(rlm::Qlearner, tsp::TSP, sᶥ, rᶥ)
    s, a, Q, α, γ, f = rlm.state, rlm.action, rlm.Q, get_alpha(rlm.α, rlm.n), rlm.γ, rlm.exploration_method
    
    # update and evaluate Qvalues
    if is_terminal_state(tsp, sᶥ)
        #@warn "terminal state reached: ($(sᶥ), $(rᶥ))"
        Q[sᶥ, nothing] = rᶥ
    end

    if !isnothing(s)
        #@info "update Qvalues: ($(s), $(a))"
        # compute: Q(s,a) = Q(s,a) + α(n_sa)[r +γ*max(q(s',a')) - Q(s,a)]
        get!(Q, (s, a), 0.0)
        
        maxval = reduce(max, [get!(Q, (sᶥ, action), 0.0) for action in actions(tsp, sᶥ)])
        Q[(s, a)] += α * (rᶥ + γ * maxval - Q[(s, a)])
    end

    # update state 
    if !isnothing(sᶥ) && (is_terminal_state(tsp, sᶥ))
        rlm.state = nothing
        rlm.action = nothing
        rlm.reward = nothing
    else
        #@info "update state: new values are state = $(sᶥ), action = $(a), reward = $(rᶥ))"
        rlm.state = sᶥ
        rlm.reward = rᶥ
        # next action accounts for exploration function
        rlm.action = get_next_action(f, Q, sᶥ, actions(tsp, sᶥ))
    end

    #@info "next action is $(rlm.action)"
    return rlm.action
end

"""
    single_trial!(rlm::V, tsp::T) where {V<:ReinformentLearningMethod, T<:AbstractMarkovDecisionProcess}

Run a single trial until a terminal state is reached. The `ReinformentLearningMethod` is updated during the process.

Return the sequence of states visited during the trial.
"""
function single_trial!(rlm::Qlearner, tsp::TSP)
    s = tsp.initial_state
    trial = [s] # not required, for didactic purpose only
    init!(rlm)
    while true
        # get reward from current state
        r = reward(tsp, s)
        # transfer state and reward to ReinformentLearningMethod and obtain the action from the policy
        a = learn!(rlm, tsp, s, r)
        if isnothing(a)
            break
        end
        # update the state
        s = take_single_action(tsp, s, a)
        push!(trial, s) # not required, for didactic purpose only
        
    end
   
    # update the utilities (if required, method should be implemented)
    update!(rlm)

    return trial
end

function update!(rlm::Qlearner) 
    rlm.n += 1
end
init!(rlm::Qlearner) = nothing

function get_best_itenerary(rlm::Qlearner, tsp::TSP)
    s = tsp.initial_state
    itenerary = [s.current_city]
    while true
        a = argmax(aᶥ -> get!(rlm.Q, (s, aᶥ), 0.0), actions(tsp, s))
        if isnothing(a)
            break
        end
        s = take_single_action(tsp, s, a)
        push!(itenerary, s.current_city)
    end

    return itenerary, s.total_distance
end

function solve(tsp_id::Int64; γ::Float64=1.0,
      α::learning_rate=constant_learning_rate(),
      exploration_function::RLExplorationMethod=simple_exploration_method(),
      convergence_threshold = 0.0001, max_episodes = 1000000 )
    
    #tsp_id, α, γ, exploration_function = 3, constant_learning_rate(0.9), 1.0, simple_exploration_method()
   
    tsp = TSP(tsp_id)
    rlm = Qlearner(tsp, γ=γ, α=α, exploration_method=exploration_function)
   
    for _ in 1:max_episodes
        # save the current Q calues to check the convergence
        current_Q = Dict(rlm.Q)
        single_trial!(rlm, tsp)
        
        #= for k in keys(rlm.Q)
            println("$k => $(rlm.Q[k])")
        end
        
        println("###########################") =#
        # check convergence
        #= if keys(current_Q) == keys(rlm.Q) && 
                maximum([abs.(current_Q[k] - rlm.Q[k]) for k in keys(current_Q)]) < convergence_threshold
            @info "convergence reached at episode $(rlm.n)"
            break
        end =#
    end
    get_best_itenerary(rlm, tsp)
end