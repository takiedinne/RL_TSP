"""
    This code it based on the project published in https://github.com/B4rtDC/DS425-PS
Thank you Bart!
"""
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
    init!(tsp)
    
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
    
    #pretty_Q_print(rlm.Q)
    # update the utilities (if required, method should be implemented)
    update!(rlm)

    return trial
end

function update!(rlm::Qlearner) 
    rlm.n += 1
    update_exploration_method!(rlm.exploration_method)
end
init!(rlm::Qlearner) = nothing

function get_best_itenerary(rlm::Qlearner, tsp::TSP)
    s = tsp.initial_state
    itenerary = [s.current_city]
    init!(tsp)
    while true
        a = argmax(aᶥ -> get!(rlm.Q, (s, aᶥ), 0.0), actions(tsp, s))
        if isnothing(a)
            break
        end
        s = take_single_action(tsp, s, a)
        push!(itenerary, s.current_city)
    end

    return itenerary, tsp.total_distance
end
