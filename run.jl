include("src/RL_TSP.jl")
using Main.RL_TSP

instance_id::Int= 3
max_episodes = 1000
trial_nbr::Int64 = 20
best_sol_each_n_episode = 50
optimal_sol = 19

# the first experiments
alpha_influence(instance_id, max_episodes=max_episodes, trial_nbr=trial_nbr, 
                best_sol_each_n_episode = best_sol_each_n_episode, optimal_sol = optimal_sol)

# the second experiments
fixed_epsilon_influence(instance_id, max_episodes=max_episodes, trial_nbr=trial_nbr, 
                best_sol_each_n_episode = best_sol_each_n_episode, optimal_sol = optimal_sol)

# the third experiments
epsilon_greedy_phase_nbr_Influence(instance_id, max_episodes=max_episodes, trial_nbr=trial_nbr, 
                best_sol_each_n_episode = best_sol_each_n_episode, optimal_sol = optimal_sol)
        
# the fourth experiments
compare_exploration_methods(instance_id, max_episodes=max_episodes, trial_nbr=trial_nbr, 
                best_sol_each_n_episode = best_sol_each_n_episode, optimal_sol = optimal_sol)                

# the fifth experiments
instance_id = 5
time_limit = 1 * 60
set_final_reward(10^4)
lr=constant_learning_rate(.2)
ef=ϵ_greedy_exploration_method(0.1, max_episodes=max_episodes, phase_nbr = 1)
a, _, _ = solve(instance_id, α = lr, exploration_method = ef, time_limit = time_limit)
