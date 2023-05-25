# ----------------------------------------------------------- #
#                   The different experiments                 #
# ----------------------------------------------------------- #
global main_results_folder_path = joinpath(@__DIR__, ".." , "results")
!isdir(main_results_folder_path) && mkdir(main_results_folder_path)

function hasTimeExpired(rlm::Qlearner, max_iter, start_time, time_limit)
    
    if time_limit == -1 && rlm.n >= max_iter
        return true
    elseif time_limit  <= (time() - start_time )
        return true
    else
        return false
    end
end
"""
    solve(tsp_id::Int64; γ::Float64=1.0,
    α::learning_rate=constant_learning_rate(),
    exploration_method::RLExplorationMethod=simple_exploration_method(),
    convergence_threshold = 0.00001, max_episodes = 100000000 )
    
    Solve the TSP instance `tsp_id` using Q learning. The parameters are:
    - `γ` the discount factor
    - `α` the learning rate method
    - `exploration_method` the exploration method
    - `convergence_threshold` the threshold to check the convergence
    - `max_episodes` the maximum number of episodes to run
"""
function solve(tsp_id::Int64; γ::Float64=1.0,
    α::learning_rate=constant_learning_rate(),
    exploration_method::RLExplorationMethod=simple_exploration_method(),
    convergence_threshold = 0.001, max_episodes = 10^6, 
    best_sol_each_n_episode = -1, time_limit=-1)
  
  #tsp_id, α, γ, exploration_method, max_episodes = 7, constant_learning_rate(0.5), 1.0, ϵ_greedy_exploration_method(), 10000
    
  tsp = TSP(tsp_id)
  rlm = Qlearner(tsp, γ=γ, α=α, exploration_method=exploration_method)
  check_best_sol = best_sol_each_n_episode > 0 ? true : false
  best_sol_length_per_iter = [0]
  converge_couter = 0
    
  start_time = time()
  while !hasTimeExpired(rlm, max_episodes, start_time, time_limit)
      (rlm.n % 100 == 0) && @info "[QL]: we are in episode = $(rlm.n))"
      
      # save the current Q calues to check the convergence
      current_Q = Dict(rlm.Q)
      single_trial!(rlm, tsp)
      
      if check_best_sol && rlm.n % best_sol_each_n_episode == 0
          push!(best_sol_length_per_iter, get_best_itenerary(rlm, tsp)[2])
      end

      # check convergence
      #= if keys(current_Q) == keys(rlm.Q) && 
          maximum([abs.(current_Q[k] - rlm.Q[k]) for k in keys(current_Q)]) < convergence_threshold
          converge_couter += 1
      else
          converge_couter = 0
      end
      if converge_couter == 1000
          @info "convergence reached at episode $(rlm.n)"
          break
      end =#
  end
  
  get_best_itenerary(rlm, tsp), rlm.n, best_sol_length_per_iter
end

#= counter = 0
for i in 1:2
    max_episodes = 1000000
    lr::learning_rate=constant_learning_rate(.5)
    ef::RLExplorationMethod=ϵ_greedy_exploration_method(0.3, max_episodes=max_episodes, phase_nbr = 1)
    a, _, _ = solve(3, α = lr, exploration_method = ef, max_episodes = max_episodes)
    if a[2] == 19
        counter += 1
    end
end
counter  =#
function pretty_Q_print(Q::Dict)
  key_list = sort!(collect(keys(Q)) , by = x -> [x[1].current_city, length(x[1].visited_cities), x[2]])  
  for k in key_list
      println("$k => $(Q[k])")
  end
end

"""
    gamma_influence(instance_id::Int)
    study the influence of gamma on the number of episodes to converge for the instance `instance_id`
"""
function gamma_influence(instance_id::Int; display_plot::Bool=true, save_results::Bool=true, max_episodes = 100000000)
    
    nbr_episodes = []
    γ_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.]
    
    for i in eachindex(γ_list)
        γ = γ_list[i]
        lr::learning_rate=constant_learning_rate(0.9)
        ef::RLExplorationMethod=ϵ_greedy_exploration_method(0.3, max_episodes=max_episodes, phase_nbr = 1)
        iterenary, n = solve(instance_id, exploration_method=ef, α=lr, max_episodes=max_episodes, γ = γ)
        push!(nbr_episodes, n)
    end
    #show the results
    if display_plot 
        p = plot(γ_list, nbr_episodes, xlabel="γ", ylabel="Number of episodes", label="gamma convergence influence")
        display(p)
    end
    #check if the folder exists
    !isdir(joinpath(main_results_folder_path, "gamma_influence")) && mkdir(joinpath(main_results_folder_path, "gamma_influence"))
    save_results && writedlm(joinpath(main_results_folder_path, "gamma_influence", "gamma_influence_$instance_id.txt"), nbr_episodes)
end

"""
    alpha_influence(instance_id::Int)
    study the influence of alpha on the number of episodes to converge for the instance `instance_id`
"""
function alpha_influence(instance_id::Int; display_plot::Bool=true, save_results::Bool=true,
         max_episodes = 100000000, trial_nbr::Int64 = 10, best_sol_each_n_episode = 100, optimal_sol = 0)
    
    #instance_id, display_plot, save_results, max_episodes, trial_nbr = 7, true, true, 10000, 10
    #best_sol_each_n_episode, optimal_sol = 100, 5
    optimal_sol == 0 && @warn "you did not define the optimal solution"
    α_list = [0.1, 0.2, 0.3 , 0.4, 0.5, 0.6, 0.8, 1.]
    nbr_best_sol_per_iter = []
    
    for i in eachindex(α_list)
        α = α_list[i] 
        lr::learning_rate=constant_learning_rate(α)
        current_alpha_best_sol_per_iter = zeros(Int64, ceil(Int64, max_episodes/best_sol_each_n_episode)+1)
        for trial in 1:trial_nbr
            ef::RLExplorationMethod=ϵ_greedy_exploration_method(0.3, max_episodes=max_episodes, phase_nbr = 5)
            iterenary, n, best_sol_each_n = solve(instance_id, exploration_method=ef, α=lr, max_episodes=max_episodes,
                         γ = 1., best_sol_each_n_episode = best_sol_each_n_episode) 
            
            current_alpha_best_sol_per_iter += Int64[x ? 1 : 0 for x in (best_sol_each_n .== optimal_sol)]
        end
        push!(nbr_best_sol_per_iter, current_alpha_best_sol_per_iter)
    end

    #check if the folder exists
    !isdir(joinpath(main_results_folder_path, "alpha_influence")) && mkdir(joinpath(main_results_folder_path, "alpha_influence"))
    #show the results
    if display_plot 
        x = collect(0:best_sol_each_n_episode:max_episodes)
        p = plot(xlabel="Iteration", ylabel="Number of times found optimal solution", title ="fixed alpha convergence influence")
        for i in eachindex(nbr_best_sol_per_iter)
            plot!(x, nbr_best_sol_per_iter[i], label = "α = $(α_list[i])")
        end
        display(p)
        #save the plot
        savefig(joinpath(main_results_folder_path, "alpha_influence", "alpha_influence_$instance_id.svg"))
    end
    
    save_results && writedlm(joinpath(main_results_folder_path, "alpha_influence", "alpha_influence_$instance_id.txt"), nbr_best_sol_per_iter)
end

"""
    epsilon_influence(instance_id::Int)
    study the influence of epsilon on the number of episodes to converge for the instance `instance_id`
"""
function fixed_epsilon_influence(instance_id::Int; display_plot::Bool=true, save_results::Bool=true,
    max_episodes = 100000000, trial_nbr::Int64 = 10, best_sol_each_n_episode = 100, optimal_sol = 0)
    
    optimal_sol == 0 && @warn "you did not define the optimal solution"
    ϵ_list = [0.1,  0.2, #= 0.3 , =# 0.4,  #= 0.5, =# 0.6, 0.8]
    nbr_best_sol_per_iter = []
    
    for i in eachindex(ϵ_list)
        ϵ = ϵ_list[i] 
        lr::learning_rate=constant_learning_rate(0.2)
        current_epsilon_best_sol_per_iter = zeros(Int64, ceil(Int64, max_episodes/best_sol_each_n_episode)+1)
        for trial in 1:trial_nbr
            ef::RLExplorationMethod=ϵ_greedy_exploration_method(ϵ, max_episodes=max_episodes, phase_nbr = 1)
            iterenary, n, best_sol_each_n = solve(instance_id, exploration_method=ef, α=lr, max_episodes=max_episodes,
                         γ = 1., best_sol_each_n_episode = best_sol_each_n_episode) 
            
            current_epsilon_best_sol_per_iter += Int64[x ? 1 : 0 for x in (best_sol_each_n .== optimal_sol)]
        end
        push!(nbr_best_sol_per_iter, current_epsilon_best_sol_per_iter)
    end
   
    #check if the folder exists
    !isdir(joinpath(main_results_folder_path, "epsilon_influence")) && mkdir(joinpath(main_results_folder_path, "epsilon_influence"))
    #show the results
    if display_plot 
        x = collect(0:best_sol_each_n_episode:max_episodes)
        p = plot(xlabel="Iteration", ylabel="Number of times QL found optimal solution", title ="Epsilon convergence influence")
        for i in eachindex(nbr_best_sol_per_iter)
            plot!(x, nbr_best_sol_per_iter[i], label = "ϵ = $(ϵ_list[i])")
        end
        display(p)
        #save the plot
        savefig(joinpath(main_results_folder_path, "epsilon_influence", "epsilon_influence_$instance_id.svg"))
    end
    save_results && writedlm(joinpath(main_results_folder_path, "epsilon_influence", "epsilon_influence_$instance_id.txt"), nbr_best_sol_per_iter)
end

"""
    epsilon_greedy_phase_nbr_Influence(instance_id::Int; display_plot::Bool=true, save_results::Bool=true,
    max_episodes = 100000000, trial_nbr::Int64 = 10, best_sol_each_n_episode = 100, optimal_sol = 0)

    study the influence of the number of phases for the epsilon greedy exploration method
"""
function epsilon_greedy_phase_nbr_Influence(instance_id::Int; display_plot::Bool=true, save_results::Bool=true,
    max_episodes = 100000000, trial_nbr::Int64 = 10, best_sol_each_n_episode = 100, optimal_sol = 0)
    
    optimal_sol == 0 && @warn "you did not define the optimal solution"
    phase_nbr_list = [1, 2, 4, 6, 8, 10]
    nbr_best_sol_per_iter = []
    
    for i in eachindex(phase_nbr_list)
        ϵ = 1. #initial value
        phase_nbr = phase_nbr_list[i]
        lr::learning_rate=constant_learning_rate(0.2)
        current_epsilon_best_sol_per_iter = zeros(Int64, ceil(Int64, max_episodes/best_sol_each_n_episode)+1)
        for trial in 1:trial_nbr
            ef::RLExplorationMethod=ϵ_greedy_exploration_method(ϵ, max_episodes=max_episodes, phase_nbr = phase_nbr)
            iterenary, n, best_sol_each_n = solve(instance_id, exploration_method=ef, α=lr, max_episodes=max_episodes,
                         γ = 1., best_sol_each_n_episode = best_sol_each_n_episode) 
            
            current_epsilon_best_sol_per_iter += Int64[x ? 1 : 0 for x in (best_sol_each_n .== optimal_sol)]
        end
        push!(nbr_best_sol_per_iter, current_epsilon_best_sol_per_iter)
    end
   
    #check if the folder exists
    !isdir(joinpath(main_results_folder_path, "epsilon_greedy_phase_nbr_influence")) && mkdir(joinpath(main_results_folder_path, "epsilon_greedy_phase_nbr_influence"))
    #show the results
    if display_plot 
        x = collect(0:best_sol_each_n_episode:max_episodes)
        p = plot(xlabel="Iteration", ylabel="Number of times QL found optimal solution", title ="Epsilon Greedy Nbr of phase convergence influence")
        for i in eachindex(nbr_best_sol_per_iter)
            plot!(x, nbr_best_sol_per_iter[i], label = "nbr phase = $(phase_nbr_list[i])")
        end
        display(p)
        #save the plot
        savefig(joinpath(main_results_folder_path, "epsilon_greedy_phase_nbr_influence", "epsilon_greedy_phase_nbr_influence_$instance_id.svg"))
    end
    save_results && writedlm(joinpath(main_results_folder_path, "epsilon_greedy_phase_nbr_influence", "epsilon_greedy_phase_nbr_influence_$instance_id.txt"), nbr_best_sol_per_iter)
end
"""
    compare_exploration_methods(instance_id::Int; display_plot::Bool=true, save_results::Bool=true,
    max_episodes = 100000000, trial_nbr::Int64 = 10, best_sol_each_n_episode = 100, optimal_sol = 0)

    study the influence of the exploration methods
"""
function compare_exploration_methods(instance_id::Int; display_plot::Bool=true, save_results::Bool=true,
    max_episodes = 100000000, trial_nbr::Int64 = 10, best_sol_each_n_episode = 100, optimal_sol = 0)
    
    optimal_sol == 0 && @warn "you did not define the optimal solution"
    
    exploration_name_list = ["simple", "ϵ-greedy", "roulette_wheel"]
    nbr_best_sol_per_iter = []
    #the simple exploration method
    
    lr=constant_learning_rate(0.2)
    ef=simple_exploration_method()
    current_epsilon_best_sol_per_iter = zeros(Int64, ceil(Int64, max_episodes/best_sol_each_n_episode)+1)
    for trial in 1:trial_nbr
        iterenary, n, best_sol_each_n = solve(instance_id, exploration_method=ef, α=lr, max_episodes=max_episodes,
                        γ = 1., best_sol_each_n_episode = best_sol_each_n_episode) 
        
        current_epsilon_best_sol_per_iter += Int64[x ? 1 : 0 for x in (best_sol_each_n .== optimal_sol)]
    end
    push!(nbr_best_sol_per_iter, current_epsilon_best_sol_per_iter)
  
    #the epsilon greedy exploration method
    ϵ = 0.1 #initial value
    lr=constant_learning_rate(0.2)
    current_epsilon_best_sol_per_iter = zeros(Int64, ceil(Int64, max_episodes/best_sol_each_n_episode)+1)
    for trial in 1:trial_nbr
        ef=ϵ_greedy_exploration_method(ϵ, max_episodes=max_episodes, phase_nbr = 1)
        iterenary, n, best_sol_each_n = solve(instance_id, exploration_method=ef, α=lr, max_episodes=max_episodes,
                        γ = 1., best_sol_each_n_episode = best_sol_each_n_episode) 
        
        current_epsilon_best_sol_per_iter += Int64[x ? 1 : 0 for x in (best_sol_each_n .== optimal_sol)]
    end
    push!(nbr_best_sol_per_iter, current_epsilon_best_sol_per_iter)
   
    #the roulette_wheel exploration method
    
    lr=constant_learning_rate(0.2)
    ef=roulette_wheel()
    current_epsilon_best_sol_per_iter = zeros(Int64, ceil(Int64, max_episodes/best_sol_each_n_episode)+1)
    for trial in 1:trial_nbr
        iterenary, n, best_sol_each_n = solve(instance_id, exploration_method=ef, α=lr, max_episodes=max_episodes,
                        γ = 1., best_sol_each_n_episode = best_sol_each_n_episode) 
        
        current_epsilon_best_sol_per_iter += Int64[x ? 1 : 0 for x in (best_sol_each_n .== optimal_sol)]
    end
    push!(nbr_best_sol_per_iter, current_epsilon_best_sol_per_iter)
    
    #check if the folder exists
    !isdir(joinpath(main_results_folder_path, "exploration_method_influence")) && mkdir(joinpath(main_results_folder_path, "exploration_method_influence"))
    #show the results
    if display_plot 
        x = collect(0:best_sol_each_n_episode:max_episodes)
        p = plot(xlabel="Iteration", ylabel="Number of times found optimal solution", title ="Epsilon Greedy Nbr of phase convergence influence")
        for i in eachindex(nbr_best_sol_per_iter)
            plot!(x, nbr_best_sol_per_iter[i], label = "nbr phase = $(exploration_name_list[i])")
        end
        display(p)
        #save the plot
        savefig(joinpath(main_results_folder_path, "exploration_method_influence", "exploration_method_influence_$instance_id.svg"))
    end
    save_results && writedlm(joinpath(main_results_folder_path, "exploration_method_influence", "exploration_method_influence_$instance_id.txt"), nbr_best_sol_per_iter)
end





