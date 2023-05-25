
# ----------------------------------------------------------- #
#                    The Learning rate methods                #
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

