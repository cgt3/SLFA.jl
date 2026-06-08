# Parameter defaults
const DEFAULT_N_1D = 250;
const DEFAULT_START_GAP = 0.0;
const DEFAULT_K_EXTREMA = 1;
const DEFAULT_MAGN_REDUCTION = 1e-4;
const MACHINE_EPS_FACTOR=1e-3;

abstract type Extremum end;
struct Minimum <: Extremum end;
struct Maximum <: Extremum end;

abstract type Monotonicity end;
struct Strict <: Monotonicity end;
struct Nonstrict <: Monotonicity end;

@inline function (::Strict)(a,b) return a < b end
@inline function (::Nonstrict)(a,b) return a <= b end

@inline function (::Strict)(a,b, ::Minimum) return a < b end
@inline function (::Strict)(a,b, ::Maximum) return a > b end

@inline function (::Nonstrict)(a,b, ::Minimum) return a <= b end
@inline function (::Nonstrict)(a,b, ::Maximum) return a >= b end

const DEFAULT_MONOTONICITY=Strict();

function rel_supr(::Maximum, X, y, i_extrema, I_support_set, I_exclude=false)
    y_extrema = y[i_extrema]
    y_next = maximum( y[~(I_support_set .|| I_exclude)] )

    return y_extrema - y_next
end

function rel_supr(::Minimum, X, y, i_extrema, I_support_set, I_exclude=false)
    return rel_supr(X, -y, i_extrema, I_support_set, I_exclude, Maximum())
end


@inline function dist(i1, i2, X::Vector{T_x}, D::AbstractMatrix) where T_x<:AbstractFloat
    D[i1, i2] != 0 ? (return D[i1, i2]) : (return norm(X[i1] - X[i2]))
end


# Helper functions ====================================================================

function get_nbr_matrix(X::Vector{T_x}; duplicate_tol=MACHINE_EPS_FACTOR*eps(T_x)) where T_x<:AbstractFloat
    n = length(X)

    # Get elements in sorted order
    I_sorted = sortperm(X)

    # Allocate memory for the neighbor matrix
    D = spzeros(n, n)

    # Use adjacent elements as neighbors provided they are distinct
    for i in eachindex(X)
        # Get the left neighbor (if it exists)
        i_left = i - 1
        while i_left > 0 && abs(X[I_sorted[i_left]] - X[I_sorted[i]]) < duplicate_tol
            i_left -= 1
        end

        # Add the neighbor and all of its duplicates if they exists
        i_left_duplicate = i_left
        while i_left_duplicate > 0 && abs(X[I_sorted[i_left_duplicate]] - X[I_sorted[i_left]]) < duplicate_tol
            D[I_sorted[i_left_duplicate], I_sorted[i]] = abs(X[I_sorted[i_left_duplicate]] - X[I_sorted[i]])
            i_left_duplicate -= 1
        end


        # Get the right neighbor (if it exists)
        i_right = i + 1
        while i_right <= n && abs(X[I_sorted[i_right]] - X[I_sorted[i]]) < duplicate_tol
            i_right += 1
        end

        # Add the neighbor and all of its duplicates if they exists
        i_right_duplicate = i_right
        while i_right_duplicate <= n && abs(X[I_sorted[i_right]] - X[I_sorted[i_right_duplicate]]) < duplicate_tol
            D[I_sorted[i_right_duplicate], I_sorted[i]] = abs(X[I_sorted[i_right_duplicate]] - X[I_sorted[i]])
            i_right_duplicate += 1
        end
    end

    return D
end

function get_support_set(X::Vector{T_x}, y::Vector{T_y}, i_extrema::Integer, D::AbstractMatrix, extremum_type::Extremum; 
    is_monotonic=DEFAULT_MONOTONICITY::Monotonicity,
    start_gap=0.0::T_x
    ) where {T_x<:AbstractFloat, T_y<:Number}

    support_set = zeros(Bool, length(y))
    support_set[i_extrema] = true

    I_next = findnz(D[:, i_extrema])[1]
    I_prev = [ i_extrema for i in 1:length(I_next) ]

    in_I_next = zeros(Bool, length(y))
    in_I_next[I_next] .= true

    I_terminal = Int64[]
    in_I_terminal = zeros(Bool, length(y))
    i = 1
    while i <= length(I_next) # Note: the size of I_next can change as the for-loop iterates
        i_nbr  = I_next[i]
        i_prev = I_prev[i]

        if !support_set[i_nbr] && dist(i_nbr, i_prev, X, D) > start_gap 
            if is_monotonic(y[i_prev], y[i_nbr], extremum_type)
                support_set[i_nbr] = true
                new_nbrs = findnz(D[:,i_nbr])[1]
                is_boundary_pt = true
                for i_new in new_nbrs
                    if !support_set[i_new] && !in_I_next[i_new] # TODO: also need sense of direction here for nD case
                        is_boundary_pt = false
                        push!(I_next, i_new)
                        push!(I_prev, i_nbr)
                    end
                end

                if is_boundary_pt && !in_I_terminal[i_nbr]
                    in_I_terminal[i_nbr] = true
                    push!(I_terminal, i_nbr)
                end
            elseif !in_I_terminal[i_prev] && i_prev != i_extrema
                in_I_terminal[i_prev] = true
                push!(I_terminal, i_prev)
            end
        end

        i += 1
    end

    return support_set, I_terminal
end

function get_k_extrema(X, y, D::AbstractMatrix; k=DEFAULT_K_EXTREMA, is_monotonic=DEFAULT_MONOTONICITY::Monotonicity)
end

function choose_extrema(X, y, support_sets, D::AbstractMatrix, score_func)
end




# SLFA functions ====================================================================
# Training parameters:
# - convergence function: RMSE, l-inf, other?
# - convergence criteria?
# - extrema ranking function (same as convergence function?)
# - monotonicity: options: strict, simple; 
# - start gap
# - omega: weighting parameter
# - globalization scheme

# Return:
# - RBFN
# - individual biases
# - residual history (in what norm/measure? RMSE, l-inf, and TV?)
# - TV history
# - neighbor graph (for nD case)

# SLFA in 1D
# function train_RBFN(X::Vector{T_x}, y::Vector{T_y}; 
#     N_max=DEFAULT_N_1D, 
#     T_RBF=Gaussian{Anisotropic{Aligned},1},  
#     conv_conditions=[x->norm(x, Inf)]::Vector{Function},
#     conv_criteria=[DEFAULT_MAGN_REDUCTION*conv_conditions[1](y)]::Vector{T_metric},
#     conv_enforcement=any::Function,
#     is_monotonic=:strict,
#     start_gap=DEFAULT_START_GAP::Real,
#     k_extrema=DEFAULT_K_EXTREMA::Integer,
#     ) where {T_x<:AbstractFloat, T_y<:Number, T_metric<:Real}

#     if length(X) != length(y)
#         throw("SLFA.train_RBFN: Number of data points and residual values do not match.")
#     end

#     # Compute neighbors


#     # Set up empty arrays
#     a = zeros(eltype(y), 1)
#     b = zeros(eltype(y), 1)
#     rbfs = T_RBF[]

#     # Train the network
#     res = copy(y)
#     N = 0
#     while N < N_max && conv_enforcement(conv_conditions(res) .< conv_criteria)
#         # get the 
#         N += 1
#     end


#     return RBFN(sum(b), a, rbfs), b, res_history, D
# end

# # nD
# function train_RBFN(X::Matrix{T_x}, y::Vector{T_y}; ) where {T_x<:AbstractFloat, T_y<:Number}
#     if size(X,1) != length(y)
#         throw("SLFA.train_RBFN: Number of data points and residual values do not match.")
#     end
# end

# # Quasi-1D
# function train_RBFN_quasi1D(X_all::Vector{Matrix{T_x}}, y_all::Vector{Vector{T_y}}; ) where {T_x<:AbstractFloat, T_y<:Number}
#     if length(X_all) != length(y_all)
#         throw("SLFA.train_RBFN: Length of X_all does not match length of y_all")
#     end

#     for k in eachindex(X_all)
#         if size(X_all[k],1) != length(y_all[k])
#             throw("SLFA.train_RBFN: Number of data points and residual values do not match.")
#         end
#     end
# end