# Parameter defaults
const DEFAULT_N_1D = 250;
const DEFAULT_START_GAP = 0.0;
const DEFAULT_K_EXTREMA = 1;
const DEFAULT_MAGN_REDUCTION = 1e-6;
const MACHINE_EPS_FACTOR=1e3;

abstract type Extremum end;
struct Minimum <: Extremum end;
struct Maximum <: Extremum end;

abstract type Monotonicity end;
struct Strict <: Monotonicity end;
struct Nonstrict <: Monotonicity end;

@inline function (::Strict)(a,b) return a > b end
@inline function (::Nonstrict)(a,b) return a >= b end

@inline function (::Strict)(a,b, ::Minimum) return a < b end
@inline function (::Strict)(a,b, ::Maximum) return a > b end

@inline function (::Nonstrict)(a,b, ::Minimum) return a <= b end
@inline function (::Nonstrict)(a,b, ::Maximum) return a >= b end

const DEFAULT_MONOTONICITY=Strict();

include("./solvers.jl")


@inline function dist!(D::AbstractMatrix, i1::Integer, i2::Integer, X::Vector{T_x}) where T_x<:AbstractFloat
    if D[i1,i2] == 0 && D[i2,i1] == 0
        D[i1,i2] = abs(X[i1] - X[i2])
        D[i2,i1] = D[i1,i2]
    elseif D[i1,i2] == 0
        D[i1,i2] = D[i2,i1]
    elseif D[i2,i1] == 0
        D[i12,i1] = D[i1,i2]
    end

    return D[i1,i2]
end

@inline function dist!(D::AbstractMatrix, i1::Integer, i2::Integer, X::Matrix{T_x}) where T_x<:AbstractFloat
    if D[i1,i2] == 0 && D[i2,i1] == 0
        D[i1,i2] = norm(X[i1,:] - X[i2,:])
        D[i2,i1] = D[i1, i2]
    elseif D[i1,i2] == 0
        D[i1,i2] = D[i2,i1]
    elseif D[i2,i1] == 0
        D[i12,i1] = D[i1,i2]
    end

    return D[i1,i2]
end


# Helper functions ====================================================================

function get_nbr_matrix1D(X::Union{Vector, Matrix}; duplicate_tol=MACHINE_EPS_FACTOR*eps(eltype(X)))
    
    # Get elements in sorted order if in true 1D
    if X isa Vector
        n = length(X)
        I_sorted = sortperm(X)
    else
        n = size(X,1)
        I_sorted = 1:n
    end


    # Allocate memory for the neighbor matrix
    A = spzeros(Bool, n, n)
    D = spzeros(n, n)

    # Use adjacent elements as neighbors provided they are distinct
    for i in eachindex(I_sorted)
        # Get the left neighbor (if it exists)
        i_left = i - 1
        while i_left > 0 && dist!(D, I_sorted[i_left], I_sorted[i], X) < duplicate_tol
            i_left -= 1
        end

        # Add the neighbor and all of its duplicates if they exists
        i_left_duplicate = i_left
        while i_left_duplicate > 0 && dist!(D, I_sorted[i_left_duplicate], I_sorted[i_left], X) < duplicate_tol
            dist!(D, I_sorted[i_left_duplicate], I_sorted[i], X)
            A[I_sorted[i_left_duplicate], I_sorted[i]] = true
            A[I_sorted[i], I_sorted[i_left_duplicate]] = true
            i_left_duplicate -= 1
        end


        # Get the right neighbor (if it exists)
        i_right = i + 1
        
        while i_right <= n && dist!(D, I_sorted[i_right], I_sorted[i], X) < duplicate_tol
            i_right += 1
        end

        # Add the neighbor and all of its duplicates if they exists
        i_right_duplicate = i_right
        while i_right_duplicate <= n && dist!(D, I_sorted[i_right_duplicate], I_sorted[i_right], X) < duplicate_tol
            dist!(D, I_sorted[i_right_duplicate], I_sorted[i], X)
            A[I_sorted[i_right_duplicate], I_sorted[i]] = true
            A[I_sorted[i], I_sorted[i_right_duplicate]] = true
            i_right_duplicate += 1
        end
    end

    return A, D
end


function get_support_set_1Dsorted(X::Vector{T_x}, res::Vector{T_y}, i_extrema::Integer, A::AbstractMatrix, D::AbstractMatrix, extremum_type::Extremum; 
    is_monotonic=DEFAULT_MONOTONICITY::Monotonicity,
    start_gap=0.0::T_x
    ) where {T_x<:AbstractFloat, T_y<:Number}

    support_set = zeros(Bool, length(res))
    support_set[i_extrema] = true
    I_terminal = Int64[]

    i_prev = i_extrema
    i_left = i_extrema - 1
    while i_left > 0 && i_prev > 0 && ( (dist!(D, i_prev, i_left, X) > start_gap && is_monotonic(res[i_prev], res[i_left], extremum_type)) || dist!(D, i_prev, i_left, X) <= start_gap )
        support_set[i_left] = true
        if dist!(D, i_prev, i_left, X) > start_gap
            i_prev -= 1
        end
        i_left -= 1
    end
    i_left += 1

    if i_left != i_extrema
        push!(I_terminal, i_left)
    end

    i_prev = i_extrema
    i_right = i_extrema + 1
    while i_right <= n && i_prev <= n && ( (dist!(D, i_prev, i_right, X) > start_gap && is_monotonic(res[i_prev], res[i_right], extremum_type)) || dist!(D, i_prev, i_right, X) <= start_gap )
        support_set[i_right] = true
        if dist!(D, i_prev, i_right, X) > start_gap
            i_prev += 1
        end
        i_right += 1
    end
    i_right -= 1

    if i_right != i_extrema
        push!(I_terminal, i_right)
    end

    return support_set, I_terminal
end

function get_support_set(X::Union{Vector{T_x}, Matrix{T_x}}, res::Vector{T_y}, i_extrema::Integer, A::AbstractMatrix, D::AbstractMatrix, extremum_type::Extremum; 
    is_monotonic=DEFAULT_MONOTONICITY::Monotonicity,
    start_gap=0.0::T_x
    ) where {T_x<:AbstractFloat, T_y<:Number}

    support_set = zeros(Bool, length(res))
    support_set[i_extrema] = true

    I_next = findnz(A[:, i_extrema])[1]
    I_prev = [ i_extrema for i in 1:length(I_next) ]   # These are the nodes to check monotonicity againsts
    I_parent = [ i_extrema for i in 1:length(I_next) ] # These are the nodes that added a given node

    in_I_next = zeros(Bool, length(res))
    in_I_next[I_next] .= true

    I2I_prev = zeros(Int64, length(res))
    I2I_prev[I_next] .= [1:length(I_next)...]

    I_terminal = Int64[]
    in_I_terminal = zeros(Bool, length(res))
    i = 1
    while i <= length(I_next) # Note: the size of I_next can change as the loop iterates
        i_nbr  = I_next[i]
        i_prev = I_prev[i]
        i_parent = I_parent[i]

        if !support_set[i_nbr] 
            if is_monotonic(res[i_prev], res[i_nbr], extremum_type)
                support_set[i_nbr] = true
                new_nbrs = findnz(A[:,i_nbr])[1]
                is_boundary_pt = true
                for i_new in new_nbrs
                    if !support_set[i_new] && !in_I_next[i_new]
                        is_boundary_pt = false
                        push!(I_next, i_new) 
                        # TODO: For nD case, a node can have multiple parent nodes; the one most aligned to the 
                        #       extremum should be used
                        push!(I_parent, i_nbr)
                        I2I_prev[i_new] = length(I_next)

                        if dist!(D, i_nbr, i_new, X) > start_gap
                            push!(I_prev, i_nbr)
                        else
                            # TODO: For nD case, the parent's lineage may not be the best way to find a comparison 
                            #       point due to the drift/the spiral effect
                            i_prev_start_gap = i_parent
                            while i_prev_start_gap != i_extrema && dist!(D, i_new, i_prev_start_gap, X) <= start_gap
                                i_prev_start_gap = I_parent[I2I_prev[i_prev_start_gap]]
                            end
                            push!(I_prev, i_prev_start_gap)
                        end
                    end
                end

                if is_boundary_pt && !in_I_terminal[i_nbr] && i_nbr != i_extrema
                    in_I_terminal[i_nbr] = true
                    push!(I_terminal, i_nbr)
                end
            elseif !in_I_terminal[i_parent] && i_parent != i_extrema
                in_I_terminal[i_parent] = true
                push!(I_terminal, i_parent)
            end
        end

        i += 1
    end

    return support_set, I_terminal
end

function get_2k_extrema(X::Union{Vector{T_x}, Matrix{T_x}}, res::Vector{T_y}, A::AbstractMatrix, D::AbstractMatrix; 
    k_max=DEFAULT_K_EXTREMA,
    is_monotonic=DEFAULT_MONOTONICITY::Monotonicity,
    start_gap=0.0::T_x
    ) where {T_x<:AbstractFloat, T_y<:Number}
    
    n = length(res)

    # Only consider points who are greater than (maxima) or less than (minima) at least one of their neighbors
    I_unprocessed = zeros(Bool, n)
    for i in eachindex(res)
        if any(res[i] .> res[A[:,i]]) || any(res[i] .< res[A[:,i]])
            I_unprocessed[i] = true
        end
    end


    extrema_types = Extremum[]
    I_extrema = Int64[]
    support_sets = Vector{Bool}[]
    I_terminal_all = Vector{Int64}[]
    I_all = [1:n...]

    k = 1
    while k <= k_max && any(I_unprocessed)
        i_max_unprocessed = argmax(res[I_unprocessed])
        i_min_unprocessed = argmin(res[I_unprocessed])

        I_all_unprocessed = I_all[I_unprocessed]
        i_max = I_all_unprocessed[i_max_unprocessed]
        i_min = I_all_unprocessed[i_min_unprocessed]
        if i_max == i_min
            return I_extrema, support_sets
        end

        push!(I_extrema, i_min)
        push!(I_extrema, i_max)

        support_set_min, I_terminal_min = get_support_set(X, res, i_min, A, D, Minimum(), is_monotonic=is_monotonic, start_gap=start_gap)
        support_set_max, I_terminal_max = get_support_set(X, res, i_max, A, D, Maximum(), is_monotonic=is_monotonic, start_gap=start_gap)
        push!(support_sets, support_set_min)
        push!(support_sets, support_set_max)

        push!(I_terminal_all, I_terminal_min)
        push!(I_terminal_all, I_terminal_max)

        push!(extrema_types, Minimum())
        push!(extrema_types, Maximum())

        # Process terminal points
        I_terminal_orig = I_unprocessed[I_terminal_min]
        I_unprocessed .= I_unprocessed .& map(!,support_set_min)
        if length(I_terminal_min) > 0
            I_unprocessed[I_terminal_min] = I_terminal_orig
        end

        I_terminal_orig = I_unprocessed[I_terminal_max]
        I_unprocessed .= I_unprocessed .& map(!, support_set_max)
        if length(I_terminal_max) > 0
            I_unprocessed[I_terminal_max] = I_terminal_orig
        end

        k += 1
    end

    return I_extrema, support_sets, I_terminal_all, extrema_types
end

function get_best_extrema(X, res::Vector{T_y}, I_extrema::Vector{Int64}, support_sets, I_terminal_all::Vector{Vector{Int64}}, extrema_types::Vector{<:Extremum}, D::AbstractMatrix, score_func) where {T_y<:Number}
    n = length(res)
    scores = zeros(n)

    k_best = 1
    max_score = score_func(X, res, I_extrema[k_best], support_sets[k_best], I_terminal_all[k_best], extrema_types[k_best], D)
    for k in 2:length(I_extrema)
        score_k = score_func(X, res, I_extrema[k], support_sets[k], I_terminal_all[k], extrema_types[k], D)
        if score_k > max_score
            max_score = score_k
            k_best = k
        end
    end

    return I_extrema[k_best], support_sets[k_best], I_terminal_all[k_best], extrema_types[k_best], max_score
end

function get_RBFN_vandermonde(X::Union{Vector{T_x}, Matrix{T_x}}, Theta::Matrix{T_x}, T_phi::Type{T_BF}) where {T_x<:Real, T_BF<:BasisFunction}
# Get the Vandermonde matrix of the basis functions
    V = zeros(T_x, size(X,1), size(Theta,1)+1)
    for i in axes(X,1)
        V[i,1] = one(T_x)
        for j in axes(Theta,1)
            V[i,j+1] = eval_phi(X[i,:], Theta[j,:], T_phi)
        end
    end

    return V
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
function train_RBFN(X::Vector{T_x}, y::Vector{T_y};
        N_max=DEFAULT_N_1D::Integer,
        T_phi=Gaussian{Isotropic, T_x, 1}::Type{<:BasisFunction},
        solver=lsq_solver::Function,
        score_extrema=rel_supr::Function,
        get_initial_guess=max_dist_theta0::Function,
        conv_conditions=bound_diff::Function,
        conv_thresholds=[DEFAULT_MAGN_REDUCTION]::Vector{<:Real},
        conv_enforcement=all::Function,
        is_monotonic=DEFAULT_MONOTONICITY,
        start_gap=DEFAULT_START_GAP::Real,
        k_extrema=DEFAULT_K_EXTREMA::Integer,
        duplicate_tol=MACHINE_EPS_FACTOR*eps(eltype(X)),
        X_validation=T_x[]::Vector{T_x},
        y_validation=T_y[]::Vector{T_y},
        print_iter=false::Bool,
        redistribute_wts_final=true::Bool
    ) where {T_x<:AbstractFloat, T_y<:Number}

    if length(X) != length(y)
        throw("SLFA.train_RBFN: Number of data points and residual values do not match.")
    end

    # Compute neighbors
    A, D = get_nbr_matrix1D(X, duplicate_tol=duplicate_tol)

    # Set up empty arrays
    num_params_RBF = size(T_phi)
    num_params = num_params_RBF + 2
    Theta0  = zeros(T_x, N_max, num_params)
    Theta  = zeros(T_x, N_max, num_params)

    # Train the network
    res = copy(y)
    res_validation = copy(y_validation)
    N = 0

    res_error = conv_conditions(res, res_validation, [], N)
    res_history = [res_error]
    while N < N_max && conv_enforcement(res_error .> conv_thresholds)
        N += 1
        if print_iter
            println("Iteration N = $N")
        end
        # Find the first 2*k extrema
        I_extrema, support_sets, I_terminal_all, extrema_types = get_2k_extrema(X, res, A, D, k_max=k_extrema, is_monotonic=is_monotonic, start_gap=start_gap)

        # Choose the highest scoring extrema
        i_extrema, support_set, I_terminal, extremum_type, _ = get_best_extrema(X, res, I_extrema, support_sets, I_terminal_all, extrema_types, D, score_extrema)

        # Construct an initial guess on the chosen support set
        theta0 = get_initial_guess(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, T_phi)
        Theta0[N,:] .= theta0

        # Solve for the RBF
        theta = solver(theta0, X, res, A, D, N, T_phi)
        Theta[N, :] .= theta

        # Update the residuals and residual history
        res .= res - theta[num_params-1]*map(x->eval_phi(x, theta, T_phi), X) .- theta[num_params]
        res_validation .= res_validation - theta[num_params-1]*map(x->eval_phi(x, theta, T_phi), X_validation) .- theta[num_params]

        res_error = conv_conditions(res, res_validation, res_history, N)
        push!(res_history, res_error)
    end

    Theta = Theta[1:N,:]
    Theta0 = Theta0[1:N,:]
    if redistribute_wts_final
        V = get_RBFN_vandermonde(X, Theta, T_phi)
        a_new = V \ y

        Theta[:, end-1] = a_new[2:end]
        a0_new = a_new[1]
        a0 = sum(Theta[:,end])
        da0 = a0_new - a0
        Theta[:,end] .+= da0 / size(Theta,1)
    end

    return Theta, res_history, res, res_validation, A, D, N, T_phi, Theta0
end

# Quasi-1D
function train_RBFN_quasi1D(X_all::Vector{Matrix{T_x}}, y_all::Vector{Vector{T_y}}; 
        N_max=DEFAULT_N_1D::Integer,
        T_phi=Gaussian{Anisotropic{Aligned}, T_x, size(X_all[1],2)}::Type{<:BasisFunction},
        solver=lsq_solver::Function,
        score_extrema=rel_supr::Function,
        get_initial_guess=max_dist_theta0::Function,
        conv_conditions=bound_diff::Function,
        conv_thresholds=[DEFAULT_MAGN_REDUCTION]::Vector{<:Real},
        conv_enforcement=all::Function,
        is_monotonic=DEFAULT_MONOTONICITY,
        start_gap=DEFAULT_START_GAP::Real,
        k_extrema=DEFAULT_K_EXTREMA::Integer,
        duplicate_tol=MACHINE_EPS_FACTOR*eps(eltype(X_all[1])),
        X_validation=T_x[]::Vector{T_x},
        y_validation=T_y[]::Vector{T_y},
        print_iter=false::Bool,
        redistribute_wts_final=true::Bool
    ) where {T_x<:AbstractFloat, T_y<:Number}

    # Check inputs
    if length(X_all) != length(y_all)
        throw("SLFA.train_RBFN: Length of X_all does not match length of y_all")
    end

    for k in eachindex(X_all)
        if size(X_all[k],1) != length(y_all[k])
            throw("SLFA.train_RBFN: Number of data points and residual values do not match.")
        end
    end

    res_flat = vcat(y_all...)
    X_flat = vcat(X_all...)
    
    # Compute neighbors for each dataset
    A_all = SparseMatrixCSC{Bool, Int64}[]
    D_all = SparseMatrixCSC{T_x, Int64}[]
    for r in eachindex(X_all)
        A_r, D_r = get_nbr_matrix1D(X_all[r], duplicate_tol=duplicate_tol)
        push!(A_all, A_r)
        push!(D_all, D_r)
    end
    
    # TODO: also compute the nD A/D matrices once the nD routines are implemented

    # Set up empty arrays
    num_params_RBF = size(T_phi)
    num_params = num_params_RBF + 2
    Theta0  = zeros(T_x, N_max, num_params)
    Theta  = zeros(T_x, N_max, num_params)

    # Train the network
    res_all = [ copy(y_all[r]) for r in eachindex(y_all) ]
    res_validation = copy(y_validation)
    N = 0

    res_error = conv_conditions(res_flat, res_validation, [], N)
    res_history = [res_error]
    while N < N_max && conv_enforcement(res_error .> conv_thresholds)
        N += 1
        if print_iter
            println("Iteration N = $N")
        end

        # Find the best extrema
        max_score = -Inf
        r_best = -1
        i_extrema_best = -1
        support_set_best = Bool[]
        I_terminal_best = Int64[]
        extremum_type_best = Maximum()
        for r in eachindex(res_all)
            # Find the first 2*k_extrema extrema for each run
            I_extrema_r, support_sets_r, I_terminal_r_all, extrema_types_r = get_2k_extrema(X_all[r], res_all[r], A_all[r], D_all[r], k_max=k_extrema, is_monotonic=is_monotonic, start_gap=start_gap)

            # Choose the highest scoring extrema in this run
            i_extrema_r, support_set_r, I_terminal_r, extremum_type_r, score_r = get_best_extrema(X_all[r], res_all[r], I_extrema_r, support_sets_r, I_terminal_r_all, extrema_types_r, D_all[r], score_extrema)
            
            if score_r > max_score
                r_best = r
                i_extrema_best = i_extrema_r
                support_set_best = support_set_r
                I_terminal_best = I_terminal_r
                extremum_type_best = extremum_type_r
                max_score = score_r
            end
        end

        # Construct an initial guess on the chosen support set
        # TODO: change to the nD get_initial_guess once it is implemented
        theta0 = get_initial_guess(X_all[r_best], res_all[r_best], A_all[r_best], D_all[r_best], i_extrema_best, support_set_best, I_terminal_best, extremum_type_best, T_phi)
        Theta0[N,:] .= theta0

        # Solve for the RBF
        theta = solver(theta0, X_flat, res_flat, A_all, D_all, N, T_phi)
        Theta[N, :] .= theta

        # Update the residuals and residual history
        for r in eachindex(X_all)
            res_all[r] .= res_all[r] - theta[num_params-1]*[ eval_phi(X_all[r][i,:], theta, T_phi) for i in axes(X_all[r],1) ] .- theta[num_params]
        end
        res_flat .= vcat(res_all...)

        if length(res_validation) > 0
            res_validation .= res_validation - theta[num_params-1]*[ eval_phi(X_validation[i,:], theta, T_phi) for i in axes(X_validation, 1) ] .- theta[num_params]
        end

        res_error = conv_conditions(res_flat, res_validation, res_history, N)
        push!(res_history, res_error)
    end

    
    Theta = Theta[1:N,:]
    Theta0 = Theta0[1:N,:]
    if redistribute_wts_final
        y_flat = vcat(y_all...)
        V = get_RBFN_vandermonde(X_flat, Theta, T_phi)
        a_new = V \ y_flat

        Theta[:, end-1] = a_new[2:end]
        a0_new = a_new[1]
        a0 = sum(Theta[:,end])
        da0 = a0_new - a0
        Theta[:,end] .+= da0 / size(Theta,1)
    end

    return Theta, res_history, res_all, res_validation, A_all, D_all, N, T_phi, Theta0
end

# # nD
# function train_RBFN(X::Matrix{T_x}, y::Vector{T_y}; ) where {T_x<:AbstractFloat, T_y<:Number}
#     if size(X,1) != length(y)
#         throw("SLFA.train_RBFN: Number of data points and residual values do not match.")
#     end
# end

