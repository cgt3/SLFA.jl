
export supr, rel_supr, max_dist_theta0
export bound_diff, squaredTV, RMSE, res_error
export initial_guess, lsq_solver, lsq_TV_solver

# Ranking functions
function supr(X, res, i_extrema, support_set, I_terminal, ::Extremum, D)
    return res[i_extrema] - norm(res[support_set], Inf)
end

function rel_supr(X, res, i_extrema, support_set, I_terminal, ::Maximum, D; tol=MACHINE_EPS_FACTOR*eps(eltype(res)))
    res_max = maximum(res)
    if res_max != res[i_extrema]
        return 0.0;
    end

    res_extrema = res[i_extrema]
    exterior_pts = map(!,support_set)
    if sum(exterior_pts) != 0
        max_next = maximum(res[exterior_pts])
    else
        max_next = minimum(res)
    end

    if abs(res_extrema - max_next) < tol
        return abs(res_extrema)
    end

    return res_extrema - max_next
end

function rel_supr(X, res, i_extrema, support_set, I_terminal, ::Minimum, D)
    return rel_supr(X, -res, i_extrema, support_set, I_terminal, Maximum(), D)
end


# Initial guess generators
function max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type::Extremum, ::Type{Gaussian{Isotropic, T_x, dim}}; tol=MACHINE_EPS_FACTOR*eps(eltype(res))) where {T_x<:Real, dim}
    diff = X[I_terminal, :] .- X[i_extrema,:]
    
    max_dist = abs(maximum(diff))
    min_dist = abs(minimum(diff))

    if max_dist < 1e-14 && min_dist < 1e-14
        max_dist = 0.5*minimum(D[A])
    end

    w0 = 2.5 ./ max(max_dist, min_dist)
    c0 = X[i_extrema,:]
    b0 = zero(eltype(res))
    if abs(res[i_extrema]) < tol
        if length(support_set) != sum(support_set)
            b0 = sum(res[map(!,support_set)]) / (length(support_set) - sum(support_set))
        else
            if extremum_type isa Maximum
                b0 = minimum(res)
            else
                b0 = maximum(res)
            end
        end
    end
    a0 = res[i_extrema] - b0

    return [c0; w0; a0; b0]
end

function max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type::Extremum, ::Type{Gaussian{Anisotropic{Aligned}, T_x, dim}}; tol=MACHINE_EPS_FACTOR*eps(eltype(res))) where {T_x<:Real, dim}
    diff = X[I_terminal, :] .- X[i_extrema,:]
    
    max_dist = abs.( maximum.( [ diff[:,i] for i in axes(diff, 2)] ) )
    min_dist = abs.( minimum.( [ diff[:,i] for i in axes(diff, 2)] ) )

    max_dist[max_dist .< 1e-14] .= 0.5*minimum(D[A])

    w0 = 2.5 ./ max.(max_dist, min_dist)
    c0 = X[i_extrema,:]
    b0 = zero(eltype(res))
    if abs(res[i_extrema]) < tol
        if length(support_set) != sum(support_set)
            b0 = sum(res[map(!,support_set)]) / (length(support_set) - sum(support_set))
        else
            if extremum_type isa Maximum
                b0 = minimum(res)
            else
                b0 = maximum(res)
            end
        end
    end
    a0 = res[i_extrema] - b0

    return [c0; w0; a0; b0]
end



# Error measures
bound_diff(res, res_validation, res_history, N) = maximum(res) - minimum(res);

function squaredTV(res, A, D)
    f_TV = zero(eltype(res))

    I_nz, J_nz, _ = findnz(A)
    for k in eachindex(I_nz)
        i = I_nz[k]
        j = J_nz[k]
        TV_ij = ( res[i] - res[j] ) / D[i,j]
        f_TV += TV_ij * TV_ij
    end

    return 0.5*f_TV
end

function RMSE(res)
    return norm(res) / sqrt(length(res))
end

function res_error(res, res_validation, res_history, N)
    return [RMSE(res), norm(res, Inf), bound_diff(res, res_validation, res_history, N)]
end


# Solvers
function initial_guess(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    return theta0
end

function lsq_solver(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    rbf(X, theta) = theta[end-1] .* [eval_phi(X[i,:], theta, T_phi) for i in axes(X,1)] .+ theta[end]
    solver_results = curve_fit(rbf, X, res, theta0)
    
    theta = coef(solver_results)
    return theta
end


function lsq_TV_solver(omega_TV, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    res_new(theta) = res - theta[end-1] .* [eval_phi(X[i,:], theta, T_phi) for i in axes(X,1)] .- theta[end]
    f_lsq_orig = norm(res_new(theta0))
    TV_orig = squaredTV(res_new(theta0), A, D)

    # Normalize/scale each term by their original value
    f_lsq(theta) = norm(res_new(theta)) / f_lsq_orig
    f_TV(theta) = squaredTV(res_new(theta), A, D) / TV_orig

    # Try constant omega
    omega = [1, omega_TV]
    f_obj(theta) = omega[1]*f_lsq(theta) + omega[2]*f_TV(theta) 

    Optim.Options(x_abstol=1e-4, f_abstol=1e-4, iterations=200*length(theta0))
    result = optimize(f_obj, theta0, NelderMead())
    
    theta = Optim.minimizer(result)

    lsq_initial = f_lsq(theta0)
    lsq_final = f_lsq(theta)
    if lsq_final > lsq_initial
        theta_lsq = lsq_solver(theta0, X, res, A, D, N, T_phi)
        if squaredTV(res_new(theta_lsq), A, D) < TV_orig
            return theta_lsq
        else
            return theta0
        end
    end

    return theta
end
