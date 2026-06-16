export bound_diff, squaredTV, RMSE, res_error
export lsq_solver, lsq_TV_solver

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
function lsq_solver(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    rbf(X, theta) = theta[end-1] .* [eval_phi(X[i,:], theta, T_phi) for i in axes(X,1)] .+ theta[end]
    solver_results = curve_fit(rbf, X, res, theta0)
    
    theta = coef(solver_results)
    return theta
end


function lsq_TV_solver(omega_TV, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    # return theta0
    # TODO: what about normalization relative to the original residual? (as opposed to the initial guess)

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

