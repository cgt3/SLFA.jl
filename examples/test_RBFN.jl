
using Pkg
pkg"activate ."

using LinearAlgebra
using Plots
using Optim

using SLFA

function RMSE(res)
    return norm(res) / sqrt(length(res))
end

function res_error(res, res_validation, res_history, N)
    return [RMSE(res), norm(res, Inf), bound_diff(res, res_validation, res_history, N)]
end

error_threshold = [0.0, 0.0, 0.0]

dx = 0.01
X = [0:dx:10...]
y = 0.8*exp.(-0.2*X) .* sin.(10*X)

N_max = 50

omega_TV = 10.0
function solver(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    f_lsq_orig = norm(res)
    TV_orig = squaredTV(res, A, D)

    res_new(theta) = res - theta[end-1] .* [eval_phi(X[i,:], theta, T_phi) for i in axes(X,1)] .- theta[end]

    # Normalize/scale each term by their original value
    f_lsq(theta) = norm(res_new(theta)) / f_lsq_orig
    f_TV(theta) = squaredTV(res_new(theta), A, D) / TV_orig

    # Try constant omega
    omega = [1, omega_TV]
    f_obj(theta) = omega[1]*f_lsq(theta) + omega[2]*f_TV(theta) 


    result = optimize(f_obj, theta0, BFGS())
    
    theta = Optim.minimizer(result)
    return theta
end


Theta, res_history, A, D, N_final, T_phi, Theta0 = train_RBFN(X, y, N_max=N_max, solver=solver, conv_conditions=res_error, conv_thresholds=error_threshold)
rbfn = RBFN(Theta, T_phi)

res = copy(y)
for N in 1:15
    global res
    y_phi = Theta[N, end-1] .* [eval_phi(X[i], Theta[N,:], T_phi) for i in eachindex(X)] .+ Theta[N, end]
    p = plot(X, res, label="Residual, N=$N", layout=(1,2), subplot=1)
    plot!(X, y_phi, label="RBF (Final), N=$N", subplot=1)
    # y_phi0 = Theta0[N, end-1] .* [eval_phi(X[i], Theta0[N,:], T_phi) for i in eachindex(X)] .+ Theta0[N, end]
    # plot!(X, y_phi0, label="RBF (Initial), N=$N")
    
    plot!(X, res-y_phi, leg=false, subplot=2)
    display(p)
    res = res - y_phi
end

y_rbfn = rbfn.(X)

# Plot the residual histories
if omega_TV == 0
    fig_RMSE   = plot(0:N_final, getindex.(res_history, 1), yaxis=:log10, label="omega=$omega_TV", title="RMSE Error")
    fig_inf    = plot(0:N_final, getindex.(res_history, 2), yaxis=:log10, label="omega=$omega_TV", title="Max Magnitude Error")
    fig_bounds = plot(0:N_final, getindex.(res_history, 3), yaxis=:log10, label="omega=$omega_TV", title="Bounds Error")
    display(fig_RMSE)
    display(fig_inf)
    display(fig_bounds)
else
    plot!(fig_RMSE,   0:N_final, getindex.(res_history, 1), yaxis=:log10, label="omega=$omega_TV")
    plot!(fig_inf,    0:N_final, getindex.(res_history, 2), yaxis=:log10, label="omega=$omega_TV")
    plot!(fig_bounds, 0:N_final, getindex.(res_history, 3), yaxis=:log10, label="omega=$omega_TV")
    display(fig_RMSE)
    display(fig_inf)
    display(fig_bounds)
end

png(fig_RMSE, "figures/convHist_RMSE_Optim.png")
png(fig_inf, "figures/convHist_inf_Optim.png")
png(fig_bounds, "figures/convHist_bounds_Optim.png")