
using Pkg
pkg"activate ."

using LinearAlgebra
using Plots
using Optim

using SLFA

dx = 0.01
X = [0:dx:10...]
y = 0.8*exp.(-0.2*X) .* sin.(10*X)

omega_TV = 0.0
function lsq_TV_solver(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
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


Theta, res_history, A, D, N_final, T_phi = train_RBFN(X, y, N_max=100, solver=lsq_TV_solver)
rbfn = RBFN(Theta, T_phi)

# res = copy(y)
# for N in 1:10
#     global res
#     y_phi = Theta[N, end-1] .* [eval_phi(X[i], Theta[N,:], T_phi) for i in eachindex(X)] .+ Theta[N, end]
#     p = plot(X, res, label="Residual, N=$N")
#     plot!(X, y_phi, label="RBF, N=$N")
#     display(p)
#     res = res - y_phi
# end

y_rbfn = rbfn.(X)

if omega_TV == 0
    fig1 = plot(X,y)
end
plot!(fig1, X, y_rbfn, label="omega=$omega_TV")
display(fig1)

if omega_TV == 0
    fig2 = plot(0:N_final, res_history, yaxis=:log10, label="omega=$omega_TV")
    display(fig2)
else
    plot!(fig2, 0:N_final, res_history, yaxis=:log10, label="omega=$omega_TV")
    display(fig2)
end