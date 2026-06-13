
using Pkg
pkg"activate ."

using Plots

using SLFA

dx = 0.01
X = [0:dx:10...]
y = 0.8*exp.(-0.2*X) .* sin.(10*X)


Theta, res_history, A, D, T_phi = train_RBFN(X, y, N_max=10)
rbfn = RBFN(Theta, T_phi)

res = copy(y)

for N in 1:10
    global res
    y_phi = Theta[N, end-1] .* [eval_phi(X[i], Theta[N,:], T_phi) for i in eachindex(X)] .+ Theta[N, end]
    p = plot(X, res, label="Residual, N=$N")
    plot!(X, y_phi, label="RBF, N=$N")
    display(p)
    res = res - y_phi
end

y_rbfn = rbfn.(X)

plot(X,y)
plot!(X, y_rbfn)