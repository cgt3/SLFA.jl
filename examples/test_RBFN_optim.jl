
using Pkg
pkg"activate ."

using LinearAlgebra
using Measures
using Plots
using Optim

using SLFA



error_threshold = [0.0, 0.0, 0.0]

dx = 0.01
X = [0:dx:10...] 
y = 0.8*exp.(-0.2*X) .* sin.(10*X)


# Set common arguments
N_max = 200
print_iter=false
monotonicity=Strict()

# Try using just the initial_guess as the final RBF
Theta_IG, res_history_IG, _, _, _, _, _ = train_RBFN(X, y, N_max=N_max, solver=initial_guess, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);

# Try using LSQ with the LSQ solver
Theta_LSQ, res_history_LSQ, _, _, _, _, _ = train_RBFN(X, y, N_max=N_max, solver=lsq_solver, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);


# Try lsq + TV penalty for various omega
omega_all = [0.0, 1.0, 10.0, 100.0]
res_hist_all = []

for omega in omega_all
    println("Starting omega=$omega...")
    solver_omega(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    Theta, res_history, A, D, N_final, T_phi, Theta0 = train_RBFN(X, y, N_max=N_max, solver=solver_omega, conv_conditions=res_error, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity);

    push!(res_hist_all, copy(res_history))
end

## Plot convergence histories

# Plot the residual histories
N_all = 0:N_max
lw = 6

# Plot the convergence history for the initial_guess solver
fig = plot(N_all, getindex.(res_history_IG, 1), 
    label="Initial Guess",
    title="RMSE Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:bottomleft,
    tickfont=18,
    titlefont=32,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=18,
    labelfontsize=24,
    layout=(1,2), 
    subplot=1,
    ylim=(1e-4, 1.0),
    bottom_margin=10mm
)
plot!(fig, getindex.(res_history_IG, 2), 
    label="Initial Guess",
    title="Max Magnitude Error", 
    linewidth=lw,
    size=(1500, 1000),
    yaxis=:log10, 
    legend=:bottomleft,
    tickfont=18,
    titlefont=32,
    xlabel="N",
    grid=true,
    minorgrid=true,
    gridalpha=0.5,
    minorgridalpha=0.15,
    legendfontsize=18,
    labelfontsize=24,
    subplot=2,
    ylim=(1e-4, 1.0),
    bottom_margin=10mm
)

# Plot the convergence history for LSQ (Levenberg-Marquardt)
plot!(fig, N_all, getindex.(res_history_LSQ, 1), linewidth=lw, label="LSQ (LM)", subplot=1)
plot!(fig, N_all, getindex.(res_history_LSQ, 2), linewidth=lw, label="LSQ (LM)", subplot=2)

# Plot the convergence histories for the LSQ-TV runs
for i = 1:length(omega_all)
    plot!(fig, N_all, getindex.(res_hist_all[i], 1), linewidth=lw, label="omega=$(omega_all[i]) (NM)", subplot=1)
    plot!(fig, N_all, getindex.(res_hist_all[i], 2), linewidth=lw, label="omega=$(omega_all[i]) (NM)", subplot=2)
end
display(fig)

png("./figures/convHist_initialGuess.png")

