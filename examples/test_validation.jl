
using Pkg
pkg"activate ."

using LinearAlgebra
using Measures
using Plots
using Optim

using SLFA

function res_w_validation(res, res_validation, res_history, N)
    if length(res_history) > 0
        RMSE_validation = RMSE(res_validation)
        supr_validation = norm(res_validation, Inf)
        return [RMSE(res), norm(res, Inf), bound_diff(res, res_validation, res_history, N), RMSE_validation, supr_validation, res_history[end][3] - RMSE_validation]
    else
        return [RMSE(res), norm(res, Inf), bound_diff(res, res_validation, res_history, N), RMSE(res_validation), norm(res_validation, Inf), 1.0]

    end

end

error_threshold = [0.0, 0.12, 0.08, 0.0, 0.0, -1.0]

dx = 0.01
xLB = 0.0
xUB = 10.0
X = [xLB:dx:xUB...]
n = length(X)

y_true =  0.8*exp.(-0.2*X) .* sin.(10*X)
noise_frac = 0.05
noise_mag = maximum(y_true)*noise_frac
noise = noise_mag*randn(n)
y = y_true .+ noise

# Set the training and validation datasets
validation_frac = 0.1
n_validation = floor(Int64, n * validation_frac)
I_rand = rand(1:n, n_validation)

I_validation = zeros(Bool, n)
I_validation[I_rand] .= true

I_train = ones(Bool, n)
I_train[I_rand] .= false

X_train = X[I_train]
y_train = y[I_train]

X_validation = X[I_validation]
y_validation = y[I_validation]


# Set common arguments
N_max = 50
start_gap = 0.1
print_iter=false
monotonicity=Strict()
redistribute_wts_final = true

# Try using just the initial_guess as the final RBF
Theta_IG, res_history_IG, res_IG, res_val_IG, A, D, N_IG, T_phi, _ = train_RBFN(X_train, y_train, N_max=N_max, solver=initial_guess, start_gap=start_gap, conv_conditions=res_w_validation, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, X_validation=X_validation, y_validation=y_validation, redistribute_wts_final=redistribute_wts_final);
rbfn_IG = RBFN(Theta_IG, T_phi)
y_IG = rbfn_IG.(X)

# Try using LSQ with the LSQ solver
Theta_LSQ, res_history_LSQ, res_LSQ, res_val_LSQ, _, _, N_LSQ, _, _ = train_RBFN(X_train, y_train, N_max=N_max, solver=lsq_solver, start_gap=start_gap, conv_conditions=res_w_validation, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, X_validation=X_validation, y_validation=y_validation, redistribute_wts_final=redistribute_wts_final);
rbfn_LSQ = RBFN(Theta_LSQ, T_phi)
y_LSQ = rbfn_LSQ.(X)

# Try lsq + TV penalty for various omega
omega_all = [0.0, 1.0, 10.0, 100.0]
res_hist_all = []
rbfns = RBFN[]

for omega in omega_all
    println("Starting omega=$omega...")
    solver_omega(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})
    Theta, res_history, _, _, _, _, _, _, _ = train_RBFN(X_train, y_train, N_max=N_max, solver=solver_omega, start_gap=start_gap, conv_conditions=res_w_validation, conv_thresholds=error_threshold, print_iter=print_iter, is_monotonic=monotonicity, X_validation=X_validation, y_validation=y_validation, redistribute_wts_final=redistribute_wts_final);

    push!(res_hist_all, copy(res_history))
    push!(rbfns, RBFN(Theta, T_phi))
end

## Plotting

# y_plot = y_IG
# rbfn_label="RBFN (IG, N=$(N_IG))"
# figure_name = "./figures/noisy_SinE_IG_trainBounds0.08_redistWts.png"

y_plot = y_LSQ
rbfn_label="RBFN (LSQ, N=$(N_LSQ))"
figure_name = "./figures/noisy_SinE_LSQ_trainBounds0.08_redistWts.png"

k = 2
y_plot = rbfns[k].(X)
rbfn_label="RBFN (omega=$(omega_all[k]), N=$(length(res_hist_all[k])-1))"
figure_name = "./figures/noisy_SinE_omega$(omega_all[k])_trainRMSE0.08_redistWts.png"

# Plot the solution  with and without noise
plot(X, y, label="Noisy Solution", 
    size=(1800, 800),
    layout=(1,2),
    subplot=1,
    xlabel="x", 
    ylabel="y", 
    ylim=(-1.0, 1.0), 
    xlim=(xLB, xUB),
    leg=:topright,
    gridalpha=0.2,
    legendfontsize=20,
    labelfontsize=24,
    left_margin=10mm,
    right_margin=10mm,
    bottom_margin=15mm,
    tickfont=18,
    linewidth=3
)
plot!(X, y_true, linewidth=3, label="True Solution", color=:red, subplot=1)
hline!([0.0], color=:black, label="", subplot=1)

plot!(X, y_plot, label=rbfn_label, 
    subplot=2,
    xlabel="x", 
    ylabel="y", 
    ylim=(-1.0, 1.0), 
    xlim=(xLB, xUB),
    leg=:topright,
    gridalpha=0.2,
    legendfontsize=20,
    labelfontsize=24,
    left_margin=10mm,
    right_margin=10mm,
    bottom_margin=15mm,
    tickfont=18,
    linewidth=5,
    color=:blue
)
plot!(X, y_true, linewidth=2, label="True Solution", subplot=2, color=:red)
hline!([0.0], color=:black, label="", subplot=2)

png(figure_name)

# Plot the residual histories
lw = 6
ylim_RMSE = (1e-2, 1.0)
ylim_supr = (1e-1, 1.0)

# Plot the convergence history for the initial_guess solver
fig = plot(getindex.(res_history_IG, 1), 
    label="Initial Guess",
    # title="Validation RMSE Error", 
    title="Training RMSE Error", 
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
    ylim=ylim_RMSE,
    bottom_margin=10mm
)
plot!(fig, getindex.(res_history_IG, 2), 
    label="Initial Guess",
    # title="Validation Max Magn. Error", 
    title="Training Max Magn. Error", 
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
    ylim=ylim_supr,
    bottom_margin=10mm
)

# Plot the convergence history for LSQ (Levenberg-Marquardt)
plot!(fig, getindex.(res_history_LSQ, 1), linewidth=lw, label="LSQ (LM)", subplot=1)
plot!(fig, getindex.(res_history_LSQ, 2), linewidth=lw, label="LSQ (LM)", subplot=2)

# Plot the convergence histories for the LSQ-TV runs
for i = 1:length(omega_all)
    plot!(fig, getindex.(res_hist_all[i], 1), linewidth=lw, label="omega=$(omega_all[i]) (NM)", subplot=1)
    plot!(fig, getindex.(res_hist_all[i], 2), linewidth=lw, label="omega=$(omega_all[i]) (NM)", subplot=2)
end
display(fig)

png("./figures/convHist_noisy_training_trainBounds0.08_sg0.1.png")


# Plot the convergence history for the initial_guess solver
fig = plot(getindex.(res_history_IG, 4), 
    label="Initial Guess",
    title="Validation RMSE Error", 
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
    ylim=ylim_RMSE,
    bottom_margin=10mm
)
plot!(fig, getindex.(res_history_IG, 5), 
    label="Initial Guess",
    title="Validation Max Magn. Error",  
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
    ylim=ylim_supr,
    bottom_margin=10mm
)

# Plot the convergence history for LSQ (Levenberg-Marquardt)
plot!(fig, getindex.(res_history_LSQ, 1), linewidth=lw, label="LSQ (LM)", subplot=1)
plot!(fig, getindex.(res_history_LSQ, 2), linewidth=lw, label="LSQ (LM)", subplot=2)

# Plot the convergence histories for the LSQ-TV runs
for i = 1:length(omega_all)
    plot!(fig, getindex.(res_hist_all[i], 1), linewidth=lw, label="omega=$(omega_all[i]) (NM)", subplot=1)
    plot!(fig, getindex.(res_hist_all[i], 2), linewidth=lw, label="omega=$(omega_all[i]) (NM)", subplot=2)
end
display(fig)

png("./figures/convHist_noisy_validation_trainBounds0.08_sg0.1.png")