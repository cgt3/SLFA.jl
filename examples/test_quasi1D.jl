
using Pkg
pkg"activate ."

using LinearAlgebra
using Measures
using Plots
using Optim

using SLFA

error_threshold = [0.0, 0.0, 0.0]

# Generate the true data
dx = 0.01
xLB = 0.0
xUB = 10.0
x_true = [xLB:dx:xUB...]
y_true = sin.(2*pi*x_true)
n_orig = length(y_true)

# Generate the flow data
num_delays = 2
X = [y_true[1:n_orig-2] y_true[2:n_orig-1]]
x_plot = x_true[3:end]
y = y_true[3:end]

X_all = [X]
y_all = [y]
error_threshold = [0.0, 0.0, 0.0]

# Set common arguments
N_max = 25
start_gap = 0.0
print_iter=false
monotonicity=Strict()
redistribute_wts_final = false
score_extrema = rel_supr


# Try using just the initial_guess as the final RBF
# omega= 0.0
# solver_omega(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})

Theta_IG, res_history_IG, _, _, _, _, N_IG, T_phi, _ = train_RBFN_quasi1D(X_all, y_all, N_max=N_max, solver=initial_guess, start_gap=start_gap, conv_conditions=res_error, conv_thresholds=error_threshold, score_extrema=score_extrema, print_iter=print_iter, is_monotonic=monotonicity, redistribute_wts_final=redistribute_wts_final);
rbfn_IG = RBFN(Theta_IG, T_phi)
y_IG = [ rbfn_IG(X[i,:]) for i in eachindex(y) ]


Theta_LSQ, res_history_LSQ, _, _, _, _, N_LSQ, T_phi, _ = train_RBFN_quasi1D(X_all, y_all, N_max=N_max, solver=lsq_solver, start_gap=start_gap, conv_conditions=res_error, conv_thresholds=error_threshold, score_extrema=score_extrema, print_iter=print_iter, is_monotonic=monotonicity, redistribute_wts_final=redistribute_wts_final);
rbfn_LSQ = RBFN(Theta_LSQ, T_phi)
y_LSQ = [ rbfn_LSQ(X[i,:]) for i in eachindex(y) ]


y_rbfn = y_IG
rbfn_label = "Initial Guess, N=$N_IG"
file_name = "./figures/quasi1D_sine_clean_IG.png"

y_rbfn = y_LSQ
rbfn_label = "LSQ, N=$N_LSQ"
file_name = "./figures/quasi1D_sine_clean_LSQ.png"

# Plot the solution
lw = 5
ylim = (-1.5, 1.75)
fig1 = plot(y_rbfn, label=rbfn_label, 
    linewidth=lw,
    size=(1800, 800),
    xlabel="i, Data Sample Index", 
    ylabel="y", 
    ylim=ylim,
    leg=:topright,
    gridalpha=0.2,
    legendfontsize=20,
    labelfontsize=24,
    left_margin=10mm,
    right_margin=10mm,
    bottom_margin=15mm,
    tickfont=18,
    color=:blue
)
plot!(y, label="True Solution", linewidth=2, color=:red)
# plot!(res[1], label="Final Residual", linewidth=2, color=:black)
display(fig1)

png(file_name)

# Plot convergence history

# Plot the residual histories
lw = 6
ylim_RMSE = (1e-3, 1.0)
ylim_supr = (1e-3, 1.0)

# Plot the convergence history for the initial_guess solver
fig2 = plot(getindex.(res_history_IG, 1), 
    label="Initial Guess",
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
plot!(fig2, getindex.(res_history_IG, 2), 
    label="Initial Guess",
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
plot!(fig2, getindex.(res_history_LSQ, 1), linewidth=lw, label="LSQ (LM)", subplot=1)
plot!(fig2, getindex.(res_history_LSQ, 2), linewidth=lw, label="LSQ (LM)", subplot=2)
display(fig2)

png("./figures/quasi1D_convHist_sine_clean.png")



# ylim = (-1.25, 1.25)
# res = copy(y)
# for N in 1:10
#     global res
#     y_phi = Theta[N, end-1] .* [eval_phi(X[i,:], Theta[N,:], T_phi) for i in axes(X,1)] .+ Theta[N, end]
#     p = plot(res, size=(1800, 700), ylim=ylim, label="Residual, N=$N", layout=(1,2), subplot=1)
#     plot!(y_phi, label="RBF (Final), N=$N", subplot=1)
    
#     plot!(res-y_phi, ylim=ylim, leg=false, subplot=2)
#     display(p)
#     res = res - y_phi
# end