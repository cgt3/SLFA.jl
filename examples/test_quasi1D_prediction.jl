
using Pkg
pkg"activate ."

using ColorSchemes
using LinearAlgebra
using Measures
using Plots
using Optim

using SLFA

error_threshold = [0.0, 0.0, 0.0]

# Generate the true data
dx = 0.01
xLB = 0.0
xUB = 1.0
x_true = [xLB:dx:xUB...]
y_true = sin.(2*pi*x_true)
n_orig = length(y_true)

# Generate the flow data
num_delays = 2
X = [y_true[1:n_orig-2] y_true[2:n_orig-1]]
y = y_true[3:end]
n = length(y)

xi(x) = [x[1] + x[2], x[1] - x[2]]
Xi = similar(X)
for i in axes(Xi,1)
    Xi[i,:] = xi(X[i,:])
end
X = Xi

X_all = [X]
y_all = [y]
num_runs = 3
noise_frac = 0.01
for r in 1:num_runs
    X_perturbed = copy(X)
    for j in axes(X,2)
        noise_mag = noise_frac*(maximum(X[:,j]) - minimum(X[:,j]))
        X_perturbed[:,j] .+= noise_mag * randn(n)
    end
    push!(X_all, X_perturbed)
    push!(y_all, y)
end

# Set common arguments
N_max = 20
start_gap = 0.01
print_iter=false
monotonicity=Strict()
redistribute_wts_final = false
score_extrema = rel_supr

omega=0.0
solver_omega(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}) = lsq_TV_solver(omega, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction})

Theta, res_history, _, _, A_all, D_all, N_LSQ, T_phi, Theta0 = train_RBFN_quasi1D(X_all, y_all, N_max=N_max, solver=lsq_solver, start_gap=start_gap, conv_conditions=res_error, conv_thresholds=error_threshold, score_extrema=score_extrema, print_iter=print_iter, is_monotonic=monotonicity, redistribute_wts_final=redistribute_wts_final);
rbfn = RBFN(Theta, T_phi)
y_LSQ = [ rbfn(X[i,:]) for i in eachindex(y) ]


y_rbfn = y_LSQ
rbfn_label = "LSQ, N=$N_LSQ"
file_name = "./figures/quasi1D_sine_prediction_LSQ.png"

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
ylim_RMSE = (1e-4, 1.0)
ylim_supr = (1e-4, 1.0)

# Plot the convergence history for the initial_guess solver
fig2 = plot(getindex.(res_history, 1), 
    label="LSQ",
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
plot!(fig2, getindex.(res_history, 2), 
    label="LSQ",
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
display(fig2)

png("./figures/quasi1D_convHist_sine_prediction.png")


res_color=:balance
rbfn_color=:balance
center_color=:yellow

lw = 4
ms = 10
res_lim = (-1, 1)
zlim=extrema(y)
dx2_plot = 0.005
dx1_plot = 0.01
x1_lim=extrema(X[:,1])
x2_lim=extrema(X[:,2])
x1 = [(x1_lim[1]-5dx1_plot):dx1_plot:(x1_lim[2]+5dx1_plot)...]
x2 = [(x2_lim[1]-dx2_plot):dx2_plot:(x2_lim[2]+2dx2_plot)...]
y_rbfn_plot = zeros(length(x2), length(x1))
xlim=extrema(x1)
ylim=extrema(x2)

res = copy(y)
res_all = [ copy(y_all[r]) for r in eachindex(y_all) ]
y_phi0 = [ copy(y_all[r]) for r in eachindex(y_all) ]
y_phi = [ copy(y_all[r]) for r in eachindex(y_all) ]
for N in 1:10
    global res
    zlim = extrema(res)
    for r in eachindex(y_phi0)
        y_phi0[r] = Theta0[N, end-1] .* [eval_phi(X_all[r][i,:], Theta0[N,:], T_phi) for i in axes(X,1)] .+ Theta0[N, end]
        y_phi[r]  = Theta[N, end-1]  .* [eval_phi(X_all[r][i,:], Theta[N,:],  T_phi) for i in axes(X,1)] .+ Theta[N, end]
    end
    thetaN = Theta[N,:]
    for j in eachindex(x2), i in eachindex(x1)
        y_rbfn_plot[j,i] = thetaN[end-1]*eval_phi([x1[i], x2[j]], thetaN[:], T_phi) + thetaN[end]
    end
    fig = heatmap(x1, x2, y_rbfn_plot, color=rbfn_color, clim=zlim, leg=false, xlim=xlim, ylim=ylim, size=(1800, 1200), layout=(2,2), subplot=1)
    for r in eachindex(X_all)
        scatter!(X_all[r][:,1],X_all[r][:,2], zcolor=res, clim=zlim, markersize=ms, markerstrokewidth=0.2, title="Residual (N=$N)", color=res_color, subplot=1)
    end
    scatter!([Theta[N,1]], [Theta[N,2]], markersize=ms, color=center_color, leg=false, label="RBF Center = ($(Theta[N,1]), $(Theta[N,2]))", subplot=1)
    scatter!([Theta0[N,1]], [Theta0[N,2]], markersize=ms, color=:green, leg=false, label="RBF Center = ($(Theta[N,1]), $(Theta[N,2]))", subplot=1)


    for r in eachindex(res_all)
        plot!(res_all[r], label="Residual[$r], N=$N", subplot=3, linewidth=lw)
    end
    plot!(y_phi0[1], label="RBF (Initial), N=$N", subplot=3, linestyle=:dash, color=:orange, linewidth=lw/2)
    plot!(y_phi[1], label="RBF (Final), N=$N", subplot=3, color=:red, linewidth=lw)
    hline!([0], color=:black, subplot=3, label="")

    

    res = res - y_phi[1]
    res_all .-= y_phi
    zlim_new = extrema(res)
    scatter!(X[:,1],X[:,2], zcolor=res, #=clim=zlim_new,=# markersize=ms, markerstrokewidth=0.2, leg=false, title="Residual New", color=res_color, subplot=2)
    
    plot!(res, ylim=zlim, leg=false, subplot=4, linewidth=lw)
    hline!([0], color=:black, subplot=4, label="")
    display(fig)
end



# lw = 4
# ylim = (-1.25, 1.25)
# res = copy(y)
# for N in 1:10
#     global res
#     y_phi0 = Theta0[N, end-1] .* [eval_phi(X[i,:], Theta0[N,:], T_phi) for i in axes(X,1)] .+ Theta0[N, end]
#     y_phi  = Theta[N, end-1] .* [eval_phi(X[i,:], Theta[N,:], T_phi) for i in axes(X,1)] .+ Theta[N, end]
#     p = plot(res, size=(1800, 700), ylim=ylim, label="Residual, N=$N", layout=(1,2), subplot=1, linewidth=lw)
#     plot!(y_phi0, label="RBF (Initial), N=$N", subplot=1, linewidth=lw)
#     plot!(y_phi, label="RBF (Final), N=$N", subplot=1, linewidth=lw)
    
#     plot!(res-y_phi, ylim=ylim, leg=false, subplot=2, linewidth=lw)
#     display(p)
#     res = res - y_phi
# end



y_prediction = similar(y)
y_delay2 = y_true[1]
y_delay1 = y_true[2]

for i in eachindex(y_prediction)
    y_prediction[i] = rbfn([y_delay2, y_delay1])
    y_delay2 = y_delay1
    y_delay1 = y_prediction[i]
end

# Plot the recurrent prediction vs true solution
lw = 5
ylim = (-1.5, 1.75)
fig3 = plot(y_prediction, label="Prediction (LSQ, N=$N_LSQ)", 
    linewidth=lw,
    size=(1800, 800),
    xlabel="i, Data Sample Index", 
    ylabel="y", 
    # xlim=(0,100),
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
display(fig3)
