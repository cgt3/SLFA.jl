using Plots

include("../src/SLFA.jl")
using Main.SLFA

x0 = [1];
w = 2;

rbf_isotropic = RBF(x0, w)

rbf_isotropic(0)

xLB = -5;
xUB = - xLB;
dx = 0.01;

x_test = xLB:dx:xUB
plot(x_test, rbf_isotropic.(x_test))

x0_2D = [0, 0];
w_2D = [1, 10];
rbf_aniso = RBF(x0_2D, w_2D)

rbf_aniso([0,0])
