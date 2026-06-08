using Plots

include("../src/SLFA.jl")
using .SLFA

x0 = [1];
w = 2;

rbf_iso1 = Gaussian(x0, w)
rbf_iso1(0)

xLB = -5;
xUB = - xLB;
dx = 0.01;

x_test = xLB:dx:xUB
plot(x_test, rbf_iso1.(x_test))

x0_2D = [0, 0];
w_2D = [1, 10];
rbf_aniso = Gaussian(x0_2D, w_2D)

rbf_aniso([0,0])


rbf_iso2 = Gaussian([2], w)
network = RBFN(0, [1, 2], [rbf_iso1, rbf_iso2])

rbfs = Gaussian{Anisotropic{Aligned}, 2}[]
