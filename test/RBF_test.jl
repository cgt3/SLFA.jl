# include("../src/SLFA.jl")
using SLFA

@testset  "Evaluating a 1D Gaussian RBF" begin
    x0 = 1.0;
    w = 2.0;

    rbf_iso = Gaussian(x0, w)
    @test rbf_iso isa Gaussian{Isotropic, typeof(x0), 1}
    @test rbf_iso(x0) == 1.0
end