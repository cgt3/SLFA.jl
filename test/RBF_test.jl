using SLFA

#Tests 1-5
@testset  "Gaussian{Isotropic, T_x, 1} Inner Constructor" begin
    x0 = 1.0;
    w = 2.0;

    rbf = Gaussian{Isotropic,typeof(x0),1}(x0,w)
    @test rbf.x0 == x0
    @test rbf.w == w
end