using SLFA

#Tests 1-5
@testset  "Gaussian{Isotropic, T_x, 1}: 1D Gaussian RBF Inner Contstructor" begin
    x0 = 1.0;
    w = 2.0;

    rbf = Gaussian{Isotropic,typeof(x0),1}(x0,w)
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0
end

@testset "Gaussian{Isotropic, T_x, dim}: ND/1D Gaussian RBF Inner Constructor" begin
    x0 = [1.0];
    w = 2.0;
    try
        rbf = Gaussian{Isotropic, typeof(x0[1]), 1.5}(x0, w)
        @test false #Fails to catch the non-integer dim
    catch e
        @test e == "SLFA.Gaussian: dim must be an integer."
    end
    
    try
        rbf = Gaussian{Isotropic, typeof(x0[1]), 2}(x0, w)
        @test false #Fails to catch the non-matching dim
    catch e
        @test e == "SLFA.Gaussian: dim does not match length(x0)."
    end
    
    try
        x_empty = Float64[]
        rbf = Gaussian{Isotropic, typeof(x0[1]), 0}(x_empty, w)
        @test false #Fails to catch the empty x0
    catch e
        @test e == "SLFA.Gaussian: x0 = []"
    end
    
    rbf = Gaussian{Isotropic, typeof(x0[1]), length(x0)}(x0, w)
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0

    rbf = Gaussian{Isotropic, typeof(x0[1]), 2}([x0[1], x0[1]], w)
    @test rbf.x0 == [1.0, 1.0]
    @test rbf.w == 2.0
end

@testset "Gaussian{Anisotropic{Aligned}, T_x, dim}: Anisotropic RBF Inner Constructor" begin
    x0 = [1.0, 1.0];
    w = [2.0, 2.0];
    try
        rbf = Gaussian{Anisotropic{Aligned}, typeof(x0[1]), 1.5}(x0, w)
        @test false #Fails to catch the non-integer dim
    catch e
        @test e == "SLFA.Gaussian: dim must be an integer."
    end
    
    try
        rbf = Gaussian{Anisotropic{Aligned}, typeof(x0[1]), 3}(x0, w)
        @test false #Fails to catch the non-matching dim
    catch e
        @test e == "SLFA.Gaussian: dim does not match length(x0)."
    end
    try
        x_empty = Float64[]
        rbf = Gaussian{Anisotropic{Aligned}, typeof(x0[1]), 0}(x_empty, w)
        @test false #Fails to catch the empty x0
    catch e
        @test e == "SLFA.Gaussian: x0 = []"
    end
    try
        rbf = Gaussian{Anisotropic{Aligned}, typeof(x0[1]), 1}([x0[1]], w)
        @test false #Fails to catch that anisotropic shouldnt be 1D
    catch e
        @test e == "SLFA.Gaussian: Anisotropic constructor should not be used for 1D Gaussians."
    end
    try
        rbf = Gaussian{Anisotropic{Aligned}, typeof(x0[1]), 2}(x0, [w[1]])
        @test false #Fails to catch that x0 and w dimensions dont match
    catch e
        @test e == "SLFA.Gaussian: length of x0 ($(length(x0))) does not match length of w ($(length([w[1]])))."
    end
    
    rbf = Gaussian{Anisotropic{Aligned}, typeof(x0[1]),length(x0)}(x0,w)
    @test rbf.x0 == [1.0, 1.0]
    @test rbf.w == [2.0,2.0]
end

@testset "Gaussian(x0::T_x, w::T_x): 1D RBF Outer Constructor" begin
    x0 = 1.0;
    w = 2.0;
    rbf = Gaussian(x0, w)
    @test rbf isa Gaussian{Isotropic, typeof(x0[1]), 1} 
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0
end