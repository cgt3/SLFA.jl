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
    w = [2.0, 3.0];
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
    @test rbf.w == [2.0,3.0]
end

@testset "Gaussian(x0::T_x, w::T_x): 1D RBF Outer Constructor" begin
    x0 = 1.0;
    w = 2.0;
    rbf = Gaussian(x0, w)
    @test rbf isa Gaussian{Isotropic, typeof(x0[1]), 1} 
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0
end

@testset "Gaussian(x0::Vector{T_x}, w::T_x): Isotropic ND RBF Outer Constructor" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = 1.5;
    
    rbf = Gaussian(x0, w)
    @test rbf isa Gaussian{Isotropic, typeof(x0[1]), length(x0)}
    @test rbf.x0[2] == 2.0
    @test rbf.x0[5] == 5.0
    @test rbf.w == 1.5
end

@testset "Gaussian(x0::Vector{T_x}, w::Vector{T_x}): Vector construction ND RBF Outer Constructor" begin
    x0 = [1.0];
    w = [2.0];
    rbf = Gaussian(x0, w) 
    @test rbf isa Gaussian{Isotropic, typeof(x0[1]), 1}
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0

    x0 = [1.0, 1.0];
    w = [2.0, 3.0];
    rbf = Gaussian(x0, w) 
    @test rbf isa Gaussian{Anisotropic{Aligned}, typeof(x0[1]), 2}
    @test rbf.x0 == [1.0, 1.0]
    @test rbf.w == [2.0, 3.0]
end

@testset "(rbf::Gaussian{Isotropic, T_x, dim})(x): Isotropic RBF Evaluation Functor" begin
    x0 = [1.0];
    w = [2.0];
    rbf = Gaussian(x0, w)
    @test isapprox(rbf(2),0.01831563888873418,atol=1e-13)
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = 1.5;
    rbf = Gaussian(x0, w)
    @test rbf(x0) == 1.0
    @test isapprox(rbf([2.0,3.0,4.0,5.0,6.0]), 1.300729765406762e-05,atol=1e-13)
end

@testset "(rbf::Gaussian{Anisotropic{Aligned}, T_x, dim})(x): Anisotropic RBF Evaluation Functor" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = [1.0, 1.5, 2.0, 2.5, 3.0];
    rbf = Gaussian(x0, w)
    @test rbf(x0) == 1.0
    @test isapprox(rbf([2.0,3.0,4.0,5.0,6.0]),1.6918979226151304e-10,atol=1e-13)
end

@testset "eval_phi(x, theta::Vector{T_theta}, ::Type{Gaussian{Isotropic, T_x, dim}}): Isotropic RBF Evaluation" begin
    x0 = [1.0];
    w = [2.0];
    theta = [x0;w]
    rbfeval1 = SLFA.eval_phi(1,theta,Gaussian{Isotropic,typeof(theta[1]),length(x0)})
    rbfeval2 = SLFA.eval_phi(2,theta,Gaussian{Isotropic,typeof(theta[1]),length(x0)})
    @test rbfeval1 == 1.0
    @test isapprox(rbfeval2,0.01831563888873418,atol=1e-13)
end

@testset "eval_phi(x, theta::Vector{T_theta}, ::Type{Gaussian{Anisotropic{Aligned}, T_x, dim}}): Isotropic RBF Evaluation" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = [1.0, 1.5, 2.0, 2.5, 3.0];
    theta = [x0;w]
    rbfeval1 = SLFA.eval_phi(x0,theta,Gaussian{Anisotropic{Aligned},typeof(theta[1]),length(x0)})
    rbfeval2 = SLFA.eval_phi([2.0,3.0,4.0,5.0,6.0],theta,Gaussian{Anisotropic{Aligned},typeof(theta[1]),length(x0)})
    @test rbfeval1 == 1.0
    @test isapprox(rbfeval2,1.6918979226151304e-10,atol=1e-13)
end

@testset "size(::Type{Gaussian{Isotropic, T_x, dim}}): Size of Isotropic RBF" begin
    @test size(Gaussian{Isotropic,  typeof(1.0), 3}) == 4
end

@testset "size(::Type{Gaussian{Anisotropic{Aligned}, T_x, dim}}): Size of Isotropic RBF" begin
    @test size(Gaussian{Anisotropic{Aligned}, typeof(1.0), 3}) == 6
end

