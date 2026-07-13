using SLFA

@testset  "Gaussian{Isotropic}: 1D Gaussian RBF Inner Constructor 1D" begin
    x0 = 1.0;
    w = 2.0;

    rbf = Gaussian{Isotropic,typeof(x0),1}(x0,w)
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0
end

@testset "Gaussian{Isotropic}: ND/1D Gaussian RBF Inner Constructor" begin
    x0 = [1.0];
    w = 2.0;
    try
        rbf = Gaussian{Isotropic, eltype(x0), 1.5}(x0, w)
        @test false #Fails to catch the non-integer dim
    catch e
        @test e == "SLFA.Gaussian: dim must be an integer."
    end
    
end

@testset "Gaussian{Isotropic}: ND/1D Gaussian RBF Inner Constructor Dim Mismatch" begin
    x0 = [1.0];
    w = 2.0;
    try
        rbf = Gaussian{Isotropic, eltype(x0), 2}(x0, w)
        @test false #Fails to catch the non-matching dim
    catch e
        @test e == "SLFA.Gaussian: dim does not match length(x0)."
    end
end

@testset "Gaussian{Isotropic}: ND/1D Gaussian RBF Inner Constructor Empty x0" begin
    x0 = [1.0];
    w = 2.0;
    try
        x_empty = Float64[]
        rbf = Gaussian{Isotropic, eltype(x0), 0}(x_empty, w)
        @test false #Fails to catch the empty x0
    catch e
        @test e == "SLFA.Gaussian: x0 = []"
    end 
end

@testset "Gaussian{Isotropic}: ND/1D Gaussian RBF Inner Constructor 1D Anisotropic" begin
    x0 = [1.0];
    w = 2.0;
    rbf = Gaussian{Isotropic, eltype(x0), length(x0)}(x0, w)
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0

    rbf = Gaussian{Isotropic, eltype(x0), 2}([x0[1], x0[1]], w)
    @test rbf.x0 == [1.0, 1.0]
    @test rbf.w == 2.0
end

@testset "Gaussian{Anisotropic{Aligned}}: Anisotropic RBF Inner Constructor Non-integer Dim" begin
    x0 = [1.0, 1.0];
    w = [2.0, 3.0];
    try
        rbf = Gaussian{Anisotropic{Aligned}, eltype(x0), 1.5}(x0, w)
        @test false #Fails to catch the non-integer dim
    catch e
        @test e == "SLFA.Gaussian: dim must be an integer."
    end
end

@testset "Gaussian{Anisotropic{Aligned}}: Anisotropic RBF Inner Constructor Dim Mismatch" begin
    x0 = [1.0, 1.0];
    w = [2.0, 3.0];
    try
        rbf = Gaussian{Anisotropic{Aligned}, eltype(x0), 3}(x0, w)
        @test false #Fails to catch the non-matching dim
    catch e
        @test e == "SLFA.Gaussian: dim does not match length(x0)."
    end
end

@testset "Gaussian{Anisotropic{Aligned}}: Anisotropic RBF Inner Constructor Empty x0" begin
    x0 = [1.0, 1.0];
    w = [2.0, 3.0];
    try
        x_empty = Float64[]
        rbf = Gaussian{Anisotropic{Aligned}, eltype(x0), 0}(x_empty, w)
        @test false #Fails to catch the empty x0
    catch e
        @test e == "SLFA.Gaussian: x0 = []"
    end
end

@testset "Gaussian{Anisotropic{Aligned}}: Anisotropic RBF Inner Constructor Anisotropic used on 1D" begin
    x0 = [1.0, 1.0];
    w = [2.0, 3.0];  
    try
        rbf = Gaussian{Anisotropic{Aligned}, eltype(x0), 1}([x0[1]], w)
        @test false #Fails to catch that anisotropic shouldnt be 1D
    catch e
        @test e == "SLFA.Gaussian: Anisotropic constructor should not be used for 1D Gaussians."
    end
end

@testset "Gaussian{Anisotropic{Aligned}}: Anisotropic RBF Inner Constructor Length Mismatch between x0 and w" begin
    x0 = [1.0, 1.0];
    w = [2.0, 3.0];
    try
        rbf = Gaussian{Anisotropic{Aligned}, eltype(x0), 2}(x0, [w[1]])
        @test false #Fails to catch that x0 and w dimensions dont match
    catch e
        @test e == "SLFA.Gaussian: length of x0 ($(length(x0))) does not match length of w ($(length([w[1]])))."
    end
end

@testset "Gaussian{Anisotropic{Aligned}}: Anisotropic RBF Inner Constructor Valid Construction" begin
    x0 = [1.0, 1.0];
    w = [2.0, 3.0];
    rbf = Gaussian{Anisotropic{Aligned}, eltype(x0),length(x0)}(x0,w)
    @test rbf.x0 == [1.0, 1.0]
    @test rbf.w == [2.0,3.0]
end

@testset "Gaussian(x0, w): 1D RBF Outer Constructor" begin
    x0 = 1.0;
    w = 2.0;
    rbf = Gaussian(x0, w)
    @test rbf isa Gaussian{Isotropic, eltype(x0), 1} 
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0
end

@testset "Gaussian(x0::Vector, w): Isotropic ND RBF Outer Constructor" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = 1.5;
    
    rbf = Gaussian(x0, w)
    @test rbf isa Gaussian{Isotropic, eltype(x0), length(x0)}
    @test rbf.x0[2] == 2.0
    @test rbf.x0[5] == 5.0
    @test rbf.w == 1.5
end

@testset "Gaussian(x0::Vector{T_x}, w::Vector{T_x}): Vector construction nD RBF Outer Constructor" begin
    x0 = [1.0];
    w = [2.0];
    rbf = Gaussian(x0, w) 
    @test rbf isa Gaussian{Isotropic, eltype(x0), 1}
    @test rbf.x0 == 1.0
    @test rbf.w == 2.0

    x0 = [1.0, 1.0];
    w = [2.0, 3.0];
    rbf = Gaussian(x0, w) 
    @test rbf isa Gaussian{Anisotropic{Aligned}, eltype(x0), 2}
    @test rbf.x0 == [1.0, 1.0]
    @test rbf.w == [2.0, 3.0]
end

@testset "(rbf::Gaussian{Isotropic})(x): 1D/nD, single samples" begin
    x0 = [1.0];
    w = [2.0];
    rbf = Gaussian(x0, w)
    @test isapprox(rbf(2), 0.01831563888873418, atol=1e-13)

    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = 1.5;
    x_eval = [2.0,3.0,4.0,5.0,6.0]
    rbf = Gaussian(x0, w)
    
    @test rbf(x0) == 1.0
    @test isapprox(rbf(x_eval), 1.300729765406762e-05, atol=1e-13)
end

@testset "(rbf::Gaussian{Anisotropic{Aligned})(x): nD, single samples" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = [1.0, 1.5, 2.0, 2.5, 3.0];
    x_eval = [2.0,3.0,4.0,5.0,6.0]
    rbf = Gaussian(x0, w)

    @test rbf(x0) == 1.0
    @test isapprox(rbf(x_eval), 1.6918979226151304e-10, atol=1e-13)
end

@testset "eval_phi(Gaussian{Isotropic}): 1D, single samples" begin
    x0 = [1.0];
    w = [2.0];
    theta = [x0; w]
    
    rbf_eval1 = SLFA.eval_phi(1, theta, Gaussian{Isotropic, typeof(theta[1]), length(x0)})
    @test isapprox(rbf_eval1, 1.0, atol=1e-13)
    
    rbf_eval2 = SLFA.eval_phi(2, theta, Gaussian{Isotropic, typeof(theta[1]), length(x0)})
    @test isapprox(rbf_eval2, 0.01831563888873418, atol=1e-13)
end

@testset "eval_phi(Gaussian{Isotropic}): 1D, multiple samples" begin
    x0 = [1.0];
    w = [2.0];
    theta = [x0; w]
    
    X_eval = [1, 2]
    rbf_eval = SLFA.eval_phi(X_eval, theta, Gaussian{Isotropic, typeof(theta[1]), length(x0)})

    @test isapprox(rbf_eval[1], 1.0, atol=1e-13)
    @test isapprox(rbf_eval[2], 0.01831563888873418, atol=1e-13)
end

@testset "eval_phi(Gaussian{Isotropic}): nD, multiple samples" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = 1.5;
    theta = [x0; w]

    x_eval = [2.0,3.0,4.0,5.0,6.0]
    X_eval = [x0 x_eval]

    rbf_eval = SLFA.eval_phi(X_eval, theta, Gaussian{Isotropic, typeof(theta[1]), length(x0)})
    @test isapprox(rbf_eval[1], 1.0, atol=1e-13)
    @test isapprox(rbf_eval[2], 1.300729765406762e-05, atol=1e-13)
end

@testset "eval_phi(Gaussian{Anisotropic{Aligned}}): nD, single samples" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = [1.0, 1.5, 2.0, 2.5, 3.0];

    x_eval = [2.0,3.0,4.0,5.0,6.0]
    theta = [x0;w]

    rbf_eval1 = SLFA.eval_phi(x0, theta, Gaussian{Anisotropic{Aligned}, typeof(theta[1]), length(x0)})
    rbf_eval2 = SLFA.eval_phi(x_eval, theta, Gaussian{Anisotropic{Aligned}, typeof(theta[1]),length(x0)})
    
    @test rbf_eval1 == 1.0
    @test isapprox(rbf_eval2, 1.6918979226151304e-10, atol=1e-13)
end


@testset "eval_phi(Gaussian{Anisotropic{Aligned}}): nD, multiple samples" begin
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    w = [1.0, 1.5, 2.0, 2.5, 3.0];

    theta = [x0;w]

    x_eval = [2.0,3.0,4.0,5.0,6.0]
    X_eval = [x0 x_eval]

    rbf_eval = SLFA.eval_phi(X_eval, theta, Gaussian{Anisotropic{Aligned}, typeof(theta[1]), length(x0)})
    
    @test rbf_eval[1] == 1.0
    @test isapprox(rbf_eval[2], 1.6918979226151304e-10, atol=1e-13)
end

@testset "dimension(Gaussian{Isotropic})" begin
    @test dimension(Gaussian{Isotropic, Float64, 3}([1.0, 1.0, 1.0], 1.0)) == 3
    @test dimension(Gaussian{Isotropic, Float64, 3}) == 3
end

@testset "dimension(Gaussian{Anisotropic{Aligned}})" begin
    @test dimension(Gaussian{Anisotropic{Aligned}, Float64, 2}([1.0, 1.0], [1.0, 1.0])) == 2
    @test dimension(Gaussian{Anisotropic{Aligned}, Float64, 2}) == 2
end

@testset "size(Gaussian{Isotropic})" begin
    @test size(Gaussian{Isotropic, Float64, 3}) == 4
end

@testset "size(Gaussian{Anisotropic{Aligned}})" begin
    @test size(Gaussian{Anisotropic{Aligned}, Float64, 3}) == 6
end

