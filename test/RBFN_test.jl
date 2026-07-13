
@testset "RBFN: Inner Constructor" begin
    a0 = 1.0
    a = [0.5, 0.3, 0.2]
    phi = [Gaussian{Isotropic, typeof(a0), 1}(1.0, 2.0), Gaussian{Isotropic, typeof(a0), 1}(2.0, 3.0), Gaussian{Isotropic, typeof(a0), 1}(3.0, 4.0)]
    
    rbfn = RBFN(a0, a, phi)
    @test rbfn.a0 == a0
    @test rbfn.a == a
    @test rbfn.phi == phi

    phi = [Gaussian{Isotropic, typeof(a0), 1}(1.0, 2.0), Gaussian{Isotropic, typeof(a0), 1}(2.0, 3.0)]
    
    try
        rbfn = RBFN(a0, a, phi)
    catch e
        @test e == "SLFA.RBFN: number of weights (3) does not match number of basis functions (2)."
    end
end 

@testset "RBFN: Outer Constructor" begin
    Theta = [1.0 1.1 1.2 1.3; 2.0 2.1 2.2 2.3; 3.0 3.1 3.2 3.3]
    rbfn_1D = RBFN(Theta, Gaussian{Isotropic, eltype(Theta), length(Theta[1,:])-3})
    @test rbfn_1D isa RBFN{Gaussian{Isotropic, eltype(Theta), 1}}
    @test rbfn_1D.a0 == sum(Theta[:,end])
    @test rbfn_1D.a == Theta[:,3]
    @test rbfn_1D.phi[1] == Gaussian{Isotropic,eltype(Theta),1}(Theta[1,1], Theta[1,2])
    @test rbfn_1D.N == size(Theta, 1)
end


@testset "dimension(RBFN)" begin
    a0 = 0.0
    a = [0.5, 0.3, 0.2]

    dim = 2
    x0 = [ [1.0, 0.0], [2.0, 0.0], [3.0, 0.0] ]
    w = [2.0, 3.0, 4.0]
    phi = [ Gaussian{Isotropic, typeof(a0), dim}(x0[1], w[1]), 
            Gaussian{Isotropic, typeof(a0), dim}(x0[2], w[2]),
            Gaussian{Isotropic, typeof(a0), dim}(x0[3], w[3])]
    
    rbfn = RBFN(a0, a, phi)

    @test dimension(rbfn) == dim
end

@testset "(network::RBFN)(x::Real): Functor for RBFN evaluation, 1D, single samples" begin
    a0 = 1.0
    a = [0.5, 0.3, 0.2]
    phi = [Gaussian{Isotropic, typeof(a0), 1}(1.0, 2.0), Gaussian{Isotropic, typeof(a0), 1}(2.0, 3.0), Gaussian{Isotropic, typeof(a0), 1}(3.0, 4.0)]
    
    rbfn = RBFN(a0, a, phi)
    @test isapprox(rbfn(1.5), 1.2155594879542804, atol=1e-13)
    @test isapprox(rbfn(2.5), 1.0353446000483493, atol=1e-13)
    @test isapprox(rbfn(3.5), 1.0036631282662591, atol=1e-13)
end

@testset "(network::RBFN)(x::Vector): Functor for RBFN evaluation, 1D, multiple samples" begin
    a0 = 1.0
    a = [0.5, 0.3, 0.2]
    phi = [Gaussian{Isotropic, typeof(a0), 1}(1.0, 2.0), Gaussian{Isotropic, typeof(a0), 1}(2.0, 3.0), Gaussian{Isotropic, typeof(a0), 1}(3.0, 4.0)]
    
    X = [1.5, 2.5, 3.5]
    rbfn = RBFN(a0, a, phi)

    y = rbfn(X)
    @test length(y) == 3
    @test isapprox(y[1], 1.2155594879542804, atol=1e-13)
    @test isapprox(y[2], 1.0353446000483493, atol=1e-13)
    @test isapprox(y[3], 1.0036631282662591, atol=1e-13)
end


@testset "(network::RBFN)(x::Vector): Functor for RBFN evaluation, nD, multiple samples" begin
    a0 = 1.0
    a = [0.5, 0.3, 0.2]

    dim = 2
    x0 = [ [1.0, 0.0], [2.0, 0.0], [3.0, 0.0] ]
    w = [2.0, 3.0, 4.0]
    phi = [ Gaussian{Isotropic, typeof(a0), dim}(x0[1], w[1]), 
            Gaussian{Isotropic, typeof(a0), dim}(x0[2], w[2]),
            Gaussian{Isotropic, typeof(a0), dim}(x0[3], w[3])]
    
    X = [1.5 2.5 3.5; 0.0 0.0 0.0]
    rbfn = RBFN(a0, a, phi)
    
    y = rbfn(X)
    @test length(y) == 3
    @test isapprox(y[1], 1.2155594879542804, atol=1e-13)
    @test isapprox(y[2], 1.0353446000483493, atol=1e-13)
    @test isapprox(y[3], 1.0036631282662591, atol=1e-13)
end