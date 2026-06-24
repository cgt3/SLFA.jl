using SLFA

@testset "RBFN(a0::T_y, a::Vector{T_y}, phi::Vector{T_phi}): RBFN Inner Constructor" begin
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

@testset "RBFN(Theta::Matrix{T_theta}, T_phi::Type{<:BasisFunction}): RBFN Outer Constructor" begin
    using SLFA
    thetas = [1.0 1.1 1.2 1.3; 2.0 2.1 2.2 2.3; 3.0 3.1 3.2 3.3]
    rbfn_1D = RBFN(thetas, Gaussian{Isotropic,typeof(thetas[1,1]),length(thetas[1,:])-3})
    @test rbfn_1D isa RBFN{Gaussian{Isotropic,typeof(thetas[1,1]),1}}
    @test rbfn_1D.a0 == sum(thetas[:,end])
    @test rbfn_1D.a == thetas[:,3]
    @test rbfn_1D.phi[1] == Gaussian{Isotropic,typeof(thetas[1,1]),1}(thetas[1,1], thetas[1,2])
    @test rbfn_1D.N == size(thetas, 1)
end

@testset "(network::RBFN)(x::Number): Functor for RBFN evaluation" begin
    a0 = 1.0
    a = [0.5, 0.3, 0.2]
    phi = [Gaussian{Isotropic, typeof(a0), 1}(1.0, 2.0), Gaussian{Isotropic, typeof(a0), 1}(2.0, 3.0), Gaussian{Isotropic, typeof(a0), 1}(3.0, 4.0)]
    
    rbfn = RBFN(a0, a, phi)
    @test isapprox(rbfn(1.5), 1.2155594879542804, atol=1e-13)
    @test isapprox(rbfn(2.5), 1.0353446000483493, atol=1e-13)
    @test isapprox(rbfn(3.5), 1.0036631282662591, atol=1e-13)
end