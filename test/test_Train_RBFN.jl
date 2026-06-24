using SLFA

@testset "(::Strict)(a,b): Return Strict Monotonicity Behavior" begin
    a = 1
    b = 1
    MonoType = Strict()
    @test MonoType(a,b)==false
end

@testset "(::Strict)(a,b): Return Non-Strict Monotonicity Behavior" begin
    a = 1
    b = 1
    MonoType = Nonstrict()
    @test MonoType(a,b)==true
end

@testset "(::Strict)(a,b, ::Minimum): Check Strict minimum Behavior" begin
    a = 1
    b = 2
    MonoType = Strict()
    @test MonoType(a,b,Minimum())==true
end

@testset "(::Strict)(a,b, ::Maximum): Check Strict maximum Behavior" begin
    a = 1
    b = 2
    MonoType = Strict()
    @test MonoType(a,b,Maximum())==false
end

@testset "(::Nonstrict)(a,b, ::Minimum): Check Non-Strict minimum Behavior" begin
    a = 1
    b = 2
    MonoType = Nonstrict()
    @test MonoType(a,b,Minimum())==true
end

@testset "(::Nonstrict)(a,b, ::Maximum): Check Non-Strict maximum Behavior" begin
    a = 1
    b = 2
    MonoType = Nonstrict()
    @test MonoType(a,b,Maximum())==false
end

@testset "dist!(D::AbstractMatrix, i1::Integer, i2::Integer, X::Vector{T_x}): D matrix modifier for 1D" begin
    D = zeros(3,3)
    X = [0.0, 0.5, 1.0]
    # Check no value assigned
    dist!(D,1,2,X)
    @test isapprox(dist!(D,1,1,X), 0, atol=1e-13)
    # Check only actual position assigned, not inverse
    D[1,3] = 5
    @test dist!(D,1,3,X) == 5
    # Check only inverse assigned, not requested position
    D[2,3] = 3
    dist!(D,3,2,X)
    @test D[3,2] == 3        
end

@testset "dist!(D::AbstractMatrix, i1::Integer, i2::Integer, X::Matrix{T_x}): D Matrix modifier for ND" begin
    D = zeros(3,3)
    X = [0.0 0.5 1.0; 0.1 0.6 1.1; 0.2 0.7 1.2]
    # Check no value assigned
    dist!(D,1,2,X)
    @test isapprox(dist!(D,1,1,X), 0, atol=1e-13)
    # Check only actual position assigned, not inverse
    D[1,3] = 5
    @test dist!(D,1,3,X) == 5
    # Check only inverse assigned, not requested position
    D[2,3] = 3
    dist!(D,3,2,X)
    @test D[3,2] == 3        
end

@testset "get_nbr_matrix1D(X::Vector{T_x}, D::AbstractMatrix, r::Real): Get neighbor matrix for 1D" begin
    using SparseArrays
    X = [0.0, 0.5, 1.0]
    A,D = get_nbr_matrix1D(X)
    # Test simple presorted case with no duplicates
    @test A == sparse([false true false; true false true; false true false])
    @test isapprox(D, sparse([0.0 0.5 0.0; 0.5 0.0 0.5; 0.0 0.5 0.0]), atol=1e-13)
    # Test for nonsorted case with duplicates
    X = [0.5, 0.0, 1.0, 0.5]
    A,D = get_nbr_matrix1D(X)
    @test A == sparse([2, 3, 1, 4, 1, 4, 2, 3], [1, 1, 2, 2, 3, 3, 4, 4], Bool[1, 1, 1, 1, 1, 1, 1, 1], 4, 4)
    @test isapprox(D, sparse([2, 3, 1, 4, 1, 4, 2, 3], [1, 1, 2, 2, 3, 3, 4, 4], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 4, 4), atol=1e-13)
end 

@testset "get_support_set(X::Union{Vector{T_x}, Matrix{T_x}}, res::Vector{T_y}, i_extrema::Integer, A::AbstractMatrix, D::AbstractMatrix, extremum_type::Extremum;): Get support set for 1D sorted" begin
    X = [0.0, 0.5, 1.0, 1.5]
    res = [1.0, 2.0, 3.0, 2.0]
    A,D = get_nbr_matrix1D(X)
    i_extrema = 3
    extremum_type = Maximum()
    support_set, I_terminal = get_support_set(X, res, i_extrema, A, D, extremum_type; is_monotonic=Strict(), start_gap=0.0)
    @test support_set == Bool[1, 1, 1, 1]
    @test I_terminal == [4, 1]
end

@testset "get_2k_extrema(X::Union{Vector{T_x}, Matrix{T_x}}, res::Vector{T_y}, A::AbstractMatrix, D::AbstractMatrix;: Get 2k extrema" begin
    using SLFA
    X = [0.0, 0.5, 1.0]
    res = [1.0, 2.0, 3.0]
    A,D = get_nbr_matrix1D(X)
    I_extrema, support_sets, I_terminal_all, extrema_types = get_2k_extrema(X, res, A, D; k_max=1, is_monotonic=Strict(), start_gap=0.0)
    @test I_extrema == [1, 3]   
    @test I_terminal_all == [[3], [1]]
    @test extrema_types == [Minimum(), Maximum()]
    @test support_sets == Vector{Bool}[[1, 1, 1], [1, 1, 1]]
end

@testset "get_best_extrema(X::Union{Vector{T_x}, Matrix{T_x}}, res::Vector{T_y}, A::AbstractMatrix, D::AbstractMatrix, score_func): Get best extrema" begin
    using SLFA
    X = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = [2.1, 2.0, 3.5, 2.5, 1.5]
    A,D = get_nbr_matrix1D(X)
    I_extremas, support_sets, I_terminal_all, extrema_types = get_2k_extrema(X, res, A, D; k_max=1, is_monotonic=Strict(), start_gap=0.0)
    I_extrema, support_set, I_terminal, extrema_type = get_best_extrema(X, res, I_extremas, support_sets, I_terminal_all, extrema_types, D, rel_supr)
    @test I_extrema == 3
    @test I_terminal == [2, 5]
    @test extrema_type == Maximum()
    @test support_set == Bool[0, 1, 1, 1, 1]
end

@testset "get_RBFN_vandermonde(X::Union{Vector{T_x}, Matrix{T_x}}, Theta::Matrix{T_x}, T_phi::Type{<:BasisFunction}): Get RBFN Vandermonde matrix" begin
    using SLFA
    X = [0.0, 0.5, 1.0]
    Theta = [1.0 2.0 3.0 4.0; 3.0 4.0 5.0 6.0; 5.0 6.0 7.0 8.0]
    T_phi = Gaussian{Isotropic, Float64, 1}
    V = get_RBFN_vandermonde(X, Theta, T_phi)
    @test size(V) == (3, 4)
end

@testset "train_RBFN(X::Vector{T_x}, y::Vector{T_y};N_max=1): Train Full 1D RBFN, single RBF" begin
    # using SLFA
    X = [0.0, 0.5, 1.0]
    y = [1.0, 2.0, 3.0]
    T_phi = Gaussian{Isotropic, Float64, 1}
    Theta, res_history, res, res_validation, A, D, N, T_phi, Theta0 = train_RBFN(X, y, N_max=1)
    # println(res)
    @test size(Theta) == (1, 4)
    @test size(res_history,1) == (2)
    @test size(res,1) == (3)
    @test size(res_validation,1) == (0)
    @test size(A) == (3, 3)
    @test size(D) == (3, 3)
    @test N == 1
    @test T_phi == Gaussian{Isotropic, Float64, 1}
end

@testset "train_RBFN(X::Vector{T_x}, y::Vector{T_y};N_max=1): Mismatch X and Y sizes" begin
    X = [0.0, 0.5, 1.0]
    y = [1.0, 2.0]
    T_phi = Gaussian{Isotropic, Float64, 1}
        try
        train_RBFN(X, y, N_max=1)
        @test false #Fails to catch the mismatch dim
    catch e
        @test e == "SLFA.train_RBFN: Number of data points and residual values do not match."
    end
end

@testset "train_RBFN(X::Vector{T_x}, y::Vector{T_y};conv_thresholds = []): Test convergence trigger for RBFN Finish" begin
    X = [0.0, 0.5, 1.0]
    y = [1.0, 2.0, 3.0]
    T_phi = Gaussian{Isotropic, Float64, 1}
    conv_thresholds = [1e10, 1e10, 1e10]
    Theta, res_history, res, res_validation, A, D, N, T_phi, Theta0 = train_RBFN(X, y; N_max=10, conv_thresholds=conv_thresholds)
    @test N == 0
    @test res_history == [2]
    @test res == y
    @test res_validation == Float64[]
    @test size(A) == (3, 3)
    @test size(D) == (3, 3)
end