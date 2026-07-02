using SLFA
using SparseArrays
using LinearAlgebra

@testset "supr(X, res, i_extrema, support_set, I_terminal, ::Extremum, D): Find extremum value for set" begin
    X = [0.0, 0.5, 1.0]
    res = [0.0, 1.0, 1.5]
    i_extrema = 2
    support_set = [true, true, true]
    I_terminal = [1, 3]
    D = 1
    extremum_type = Maximum()
    @test supr(X,res,i_extrema,support_set,I_terminal,extremum_type,D) == -0.5
end

@testset "rel_supr(X, res, i_extrema, support_set, I_terminal, ::Maximum, D): Relative supremum for maximum" begin
    # Test when the i_extrema is not at the residual maximum
    X = [0.0, 0.5, 1.0]
    res = [0.0, 1.0, 0.5]
    i_extrema = 3
    support_set = [true, true, true]
    I_terminal = [1, 3]
    D = 1
    @test rel_supr(X, res, i_extrema, support_set, I_terminal, Maximum(), D) == 0
    
    # Test when the support set is a subset of X 
    X = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = [0.0, 1.0, 0.5, 0.8, 0.0]
    i_extrema = 2 # Cannot be a vector
    support_set = [true, true, true, false, false]

    @test isapprox(rel_supr(X, res, i_extrema, support_set, I_terminal, Maximum(), D), .20, atol=1e-13)

    # Test when the support set is the full X
    X = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = [0.0, 1.0, 0.5, 0.25, 0.0]
    i_extrema = 2
    support_set = [true, true, true, true, true]
    @test rel_supr(X, res, i_extrema, support_set, I_terminal, Maximum(), D) == 1.0
end

@testset "rel_supr(X, res, i_extrema, support_set, I_terminal, ::Minimum, D): Relative supremum for minimum" begin
    # Test when the i_extrema is not at the residual minimum
    X = [0.0, 0.5, 1.0]
    res = [0.0, -1.0, -0.5]
    i_extrema = 3
    support_set = [true, true, true]
    I_terminal = [1, 3]
    D = 1
    @test rel_supr(X, res, i_extrema, support_set, I_terminal, Minimum(), D) == 0
    
    # Test when the support set is a subset of X 
    X = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = [0.0, -1.0, -0.5, -0.8, 0.0]
    i_extrema = 2 # Cannot be a vector
    support_set = [true, true, true, false, false]

    @test isapprox(rel_supr(X, res, i_extrema, support_set, I_terminal, Minimum(), D), .20, atol=1e-13)

    # Test when the support set is the full X
    X = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = [0.0, -1.0, -0.5, -0.25, 0.0]
    i_extrema = 2
    support_set = [true, true, true, true, true]
    @test rel_supr(X, res, i_extrema, support_set, I_terminal, Minimum(), D) == 1.0
end

@testset "max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type::Extremum, ::Type{Gaussian{Isotropic, T_x, dim}}) where {T_x, dim}: Initial Guess RBF using maximum distance" begin
    X = [0.0, 0.5, 1.0]
    res = [0.0, 0.5, 0.0]
    A = [[false, true, false], [true, false, true], [false, false, true]]
    D = [[0.0, sqrt(0.5^2+0.5^2), sqrt(0.5^2+1.0^2)], [sqrt(0.5^2+0.5^2), 0.0, sqrt(0.5^2+0.5^2)], [sqrt(0.5^2+1.0^2), sqrt(0.5^2+0.5^2), 0.0]]
    i_extrema = 2
    support_set = [true, true, true]
    I_terminal = [1, 3]
    extremum_type = Maximum()
    # Tests general case with maximum
    @test max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, Gaussian{Isotropic, Float64, 1}) == [0.5, 5.0, only(X[i_extrema,:]), 0.0]
end

@testset "max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type::Extremum, ::Type{Gaussian{Isotropic, T_x, dim}}) where {T_x, dim}: Initial Guess RBF using maximum distance - Small extrema" begin
    X = [0.0, 0.5, 1.0]
    res = [0.0, .020446049250313e-13, 1.0]
    A = [[false, true, false], [true, false, true], [false, false, true]]
    D = [[0.0, sqrt(0.5^2+0.5^2), sqrt(0.5^2+1.0^2)], [sqrt(0.5^2+0.5^2), 0.0, sqrt(0.5^2+0.5^2)], [sqrt(0.5^2+1.0^2), sqrt(0.5^2+0.5^2), 0.0]]
    i_extrema = 2
    I_terminal = [1, 3]
    extremum_type = Maximum()
    support_set = [true, true, false]
    # Test if support set is not full data set
    @test max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, Gaussian{Isotropic, Float64, 1}) == [0.5, 5.0, only(res[i_extrema])-1, 1.0] 
    # Test if extremum is a maximum
    support_set = [true, true, true]
    @test max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, Gaussian{Isotropic, Float64, 1}) == [0.5, 5.0, res[i_extrema]-minimum(res), minimum(res)] 
    # Test if extremum is a minimum
    # res = [0.0, -2.020446049250313e-13, -1.0]
    extremum_type = Minimum()
    @test max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, Gaussian{Isotropic, Float64, 1}) == [0.5, 5.0, res[i_extrema]-maximum(res), maximum(res)] 
end

@testset "squaredTV(res, A, D): Squared TV Penalty Value" begin
    using SparseArrays
    res = [0.0, 0.5, 0.0]
    A = sparse([false true false; true false true; false true false])
    D = sparse([0.0 sqrt(0.5^2+0.5^2) sqrt(0.5^2+1.0^2); sqrt(0.5^2+0.5^2) 0.0 sqrt(0.5^2+0.5^2); sqrt(0.5^2+1.0^2) sqrt(0.5^2+0.5^2) 0.0])
    @test isapprox(squaredTV(res,A,D),1.0,atol=1e-8)
end

@testset "RMSE(res): RMSE Value" begin
    using LinearAlgebra
    res = [0.0, 0.5, 0.0]
    @test RMSE(res) == norm(res) / sqrt(length(res))
end

@testset "res_error(res, res_validation, res_history, N): Set of Error Measurements" begin
    res = [0.0, 0.5, 0.0]
    res_validation = res
    res_history = res
    N = 1
    @test res_error(res, res_validation, res_history, N) == [RMSE(res), maximum(res), maximum(res)-minimum(res)]
end

@testset "initial_guess(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}): Get Initial Guess and Isolate Theta 0" begin
    X = [0.0, 0.5, 1.0]
    res = [0.0, 0.5, 0.0]
    A = [[false, true, false], [true, false, true], [false, false, true]]
    D = [[0.0, sqrt(0.5^2+0.5^2), sqrt(0.5^2+1.0^2)], [sqrt(0.5^2+0.5^2), 0.0, sqrt(0.5^2+0.5^2)], [sqrt(0.5^2+1.0^2), sqrt(0.5^2+0.5^2), 0.0]]
    i_extrema = 2
    support_set = [true, true, true]
    I_terminal = [1, 3]
    extremum_type = Maximum()
    N = 1
    theta0 = max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, Gaussian{Isotropic, Float64, 1})
    @test initial_guess(theta0, X, res, A, D, N, Gaussian{Isotropic, Float64, 1}) == theta0
end

@testset "lsq_solver(theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}): RBF Simple LSQ Solver" begin
    X = [0.0, 0.5, 1.0]
    res = [0.0, 0.5, 0.0]
    A = sparse([false true false; true false true; false true false])
    D = sparse([0.0 sqrt(0.5^2+0.5^2) sqrt(0.5^2+1.0^2); sqrt(0.5^2+0.5^2) 0.0 sqrt(0.5^2+0.5^2); sqrt(0.5^2+1.0^2) sqrt(0.5^2+0.5^2) 0.0])
    i_extrema = 2
    support_set = [true, true, true]
    I_terminal = [1, 3]
    extremum_type = Maximum()
    N = 1
    theta0 = max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, Gaussian{Isotropic, Float64, 1})
    @test all(isa.(lsq_solver(theta0, X, res, A, D, N, Gaussian{Isotropic, Float64, 1}),Number))
end

@testset "lsq_TV_solver(omega_TV, theta0, X, res, A, D, N, T_phi::Type{<:BasisFunction}): RBF Combined TV Penalty Solver" begin
    X = [0.0, 0.5, 1.0]
    res = [0.0, 0.5, 0.0]
    A = sparse([false true false; true false true; false true false])
    D = sparse([0.0 sqrt(0.5^2+0.5^2) sqrt(0.5^2+1.0^2); sqrt(0.5^2+0.5^2) 0.0 sqrt(0.5^2+0.5^2); sqrt(0.5^2+1.0^2) sqrt(0.5^2+0.5^2) 0.0])
    i_extrema = 2
    support_set = [true, true, true]
    I_terminal = [1, 3]
    extremum_type = Maximum()
    omega_TV = 5
    N = 1
    theta0 = max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, Gaussian{Isotropic, Float64, 1})
    @test all(isa.(lsq_TV_solver(omega_TV, theta0, X, res, A, D, N, Gaussian{Isotropic, Float64, 1}),Number))
end