using SLFA

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
    A = sparse([false, true, false; true, false, true; false, false, true])
    D = sparse(0.0, sqrt(0.5^2+0.5^2), sqrt(0.5^2+1.0^2); sqrt(0.5^2+0.5^2), 0.0, sqrt(0.5^2+0.5^2); sqrt(0.5^2+1.0^2), sqrt(0.5^2+0.5^2), 0.0)
    i_extrema = 2
    support_set = [true, true, true]
    I_terminal = [1, 3]
    extremum_type = Maximum()
    @test max_dist_theta0(X, res, A, D, i_extrema, support_set, I_terminal, extremum_type, Gaussian{Isotropic, Float64, 1}) == (2.5 / sqrt(0.5^2 + 0.5^2), X[i_extrema,:])
end