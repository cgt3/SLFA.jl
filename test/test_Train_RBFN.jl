using SLFA

@testset "rel_supr(X, res, i_extrema, support_set, I_terminal, ::Maximum, D): Relative support set test on maximums" begin
    # Test when the i_extrema is not at the residual maximum
    X = [0.0, 0.5, 1.0]
    res = [0.0, 1.0, 0.5]
    i_extrema = 3
    support_set = [true, true, true]
    I_terminal = [1, 3]
    D = 1
    @test rel_supr(X, res, i_extrema, support_set, I_terminal, Maximum(), D) == 0
    
    # Test when the support set is a subset of X 
    using SLFA
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