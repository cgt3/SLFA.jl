using SLFA

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