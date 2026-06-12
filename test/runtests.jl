using SLFA
using SafeTestsets

@safetestset "SLFA:RBF" begin
    # Write your tests here.
    include("./RBF_test.jl")
end
