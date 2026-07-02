using SLFA
using SafeTestsets

@safetestset "SLFA:RBF" begin
    # Write your tests here.
    include("./RBF_test.jl")
    include("./RBFN_test.jl")
    include("./solvers_test.jl")
    include("./train_RBFN_test.jl")
end

