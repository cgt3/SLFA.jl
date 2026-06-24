using SLFA
using SafeTestsets

@safetestset "SLFA:RBF" begin
    # Write your tests here.
    include("./RBF_test.jl")
    include("./RBFN_test.jl")
    include("./HelperFunctions_test.jl")
    include("./test_Train_RBFN.jl")
end

