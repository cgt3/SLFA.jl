include("../src/SLFA.jl")
using .SLFA


X = [1, 1, 1.5, 3, 4];

D = SLFA.get_nbr_matrix(X )
D_dense = Array(D)