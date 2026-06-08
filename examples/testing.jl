
using SparseArrays

include("../src/SLFA.jl")
using .SLFA


X = Float64[1, 2, 3, 4, 5, 6]
y = -[3, 0, 1, 4, 3, 4]

D = SLFA.get_nbr_matrix(X)

support_set, I_terminal = get_support_set(X, y, 4, D, SLFA.Minimum())

