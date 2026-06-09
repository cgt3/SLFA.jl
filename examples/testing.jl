
using SparseArrays

include("../src/SLFA.jl")
using .SLFA


X = Float64[1:11...]
y = [1, 2, 3, 4, 5, 7.5, 7, 8.5, 8, 10, 9]

A, D = SLFA.get_nbr_matrix(X)

i_extrema = 10
support_set, I_terminal = get_support_set(X, y, i_extrema, A, D, SLFA.Maximum(), start_gap = 1.0)

