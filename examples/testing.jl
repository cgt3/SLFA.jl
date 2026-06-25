
using SparseArrays

include("../src/SLFA.jl")
using .SLFA


# X = Float64[1:11...]
# y = [11, 2, 3, 4, 5, 7.5, 7, 8.5, 8, 10, 9]

# y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

X = Float64[1:10...]
y = [3, 2, 1, 2, 3, 4, 5, 4, 3, 2 ]
A, D = SLFA.get_nbr_matrix_1D(X)

i_extrema = 7
support_set, I_terminal = get_support_set(X, y, i_extrema, A, D, SLFA.Maximum(), start_gap = 0.0)

