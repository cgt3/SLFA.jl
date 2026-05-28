module SLFA

# Abstract types
export RBF_Orientation, Rotated, Aligned
export RBF_Shape, Isotropic, Anisotropic
export RBF_Sparsity, Sparse, Dense
export RBF_Type, Gaussian

# Data structures
export RBF, RBFN

# Functions

# Abstract data types for classifying/parameterizing RBFs
abstract type RBF_Orientation end
abstract type Rotated <:RBF_Orientation end
abstract type Aligned <:RBF_Orientation end

abstract type RBF_Shape end
abstract type Isotropic   <:RBF_Shape end
abstract type Anisotropic{T_orientation} <:RBF_Shape end

abstract type RBF_Sparsity end
abstract type Sparse <: RBF_Sparsity end
abstract type Dense  <: RBF_Sparsity end

# Abstract type for setting RBF parameters
abstract type RBF end

struct Gaussian{T_shape, T_sparsity, dim} <: RBF
    x0::Union{Vector{<:Real}, Real}
    w::Union{Vector{<:Real}, Real}

    function Gaussian{Isotropic, Dense, 1}(x0::Real, w::Real)
        return new{Isotropic, Dense, 1}(x0, w)
    end

    function Gaussian{Isotropic, Dense, dim}(x0::Vector{T}, w::Real) where {dim, T<:Real}
        if !(dim isa Integer)
            throw("SLFA.Gaussian: dim must be an integer.")
        end

        if dim != length(x0)
            throw("SLFA.Gaussian: dim does not match length(x0).")
        end

        if length(x0) ==  0
            throw("SLFA.Gaussian: x0 = []")
        end

        if length(x0) == 1
            return new{Isotropic, Dense, dim}(x0[1], w)
        else
            return new{Isotropic, Dense, dim}(x0, w)
        end
    end
 
    function Gaussian{Anisotropic{Aligned}, Dense, dim}(x0::Vector{T}, w::Vector{T}) where {dim,T<:Real}
        if !(dim isa Integer)
            throw("SLFA.Gaussian: dim must be an integer.")
        end

        if dim != length(x0)
            throw("SLFA.Gaussian: dim does not match length(x0).")
        end

        if length(x0) == 0
            throw("SLFA.Gaussian: x0 = []")
        elseif length(x0) == 1
            throw("SLFA.Gaussian: Anisotropic constructor should not be used for 1D Gaussians.")
        end

        if length(x0) != length(w)
            throw("SLFA.Gaussian: length of x0 ($(length(x0))) does not match length of w ($(length(w))).")
        end

        return new{Anisotropic{Aligned}, Dense, dim}(x0, w)
    end
end

# Non-parametric constructors
function Gaussian(x0::Real, w::Real)
    return Gaussian{Isotropic, Dense, 1}(x0, w)
end

function Gaussian(x0::Vector{T}, w::Real) where T<:Real
    return Gaussian{Isotropic, Dense, length(x0)}(x0, w)
end

function Gaussian(x0::Vector{T}, w::Vector{T}) where T<:Real
    if length(x0) == 1 && length(w) == 1
        return Gaussian{Isotropic, Dense, 1}(x0[1], w[1])
    end

    return Gaussian{Anisotropic{Aligned}, Dense, length(x0)}(x0, w)
end

# Functors for evaluating individual RBFs
function (rbf::Gaussian{Isotropic, Dense, dim})(x) where dim
    return exp( -sum( (rbf.w .*(x - rbf.x0)) .^ 2 ) )
end

function (rbf::Gaussian{Anisotropic{Aligned}, Dense, dim})(x) where dim
    return exp( -sum( (rbf.w .*(x - rbf.x0)) .^ 2 ) ) 
end


# Functions for constructing RBFN
struct RBFN{T_RBF}
    N::Integer
    a0::Number
    a::Vector{<:Number}
    rbf::Vector{T_RBF}

    function RBFN(a0::Real, a::Vector{T}, rbf::Vector{T_RBF}) where {T<:Real, T_RBF<:RBF}
        if length(a) != length(rbf)
            throw("SLFA.RBFN: number of weights ($(length(a)))does not match number of basis functions ($(length(rbf))).")
        end

        return new{eltype(rbf)}(length(a), a0, a, rbf)
    end
end

function (network::RBFN)(x::Number)
    result = network.a0
    
    for k = 1:network.N
        result += a[k]*network.rbf[k](x)
    end
end

#include("train_RBFN.jl")


end # module