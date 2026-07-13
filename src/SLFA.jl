module SLFA

using LinearAlgebra
using LsqFit
using Optim
using SparseArrays

import Base.size

# Abstract types
export RBF_Orientation, Rotated, Aligned
export RBF_Shape, Isotropic, Anisotropic
export RBF_Sparsity, Sparse, Dense
export BasisFunction, RBF, Gaussian

export Monotonicity, Strict, Nonstrict
export Extremum, Maximum, Minimum

# Data structures
export RBF, RBFN

# Functions
export dist!, num_samples, getsample, size, dimension
export get_nbr_matrix1D, get_support_set, get_2k_extrema, get_best_extrema 
export get_RBFN_vandermonde, eval_phi, train_RBFN, train_RBFN_quasi1D

# Abstract data types for classifying/parameterizing RBFs
abstract type RBF_Orientation end
abstract type Rotated <:RBF_Orientation end
abstract type Aligned <:RBF_Orientation end

abstract type RBF_Shape end
abstract type Isotropic <:RBF_Shape end
abstract type Anisotropic{T_orientation} <:RBF_Shape end

# Abstract type for setting RBF parameters
abstract type BasisFunction{dim} end
abstract type RBF{dim} <: BasisFunction{dim} end

struct Gaussian{T_shape, T_x<:Real, dim} <: RBF{dim}
    x0::Union{Vector{T_x}, T_x}
    w::Union{Matrix{T_x}, Vector{T_x}, T_x}

    function Gaussian{Isotropic, T_x, 1}(x0::T_x, w::T_x) where T_x<:Real
        return new{Isotropic, T_x, 1}(x0, w)
    end

    function Gaussian{Isotropic, T_x, dim}(x0::Vector{T_x}, w::T_x) where {dim, T_x<:Real}
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
            return new{Isotropic, T_x, dim}(x0[1], w)
        else
            return new{Isotropic, T_x, dim}(x0, w)
        end
    end
 
    function Gaussian{Anisotropic{Aligned}, T_x, dim}(x0::Vector{T_x}, w::Vector{T_x}) where {dim, T_x<:Real}
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

        return new{Anisotropic{Aligned}, T_x, dim}(x0, w)
    end
end

# Non-parametric constructors
function Gaussian(x0::T_x, w::T_x) where T_x<:Real
    return Gaussian{Isotropic, T_x, 1}(x0, w)
end

function Gaussian(x0::Vector{T_x}, w::T_x) where T_x<:Real
    return Gaussian{Isotropic, T_x, length(x0)}(x0, w)
end

function Gaussian(x0::Vector{T_x}, w::Vector{T_x}) where T_x<:Real
    if length(x0) == 1 && length(w) == 1
        return Gaussian{Isotropic, T_x, 1}(x0[1], w[1])
    end

    return Gaussian{Anisotropic{Aligned}, T_x, length(x0)}(x0, w)
end

# Constructors for arrays of parameters
function Gaussian{Isotropic, T_x, dim}(theta::Vector{T_x}) where {T_x<:Real, dim}
    return Gaussian(theta[1:dim], theta[dim+1])
end

function Gaussian{Anisotropic{Aligned}, T_x, dim}(theta::Vector{T_x}) where {T_x<:Real, dim}
    return Gaussian(theta[1:dim], theta[dim+1:2*dim])
end


# Functors for evaluating individual RBFs
function (rbf::Gaussian{Isotropic, T_x, dim})(x) where {dim, T_x<:Real}
    return exp( -sum( (rbf.w .* (x - rbf.x0)) .^ 2 ) )
end

function (rbf::Gaussian{Anisotropic{Aligned}, T_x, dim})(x) where {dim, T_x<:Real}
    return exp( -sum( (rbf.w .* (x .- rbf.x0)) .^ 2 ) ) 
end

# Functions for evaluating arrays of parameters as RBFs
function eval_phi(x::Real, theta::Vector{T_theta}, ::Type{Gaussian{Isotropic, T_x, 1}}) where {T_theta<:Number, T_x<:Real}
    return exp( -(theta[2]*(x - theta[1]))^2 )
end

function eval_phi(x::Vector{<:Real}, theta::Vector{T_theta}, ::Type{Gaussian{Isotropic, T_x, dim}}) where {dim, T_theta<:Number, T_x<:Real}
    if isempty(x)
        return T_theta[]
    end

    if dim == 1
        return exp.( - (theta[dim+1] .*(x .- theta[1:dim])) .^ 2 )
    else
        return exp( -sum( (theta[dim+1] .*(x .- theta[1:dim])) .^ 2 ) )
    end
end

function eval_phi(x::Vector{<:Real}, theta::Vector{T_theta}, ::Type{Gaussian{Anisotropic{Aligned}, T_x, dim}}) where {dim, T_theta<:Number, T_x<:Real}
    if isempty(x)
        return T_theta[]
    end
    
    return exp( -sum( (theta[(dim+1):2*dim] .*(x .- theta[1:dim])) .^ 2 ) )
end

function eval_phi(X::Matrix{<:Real}, theta::Vector{T_theta}, ::Type{Gaussian{Isotropic, T_x, dim}}) where {dim, T_theta<:Number, T_x<:Real}
    if isempty(X)
        return T_theta[]
    end
    
    n = num_samples(X);
    result = zeros(T_theta, n)
    for i in eachindex(result)
        result[i] = eval_phi(getsample(X, i), theta, Gaussian{Isotropic, T_x, dim})
    end
    return result
end

function eval_phi(X::Matrix{<:Real}, theta::Vector{T_theta}, ::Type{Gaussian{Anisotropic{Aligned}, T_x, dim}}) where {dim, T_theta<:Number, T_x<:Real}
    n = num_samples(X);
    result = zeros(T_theta, n)
    for i in eachindex(result)
        result[i] = eval_phi(getsample(X, i), theta, Gaussian{Anisotropic{Aligned}, T_x, dim})
    end
    return result
end

# Functions for getting the dimension of a basis function
@inline function dimension(::Type{<:BasisFunction{dim}}) where {dim}
    return dim 
end

@inline function dimension(phi::BasisFunction)
    return dimension(typeof(phi))
end

# Functions for getting the number of parameters 
@inline function size(::Type{Gaussian{Isotropic, T_x, dim}}) where {dim, T_x<:Real}
    return dim + 1
end

@inline function size(::Type{Gaussian{Anisotropic{Aligned}, T_x, dim}}) where {dim, T_x<:Real}
    return 2*dim
end

# Functions for constructing RBFN
struct RBFN{T_phi, T_y} 
    N::Integer
    a0::T_y
    a::Vector{T_y}
    phi::Vector{T_phi}

    function RBFN(a0::T_y, a::Vector{T_y}, phi::Vector{T_phi}) where {T_y<:Number, T_phi<:RBF}
        if length(a) != length(phi)
            throw("SLFA.RBFN: number of weights ($(length(a))) does not match number of basis functions ($(length(phi))).")
        end

        return new{eltype(phi), T_y}(length(a), a0, a, phi)
    end
end

function RBFN(Theta::Matrix{T_theta}, T_phi::Type{<:BasisFunction}) where T_theta<:Number
    num_params = size(Theta,2)
    a = Theta[:, num_params-1]
    a0 = sum(Theta[:, num_params])

    phi_all = [ T_phi(Theta[i,:]) for i in axes(Theta,1) ]
    return RBFN(a0, a, phi_all)
end

@inline function dimension(::RBFN{T_phi, T_y}) where {T_phi<:BasisFunction, T_y<:Number}
    return dimension(T_phi)
end

# Functors for RBFNs
function (network::RBFN{T_phi, T_y})(x::Real) where {T_phi<:BasisFunction, T_y<:Number}
    result = network.a0
    for k = 1:network.N
        result += network.a[k]*network.phi[k](x)
    end
    return result
end

function (network::RBFN{T_phi, T_y})(X::Vector{T_x}) where {T_phi<:BasisFunction{1}, T_y<:Number, T_x<:Real}
    n = num_samples(X)
    result = zeros(T_y, n)
    for i in 1:n
        result[i] = network(getsample(X,i))
    end
    return result
end

function (network::RBFN{T_phi, T_y})(x::Vector{T_x}) where {dim, T_phi<:BasisFunction{dim}, T_y<:Number, T_x<:Real}
    result = network.a0 
    for k = 1:network.N
        result += network.a[k]*network.phi[k](x)
    end
    return result
end

function (network::RBFN{T_phi, T_y})(X::Matrix{T_x}) where {dim, T_phi<:BasisFunction{dim}, T_y<:Number, T_x<:Real}
    n = num_samples(X)
    result = zeros(T_y, n)
    for i in 1:n
        result[i] = network(getsample(X,i))
    end
    return result
end

include("train_RBFN.jl")


end # module