module SLFA

# Data structures
export RBF

# Functions

# Abstract data types for RBFs
abstract type RBF_Orientation end
abstract type Rotated <:RBF_Orientation end
abstract type Aligned <:RBF_Orientation end

abstract type RBF_Shape end
abstract type Isotropic   <:RBF_Shape end
abstract type Anisotropic{T<:RBF_Orientation} <:RBF_Shape end;

abstract type RBF_Type end
abstract type Gaussian{T<:RBF_Shape} <: RBF_Type end;


struct RBF{T<:RBF_Type}
    x0
    params

    # Parameter-specific constructors
    function RBF{Gaussian{Anisotropic{Aligned}}}(x0::Vector{<:Real}, w::Vector{<:Real})
        if length(x0) != length(w)
            throw("SLFA.RBF{Gaussian{Anisotropic{Aligned}}}: length of x0 ($(length(x0))) does not match length of w ($(length(w))).")
        end

        return new(x0, (; w))
    end

    function RBF{Gaussian{Isotropic}}(x0::Vector{<:Real}, w::Real)
        if length(x0) == 1
            return new(x0[1], (; w))
        else
            return new(x0, (; w))
        end
    end

    # Non-parameterized constructors
    function RBF(x0::Vector{<:Real}, w::Real )
        return RBF{Gaussian{Isotropic}}(x0, w)
    end

    function RBF(x0::Vector{<:Real}, w::Vector{<:Real}, type=:gaussian)
        if length(w) > 1
            return RBF{Gaussian{Anisotropic{Aligned}}}(x0, w)
        else
            return RBF{Gaussian{Isotropic}}(x0, w)
        end
    end
end


function (rbf::RBF{Gaussian{Isotropic}})(x)
    return exp( -sum( (rbf.params.w .*(x - rbf.x0) .^ 2) ) )
end

function (rbf::RBF{Gaussian{Anisotropic{Aligned}}})(x)
    return exp( -sum( (rbf.params.w .*(x - rbf.x0) .^ 2) ) )
end


end # module