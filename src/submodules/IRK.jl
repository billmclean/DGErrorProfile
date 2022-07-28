module IRK

using OffsetArrays
using Printf
import LinearAlgebra
import LinearAlgebra: diagm, dot
import SparseArrays: spdiagm

struct Tableau{T<:AbstractFloat}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
    descr::String
end

function Gauss2(::Type{T}) where T <: AbstractFloat
    three = parse(T, "3")
    β = sqrt(three) / 6
    quarter = one(T) / 4
    half = one(T) / 2
    A = [ quarter   (quarter-β) 
         (quarter+β) quarter   ]
    b = [ half, half ]
    c = [ (half-β), (half+β) ]
    tbl = Tableau(A, b, c, "Gauss2")
    return tbl
end

function Gauss3(::Type{T}) where T <: AbstractFloat
    two = parse(T, "2")
    four = parse(T, "4")
    five = parse(T, "5")
    half = one(T) / 2
    root15 = sqrt(parse(T, "15"))
    A = [ five/36            (two/9-root15/15)  (five/36-root15/30)
         (five/36+root15/24)  two/9             (five/36-root15/24)
         (five/36+root15/30) (two/9+root15/15)   five/36 ]
    b = [ five/18, four/9, five/18 ]
    c = [ (half-root15/10), half, (half+root15/10) ]
    tbl = Tableau(A, b, c, "Gauss2")
    return tbl
end

function RadauIIA(::Type{T}, nstages::Integer) where T <: AbstractFloat
    if nstages == 1
	A = [one(T);;]
	b = [ one(T) ]
	c = [ one(T) ]
    elseif nstages == 2
	three = parse(T, "3")
	four = parse(T, "4")
	five = parse(T, "5")
	A = [ five/12  -one(T)/12
	      3/four       1/four ]
	b = [ 3/four, 1/four ]
	c = [ 1/three, one(T) ]
    elseif nstages == 3
	six = parse(T, "6")
	rtsix = sqrt(six)
	A = [   (88-7*rtsix)/360   (296-169*rtsix)/1800  (-2+3*rtsix)/225
	     (296+169*rtsix)/1800     (88+7*rtsix)/360   (-2-3*rtsix)/225
   	          (16-rtsix)/36         (16+rtsix)/36         one(T)/9    ]
	b = [ (16-rtsix)/36, (16+rtsix)/36, one(T)/9 ]
	c = [ (4-rtsix)/10, (4+rtsix)/10, one(T) ]
    else
	error("Not implemented: nstages must be 1, 2 or 3.")
    end
    tbl = Tableau(A, b, c, "RadauIIA")
    return tbl
end

function ODE_IRK(λ::T, tmax::T, f::Function, u0::T, 
        Nt::Integer, tbl::Tableau{T}) where T <: AbstractFloat
    U = OffsetVector{T}(undef, 0:Nt)
    k = tmax / Nt
    nstages = length(tbl.c)
    t = OffsetVector{T}(range(zero(T), tmax, Nt+1), 0:Nt)
    I = diagm(ones(T, nstages))
    C = I + k * λ * tbl.A
    rhs = Vector{T}(undef, nstages)
    Fact = LinearAlgebra.lu(C)
    U[0] = u0
    fstage = Vector{T}(undef, nstages)
    for n = 1:Nt
        for i = 1:nstages
            fstage[i] = f(t[n-1] + tbl.c[i]*k)
        end
        for i = 1:nstages
            rhs[i] = U[n-1] + k * dot(tbl.A[i,:], fstage)
        end
        stageU = Fact \ rhs
        U[n] = U[n-1] + k * dot(tbl.b, fstage - λ * stageU)
    end
    return U, t
end

function SystemODE_IRK(A::AbstractMatrix{T}, max_t::T, u0vec::Vector{T}, 
        source!::Function, Nt::Integer, tbl::Tableau{T}, 
        xparams...) where T <: AbstractFloat
    t = OffsetVector{T}(range(zero(T), max_t, Nt+1), 0:Nt)
    Ns = size(A, 1)
    k = max_t / Nt
    nstages = size(tbl.A, 1) 
    I = spdiagm(ones(T, nstages*Ns))
    C = I + k * kron(tbl.A, A)
    Fact = LinearAlgebra.lu(C)
    U = OffsetVector{Vector{T}}(undef, 0:Nt)
    for n = 0:Nt
	U[n] = Array{T}(undef, Ns)
    end
    U[0] .= u0vec
    rhs = Vector{T}(undef, nstages*Ns)
    fvec = Vector{Vector{T}}(undef, nstages)
    for n = 1:Nt
        for i = 1:nstages
    	    fvec[i] = Array{T}(undef, Ns)
	    source!(fvec[i], t[n-1] + tbl.c[i] * k, xparams...)
        end
	lo, hi = 1, Ns
	for i = 1:nstages
	    rhs[lo:hi] = U[n-1]
	    for j = 1:nstages
		rhs[lo:hi] += ( k * tbl.A[i,j] ) * fvec[j]
	    end
	    lo += Ns
	    hi += Ns
	end
        ξ = Fact \ rhs
	U[n] .= U[n-1]
	lo, hi = 1, Ns
	for j = 1:nstages
	    U[n] .+= ( k * tbl.b[j] ) * ( fvec[j] - A * ξ[lo:hi] )
	    lo += Ns
	    hi += Ns
	end
    end

    return U, t
end

function Richardson!(U::OffsetVector{Vector{T}}, 
	             Ufine::OffsetVector{Vector{T}}) where T <: AbstractFloat
    Nt = length(U) - 1
    Threads.@threads for n = 0:Nt
        Ns = length(U[n])
        for p = 1:Ns
            U[n][p] = Ufine[n][2p] + ( Ufine[n][2p] - U[n][p] )/3
        end
    end
end

end # module
