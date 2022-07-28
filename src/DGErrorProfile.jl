module DGErrorProfile

import FractionalTimeDG: legendre_polys!, reconstruction_pts, coef_G, coef_K
import OffsetArrays: OffsetVector
import SparseArrays: SparseMatrixCSC, spdiagm
import LinearAlgebra: diagm, lu, SymTridiagonal, ldiv!
import LinearAlgebra.BLAS: axpy!, scal!
import GaussQuadrature
using ArgCheck

function max_order(U::Vector{Vector{T}}) where T <: AbstractFloat
    N = length(U)
    r = length(U[1])
    for n = 2:N
	rn = max(r, length(U[n]))
    end
    return r
end

function max_order(U::Vector{Array{T}}) where T <: AbstractFloat
    N = length(U)
    r = size(U[1], 1)
    for n = 2:N
	rn = max(r, size(U[n], 1))
    end
    return r
end

function pcwise_times(t::OffsetVector{T}, τ::AbstractVector{T}
    ) where T <: AbstractFloat
    Nt = length(t) - 1
    pts_per_interval = length(τ)
    pcwise_t = Matrix{T}(undef, pts_per_interval, Nt)
    for n = 1:Nt, m = 1:pts_per_interval
        pcwise_t[m,n] = ( (1-τ[m])*t[n-1] + (1+τ[m])*t[n] ) / 2
    end
    return pcwise_t
end

function evaluate_pcwise_poly(U::Vector{Vector{T}}, τ::AbstractVector{T}
    ) where T <: AbstractFloat
    Nt = length(U)
    pts_per_interval = length(τ)
    pcwise_U = Matrix{T}(undef, pts_per_interval, Nt)
    r = max_order(U)
    P = Matrix{T}(undef, r, pts_per_interval)
    legendre_polys!(P, τ)
    for n = 1:Nt
	rn = length(U[n])
        for m = 1:pts_per_interval
            s = zero(T)
            for j = 1:rn
                s += U[n][j] * P[j,m]
            end
            pcwise_U[m,n] = s
        end
    end
    return pcwise_U
end

function evaluate_pcwise_poly(U::Vector{Array{T}}, τ::AbstractVector{T}
    ) where T <: AbstractFloat
    Nt = length(U)
    Ns, r = size(U[1])
    pts_per_interval = length(τ)
    P = Matrix{T}(undef, r, pts_per_interval)
    legendre_polys!(P, τ)
    pts_per_interval = length(τ)
    P = Matrix{T}(undef, r, pts_per_interval)
    legendre_polys!(P, τ)
    pcwise_t = Matrix{T}(undef, pts_per_interval, Nt)
    pcwise_U = Array{T}(undef, Ns, pts_per_interval, Nt)
    fill!(pcwise_U, zero(T))
    Threads.@threads for n = 1:Nt
        Ns, rn = size(U[n])
	for m = 1:pts_per_interval
	    for j = 1:rn 
		for p = 1:Ns
		    pcwise_U[p,m,n] += U[n][p,j] * P[j,m]
                end
	    end
	end
    end
    return pcwise_U
end

function superconvergence_pts(t::OffsetVector{T}, 
	r::Integer) where T <: AbstractFloat
    τ = reconstruction_pts(T, r)
    Nt = length(t) - 1
    super_pts = Vector{T}(undef, r*Nt)
    j = 1
    for n = 1:Nt
	for i = 1:r
	    super_pts[j] = ( (1-τ[i]) * t[n-1] + (1+τ[i]) * t[n] ) / 2
	    j += 1
	end
    end
    return super_pts
end

"""
    SystemODEdG(A, max_t, u0, source!, Nt, r, Gauss_pts, xparams...)
"""
function SystemODEdG(A::AbstractMatrix{T}, max_t::T, u0vec::Vector{T},
	source!::Function, Nt::Integer, r::Integer, Gauss_pts::Integer,
        xparams...) where T <: AbstractFloat
    Ns = size(A, 1) # spatial degrees of freedom
    G = coef_G(T, r)
    H = diagm( T[ one(T)/(2j-1) for j=1:r ] )
    I = spdiagm(ones(T, Ns))
    k = max_t / Nt
    C = kron(G, I) + k * kron(H, A)
    Fact = lu(C)

    U = Vector{Array{T}}(undef, Nt)
    storage = Array{T}(undef, Ns, r, Nt)
    for n = 1:Nt
	U[n] = view(storage, 1:Ns, 1:r, n)
    end
    fvec = Vector{T}(undef, Ns)
    rhs_integral = Matrix{T}(undef, Ns, r)
    τ, wτ = GaussQuadrature.legendre(T, Gauss_pts)
#    Replace by the following to reproduce Radau IIA IRK scheme.
#    M = r
#    τ, wτ = GaussQuadrature.legendre(T, Gauss_pts, GaussQuadrature.right)
    P = Matrix{T}(undef, r, Gauss_pts)
    legendre_polys!(P, τ)
    t = OffsetVector{T}(collect(range(0, max_t, length=Nt+1)), 0:Nt)

    # accumlate the rhs in U[1]
    even = true
    for i = 0:r-1
	if even
	    U[1][:,i+1] = u0vec
	else
	    U[1][:,i+1] = -u0vec
	end
	even = !even
    end
    get_rhs_integral!(rhs_integral, fvec, source!, (t[0],t[1]), τ, wτ, P, 
                      xparams...)
    U[1] .+= rhs_integral 
    rhs = reshape(U[1], Ns*r)
    ldiv!(Fact, rhs)  # overwrite rhs with C \ rhs thereby computing U[1]
    Usum = Vector{T}(undef, Ns)
    for n = 2:Nt
	fill!(Usum, zero(T))
	for j = 0:r-1
	    Usum .+= U[n-1][:,j+1]
	end
	even = true
	for i = 0:r-1
	    if even
		U[n][:,i+1] = Usum
	    else
		U[n][:,i+1] = -Usum
	    end
	    even = !even
	end
        get_rhs_integral!(rhs_integral, fvec, source!, (t[n-1],t[n]), τ, wτ, P,
                         xparams...)
        U[n] .+= rhs_integral 
        rhs = reshape(U[n], Ns*r)
	ldiv!(Fact, rhs)  # overwrite rhs with C \ rhs, updates U[n]
    end
    return U, t
end

"""
    get_rhs_integral!(rhs_integral, fvec, source!, In, τ, wτ, P, xparams...)

Evaluates the integrals

    rhs_integral[k,i] = ∫ fₖ(t) pₙᵢ(t) dt 

over the interval `In = (tₙ₋₁,tₙ)` using the (Gauss) quadrature rule with 
points `τ` and weights `wτ` for the interval `(-1,1)`.  The matrix of
Legendre polynomial entries `P[j+1,m] = Pⱼ(τₘ)` is computed by calling
`legendre_polys!(P, τ)`.  The argument `source!` is a function that
computes the vector `f(t)` via a call `source!(fvec, t, xparams...)`.
"""
function get_rhs_integral!(rhs_integral::Matrix{T}, fvec::Vector{T},
	source!::Function, In::Tuple{T,T}, τ::Vector{T}, wτ::Vector{T}, 
	P::Matrix{T}, xparams...) where T <: AbstractFloat
    r, M = size(P)
    Ns = length(fvec)
    tnm1, tn = In
    fill!(rhs_integral, zero(T))
    for m = 1:M
	tm = ( (1-τ[m]) * tnm1 + (1+τ[m]) * tn ) / 2
	for i = 1:r
	    source!(fvec, tm, xparams...)
	    rhsi = view(rhs_integral, 1:Ns, i)
	    axpy!(wτ[m]*P[i,m], fvec, rhsi)
	    # rhs_integral[:,i] .+= ( wτ[m] * P[i,m] ) * fvec
	end
    end
    k = tn - tnm1
    scal!(k/2, rhs_integral)
end

D2matrix(P, h, κ) = D2matrix(Float64, P, h, κ)

function D2matrix(::Type{T}, P, h::T, κ::T) where T <: Number
    dv = Vector{T}(undef, P-1)
    ev = Vector{T}(undef, P-2)
    fill!(dv, 2κ/h^2)
    fill!(ev, -κ/h^2)
    return spdiagm(-1 => ev, 0 => dv, 1 => ev)
end

#function refsoln(x::T, t::T, uhat::Function, Nq::Integer
#    ) where T <: AbstractFloat
#    return Bromwich_integral(t, z -> uhat(x, z), Nq)
#end

"""
    Bromwich_integral(t, F, N)

Evaluates the inverse Laplace transform of `F(z)` by approximating
`1/(2πi)` times the integral of `exp(zt)F(z) dz` along a
Hankel contour, assuming `F(z)` is analytic in the cut plane
`|arg(z)|<π` and `F(conj(z)) = conj(F(z))`.
"""
function Bromwich_integral(t::Float64, F::Function, N::Integer)
    μ = 4.492075 * N / t
    ϕ = 1.172104
    h = 1.081792 / N
    z0, w0 = integration_contour(0.0, μ, ϕ)
    Σ = real( w0 * exp(z0*t) * F(z0) ) / 2
    for n = 1:N
        zn, wn = integration_contour(n*h, μ, ϕ)
        Σ += real( wn * exp(zn*t) * F(zn) )
    end
    return h * Σ
end

function integration_contour(u, μ, ϕ) 
    z = μ * ( 1 + sin(Complex(-ϕ, u)) )
    w = (μ/π) * cos(Complex(-ϕ, u))
    return z, w
end

function reconstruction(U::Vector{Array{Float64}}, u0::Vector{Float64})
    Nt = length(U)
    Ns, r = size(U[1])
    Uhat = Vector{Array{Float64}}(undef, Nt)
    for n = 1:Nt
	Uhat[n] = Array{Float64}(undef, Ns, r+1)
    end
    jumpU = OffsetVector{Vector{Float64}}(undef, 0:Nt-1)
    for n = 0:Nt-1
	jumpU[n] = Vector{Float64}(undef, Ns)
    end
    pow = OffsetVector{Float64}(undef, 0:r)
    pow[0:2:end] .=  1.0
    pow[1:2:end] .= -1.0
    U_left  = zeros(Ns)
    U_right = zeros(Ns)
    for j = 1:r
        U_left .+= pow[j-1] * U[1][:,j]
        U_right .+= U[1][:,j]
    end
    jumpU[0] .= U_left - u0
    Uhat[1][:,1:r] .= U[1][:,1:r]
    Uhat[1][:,r] .+= pow[r] * jumpU[0] / 2
    Uhat[1][:,r+1] .= - pow[r] * jumpU[0] / 2
    for n = 2:Nt
        fill!(U_left, 0.0)
        for j = 1:r
  	    U_left .+= pow[j-1] * U[n][:,j]
        end
        jumpU[n-1] .= U_left - U_right
	Uhat[n][:,1:r] .= U[n][:,1:r]
	Uhat[n][:,r]  .+=  pow[r] * jumpU[n-1] / 2
	Uhat[n][:,r+1] .= -pow[r] * jumpU[n-1] / 2
        fill!(U_right, 0.0)
        for j = 1:r
	    U_right .+= U[n][:,j]
        end
    end
    return Uhat, jumpU
end

include("submodules/Utils1D.jl")
include("submodules/Utils2D.jl")
include("submodules/IRK.jl")
include("submodules/Examples.jl")

end # module
