module Toy_ODE

T = Float64
#T = BigFloat

λ = 1 / parse(T, "2")
u0 = one(T)
ampl = one(T)
f(t) = ampl * cos(π*t)
c =  λ^2 + big(π)^2
u(t) = ( ( u0 - ampl * λ / c ) * exp(-λ*t)
        + ampl * ( λ * cos(π*t) + π * sin(π*t) ) / c )
max_t = parse(T, "2")

end # module

module No_Source_1D

L = 1.0
max_t = 2.0
κ = (L/π)^2   # ensure smallest decay constant equals 1

initial_data(x) = sin(π*x/L)
f(x, t) = 0.0
u(x, t) = exp(-t) * sin(π*x/L)

end # module

module Zero_IC_1D

const L = 2.0
const max_t = 2.0
const κ = (L/π)^2   # Ensure smallest eigenvalue of -κ(d/dx)^2 equals 1

initial_data(x) = 0.0
f(x, t) = sin((π/L)*x)
u(x, t) = (1 - exp(-t)) * sin((π/L)*x)

end # module

module General_1D

import ..DGErrorProfile: Bromwich_integral

const L = 2.0
const max_t = 2.0
const κ = (L/π)^2

initial_data(x, Cu0) = Cu0 * x * (L-x)
source_func(x, t, Cf0, Cf1) = ( Cf0 + Cf1 * t ) * exp(-t)

function uhat(x::T, z::Complex{T}, κ::T,
	Cu0::T, Cf0::T, Cf1::T) where T <: AbstractFloat
    ω = sqrt(z/κ)
    ρ1(x) = ( ω * x * (L-x) - 2/ω ) * cosh(ω*x) + (2x-L) * sinh(ω*x) + 2 / ω
    ρ2(x) = cosh(ω*x) - 1
    rzp1 = 1 / (z+1)
    fhat = Cf0 * rzp1 + Cf1 * rzp1^2 
    return ( (Cu0/ω) * ( ρ1(x) * sinh(ω*(L-x)) + ρ1(L-x) * sinh(ω*x) )
           + fhat * ( ρ2(x) * sinh(ω*(L-x)) + ρ2(L-x) * sinh(ω*x) )
          ) / ( z * sinh(ω*L) )
end

Nq = 14

function exact_soln(x, t, Cu0, Cf0, Cf1) 
    if t<10*eps(Float64)
	return initial_data(x, Cu0)  
    else
	return Bromwich_integral(t, z -> uhat(x, z, κ, Cu0, Cf0, Cf1), Nq) 
    end
end

end # module

module No_Source_2D

import ..DGErrorProfile.Utils2D: Grid2D

Lx = 1.0
Ly = 1.5
max_t = 2.0
κ = 1 / ( (π/Lx)^2 + (π/Ly)^2 )   # ensure smallest decay constant equals 1

initial_data(x, y) = sin((π/Lx)*x) * sin((π/Ly)*y)
f(x, y, t) = 0.0
u(x, y, t) = exp(-t) * sin((π/Lx)*x) * sin((π/Ly)*y)

function u_snapshot!(umat::AbstractMatrix{T}, t::T, gr::Grid2D{T}
	             ) where T <: AbstractFloat
    Px, Py, x, y = gr.Px, gr.Py, gr.x, gr.y
    for q = 1:Py-1, p = 1:Px - 1
	umat[p,q] = u(x[p], y[q], t)
    end
end

end # module

module Zero_IC_2D

import ..DGErrorProfile.Utils2D: Grid2D

Lx = 1.0
Ly = 1.5
max_t = 2.0
κ = 1 / ( (π/Lx)^2 + (π/Ly)^2 )   # ensure smallest decay constant equals 1

initial_data(x, y) = 0.0
f(x, y, t) = (1 - exp(-t/2)/2) * sin((π/Lx)*x) * sin((π/Ly)*y)
u(x, y, t) = (1 - exp(-t/2)) * sin((π/Lx)*x) * sin((π/Ly)*y)

function u_snapshot!(umat::AbstractMatrix{T}, t::T, gr::Grid2D{T}
	             ) where T <: AbstractFloat
    Px, Py, x, y = gr.Px, gr.Py, gr.x, gr.y
    for q = 1:Py-1, p = 1:Px - 1
	umat[p,q] = u(x[p], y[q], t)
    end
end

end # module

module General_2D

import ..DGErrorProfile: integration_contour
import ..DGErrorProfile.Utils2D: Poisson_matrix, Grid2D
import SparseArrays: spdiagm, lu, lu!, ldiv!

const max_t = 2.0
const Lx = 2.0
const Ly = 2.0
const κ = 1/( (π/Lx)^2 + (π/Ly)^2 )

initial_data(x, y) = x * (Lx-x) * y * (Ly-y)
f(x, y, t) = ( 1 + t ) * exp(-t)

function evaluate_g!(gvec::Vector{Complex{T}}, z::Complex{T}, gr::Grid2D{T}
               ) where T <: AbstractFloat
    Px, Py, x, y = gr.Px, gr.Py, gr.x, gr.y
    gmat = reshape(gvec, Px-1, Py-1)
    fhat = 1/(z+1) + 1/(z+1)^2
    for q = 1:Py-1, p = 1:Px-1
	gmat[p,q] = initial_data(x[p], y[q]) + fhat 
    end
end

function u_snapshot!(umat::AbstractMatrix{T}, t::T, gr::Grid2D{T}
                  ) where T <: AbstractFloat
    Px, Py = gr.Px, gr.Py
    if t < 10eps(T)
	x, y = gr.x, gr.y
	for q = 1:Py-1, p = 1:Px-1
	    umat[p,q] = initial_data(x[p], y[q])
	end
	return
    end
    # Contour integration parameters
    Nq = 14
    μ = 4.492075 * Nq / t
    ϕ = 1.172104
    h = 1.081792 / Nq

    M = (Px-1)*(Py-1)
    A = Poisson_matrix(gr, κ)
    I = spdiagm(ones(T, M))

    uh = reshape(umat, M)
    term = Vector{Complex{T}}(undef, M)
    z0, w0 = integration_contour(0.0, μ, ϕ)
    zIpA = z0 * I + A
    F = lu(zIpA)
    rhs = term
    evaluate_g!(rhs, z0, gr)  # set rhs = g(z0)
    ldiv!(F, rhs)             # overwite term with (z0 I + A) \ rhs
    term .= (w0 * exp(z0*t)) * term
    uh .= real.(term) / 2
    for n = 1:Nq
        zn, wn = integration_contour(n*h, μ, ϕ)
        zIpA .= zn * I .+ A
        lu!(F, zIpA)
        rhs = term
        evaluate_g!(rhs, zn, gr)
        ldiv!(F, rhs)
        term .= (wn * exp(zn*t)) * term
        uh .+= real.(term)
    end
    uh .= h * uh
end

end # module
