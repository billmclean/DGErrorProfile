module Utils1D

import ..DGErrorProfile: D2matrix, SystemODEdG, pcwise_times, 
			 evaluate_pcwise_poly, reconstruction
import LinearAlgebra: norm
import Printf: @printf
using OffsetArrays

function source!(fvec::Vector{T}, t::T, 
        x::OffsetVector{T}, f::Function) where T <: AbstractFloat
    Ns = length(fvec)
    for p = 1:Ns
	fvec[p] = f(x[p], t)
    end
end

function evaluate_ref_soln(u::Function, x::OffsetVector{T}, 
	pcwise_t::Matrix{T}) where T <: AbstractFloat
    pts_per_interval, Nt = size(pcwise_t)
    Nx = length(x) - 1
    Ns = Nx - 1
    pcwise_u = Array{T}(undef, Ns, pts_per_interval, Nt)
    Threads.@threads for n = 1:Nt
        for m = 1:pts_per_interval
	    pcwise_u[:,m,n] = u.(x[1:Ns], pcwise_t[m,n])
        end
    end
    return pcwise_u
end

function find_approx_soln(Nx::Integer, Nt::Integer, r::Integer, M::Integer, 
	initial_data::Function, f::Function, 
	L::T, κ::T, max_t::T) where T <: AbstractFloat
    h = L / Nx
    A = D2matrix(Nx, h, κ)
    x = OffsetVector{Float64}( collect(range(0, L, length=Nx+1)), 0:Nx)
    Ns = Nx - 1
    u0vec = initial_data.(x[1:Ns])
    U, t = SystemODEdG(A, max_t, u0vec, source!,
                       Nt, r, M, x, f)
    return U, x, t, u0vec
end

function soln_error(U::Vector{Array{T}}, u::Function, 
	x::OffsetVector{T}, t::OffsetVector{T}, τ::AbstractVector{T}
	) where T <: AbstractFloat
    pcwise_t = pcwise_times(t, τ)
    pcwise_U = evaluate_pcwise_poly(U, τ)
    pcwise_u = evaluate_ref_soln(u, x, pcwise_t)
    Ns, pts_per_interval, Nt = size(pcwise_U)
    Nx = Ns + 1
    L = x[Nx]
    h = L / Nx
    sqrt_h = sqrt(h)
    pcwise_err = Array{Float64}(undef, pts_per_interval, Nt)
    for n = 1:Nt
        for m = 1:pts_per_interval
            pcwise_err[m,n] = sqrt_h*norm(pcwise_U[:,m,n]-pcwise_u[:,m,n])
        end
    end
    return pcwise_err, pcwise_t
end

"""
    Richardson!(U, Ufine)

Updates `U` by performing one step of Richardson extrapolation, assuming
the spatial error is proportional to `Δx²`.
"""
function Richardson!(U::Vector{Array{T}}, Ufine::Vector{Array{T}}
    ) where T <: AbstractFloat
    Nt = length(U)
    for n = 1:Nt
        Ns, rn = size(U[n])
        for j = 1:rn, p = 1:Ns
            U[n][p,j] = Ufine[n][2p,j] + ( Ufine[n][2p,j] - U[n][p,j] )/3
        end
    end
end

function print_table(cutoff::Bool, r::Integer, nrows::Integer, 
	u0::Function, f::Function, u::Function,
	L::T, κ::T, max_t::T, Nx::Integer, M::Integer, 
	pts_per_interval::Integer) where T <: AbstractFloat
    τ = range(-1.0, 1.0, length=pts_per_interval)
    if cutoff
        @printf("Cutting off errors for t < T/4.\n")
    else
	@printf("Weighting errors.\n")
        α = r - 5/4
        αstar = r + 1 - 5/4
        αnode = 2r - 1- 5/4
	@printf("\tα = %g, α star = %g, α node = %g\n", α, αstar, αnode)
    end
    @printf("\n%4s  %6s  %17s  %17s  %17s\n\n",
            "Nt", "Nx", "DG error    ", "Reconstr. error ", "Nodal error   ")
    err = Vector{Float64}(undef, nrows)
    err_Ustar = similar(err)
    err_node = similar(err)
    for row = 1:nrows
        Nt = 4 * 2^row
        U, x, t, u0vec = find_approx_soln(Nx, Nt, r, M, u0, f,
                                        L, κ, max_t)
        Ufine, xfine, tfine, u0finevec = find_approx_soln(2Nx, Nt, r, M,
                                        u0, f, L, κ, max_t)
        Richardson!(U, Ufine)
        pcwise_err, pcwise_t = soln_error(U, u, x, t, τ)
        Ustar, jumpU = reconstruction(U, u0vec)
        pcwise_Ustar_err, pcwise_t = soln_error(Ustar, u, x, t, τ)
	if cutoff
            n_min = div(Nt, 4)  # cutoff from t = L/4
            err[row] = maximum(pcwise_err[:,n_min:Nt])
            err_Ustar[row] = maximum(pcwise_Ustar_err[:,n_min:Nt])
            err_node[row] = maximum(pcwise_Ustar_err[end,n_min:Nt])
	else
            pcwise_wt = min.(pcwise_t, 1.0)
	    err[row] = maximum(pcwise_wt.^α.* pcwise_err)
	    err_Ustar[row] = maximum(pcwise_wt[:,2:Nt].^αstar .* 
				     pcwise_Ustar_err[:,2:Nt])
	    err_node[row] = maximum(pcwise_wt[end,:].^αnode .*
                                    pcwise_Ustar_err[end,:])

	end
        if row == 1
            @printf("%4d &%6d &%9.2e&%7s &%9.2e&%7s &%9.2e&%7s\\\\\n",
                    Nt, Nx, err[row], "", err_Ustar[row], "", err_node[row],
		    "")
        else
            rate = log2(err[row-1]/err[row])
            rate_Ustar = log2(err_Ustar[row-1]/err_Ustar[row])
            rate_node = log2(err_node[row-1]/err_node[row])
            @printf("%4d &%6d &%9.2e&%7.3f &%9.2e&%7.3f &%9.2e&%7.3f\\\\\n",
                    Nt, Nx, err[row], rate, err_Ustar[row], rate_Ustar,
                    err_node[row], rate_node)
        end
    end
end

end # module
