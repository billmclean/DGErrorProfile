module Utils2D

import ..DGErrorProfile: D2matrix, SystemODEdG, pcwise_times, 
			 evaluate_pcwise_poly, reconstruction
import LinearAlgebra: norm
import SparseArrays: spdiagm, SparseMatrixCSC
using OffsetArrays
using Printf

"""
A `Px-by-Py` grid on the rectangle `[0,Lx] x [0,Ly]`.
"""
struct Grid2D{T<:AbstractFloat}
    Px::Integer
    Py::Integer
    Lx::T
    Ly::T
    Δx::T
    Δy::T
    x::OffsetVector{T}
    y::OffsetVector{T}
end

function Grid2D(::Type{T}, Px, Py, Lx, Ly) where T <: AbstractFloat
    Δx = Lx/Px
    Δy = Ly/Py
    x = OffsetVector(range(0, Lx, Px+1), 0:Px)
    y = OffsetVector(range(0, Ly, Py+1), 0:Py)
    return Grid2D(Px, Py, Lx, Ly, Δx, Δy, x, y)
end

Grid2D(Px, Py, Lx, Ly) = Grid2D(Float64, Px, Py, Lx, Ly)

function Poisson_matrix(gr::Grid2D{T}, κ::T) where T <: AbstractFloat
    Ax = D2matrix(gr.Px, gr.Δx, κ)
    Ay = D2matrix(gr.Py, gr.Δy, κ)
    Ix = spdiagm(ones(gr.Px-1))
    Iy = spdiagm(ones(gr.Py-1))
    A = kron(Iy, Ax) + kron(Ay, Ix)
    return A
end

function source!(fvec::Vector{T}, t::T, 
        gr::Grid2D{T}, f::Function) where T <: AbstractFloat
    Px, Py, x, y = gr.Px, gr.Py, gr.x, gr.y
    fmat = reshape(fvec, (Px-1,Py-1))
    Threads.@threads for q = 1:Py-1 
        for p = 1:Px-1
	    fmat[p,q] = f(x[p], y[q], t)
        end
    end
end

function evaluate_reference_soln(u_snapshot!::Function, gr::Grid2D{T},
	pcwise_t::Matrix{T}) where T <: AbstractFloat
    Px, Py, x, y = gr.Px, gr.Py, gr.x, gr.y
    pts_per_interval, Nt = size(pcwise_t)
    pcwise_u = Array{T}(undef, Px-1, Py-1, pts_per_interval, Nt)
    Threads.@threads for n = 1:Nt
	for m = 1:pts_per_interval
	    umat = view(pcwise_u, :, :, m, n)
	    u_snapshot!(umat, pcwise_t[m,n], gr)
	end
    end
    return pcwise_u
end

function find_approx_soln(gr::Grid2D{T}, Nt::Integer, r::Integer, M::Integer,
	initial_data::Function, f::Function, 
	κ::T, max_t::T) where T <: AbstractFloat
    Px, Py, x, y = gr.Px, gr.Py, gr.x, gr.y
    A = Poisson_matrix(gr, κ)
    u0mat = T[ initial_data(x[p], y[q]) for p=1:Px-1, q=1:Py-1 ]
    Ns = (Px - 1) * (Py - 1)
    u0vec = reshape(u0mat, Ns)
    U, t = SystemODEdG(A, max_t, u0vec, source!, Nt, r, M, gr, f)
    return U, x, t, u0vec
end

function soln_error(U::Vector{Array{T}}, u_snapshot!::Function,
        gr::Grid2D{T}, t::OffsetVector{T}, τ::AbstractVector{T},
	) where T <: AbstractFloat
    Px, Py, Δx, Δy = gr.Px, gr.Py, gr.Δx, gr.Δy
    pcwise_t = pcwise_times(t, τ)
    pcwise_U = evaluate_pcwise_poly(U, τ)
    Ns, pts_per_interval, Nt = size(pcwise_U)
    pcwise_u = evaluate_reference_soln(u_snapshot!, gr, pcwise_t)
    pcwise_uvec = reshape(pcwise_u, Ns, pts_per_interval, Nt)
    nt = Threads.nthreads()
    E = Array{T}(undef, Ns, nt)
    sqrtΔA = sqrt(Δx * Δy)
    pcwise_err = Array{Float64}(undef, pts_per_interval, Nt)
    Threads.@threads for n = 1:Nt 
        tid = Threads.threadid()
        for m = 1:pts_per_interval
            E[:,tid] .= pcwise_U[:,m,n] .- pcwise_uvec[:,m,n]
            pcwise_err[m,n] = sqrtΔA * norm(E[:,tid])
        end
    end
    return pcwise_err, pcwise_t
end

function Richardson!(U::Vector{Array{T}}, Ufine::Vector{Array{T}}, 
        gr::Grid2D{T}) where T <: AbstractFloat
    Nt = length(U)
    Px, Py = gr.Px, gr.Py
    Threads.@threads for n = 1:Nt
        rn = size(U[n], 2)
        U_mat = reshape(U[n], Px-1, Py-1, rn)
        Ufine_mat = reshape(Ufine[n], 2Px-1, 2Py-1, rn)
        for j = 1:rn, q = 1:Py-1, p = 1:Px-1
            U_mat[p,q,j] = ( Ufine_mat[2p,2q,j] + 
                           ( Ufine_mat[2p,2q,j] - U_mat[p,q,j] )/3 )
        end
    end
end

function print_table(cutoff::Bool, r::Integer, nrows::Integer,
        u0::Function, f::Function, u_snapshot!::Function, gr::Grid2D,
        κ::T, max_t::T, Gauss_pts::Integer, pts_per_interval::Integer, 
	extrapolate=true) where T <: AbstractFloat
    τ = range(-1.0, 1.0, length=pts_per_interval)
    if cutoff
        @printf("Cutting off errors for t < T/4.\n")
    else
        @printf("Weighted errors.\n")
        α = r - 5/4
        αstar = r + 1 - 5/4
        αnode = 2r - 1- 5/4
    end
    @printf("\n%4s  %6s  %6s  %17s  %17s  %17s\n\n", "Nt", "Px", "Py", 
            "DG error    ", "Reconstr. error ", "Nodal error   ")
    err = Vector{Float64}(undef, nrows)
    err_Ustar = similar(err)
    err_node = similar(err)
    Px, Py, Lx, Ly = gr.Px, gr.Py, gr.Lx, gr.Ly
    gr_fine = Grid2D(2Px, 2Py, Lx, Ly)
    for row = 1:nrows
        Nt = 4 * 2^row
        U, x, t, u0vec = find_approx_soln(gr, Nt, r, Gauss_pts, u0, f, κ, max_t)
	if extrapolate
            Ufine, xfine, tfine, u0finevec = find_approx_soln(gr_fine, Nt, r, 
                                             Gauss_pts, u0, f, κ, max_t)
            Richardson!(U, Ufine, gr)
	end
        pcwise_err, pcwise_t = soln_error(U, u_snapshot!, gr, t, τ)
        Ustar, jumpU = reconstruction(U, u0vec)
        pcwise_Ustar_err, pcwise_t = soln_error(Ustar, u_snapshot!, gr, t, τ)
        if cutoff
            n_min = div(Nt, 4)  # cutoff from t = T/4
            err[row] = maximum(pcwise_err[:,n_min:Nt])
            err_Ustar[row] = maximum(pcwise_Ustar_err[:,n_min:Nt])
            err_node[row] = maximum(pcwise_Ustar_err[end,n_min:Nt])
        else
            pcwise_wt = min.(pcwise_t, 1.0)
            err[row] = maximum(pcwise_wt.^α.* pcwise_err)
            err_Ustar[row] = maximum(pcwise_wt.^αstar .* pcwise_Ustar_err)
            err_node[row] = maximum(pcwise_wt[end,:].^αnode .*
                                    pcwise_Ustar_err[end,:])

        end
        if row == 1
            @printf("%4d &%6d &%6d &%9.2e&%7s &%9.2e&%7s &%9.2e&%7s\\\\\n",
                    Nt, Px, Py, err[row], "", err_Ustar[row], "", err_node[row],
                    "")
        else
            rate = log2(err[row-1]/err[row])
            rate_Ustar = log2(err_Ustar[row-1]/err_Ustar[row])
            rate_node = log2(err_node[row-1]/err_node[row])
            @printf("%4d &%6d &%6d &%9.2e&%7.3f \
                    &%9.2e&%7.3f &%9.2e&%7.3f\\\\\n",
                    Nt, Px, Py, err[row], rate, err_Ustar[row], rate_Ustar,
                    err_node[row], rate_node)
        end
    end
end

end # module
