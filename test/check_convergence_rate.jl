using OffsetArrays
using Printf
import DGErrorProfile: D2matrix, SystemODEdG, pcwise_times, evaluate_pcwise_poly
import DGErrorProfile.Zero_IC_1D: L, max_t, κ, initial_data, f, u
import FractionalTimeDG: reconstruction_pts

r = 2
Nx = 800
h = L / Nx
Ns = Nx - 1
pts_per_interval = 5 # samples per time interval
M = 5

A = D2matrix(Nx, h, κ)
x = OffsetVector{Float64}( collect(range(0, L, length=Nx+1)), 0:Nx)
u0 = initial_data.(x[1:Ns])

function source!(fvec, t)
    Ns = length(fvec)
    for p = 1:Ns
        fvec[p] = f(x[p], t)
    end
end

function evaluate_soln(u, x, pcwise_t)
    pts_per_interval, Nt = size(pcwise_t)
    Nx = length(x) - 1
    Ns = Nx - 1
    pcwise_u = Array{Float64}(undef, Ns, pts_per_interval, Nt)
    for n = 1:Nt
	for m = 1:pts_per_interval
	    pcwise_u[:,m,n] = u.(x[1:Ns], pcwise_t[m,n])
	end
    end
    return pcwise_u
end

nrows = 4
τ = range(-1.0, 1.0, length=pts_per_interval)
τ_rec = reconstruction_pts(Float64, r)
err = Vector{Float64}(undef, nrows)
err_rec = similar(err)
for row = 1:nrows
    Nt = 4 * 2^row
    U, t = SystemODEdG(A, max_t, u0, source!, Nt, r, M)
    pcwise_t = pcwise_times(t, τ)
    pcwise_U = evaluate_pcwise_poly(U, τ)
    pcwise_u = evaluate_soln(u, x, pcwise_t)
    err[row] = maximum(abs, pcwise_U-pcwise_u)
    pcwise_t_rec = pcwise_times(t, τ_rec[1:r])
    pcwise_U_rec = evaluate_pcwise_poly(U, τ_rec[1:r])
    pcwise_u_rec = evaluate_soln(u, x, pcwise_t_rec)
    err_rec[row] = maximum(abs, pcwise_U_rec-pcwise_u_rec)
    if row == 1
	@printf("%6d  %6d  %10.2e  %8s  %10.2e\n", 
		Nt, Nx, err[row], "", err_rec[row])
    else
	rate = log2(err[row-1]/err[row])
	rate_rec = log2(err_rec[row-1]/err_rec[row])
        @printf("%6d  %6d  %10.2e  %8.4f  %10.2e  %8.4f\n", 
		Nt, Nx, err[row], rate, err_rec[row], rate_rec)
    end
end
