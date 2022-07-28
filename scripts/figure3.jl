using OffsetArrays
using PyPlot
using Printf
import DGErrorProfile: D2matrix, SystemODEdG, evaluate_pcwise_poly, 
		       reconstruction, pcwise_times
import DGErrorProfile.Utils1D: find_approx_soln, Richardson!, soln_error
import DGErrorProfile.General_1D: L, max_t, κ, initial_data, source_func, 
                                  exact_soln
import FractionalTimeDG: reconstruction_pts
import LinearAlgebra: norm

plot_soln = true
plot_error = true
print_table = false
plot_weighted_error = false

nt = Threads.nthreads()
if nt > 1
    @printf("Using %0d threads\n", nt)
end
Cu0, Cf0, Cf1 = 1.0, 1.0, 1.0
u(x, t) = exact_soln(x, t, Cu0, Cf0, Cf1)
u0(x) = initial_data(x, Cu0)
f(x, t) = source_func(x, t, Cf0, Cf1)
r=3
@printf("\tr = %g, Cu0 = %g, Cf0 = %g, Cf1 = %g\n", r, Cu0, Cf0, Cf1)

if plot_soln
    figure(1)
    Nx = 20
    Nt = 20
    x = OffsetVector{Float64}(range(0, L, length=Nx+1), 0:Nx)
    t = OffsetVector{Float64}(range(0, max_t, length=Nt+1), 0:Nt)
    uvals = Float64[ u(x[p],t[n]) for p = 0:Nx, n = 0:Nt ]
    plot_wireframe(t[0:Nt], x[0:Nx], uvals)
    xlabel(L"$t$")
    ylabel(L"$x$")
end

if print_table
    r = 3
    Nx = 500
    M = 5
    pts_per_interval = 10
    τ = range(-1.0, 1.0, length=pts_per_interval)
    nrows = 5
    weighted = true
    if weighted
	@printf("Using weighted errors.\n")
	α = r - 5/4
	αstar = r +1 - 5/4
	αnode = 2r - 1- 5/4
	@printf("\tα: DG %g, Reconstr %g, Nodal %g\n", α, αstar, αnode)
    else
	@printf("Cutting off errors for t < T/4.\n")
    end
    @printf("\n%4s  %6s  %17s  %17s  %17s\n\n", 
	    "Nt", "Nx", "DG error    ", "Reconstr. error ", "Nodal error   ")
    err = Vector{Float64}(undef, nrows)
    err_Ustar = similar(err)
    err_node = similar(err)
    for row = 1:nrows
        local Nt, x, t, u0vec, U, pcwise_t, pcwise_U, pcwise_u, pcwise_err
        local pcwise_t, pcwise_Ustar, pcwise_Ustar_err, jumpU, Ustar
	global Ufine, xfine, tfine, u0vecfine
        Nt = 4 * 2^row
        U, x, t, u0vec = find_approx_soln(Nx, Nt, r, M, u0, f, L, κ, max_t)
	Ufine, xfine, tfine, u0vecfine = find_approx_soln(2Nx, Nt, r, M,
						       u0, f, L, κ, max_t)
	Richardson!(U, Ufine)
	pcwise_err, pcwise_t = soln_error(U, u, x, t, τ)
        Ustar, jumpU = reconstruction(U, u0vec)
        pcwise_Ustar_err, pcwise_t = soln_error(Ustar, u, x, t, τ)
	if weighted
     	    pcwise_wt = min.(pcwise_t, 1.0)
	    err[row] = maximum(pcwise_wt.^α.* pcwise_err)
	    err_Ustar[row] = maximum(pcwise_wt.^αstar .* pcwise_Ustar_err)
	    err_node[row] = maximum(pcwise_wt[end,:].^αnode .* 
	    			    pcwise_Ustar_err[end,:])
        else
            n_min = div(Nt, 4)  # cutoff from t = L/4
            err[row] = maximum(pcwise_err[:,n_min:Nt])
            err_Ustar[row] = maximum(pcwise_Ustar_err[:,n_min:Nt])
  	    err_node[row] = maximum(pcwise_Ustar_err[end,n_min:Nt])
	end
        if row == 1
   	    @printf("%4d &%6d &%9.2e&%7s &%9.2e&%7s &%9.2e\\\\\n", 
		    Nt, Nx, err[row], "", err_Ustar[row], "", err_node[row])
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

if plot_error
    r = 3
    Nx = 500
    Nt = 8
    M = 5
    pts_per_interval = 30
    figure(2)
    U, x, t, u0vec = find_approx_soln(Nx, Nt, r, M, u0, f, L, κ, max_t)
    τ_unif = range(-1.0, 1.0, length=pts_per_interval)
    τ_rec = reconstruction_pts(Float64, r)
    τ = sort!([τ_unif; τ_rec[1:r]])
    println(τ)
    pcwise_err, pcwise_t = soln_error(U, u, x, t, τ)
    pcwise_err_rec, pcwise_t_rec = soln_error(U, u, x, t, τ_rec[1:r])
    Ustar, jumpU = reconstruction(U, u0vec)
    pcwise_Ustar_err, pcwise_t = soln_error(Ustar, u, x, t, τ)
    line1 = semilogy(pcwise_t, pcwise_err, "C0-") 
    line2 = semilogy(pcwise_t, pcwise_Ustar_err, "k:") 
    line3 = semilogy(pcwise_t_rec, pcwise_err_rec, "C1o")
    legend((line1[1], line2[1]), 
	   (L"$||{\bf U}(t)-{\bf u}(t)||_h$", 
	    L"$||{\bf U}_*(t)-{\bf u}(t)||_h$"))
    v = axis()
    v = ( v[1], v[2], minimum(pcwise_err[:]), v[4])
    axis(v)
    grid(true)
    xlabel(L"$t$")
    savefig("figure3.pdf")
end

if plot_weighted_error
    r = 3
    Nx = 500
    Nt = 64
    M = 5
    pts_per_interval = 10
    τ = range(-1.0, 1.0, length=pts_per_interval)
    τ_rec = reconstruction_pts(Float64, r)
    U, x, t, u0 = find_approx_soln(Nx, Nt, r, M)
    Ufine, xfine, tfine, u0fine = find_approx_soln(2Nx, Nt, r, M)
    Richardson!(U, Ufine)
    pcwise_err, pcwise_t = soln_error(U, u, x, t, τ)
    Ustar, jumpU = reconstruction(U, u0)
    pcwise_Ustar_err, pcwise_t = soln_error(Ustar, u, x, t, τ)
    figure(3)
    line1 = semilogy(pcwise_t, pcwise_t.^(r-1.5) .* pcwise_err, "C0-") 
    line2 = semilogy(pcwise_t, pcwise_t.^(r-0.5) .* pcwise_Ustar_err, "k:") 
    line3 = semilogy(pcwise_t[end,:], 
		    pcwise_t[end,:].^(2r-2.5) .* pcwise_Ustar_err[end,:], "C1o")
    legend((line1[1], line2[1], line3[1]), 
	   ("U error", "U* error", "Uⁿ₋ error"))
    v = axis()
    v = ( v[1], v[2], minimum(pcwise_err[:]), v[4])
    axis(v)
    grid(true)
    xlabel(L"$t$")
    savefig("PDE_1D_soln_error.pdf")
end
