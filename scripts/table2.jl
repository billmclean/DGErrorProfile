using Printf
import DGErrorProfile.Utils1D: print_table
import DGErrorProfile.General_1D: L, max_t, κ, initial_data, source_func, 
                                  exact_soln

Cu0, Cf0, Cf1 = 1.0, 1.0, 1.0
u(x, t) = exact_soln(x, t, Cu0, Cf0, Cf1)
u0(x) = initial_data(x, Cu0)
f(x, t) = source_func(x, t, Cf0, Cf1)
r = 3
@printf("\tr = %g, Cu0 = %g, Cf0 = %g, Cf1 = %g\n", r, Cu0, Cf0, Cf1)

nt = Threads.nthreads()
if nt > 1
    @printf("Using %0d threads\n", nt)
end

Nx = 500  # Number of spatial subintervals
Gauss_pts = 5
pts_per_interval = 10
nrows = 5
cutoff = true

print_table(cutoff, r, nrows, u0, f, u,
	    L, κ, max_t, Nx, Gauss_pts, pts_per_interval)
