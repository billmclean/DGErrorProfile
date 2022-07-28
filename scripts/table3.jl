using Printf
import DGErrorProfile.Utils1D: print_table
import DGErrorProfile.General_1D: L, max_t, κ, initial_data, source_func, 
                                  exact_soln

nt = Threads.nthreads()
if nt > 1
    @printf("Using %0d threads\n", nt)
end

r = 3
Nx = 500
Gauss_pts = 5
#pts_per_interval = 10
pts_per_interval = 2
nrows = 5
cutoff = false

Cu0, Cf0, Cf1 = 1.0, 0.0, 0.0
u0(x) = initial_data(x, Cu0)
f_top(x, t) = 0.0
u_top(x, t) = exact_soln(x, t, Cu0, Cf0, Cf1)
@printf("Homogeneous problem: ")
@printf("r = %g, Cu0 = %g, Cf0 = %g, Cf1 = %g\n", r, Cu0, Cf0, Cf1)
print_table(cutoff, r, nrows, u0, f_top, u_top,
	    L, κ, max_t, Nx, Gauss_pts, pts_per_interval)

Cu0, Cf0, Cf1 = 1.0, 1.0, 1.0
f_bottom(x, t) = source_func(x, t, Cf0, Cf1)
u_bottom(x, t) = exact_soln(x, t, Cu0, Cf0, Cf1)
@printf("\n\nInhomogeneous problem: ")
@printf("r = %g, Cu0 = %g, Cf0 = %g, Cf1 = %g\n", r, Cu0, Cf0, Cf1)
print_table(cutoff, r, nrows, u0, f_bottom, u_bottom,
	    L, κ, max_t, Nx, Gauss_pts, pts_per_interval)
