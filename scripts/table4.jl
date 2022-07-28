using Printf
import DGErrorProfile.Utils2D: Grid2D, print_table 
import DGErrorProfile.General_2D: max_t, Lx, Ly, κ, initial_data, f, 
                                  u_snapshot!

nt = Threads.nthreads()
if nt > 1
    @printf("Using %0d threads\n", nt)
end

r = 3
Gauss_pts = 5
pts_per_interval = 4
nrows = 5
cutoff = true
Px = 50
Py = 50
gr = Grid2D(Px, Py, Lx, Ly)

print_table(cutoff, r, nrows, initial_data, f, u_snapshot!, gr,
	    κ, max_t, Gauss_pts, pts_per_interval, false)

