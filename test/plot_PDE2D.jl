import DGErrorProfile.Utils2D: Grid2D, find_approx_soln, soln_error,
                              evaluate_reference_soln, Richardson!
import DGErrorProfile: evaluate_pcwise_poly
import DGErrorProfile.No_Source_2D: Lx, Ly, max_t, κ, initial_data, f, 
                                    u_snapshot!
using Printf
using PyPlot

r = 3           # polynomial degree r-1 in time
Nt = 5          # time intervals
Px, Py = 100, 160 # spatial intervals
M = 5           # Gauss points
plot_type = 1   # 1. wireframe  2. surface  3. contour

gr = Grid2D(Px, Py, Lx, Ly)
U, x, t, u0vec = find_approx_soln(gr, Nt, r, M, initial_data, f, κ, max_t)
gr_fine = Grid2D(2Px, 2Py, Lx, Ly)
Ufine, xfine, tfine, u0finevec = find_approx_soln(gr_fine, Nt, r, M, 
                                                  initial_data, f, κ, max_t)
Richardson!(U, Ufine, gr)
pts_per_interval = 50
τ = range(-1.0, 1.0, length=pts_per_interval)
pcwise_err, pcwise_t = soln_error(U, u_snapshot!, gr, t, τ)

figure(1)
X = [ gr.x[p] for p = 0:Px, q = 0:Py ]
Y = [ gr.y[q] for p = 0:Px, q = 0:Py ]
E = zeros(Px+1, Py+1)
m = div(pts_per_interval,2)
n = div(Nt,2)
pcwise_U = evaluate_pcwise_poly(U, τ)
pcwise_U_mat = reshape(pcwise_U, Px-1, Py-1, pts_per_interval, Nt)
pcwise_u = evaluate_reference_soln(u_snapshot!, gr, pcwise_t)
E[2:Px,2:Py] = pcwise_U_mat[:,:,m,n] - pcwise_u[:,:,m,n]
contourf(X, Y, E)
colorbar()
xlabel(L"$x$")
ylabel(L"$y$")
str = @sprintf("Error at time %g", pcwise_t[m,n])
title(str)
show()

figure(2)
plot(pcwise_t, pcwise_err)
xlabel(L"$t$")
grid(true)
