using OffsetArrays
import DGErrorProfile: D2matrix, SystemODEdG, evaluate_pcwise_poly, pcwise_times
import DGErrorProfile.Utils1D: source!
import DGErrorProfile.General_1D: L, max_t, κ, initial_data, source_func
using PyPlot

Cu0, Cf0, Cf1 = 1.0, 1.0, 1.0
f(x, t) = source_func(x, t, Cf0, Cf1)

r = 3         # polynomial degree r-1 in time
Nt = 3        # time intervals
Nx = 10       # spatial intervals
Ns = Nx - 1   # spatial degrees of freedom
M = 5         # Gauss points
plot_type = 1 # 1. wireframe  2. surface  3. contour

x = OffsetVector{Float64}( collect(range(0, L, length=Nx+1)), 0:Nx)

h = L / Nx
A = D2matrix(Nx, h, κ)
Ns = Nx-1 # spatial degrees of freedom
u0vec = initial_data.(x[1:Ns], Cu0)

U, t = SystemODEdG(A, max_t, u0vec, source!, Nt, r, M, x, f)

pts_per_interval = 5
τ = range(-1.0, 1.0, length=pts_per_interval)
pcwise_t = pcwise_times(t, τ)
pcwise_U = evaluate_pcwise_poly(U, τ)

figure(1)
maxU = maximum(pcwise_U)
minU = minimum(pcwise_U)
for n = 1:Nt
    Un = zeros(Nx+1, pts_per_interval)
    Un[2:Nx,:] = pcwise_U[:,:,n]
    if plot_type == 1
        plot_wireframe(x[0:Nx], pcwise_t[:,n], Un';
    		       rcount=pts_per_interval, ccount=Ns)
    elseif plot_type == 2
        surf(x[0:Nx], pcwise_t[:,n], Un';
             rcount=pts_per_interval, ccount=Ns)
    else
        contourf(x[0:Nx], pcwise_t[:,n], Un', 10; vmin=minU, vmax=maxU)
	if n == 1
	    colorbar()
	end
    end
end
xlabel("x")
ylabel("t")
