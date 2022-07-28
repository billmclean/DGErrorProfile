using OffsetArrays
import DGErrorProfile: D2matrix
import DGErrorProfile.IRK: Gauss2, Gauss3, SystemODE_IRK
import DGErrorProfile.Utils1D: source!
import DGErrorProfile.General_1D: L, max_t, κ, initial_data, source_func
using PyPlot

tbl = Gauss2(Float64)
Cu0, Cf0, Cf1 = 1.0, 1.0, 1.0
f(x, t) = source_func(x, t, Cf0, Cf1)

Nt = 10       # time intervals
Nx = 10       # spatial intervals
Ns = Nx - 1   # spatial degrees of freedom
plot_type = 1 # 1. wireframe  2. surface  3. contour

x = OffsetVector( collect(range(0, L, length=Nx+1)), 0:Nx)

h = L / Nx
A = D2matrix(Nx, h, κ)
u0vec = initial_data.(x[1:Ns], Cu0)

U, t = SystemODE_IRK(A, max_t, u0vec, source!, Nt, tbl, x, f)

figure(1)
Ux = zeros(Nx+1, Nt+1)
for n = 0:Nt
    Ux[2:Nx,n+1] = U[n]
end
maxU = maximum(Ux)
minU = minimum(Ux)
if plot_type == 1
    plot_wireframe(x[0:Nx], t[0:Nt], Ux';
    		       rcount=Nt+1, ccount=Ns)
elseif plot_type == 2
    surf(x[0:Nx], t[0:Nt], Ux'; rcount=Nt+1, ccount=Ns)
else
    contourf(x[0:Nx], t[0:Nt], Ux'; rcount=Nt+1, ccount=Ns)
    colorbar()
end
xlabel("x")
ylabel("t")
