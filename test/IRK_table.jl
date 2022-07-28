using Printf
import DGErrorProfile: D2matrix
import DGErrorProfile.IRK: Gauss2, Gauss3, RadauIIA, SystemODE_IRK, Richardson!
import DGErrorProfile.Utils1D: source!
import OffsetArrays: OffsetVector
import LinearAlgebra: norm

include("../inputs/general_1d.jl")
Cu0, Cf0, Cf1 = 1.0, 0.0, 0.0
@printf("\tCu0 = %g, Cf0 = %g, Cf1 = %g\n", Cu0, Cf0, Cf1)
u(x, t) = exact_soln(x, t, Cu0, Cf0, Cf1)
u0(x) = initial_data(x, Cu0)
f(x, t) = source_func(x, t, Cf0, Cf1)

nt = Threads.nthreads()
if nt > 1
    @printf("Using %0d threads\n", nt)
end

tbl2 = RadauIIA(Float64, 2)
tbl3 = RadauIIA(Float64, 3)
@printf("\tUsing RadauIIA with 2 and 3 stages\n\n")
Nx = 500  # Number of subintervals in the spatial grid
Ns = Nx - 1
h = L / Nx
A = D2matrix(Nx, h, κ)
x = OffsetVector(collect(range(0, L, Nx+1)), 0:Nx)
Nxfine = 2Nx
Nsfine = Nxfine - 1
hfine = L / Nxfine
Afine = D2matrix(Nxfine, hfine, κ)
xfine = OffsetVector(collect(range(0, L, Nxfine+1)), 0:Nxfine)
u0vec = u0.(x[1:Ns])
u0vecfine = u0.(xfine[1:Nsfine])

function L2_error(U, t, h, cutoff)
    max_err = 0.0
    Nt = length(U) - 1
    En = Vector{Float64}(undef, Ns)
    idx = minimum(findall(s -> s ≥ cutoff, t))
    Threads.@threads for n = idx:Nt
   	En .= U[n] - u.(x[1:Ns], t[n])
	err = sqrt(h) * norm(En, 2)
	max_err = max(max_err, err)
    end
    return max_err
end

nrows = 5
Nt = 8
cutoff = L/4

err = zeros(nrows,2)
for row = 1:nrows
    global Nt, err
    local U, Ufine, t
    for (i,tbl) in zip(1:2,[tbl2, tbl3])
        U, t = SystemODE_IRK(A, max_t, u0vec, source!, Nt, tbl, x, f)
        Ufine, t = SystemODE_IRK(Afine, max_t, u0vecfine, source!,
   			         Nt, tbl, xfine, f)
        Richardson!(U, Ufine)
        err[row,i] = L2_error(U, t, h, cutoff)
    end
    if row == 1
	@printf("%5d &%5d &%10.2e &%8s &%10.2e &%8s\\\\\n", 
		Nt, Nx, err[row,1], "", err[row,2], "")
    else
	rate = log2.(err[row-1,:] ./ err[row,:])
	@printf("%5d &%5d &%10.2e &%8.3f &%10.2e &%8.3f\\\\\n", 
		Nt, Nx, err[row,1], rate[1], err[row,2], rate[2])
    end
    Nt *= 2
end
