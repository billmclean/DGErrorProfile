import DGErrorProfile.IRK: Gauss2, Gauss3, ODE_IRK
import  DGErrorProfile.Toy_ODE: λ, u0, f, u, max_t

using PyPlot
using Printf

#tbl = Gauss2(Float64)
tbl = Gauss3(Float64)
N = 5
U, t = ODE_IRK(λ, max_t, f, u0, N, tbl)

figure(1)
tt = range(0, max_t, length=201)
line1 = plot(tt, u.(tt), "k:")
line2 = plot(t, U, "co")
legend((line1[1], line2[1]), (L"$u$", L"$U^n$"))
xlabel(L"$t$")
xticks(range(0, max_t, length=N+1))
grid(true)
show()

nrows = 6
N = 2
Uerr = Vector{Float64}(undef, nrows)
for row = 1:nrows
    global N, Uerr
    local U, t, rate
    N *= 2
    U, t = ODE_IRK(λ, max_t, f, u0, N, tbl)
    Uerr[row] = maximum(abs, U-u.(t))
    if row == 1
        @printf("%4d  %10.2e\n", N, Uerr[row])
    else
        rate = log2(Uerr[row-1] / Uerr[row])
        @printf("%4d  %10.2e  %8.3f\n", N, Uerr[row], rate)
    end
end
