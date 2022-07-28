import FractionalTimeDG
using PyPlot

rmax = 5
M = 201
τ = range(-1, 1, length=M)
Ψ = zeros(rmax, M)
FractionalTimeDG.legendre_polys!(Ψ, τ)

figure(1)
fmt = [":", "--", "-.", "-"]
for r = 1:rmax-1
    plot(τ, Ψ[r+1,:]-Ψ[r,:], fmt[r])
end
grid(true)
xlabel(L"$\tau$")
s = LaTeXString[latexstring("\$r=", r, "\$") for r = 1:rmax-1]
legend(s)
savefig("figure1.pdf")

