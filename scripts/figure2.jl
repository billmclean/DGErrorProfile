import FractionalTimeDG: ODEdG!, jumps, dG_error_estimator
import DGErrorProfile: superconvergence_pts, pcwise_times, evaluate_pcwise_poly
import DGErrorProfile.Toy_ODE: T, λ, max_t, u0, f, u	

using PyPlot

N = 5   # Number of subintervals on the time axis
r = 4   # Piecewise cubics
Gauss_pts = 4   # Number of Gauss quadrature points per interval
pts_per_interval = 50  # Number of evaluation points per interval
t, U = ODEdG!(λ, max_t, f, u0, N, r, Gauss_pts)
τ = range(-1.0, 1.0, length=pts_per_interval)
pcwise_t = pcwise_times(t, τ)
pcwise_U = evaluate_pcwise_poly(U, τ)
pcwise_u = T[ u(t) for t in pcwise_t ]

figure(1)
line1 = plot(pcwise_t, pcwise_u, "k:")
line2 = plot(pcwise_t, pcwise_U, "c")
legend((line1[1], line2[1]), (L"$u$", L"$U$"))
xlabel(L"$t$", size=14)
ylabel(L"$U$", size=14)
xticks(range(0, max_t, length=N+1))
grid(true)
show()

figure(2)
super_pts = superconvergence_pts(t, r)
JU = jumps(U, t, u0)
pcwise_tx, pcwise_approx_err = dG_error_estimator(JU, t, r, pts_per_interval)
line1 = plot(pcwise_t, pcwise_U-pcwise_u, "C0")
line2 = plot(pcwise_tx, pcwise_approx_err, "C3--")
line3 = plot(super_pts, zeros(size(super_pts)), "oC1", markersize=4)
legend((line1[1], line2[1], line3[1]), 
       (L"$U-u$", L"$U-U_*$", "superconvergence points"))
grid(true)
xticks(range(0, max_t, length=N+1))
xlabel(L"$t$", size=14)
savefig("figure2.pdf")
