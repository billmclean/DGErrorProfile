import Printf: @printf
import FractionalTimeDG: ODEdG!, reconstruction
import DGErrorProfile: evaluate_pcwise_poly, pcwise_times
import DGErrorProfile.Toy_ODE: T, λ, max_t, u0, f, u

nrows = 6
N = 2
r = 4
M = r+1
pts_per_interval = 50
τ = range(-one(T), one(T), length=pts_per_interval)
Uerr = Vector{Float64}(undef, nrows)
Ustarerr = similar(Uerr)
Unodeerr = similar(Uerr)

@printf("ODEdG with r = %0d\n", r)
@printf("%4s  %10s  %8s  %10s  %8s  %10s  %8s\n\n", 
	"N", "max|U-u|", "rate", "max|U*-u|", "rate", "Uⁿ₋-u(tₙ)", "rate")
for row in 1:nrows
    global N
    local t, U, pcwise_t, pcwise_U, pcwise_u, pcwise_Ustar
    N *= 2
    t, U = ODEdG!(λ, max_t, f, u0, N, r, M)
    pcwise_t = pcwise_times(t, τ)
    pcwise_U = evaluate_pcwise_poly(U, τ)
    pcwise_u = T[ u(t) for t in pcwise_t ]
    Ustar = reconstruction(U, u0, r+1)
    pcwise_Ustar = evaluate_pcwise_poly(Ustar, τ)
    Uerr[row] = maximum(abs, pcwise_U - pcwise_u)
    Ustarerr[row] = maximum(abs, pcwise_Ustar-pcwise_u)
    Unodeerr[row] = maximum(abs, pcwise_Ustar[end,:]-pcwise_u[end,:])
    if row == 1
	@printf("%4d &%10.2e &%8s &%10.2e &%8s &%10.2e\\\\\n", 
		N, Uerr[row], "", Ustarerr[row], "", Unodeerr[row])
    else
	Urate = log2(Uerr[row-1]/Uerr[row])
	Ustarrate = log2(Ustarerr[row-1]/Ustarerr[row])
	Unoderate = log2(Unodeerr[row-1]/Unodeerr[row])
        @printf("%4d &%10.2e &%8.3f &%10.2e &%8.3f &%10.2e &%8.3f\\\\\n", 
		N, Uerr[row], Urate, Ustarerr[row], Ustarrate, 
		Unodeerr[row], Unoderate)
    end
end
