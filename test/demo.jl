using StructureFunctions

using YPlot, PyPlot
using LinearAlgebra, OffsetArrays

# Create a structure function for the Kolmogorov law.
r0 = 1.0; # Fried's parameter
Dᵩ = KolmogorovStructFunc(r0);

# Create a support as a circular pupil with a central aperture.
S = [16.5 ≤ sqrt(x^2 + y^2) < 50.5 for x in -51:51, y in -51:51];

# Create a packed lazy covariance.
σ = 0.1; # piston variance
Cᵩ = StructureFunctions.PackedLazyCovariance(Dᵩ, S, σ);

# Display the variance.
v = fill!(similar(Cᵩ.mask, eltype(Cᵩ)), 0);
v[Cᵩ.mask] = var(Cᵩ);
plimg(v; fig=1, title="Variance");
sleep(0.1);

# Cholesky decomposition of the covariance.
LL′ = cholesky(Cᵩ); # Cᵩ = L*L'
L = LL′.L;

# Generate a turbulent wavefront.
w = fill!(similar(Cᵩ.mask, eltype(Cᵩ)), 0);
w[Cᵩ.mask] = L*randn(eltype(L), size(L,2));
plimg(w; fig=2, title="Wavefront");
sleep(0.1);

# Empirically estimate the structure function.
n = 100;
printstyled("Computing empirical structure function from $n observations.\nHold your breath...\n";
            bold=true, color=:yellow)
esf = EmpiricalStructFunc(S);
for i in 1:n;
    w[Cᵩ.mask] = L*randn(eltype(L), size(L,2));
    push!(esf, w);
end
plimg(esf.values; fig=3, title="Empirical Structure Function ($n samples)");
plimg(esf.weights; fig=4, title="Cumulated Weights ($n samples)");
sleep(0.1);

# Compute ground truth structure function.
gtsf = similar(esf)
for Δr in CartesianIndices(gtsf)
    gtsf[Δr] = iszero(esf.weights[Δr]) ? zero(eltype(gtsf)) : Dᵩ(Δr)
end
plimg(gtsf; fig=5, title="Theoretical Structure Function");
sleep(0.1);
