using StructureFunctions

using YPlot, PyPlot
using LinearAlgebra, OffsetArrays

# Create a structure function for the Kolmogorov law.
r0 = 1.0; # Fried's parameter
f = KolmogorovStructFunc(r0);

# Create a support as a circular pupil with a central aperture.
S = [16.5 ≤ sqrt(x^2 + y^2) < 50.5 for x in -51:51, y in -51:51];

# Create a packed lazy covariance.
σ = 0.1; # piston variance
Cov = StructureFunctions.PackedLazyCovariance(f, S, σ);

# Display the variance.
v = fill!(similar(Cov.mask, eltype(Cov)), 0);
v[Cov.mask] = var(Cov);
plimg(v; fig=1, title="Variance");
sleep(0.1);

# Cholesky decomposition of the covariance.
LL′ = cholesky(Cov); # C = L*L'
L = LL′.L;

# Generate a turbulent wavefront.
w = fill!(similar(Cov.mask, eltype(Cov)), 0);
w[Cov.mask] = L*randn(eltype(L), size(L,2));
plimg(w; fig=2, title="Wavefront");
sleep(0.1);

# Empirically estimate the structure function.
n = 100;
printstyled("Computing empirical structure function from $n observations.\nHold your breath...\n";
            bold=true, color=:yellow)
esf = EmpiricalStructureFunction(S);
for i in 1:n;
    w[Cov.mask] = L*randn(eltype(L), size(L,2));
    push!(esf, w);
end
plimg(esf.values; fig=3, title="Empirical Structure Function ($n samples)");
plimg(esf.weights; fig=4, title="Cumulated Weights ($n samples)");
sleep(0.1);

# Compute groun truth structure fucntion.
gtsf = similar(esf)
for Δr in CartesianIndices(gtsf)
    gtsf[Δr] = iszero(esf.weights[Δr]) ? zero(eltype(gtsf)) : f(Δr)
end
plimg(gtsf; fig=5, title="Theoretical Structure Function");
sleep(0.1);
