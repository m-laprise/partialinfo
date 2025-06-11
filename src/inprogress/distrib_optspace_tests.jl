using LinearAlgebra
using Random
import Distributions: Dirichlet, mean, std, var
using Plots
using Lux

function trim_matrix(A::AbstractMatrix)
    m, n = size(A)
    k = sum(A .!= 0)
    for col in eachcol(A)
        if sum(col .!= 0)  > (2*k) / n
            col .= 0
        end
    end
    for row in eachrow(A)
        if sum(row .!= 0) > (2*k) / m
            row .= 0
        end
    end
    return A
end

function project_r(A::AbstractArray, r::Int)
    u, s, v = svd(A)
    return u[:, 1:r] * Diagonal(s[1:r]) * v[:, 1:r]'
end

function project_r!(A::AbstractArray, r::Int)
    try
        u, s, v = svd(A)
        A .= u[:, 1:r] * Diagonal(s[1:r]) * v[:, 1:r]'
    catch 
        u, s, v = svd(A; alg = LinearAlgebra.QRIteration())
        A .= u[:, 1:r] * Diagonal(s[1:r]) * v[:, 1:r]'
    end
end

function correct_known!(grassman_proj, sparsematrix)
    for i = axes(sparsematrix, 1)
        for j = axes(sparsematrix, 2)
            if sparsematrix[i, j] != 0
                grassman_proj[i, j] = sparsematrix[i, j]
            end
        end
    end
end

function indiv_embed(sparsevec::AbstractVector, r, iter=5)
    n2 = length(sparsevec)
    n = Int(sqrt(n2))
    sparsematrix = reshape(sparsevec, n, n)
    grassman_proj = project_r(trim_matrix(sparsematrix), r)
    # For iter times, correct the known entries and project back to the Grassman manifold again
    for _ in 1:iter
        correct_known!(grassman_proj, sparsematrix)
        project_r!(grassman_proj, r)
    end
    correct_known!(grassman_proj, sparsematrix)
    return grassman_proj
end

function indiv_embed(sparsematrix::AbstractMatrix, r, iter=5)
    grassman_proj = project_r(trim_matrix(sparsematrix), r)
    # For iter times, correct the known entries and project back to the Grassman manifold again
    for _ in 1:iter
        correct_known!(grassman_proj, sparsematrix)
        project_r!(grassman_proj, r)
    end
    correct_known!(grassman_proj, sparsematrix)
    return grassman_proj
end

function combine(dataX, k::Int)
    # Combine the first k agents' data vectors into a single vector
    # If an entry is known by multiple agents, keep the entry that has the maximum absolute value
    data = dataX[:, 1:k]
    _, idx = findmax(abs, data, dims=2)
    return data[idx]
end

function symmetrize!(X::AbstractArray{<:Real, 2})
    @assert size(X, 1) == size(X, 2)
    for i in axes(X, 1)
        for j in axes(X, 2)
            if i < j
                val = maximum([abs(X[i,j]), abs(X[j,i])])
                X[i,j] = X[j,i] = val
            end
        end
    end
end

function symmetrize!(X::AbstractArray{<:Real, 2}, N)
    @assert Int(sqrt(size(X, 1))) == N
    for partialX in eachslice(X, dims=2)
        x = reshape(partialX, N, N)
        for i in axes(x, 1)
            for j in axes(x, 2)
                if i < j
                    val = maximum([abs(x[i,j]), abs(x[j,i])])
                    x[i,j] = x[j,i] = val
                end
            end
        end
        @views partialX .= vec(x)
    end
end

function symmetrize!(X::AbstractVector, N)
    @assert Int(sqrt(length(X))) == N
    x = reshape(X, N, N)
    for i in axes(x, 1)
        for j in axes(x, 2)
            if i < j
                val = maximum([abs(x[i,j]), abs(x[j,i])])
                x[i,j] = x[j,i] = val
            end
        end
    end
    X .= vec(x)
end

# Hyperparameters and constants
K::Int = 50 # Try 9 (182 entries per agent) or 8 (205 entries per agent)
N::Int = 128
const DATASETSIZE::Int = 1
const KNOWNENTRIES::Int = 2500

mylw = 2
datarng = Random.MersenneTwister(4021)
k = K
is = 1:k

_3D(y) = reshape(y, N, N, size(y, 2))
_2D(y) = reshape(y, N*N, size(y, 3))
_3Dslices(y) = eachslice(_3D(y), dims=3)
l(f, y) = mean(f.(_3Dslices(y)))

include("/Users/mlaprise/dev/socialcomputation/scripts/datacreation_LuxCPU.jl")

function l_optspace(X, Y, RANK, niter)
    A = indiv_embed(X, RANK, niter)
    return MAELoss()(A, Y)
end

function gen_results(X, Y, RANK, is)
    MAEs1 = []
    MAEs10 = []
    MAEs100 = []
    MAEs1000 = []
    for i in is
        combinedX = vec(combine(X, i))
        println(sum(combinedX .!= 0))
        #symmetrize!(combinedX, N)
        #println(sum(combinedX .!= 0))
        push!(MAEs1, l_optspace(combinedX, Y, RANK, 1))
        push!(MAEs10, l_optspace(combinedX, Y, RANK, 10))
        push!(MAEs100, l_optspace(combinedX, Y, RANK, 100))
        push!(MAEs1000, l_optspace(combinedX, Y, RANK, 1000))
        println("Best MAE, loop ", i, ": ", MAEs1000[end])
    end
    return MAEs1, MAEs10, MAEs100, MAEs1000
end

#====================#

RANK::Int = 1
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng; symmetric=true)
#maskmatrix = masktuple2array(masktuples, N, N)
#nonzeroidx = findall((vec(maskmatrix)) .!= 0)
size(dataX), size(dataY)
X = dataX[:,:,1]
Y = dataY[:,:,1]
#=
A = indiv_embed(vec(sum(X[:,1:2], dims=2)), 1, 100)
MAELoss()(A, Y)=#

MAEs1, MAEs10, MAEs100, MAEs1000 = gen_results(X, Y, RANK, is)

x = cumsum(knowledgedistr[1:k])
p1 = Plots.plot(x, MAEs1, label = "1 iteration", lw = mylw)
Plots.plot!(x, MAEs10, label = "10 iterations", lw = mylw)
Plots.plot!(x, MAEs100, label = "100 iterations", lw = mylw)
Plots.plot!(x, MAEs1000, label = "1,000 iterations", lw = mylw)
rn = N * RANK 
nlogn = N * log(N)
Plots.vline!([rn], color=:black, linestyle=:dash, label="x = r n", lw = mylw)
Plots.vline!([nlogn], color=:red, linestyle=:dash, label="x = n log(n)", lw = mylw)
Plots.xlabel!("Number of known entries")
Plots.ylabel!("MAE")
Plots.title!("Rank $(RANK)")
Plots.ylims!(0,0.9)
#====================#
RANK::Int = 2
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(
    N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng; symmetric=true)
X = dataX[:,:,1]
Y = dataY[:,:,1]
#symmetrize!(X, N)

MAEs1, MAEs10, MAEs100, MAEs1000 = gen_results(X, Y, RANK, is)

x = cumsum(knowledgedistr[1:k])
p2 = Plots.plot(x, MAEs1, label = "1 iteration", lw = mylw)
Plots.plot!(x, MAEs10, label = "10 iterations", lw = mylw)
Plots.plot!(x, MAEs100, label = "100 iterations", lw = mylw)
Plots.plot!(x, MAEs1000, label = "1,000 iterations", lw = mylw)
rn = N * RANK 
nlogn = N * log(N)
rnlogn = N * RANK * log(N)
Plots.vline!([rn], color=:black, linestyle=:dash, label="x = r n", lw = mylw)
Plots.vline!([nlogn], color=:red, linestyle=:dash, label="x = n log(n)", lw = mylw)
Plots.vline!([rnlogn], color=:grey, linestyle=:dash, label="x = r n log(n)", lw = mylw)
Plots.xlabel!("Number of known entries")
Plots.ylabel!("MAE")
Plots.title!("Rank $(RANK)")
Plots.ylims!(0,0.9)
savefig(p2, "optspace_$(N)x$(N)_sym_r2_nocopy.png")
#====================#
RANK::Int = 4
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng; symmetric=true)
X = dataX[:,:,1]
Y = dataY[:,:,1]

MAEs1, MAEs10, MAEs100, MAEs1000 = gen_results(X, Y, RANK, is)

x = cumsum(knowledgedistr[1:k])
p3 = Plots.plot(x, MAEs1, label = "1 iteration", lw = mylw)
Plots.plot!(x, MAEs10, label = "10 iterations", lw = mylw)
Plots.plot!(x, MAEs100, label = "100 iterations", lw = mylw)
Plots.plot!(x, MAEs1000, label = "10,000 iterations", lw = mylw)
rn = N * RANK 
nlogn = N * log(N)
Plots.vline!([rn], color=:black, linestyle=:dash, label="x = r n", lw = mylw)
Plots.vline!([nlogn], color=:red, linestyle=:dash, label="x = n log(n)", lw = mylw)
Plots.xlabel!("Number of known entries")
Plots.ylabel!("MAE")
Plots.title!("Rank $(RANK)")
Plots.ylims!(0,0.9)
#====================#
RANK::Int = 8
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng; symmetric=true)
X = dataX[:,:,1]
Y = dataY[:,:,1]

MAEs1, MAEs10, MAEs100, MAEs1000 = gen_results(X, Y, RANK, is)

x = cumsum(knowledgedistr[1:k])
p4 = Plots.plot(x, MAEs1, label = "1 iteration", lw = mylw,
                legend_position = :bottomleft)
Plots.plot!(x, MAEs10, label = "10 iterations", lw = mylw)
Plots.plot!(x, MAEs100, label = "100 iterations", lw = mylw)
Plots.plot!(x, MAEs1000, label = "10,000 iterations", lw = mylw)
rn = N * RANK 
nlogn = N * log(N)
Plots.vline!([rn], color=:black, linestyle=:dash, label="x = r n", lw = mylw)
Plots.vline!([nlogn], color=:red, linestyle=:dash, label="x = n log(n)", lw = mylw)
Plots.xlabel!("Number of known entries")
Plots.ylabel!("MAE")
Plots.title!("Rank $(RANK)")
Plots.ylims!(0,0.9)

#====================#

fig = plot(p1, p2, p3, p4, layout=(2,2), size=(850,600),
            suptitle = "Completion of one random $(N)x$(N) matrix (Optspace)")
savefig(fig, "optspace_$(N)x$(N)_r1r2r4r8.png")
#, title = )

#====================#
K::Int = 3 # Try 9 (182 entries per agent) or 8 (205 entries per agent)
N::Int = 100
RANK::Int = 2
dataX, dataY, masktuples, knowledgedistr = datasetgeneration(N, N, RANK, DATASETSIZE, KNOWNENTRIES, K, datarng; symmetric=false)

cumsum(knowledgedistr[1:K])
X = dataX[:,:,1]
Y = dataY[:,:,1]

sols = zeros(N, N, K)
niter = 1000
for agent in 1:K
    A = indiv_embed(X[:,agent], RANK, niter)
    sols[:, :, agent] = A
    println("MAE for agent $(agent): ", MAELoss()(A, Y))
end

FR = FixedRankMatrices(N, N, RANK)
mean([MAELoss()(sol, Y) for sol in eachslice(sols, dims=3)])
var([MAELoss()(sol, Y) for sol in eachslice(sols, dims=3)])
mean_sol = reshape(mean(sols, dims=3), N, N)
sum_sol = reshape(sum(sols, dims=3), N, N)
MAELoss()(mean_sol, Y)
MAELoss()(embed(FR, SVDMPoint(mean_sol, RANK)), Y)

#===================================#
using LinearAlgebra
using Manopt, Manifolds, ManifoldDiff

# Define the Grassmann manifold
n, k = 10, 2
Gr = Grassmann(n, k)
St = Stiefel(n, k)
FR = FixedRankMatrices(n, n, k)
SPSfr = SymmetricPositiveSemidefiniteFixedRank(n, k)
# Random initial pair of points on the Grassmann manifold
(X0a, X0b) = (rand(Gr), rand(Gr))
(S0a, S0b) = (rand(St), rand(St))

a = rand(FR)
b = rand(10, 2) * rand(2, 10)
c = rand(10,10) 
SVDMPoint(b, 2)
check_point(FR, b)
check_point(FR, c)
embed(FR, SVDMPoint(c,2))

# Define the cost function as the Frobenius norm of 
# the difference between points and their projections onto the Grassmann manifold
function cost_function((Ma, Mb), X)
    a = norm(Ma - X * X' * Ma, Fro)^2
    b = norm(Mb - X * X' * Mb, Fro)^2
    return (a + b) / 2
end

# Define the gradient of the cost function
function grad_cost_function((Ma, Mb), X)
    a = Ma - X * X' * Ma
    b = Mb - X * X' * Mb
    return (a * Ma' + b * Mb') / 2
end

# Perform gradient descent
X_closest = gradient_descent(Gr, 
    cost_function,
    grad_cost_function,
    (X0a, X0b)
)
