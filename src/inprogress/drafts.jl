
function Lux.initialstates(rng::AbstractRNG, l::GraphAttention)
    #h = l.init(rng, l.m, l.k; gain = 0.01f0)
    (
        #=H = h,
        A = l.init_ones(l.m, l.k),
        Xproj = l.init_zeros(l.m, l.k),
        oldH = deepcopy(h),
        init = deepcopy(h)=#
    ) 
end


#=  Dot product attention: Attention(Q,K,V)=softmax( Q K^T / sqrt(d_k) ) V =#
K::Int = 5 
BATCH_SIZE::Int = 1
QK_DIM::Int = 10 # query/key feature dimension: nb of features in each query/key vector
SEQ_LEN::Int = 16 # length of the input sequence (number of query positions)

q = rand(Float32, QK_DIM, SEQ_LEN, BATCH_SIZE)

KV_LEN:: Int = 30 # length of the context sequence (number of key/value positions)
k = rand(Float32, QK_DIM, KV_LEN, BATCH_SIZE)

V_DIM::Int = 20 # value feature dimension: dimension of each value vector. Output will inherit this dimension.
v = rand(Float32, V_DIM, KV_LEN, BATCH_SIZE)

# Unmasked
y, α = dot_product_attention(q, k, v; mask = nothing, nheads = K)

# Adjacency matrix:
A = rand(0:1, SEQ_LEN, SEQ_LEN)

# NOTE: This does not work. The mask should be a boolean array broadcastable to size (kv_len, seq_len, nheads, batch_size)
function attention_mask_from_adjacency(A::AbstractMatrix{<:Integer}; nheads = 1, batch_size = 1)
    N = size(A, 1)  # Number of nodes
    mask = BitArray(undef, (N, N, nheads, batch_size))  # (kv_len, q_len, nheads, batch)
    for i in 1:N         # query index
        for j in 1:N     # key index
            if A[i, j] != 0
                mask[j, i, :, :] .= 1  # allow attention from node i to j
            end
        end
    end
    return mask
end
attention_mask = attention_mask_from_adjacency(A; nheads = K, batch_size = BATCH_SIZE)
y, α = dot_product_attention(q, k, v; mask = attention_mask, nheads = K)

# Returns the attention scores after softmax of size (kv_len, seq_len, nheads, batch_size...)
# For each query position (SEQ_LEN), we get a distribution over all key/value positions (KV_LEN).
# (How much does each query attend to each key?) -> Apply softmax along each query column to turn raw scores into weights over the 30 keys.
size(α)

# Returns the attention output array of size (v_dim, seq_len, batch_size...) 
# For each query position (SEQ_LEN), we get a weighted combination of value vectors (each of dim V_DIM).
# (For each column (query position), take a weighted sum of all v vectors using the attention weights.)
size(y)


function forwardpass!(st, ps, m::A, x) where A <: Lux.AbstractLuxLayer
    Lux.apply(m, x, ps, st)[1]
end

myloss(ps, st, m, x, y, steps) = Lux.MSELoss()(forwardpass!(st, ps, m, x), y)

#yhat = forwardpass!(st, ps, model, x; steps=steps)

#=model = Chain(
    x -> x[1],
    LayerNorm((N,)),
)=#

# ========== Simulate Random Input ========== #
#=
# Feature dimension per node
const F_IN::Int = 16
# Project down to this number of features
const F_OUT::Int = 8
# Number of nodes in the graph
const N::Int = 32
rng = MersenneTwister(1234)

# Feature matrix
const X::Matrix{Float32} = randn(rng, Float32, F_IN, N)  # Node features

const EDGES, ADJ = create_adj(rng, N, 0.5f0)
=#
# ========== Instantiate GAT Layer ========== #
#=
const gat = GATLayer(F_IN, F_OUT; attention_activation = leakyrelu, head_activation = sigmoid)
ps1, st1 = Lux.setup(rng, gat)

# Run forward pass
(output, _), _ = gat((X, ADJ), ps1, st1)

println("Input shape: ", size(X))        # (F_in, N)
println("Output shape: ", size(output)) # (F_out, N)

=#
# ========== Instantiate Multi-Head GAT ==========#
#=
const NUM_HEADS::Int = 3
# Set concat_heads = false to average instead of concatenate outputs.
const CONCAT_HEADS::Bool = true
const HEADS = ntuple(_ -> GATLayer(
    F_IN, F_OUT; 
    attention_activation = leakyrelu, head_activation = sigmoid), NUM_HEADS)
model = MultiHeadGAT(HEADS, CONCAT_HEADS, identity)
ps, st = Lux.setup(rng, model)

const chainedmodel = Lux.Chain(
    # One layer which concatenates, with the head activations
    MultiHeadGAT(
        ntuple(_ -> GATLayer(
            F_IN, F_OUT; 
            attention_activation = leakyrelu, 
            head_activation = sigmoid), NUM_HEADS), 
        true, 
        identity),
    # One layer which averages, with no head activation and a final softmax activation
    MultiHeadGAT(
        ntuple(_ -> GATLayer(
            F_OUT * NUM_HEADS, F_OUT; 
            attention_activation = leakyrelu, 
            head_activation = identity), NUM_HEADS), 
        false, 
        softmax)
)
ps_chain, st_chain = Lux.setup(rng, chainedmodel)
=#
# ========== Forward Pass ========== #
#=
(output, _), _ = model((X, ADJ), ps, st)
(output_chain, _), _ = chainedmodel((X, ADJ), ps_chain, st_chain)

println("Input shape:  ", size(X))
println("Output shape: ", size(output))  # Should be (F_out * heads, N) if concat
println("Output shape: ", size(output_chain))

using BenchmarkTools

@btime gat($(X, ADJ), $ps1, $st1)
# 65.291 μs (3322 allocations: 203.52 KiB)
@btime model($(X, ADJ), $ps, $st) 
# 224.750 μs (9989 allocations: 614.34 KiB)
@btime chainedmodel($(X, ADJ), $ps_chain, $st_chain)
# 412.917 μs (19978 allocations: 1.20 MiB)
=#

#= Benchmarks =#

#@code_warntype gat((X, ADJ), ps1, st1)
#@code_warntype model((X, ADJ), ps, st)
#@code_warntype chainedmodel((X, ADJ), ps_chain, st_chain)

function myloss(ps, st, model, x)
    y, _ = model(x, ps, st)
    return sum(sum, y)
end

val, grad = DI.value_and_gradient(
    myloss, 
    DI.AutoMooncake(;config=nothing), 
    ps, DI.Constant(st), DI.Constant(model), DI.Constant((X, ADJ))
)

val2, grad2 = DI.value_and_gradient(
    myloss, 
    DI.AutoEnzyme(mode=set_runtime_activity(Enzyme.Reverse)),
    ps, DI.Constant(st), DI.Constant(model), DI.Constant((X, ADJ))
)

@btime DI.value_and_gradient(
    $myloss, $DI.AutoMooncake(;config=nothing), 
    $ps, $DI.Cache(st), $DI.Constant(model), $DI.Constant((X, ADJ))
)
# 6.156 ms (73339 allocations: 23.71 MiB)
@btime DI.value_and_gradient(
    $myloss, $DI.AutoEnzyme(mode=set_runtime_activity(Enzyme.Reverse)), 
    $ps, $DI.Constant(st), $DI.Constant(model), $DI.Constant((X, ADJ))
)
# 9.313 ms (88106 allocations: 10.07 MiB)

@btime DI.value_and_gradient(
    $myloss, $DI.AutoMooncake(;config=nothing), 
    $ps_chain, $DI.Cache(st_chain), $DI.Constant(chainedmodel), $DI.Constant((X, ADJ))
)
# 
@btime DI.value_and_gradient(
    $myloss, $DI.AutoEnzyme(mode=set_runtime_activity(Enzyme.Reverse)), 
    $ps_chain, $DI.Constant(st_chain), $DI.Constant(chainedmodel), $DI.Constant((X, ADJ))
)

