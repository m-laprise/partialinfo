using Lux, Random, NNlib, LinearAlgebra
using Statistics: mean
import DifferentiationInterface as DI
using Mooncake, Enzyme

import OneHotArrays: onecold, onehotbatch
import MLDatasets: Cora
import Optimisers: Adam, setup, adjust!, update!
using Printf

import LoopVectorization: @turbo

# ========== Define helper functions ========== #

function create_adj(rng, N::Int, threshold::Float32; undirected = true, selfloops = true)
    # Create random edge list
    edges::Vector{Tuple{Int64, Int64}} = [(i, j) for i in 1:N, j in 1:N if i != j && rand(rng) < threshold]
    A = falses(N, N)
    for (i, j) in edges
        A[i, j] = true
        undirected && i != j && (A[j, i] = true)
    end
    if selfloops
        for i in 1:N
            A[i, i] = true
        end
    end
    return (edges, A)
end

function create_adj(edges::Vector{Tuple{Int64, Int64}}, N::Int; undirected = true, selfloops = true)
    A = falses(N, N)
    for (i, j) in edges
        A[i, j] = true
        undirected && i != j && (A[j, i] = true)
    end
    if selfloops
        for i in 1:N
            A[i, i] = true
        end
    end
    A
end

function attention!(
    E::Matrix{Float32}, 
    H_out::Matrix{Float32},
    H::Matrix{Float32}, # node features (value matrix)
    Q::Matrix{Float32},  # query matrix
    K::Matrix{Float32},  # key matrix
    A::AbstractArray{Bool, 2}, # adjacency matrix
    d::Float32; 
    ϵ::Tuple{Float32, Float32} = (log(1f0 + 1f-6), log(1f-6))
)
    mul!(E, Q', K)

    @turbo for i in axes(E, 1), j in axes(E, 2) # Normalize
        E[i, j] = E[i, j] / d
    end

    @. E += ifelse(A, ϵ[1], ϵ[2]) # Add log(A + ϵ) efficiently
    
    @turbo for i in axes(E, 1), j in axes(E, 2) # Faster exp
        E[i, j] = exp(E[i, j])
    end

    for j in axes(E, 2)           # Non allocating softmax
        sum_j = 0f0
        @turbo for i in axes(E, 1)
            sum_j += E[i, j]
        end
        @turbo for i in axes(E, 1)
            E[i, j] /= sum_j
        end
    end
    
    mul!(H_out, H, E)
end

Q = ps.Wq * H
K = ps.Wk * H

E = zeros(Float32, N, N)
H_out = zeros(Float32, f_in, N)

@btime attention!($E, $H_out, $H, $Q, $K, $A, $model.d)
#78.362 ms (0 allocations: 0 bytes)

attention!(E, H_out, H, Q, K, A, model.d) 
elu(H_out)

# ========== Define GAT Layer ========== #

struct GATLayer{F0, F1, F2} <: Lux.AbstractLuxLayer
    F_in::Int
    F_hid::Int
    F_out::Int
    init_params::F0
    init_gain::Float32
    attention_activation::F1
    head_activation::F2
    leaky_slope::Float32
    d::Real
    N::Int
end

function GATLayer(
    F_in::Int, 
    F_hid::Int,
    F_out::Int;
    init_params = glorot_uniform, 
    init_gain = 1f0,
    attention_activation = leakyrelu,
    head_activation = identity,
    leaky_slope = 0.2,
    N = 0
)
    GATLayer{typeof(init_params), typeof(attention_activation), typeof(head_activation)}(
        F_in, F_hid, F_out, init_params, init_gain, attention_activation, head_activation, leaky_slope, Float32(sqrt(F_in)), N
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::GATLayer)
    return (
        Wq = l.init_params(rng, Float32, l.F_hid, l.F_in; 
                          gain=l.init_gain), 
        Wk = l.init_params(rng, Float32, l.F_hid, l.F_in;
                           gain=l.init_gain)
    )
end

function Lux.initialstates(rng::AbstractRNG, l::GATLayer)
    return (
        Q = zeros(Float32, l.F_hid, l.N),
        K = zeros(Float32, l.F_hid, l.N),
        E = zeros(Float32, l.N, l.N),
        H_out = zeros(Float32, l.F_in, l.N)
    )
end

"""
Forward pass for a single graph attention head. 
The input x is a tuple of (node features, adjacency matrix).
The node features is a matrix of shape (F_in, N), where F_in is 
the number of features per node and N is the number of nodes; the 
matrix is indexed from 1.

The adjacency matrix is a dictionary mapping each node ID to a list 
of neighbor IDs, but the IDs are not restricted to the range 1 to N.

The order of the nodes in the adjacency list and the order of the 
columns in the node features matrix MUST be the same.
"""
function (l::GATLayer)(x, ps, st) 
    # Unpack inputs
    H, A = x 
    @assert size(H, 2) == l.N "Node features must be of size (F_in, N)"

    # Project features (each col represents a node) into key and query vectors
    #Q = ps.Wq * H
    #K = ps.Wk * H
    mul!(st.Q, ps.Wq, H)
    mul!(st.K, ps.Wk, H)

    # Apply attention
    #E = zeros(Float32, l.N, l.N)
    #H_out = zeros(Float32, l.F_in, l.N)
    attention!(st.E, st.H_out, H, st.Q, st.K, A, l.d) 

    return (l.head_activation(st.H_out), A), st
end

# ========== Build Multi-Head GAT ==========

struct MultiHeadGAT{N, H<:GATLayer, Bool, F} <: Lux.AbstractLuxLayer
    heads::NTuple{N, H}
    concat::Bool #  if false, average outputs
    final_activation::F  # should be identity if concat is true
end

function Lux.initialparameters(rng::AbstractRNG, l::MultiHeadGAT)
    heads = l.heads
    fieldnames = Tuple(Symbol("head$i") for i in 1:length(heads))
    ps_values = ntuple(i -> Lux.initialparameters(rng, heads[i]), length(heads))
    ps = NamedTuple{fieldnames}(ps_values)
    return ps
end
#Lux.initialstates(rng::AbstractRNG, l::MultiHeadGAT) = NamedTuple()
function Lux.initialstates(rng::AbstractRNG, l::MultiHeadGAT)
    heads = l.heads
    fieldnames = Tuple(Symbol("head$i") for i in 1:length(heads))
    st_values = ntuple(i -> Lux.initialstates(rng, heads[i]), length(heads))
    st = NamedTuple{fieldnames}(st_values)
    return st
end

(l::MultiHeadGAT)(
    x, ps, st; concat = l.concat) = l(x, ps, st, Val(concat))

function (l::MultiHeadGAT)(
    x, ps, st, ::Val{true}
)
    x, adj = x
    outputs = map(1:length(l.heads)) do i
        head = l.heads[i]
        pname = Symbol("head$i")
        head_ps = ps[pname]
        head_st = st[pname]
        (out, _), _ = head((x, adj), head_ps, head_st)
        #(out, _), _ = head((x, adj), head_ps, NamedTuple())
        #head_st = st[pname]
        #out, _ = head(x, head_ps, head_st)
        return out
    end

    return (reduce(vcat, outputs)::Matrix{Float32}, adj), st    # (F_out * num_heads, N)
end

function (l::MultiHeadGAT)(
    x, ps, st, ::Val{false}
)
    x, adj = x
    outputs = map(1:length(l.heads)) do i
        head = l.heads[i]
        pname = Symbol("head$i")
        head_ps = ps[pname]
        head_st = st[pname]
        (out, _), _ = head((x, adj), head_ps, head_st)
        #(out, _), _ = head((x, adj), head_ps, NamedTuple())
        #head_st = st[pname]
        #out, _ = head(x, head_ps, head_st)
        return out
    end

    stacked = cat(outputs...; dims=3)               # (F_out, N, num_heads)
    final = dropdims(mean(stacked; dims=3), dims=3) # (F_out, N)
    return (l.final_activation(final)::Matrix{Float32}, adj), st
end

# ==== Try with Cora data ===== #

function cora_data()
    data = Cora()
    gph = data.graphs[1]
    classes = data.metadata["classes"]
    H = gph.node_data.features
    y = onehotbatch(gph.node_data.targets, classes)

    n = gph.num_nodes
    f_in = size(gph.node_data.features, 1)
    f_out = size(y, 1)

    src = gph.edge_index[1]
    dst = gph.edge_index[2]

    edges = sort(collect(zip(src, dst)))
    A = create_adj(edges, n; undirected = true, selfloops = true)

    train_idx, val_idx, test_idx = (1:140, 141:640, 1709:2708)

    return H, y, A, (f_in, f_out), (train_idx, val_idx, test_idx)
end

function main(;
    num_heads = 2,
    hidden_dim = 64,
    learning_rate = 0.005,
    epochs = 100, 
    patience = 20
)
    H, y, A, (f_in, f_out), (train_idx, val_idx, test_idx) = cora_data()

    rng = Random.MersenneTwister(23)

    model = Lux.Chain(
        # One layer which concatenates, with the head activations
        MultiHeadGAT(
            ntuple(_ -> GATLayer(
                f_in, hidden_dim, f_out; 
                attention_activation = elu, 
                head_activation = tanh_fast), num_heads), 
            true, 
            identity),
        # One layer which averages, with no head activation and a final softmax activation
        MultiHeadGAT(
            ntuple(_ -> GATLayer(
                f_out * num_heads, hidden_dim, f_out; 
                attention_activation = elu, 
                head_activation = identity), 1), 
            false, 
            identity)
    )
    ps, st = Lux.setup(rng, model)

    opt = Adam(learning_rate, (0.9, 0.9))
    opt_state = setup(opt, ps)

    @printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps) / 1.0e6)

    function masked_loss(y_pred, y_true, mask)
        CrossEntropyLoss(; agg=mean, logits=Val(true))(
            y_pred[:, mask], y_true[:, mask]
        )
    end

    function loss_function(ps, st, model::Chain, (x, y, adj, mask))
        (y_pred, _), st = model((x, adj), ps, st)
        loss = masked_loss(y_pred, y, mask)
        return loss, st, (; y_pred)
    end

    accuracy(y_pred, y) = mean(onecold(y_pred) .== onecold(y)) * 100

    train_loss(ps, st, model::Chain, (X, y, adj, mask)) = loss_function(ps, st, model::Chain, (X, y, adj, mask))[1]

    best_loss_val = Inf
    cnt = 0

    backend = DI.AutoMooncake(; config=nothing)
    #backend = DI.AutoEnzyme(mode=set_runtime_activity(Enzyme.Reverse))

    prep = DI.prepare_gradient(
        train_loss, backend, 
        ps, DI.Constant(st), DI.Constant(model), 
        DI.Constant((H, y, A, train_idx))
    )

    for epoch in 1:epochs
        loss, grads = DI.value_and_gradient(
            train_loss, prep, backend,
            ps, DI.Constant(st), DI.Constant(model),
            DI.Constant((H, y, A, train_idx))
        )

        if !isfinite(loss)
            @warn "Loss is $loss on epoch $(epoch)" 
        else
            opt_state, ps = update!(opt_state, ps, grads)
        end

        val_l = first(
            loss_function(
                ps, Lux.testmode(st), model, 
                (H, y, A, val_idx),
            ),
        )

        @printf "Epoch %3d\tTrain Loss: %.6f\tVal Loss: %.6f\n" epoch loss val_l

        if val_l < best_loss_val
            best_loss_val = val_l
            cnt = 0
        else
            cnt += 1
            if cnt == patience
                @printf "Early Stopping at Epoch %d\n" epoch
                break
            end
        end
    end

    test_loss = loss_function(
        ps, Lux.testmode(st), model, 
        (H, y, A, test_idx),
    )[1]
    test_acc = accuracy(
        model(
                (H, A, test_idx),
                ps, Lux.testmode(st),
            )[1][1],
        Array(y_test),
    )
    @printf "Test Loss: %.6f\tTest Acc: %.4f%%\n" test_loss test_acc
    return test_loss, test_acc
end

# Enzyme.API.strictAliasing!(false)
test_loss, test_acc = main(
    num_heads = 8,
    learning_rate = 0.005,
    epochs = 20, 
    patience = 100
)

# ==== tests ==== #
H, y, A, (f_in, f_out), (train_idx, val_idx, test_idx) = cora_data()
N = size(H, 2)
rng = Random.MersenneTwister(23)

model = GATLayer(f_in, 2*f_out, f_out; N = N)
ps, st = Lux.setup(rng, model)

y_pred, st = model((H, A), ps, st)

using BenchmarkTools
@btime model((H, A), ps, st)
# 4.110 ms (96188 allocations: 5.33 MiB)
# 80.204 ms (2 allocations: 96 bytes)
