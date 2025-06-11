using Lux, Random, NNlib, LinearAlgebra
using Statistics: mean
import DifferentiationInterface as DI
using Mooncake, Enzyme

import OneHotArrays: onecold, onehotbatch
import MLDatasets: Cora
import Optimisers: Adam, setup, adjust!, update!
using Printf

# ========== Define helper functions ========== #

function create_adj(rng, N::Int, threshold::Float32)
    # Create random edge list
    edges::Vector{Tuple{Int64, Int64}} = [(i, j) for i in 1:N, j in 1:N if i != j && rand(rng) < threshold]
    # Dictionary mapping each node ID (1 to N) to a list of neighbor IDs
    # Sparse neighbor mapping / connectivity mask (i => neighbors / i attends to neighbors)
    adj::Dict{Int64, Vector{Int64}} = Dict(i => Int[] for i in 1:N)
    for (i, j) in edges
        push!(adj[i], j)
        push!(adj[j], i)
    end
    # Add self-loops if any missing
    for i in 1:N
        if !haskey(adj, i)
            adj[i] = Int[]
        end
        # locate key i, check if i is in the values, if not push it
        if i in values(adj)
            continue
        else
            push!(adj[i], i)
        end
    end
    return (edges, adj)
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
    #=adj::Dict{Int64, Vector{Int64}} = Dict(i => Int[] for i in 1:N)
    for (i, j) in edges
        push!(adj[i], j)
        undirected && i != j && push!(adj[j], i)
    end
    return sort(adj)=#
end

"""
The order of the columns of WH MUST correspond to the order of the keys of adj,
however the indexes are not the same. WH is 1-indexed and adj can contain arbitrary keys.
"""
function attention_scores!(
    e::Dict{Tuple{Int,Int}, Float32},
    WH::Matrix{Float32}, # node features
    a::Vector{Float32},  # attention weights
    adj::Dict{Int, Vector{Int}},
    activation::F,
    leaky_slope::Float32
) where F
    key_to_index = Dict(k => i for (i, k) in enumerate(keys(adj)))
    for (idx_i, key_i) in enumerate(keys(adj)) # For each node i,
        hi = view(WH, :, idx_i) 
        for key_j in adj[key_i]     # for each neighbor j of node i
            if haskey(adj, key_j)
                idx_j = key_to_index[key_j]
                hj = view(WH, :, idx_j)
                e[(key_i, key_j)] = activation(dot(a, vcat(hi, hj)), leaky_slope)
            end
        end
    end
end

function normalize_scores!(
    e::Dict{Tuple{Int,Int}, Float32}, 
    adj::Dict{Int, Vector{Int}}
)
    # For each node i, normalize the attention scores over its neighbors
    for key_i in keys(adj)
        neighborkeys = [key_j for key_j in adj[key_i] if haskey(e, (key_i, key_j))]
        if length(neighborkeys) > 0
            neighborscores = [e[(key_i, key_j)] for key_j in neighborkeys]
            α = mysoftmax(neighborscores)
            for (key_j, α_j) in zip(neighborkeys, α)
                e[(key_i, key_j)] = α_j
            end
        end
    end
end

mysoftmax(x) = softmax(x)
mysoftmax(x::R) where R<:Real = softmax([x])

function aggregate_features!(
    H_out::Matrix{Float32}, 
    WH::Matrix{Float32}, 
    e::Dict{Tuple{Int,Int}, Float32}, 
    adj::Dict{Int, Vector{Int}}
)
    key_to_index = Dict(k => i for (i, k) in enumerate(keys(adj)))
    # For each node i, aggregate the features of its neighbors
    for (idx_i, key_i) in enumerate(keys(adj))
        neighborkeys = [key_j for key_j in adj[key_i] if haskey(e, (key_i, key_j))]
        # By adding each neighbor's features weighted by its attention score
        for key_j in neighborkeys
            #idx_j = findfirst(keys(adj) .== key_j)
            idx_j = key_to_index[key_j]
            @inbounds @views H_out[:, idx_i] += e[(key_i, key_j)] .* WH[:, idx_j]
        end
    end
end

# ========== Define GAT Layer ========== #

struct GATLayer{F0, F1, F2} <: Lux.AbstractLuxLayer
    F_in::Int
    F_out::Int
    init_params::F0
    init_gain::Float32
    attention_activation::F1
    head_activation::F2
    leaky_slope::Float32
    N::Int
end

function GATLayer(
    F_in::Int, 
    F_out::Int;
    init_params = glorot_uniform, 
    init_gain = 1f0,
    attention_activation = leakyrelu,
    head_activation = identity,
    leaky_slope = 0.2,
    N = 0
)
    GATLayer{typeof(init_params), typeof(attention_activation), typeof(head_activation)}(
        F_in, F_out, init_params, init_gain, attention_activation, head_activation, leaky_slope, N
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::GATLayer)
    return (
        W = l.init_params(rng, Float32, l.F_out, l.F_in; 
                          gain=l.init_gain), 
        a = l.init_params(rng, Float32, 2 * l.F_out; 
                          gain=l.init_gain)
    )
end

function Lux.initialstates(rng::AbstractRNG, l::GATLayer)
    if l.N > 0
        WH = zeros(Float32, l.F_out, l.N)
    else
        WH = nothing
    end
    return (WH = WH,)
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
    H, adj = x # H: (F_in, N)
    N = size(H, 2)
    F_out = l.F_out

    # Project features (each col represents a node)
    WH = ps.W * H  # shape: (F_out, N)   

    # Compute unnormalized attention coefficients
    e = Dict{Tuple{Int,Int}, Float32}()
    attention_scores!(e, WH, ps.a, adj, l.attention_activation, l.leaky_slope) # 724.834 μs (21119 allocations: 1.19 MiB)
    # Softmax normalization
    normalize_scores!(e, adj) # 826.375 μs (24525 allocations: 1.04 MiB)
    # Aggregate features
    H_out = zeros(Float32, F_out, N)
    aggregate_features!(H_out, WH, e, adj) # 1.031 ms (50508 allocations: 2.47 MiB)

    return (l.head_activation(H_out), adj), st
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
Lux.initialstates(rng::AbstractRNG, l::MultiHeadGAT) = NamedTuple()

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
        (out, _), _ = head((x, adj), head_ps, NamedTuple())
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
        (out, _), _ = head((x, adj), head_ps, NamedTuple())
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
    learning_rate = 0.005,
    epochs = 100, 
    patience = 20,
    #drop_rate = 0.6,
    transductive = true
)
    H, y, adj, (f_in, f_out), (train_idx, val_idx, test_idx) = cora_data()

    if transductive
        adj_train = adj_val = adj_test = adj
        X_train = X_val = X_test = H
        y_train = y_val = y_test = y
    else
        X_train, X_val, X_test = H[:, train_idx], H[:, val_idx], H[:, test_idx]
        y_train, y_val, y_test = y[:, train_idx], y[:, val_idx], y[:, test_idx]
        adj_train = Dict((key, value) for (key, value) in adj if key in train_idx)
        adj_val = Dict((key, value) for (key, value) in adj if key in val_idx)
        adj_test = Dict((key, value) for (key, value) in adj if key in test_idx)
    end

    rng = Random.MersenneTwister(23)

    model = Lux.Chain(
        # One layer which concatenates, with the head activations
        MultiHeadGAT(
            ntuple(_ -> GATLayer(
                f_in, f_out; 
                attention_activation = elu, 
                head_activation = tanh_fast), num_heads), 
            true, 
            identity),
        # One layer which averages, with no head activation and a final softmax activation
        MultiHeadGAT(
            ntuple(_ -> GATLayer(
                f_out * num_heads, f_out; 
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
        DI.Constant((X_train, y_train, adj_train, train_idx))
    )

    for epoch in 1:epochs
        loss, grads = DI.value_and_gradient(
            train_loss, prep, backend,
            ps, DI.Constant(st), DI.Constant(model),
            DI.Constant((X_train, y_train, adj_train, train_idx))
        )

        if !isfinite(loss)
            @warn "Loss is $loss on epoch $(epoch)" 
        else
            opt_state, ps = update!(opt_state, ps, grads)
        end

        val_l = first(
            loss_function(
                ps, Lux.testmode(st), model, 
                (X_val, y_val, adj_val, val_idx),
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
        (X_test, y_test, adj_test, test_idx),
    )[1]
    test_acc = accuracy(
        model(
                (X_test, adj_test, test_idx),
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
    patience = 100,
    transductive = true
)

# ==== tests ==== #
H, y, A, (f_in, f_out), (train_idx, val_idx, test_idx) = cora_data()
N = size(H, 2)
rng = Random.MersenneTwister(23)

model = GATLayer(f_in, f_out; N = N)
model_no = GATLayer(f_in, f_out; N = 0)
ps, st = Lux.setup(rng, model)
ps_no, st_no = Lux.setup(rng, model_no)

y_pred, st = model((H, A), ps, st)
y_pred_no, st_no = model_no((H, A), ps_no, st_no)

using BenchmarkTools
@btime model((H, A), ps, st)
# 4.110 ms (96188 allocations: 5.33 MiB)

@btime model_no((H, A), ps_no, st_no)
#