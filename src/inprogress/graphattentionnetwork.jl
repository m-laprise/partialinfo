using Lux, Random, NNlib, LinearAlgebra
using Statistics: mean
import DifferentiationInterface as DI
using Mooncake, Enzyme

import OneHotArrays: onecold, onehotbatch
import MLDatasets: Cora
import Optimisers: AdamW, setup, adjust!, update!
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

#=function attention!(
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
end=#

function attention!v2(
    E::Matrix{Float32}, 
    Q::Matrix{Float32},  # query matrix
    K::Matrix{Float32},  # key matrix
    grouped_pairs::Vector{Vector{Tuple{Int, Int}}},
    d::Float32
)
    # Compute and exponentiate logits
    @inbounds for group in grouped_pairs
        for (i, j) in group
            @views qi = Q[:, i]
            @views kj = K[:, j]
            E[i, j] = exp(dot(qi, kj) / d)
        end
    end
    # Apply colwise non-allocating softmax
    @inbounds for j in eachindex(grouped_pairs)
        group = grouped_pairs[j]
        sum_j = 0f0
        for (i, _) in group
            sum_j += E[i, j]
        end
        for (i, _) in group
            E[i, j] /= sum_j
        end
    end
end

function right_sparse_mul!(H_out, H, E, A)
    @inbounds for node in axes(H_out, 2) # For each node
        for (ngh_idx, _) in A[node] # Iterate over neighbors attention scores (non-zero entries of the column)
            @turbo for feature_idx in axes(H_out, 1) # Feature by feature, add to H_out the value * its attention weight
                H_out[feature_idx, node] += H[feature_idx, ngh_idx] * E[ngh_idx, node]
            end
        end
    end
end

#=function MCK_right_sparse_mul!(H_out, H, E, A)
    @inbounds for node in axes(H_out, 2) # For each node
        for (ngh_idx, _) in A[node] # Iterate over neighbors attention scores (non-zero entries of the column)
            for feature_idx in axes(H_out, 1) # Feature by feature, add to H_out the value * its attention weight
                H_out[feature_idx, node] += H[feature_idx, ngh_idx] * E[ngh_idx, node]
            end
        end
    end
end
Mooncake.@mooncake_overlay right_sparse_mul!(H_out, H, E, A) = MCK_right_sparse_mul!(H_out, H, E, A)=#

# ========== Define GAT Layer ========== #
struct GATLayer{F0, F1} <: Lux.AbstractLuxLayer
    F_in::Int
    F_hid::Int
    init_params::F0
    init_gain::Float32
    attention_activation::F1
    leaky_slope::Float32
    d::Float32
    N::Int
end

function GATLayer(
    F_in::Int, 
    F_hid::Int;
    init_params = glorot_uniform, 
    init_gain::Float32 = 1f0,
    attention_activation = leakyrelu,
    leaky_slope::Float32 = 0.2f0,
    N::Int = 0,
)
    GATLayer{typeof(init_params), typeof(attention_activation)}(
        F_in, F_hid, init_params, init_gain, attention_activation, leaky_slope, 
        Float32(sqrt(F_in)), N, #attention_dropout
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
        H_out = zeros(Float32, l.F_in, l.N),
        training = true
    )
end

"""
Forward pass for a single graph attention head. 
The input x is a tuple of (node features, adjacency matrix).
The node features is a matrix of shape (F_in, N), where F_in is 
the number of features per node and N is the number of nodes.
"""
function (l::GATLayer)(x, ps, st) 
    H, A = x 
    fill!(st.E, 0f0)
    fill!(st.H_out, 0f0)

    # Project features (each col represents a node) into key and query vectors
    mul!(st.Q, ps.Wq, H)
    mul!(st.K, ps.Wk, H)

    # Get attention scores
    attention!v2(
        st.E,
        st.Q, st.K, 
        A,
        l.d
    )

    right_sparse_mul!(st.H_out, H, st.E, A)

    return (st.H_out, A), st
end

# ========== Build Multi-Head GAT ==========

struct MultiHeadGAT{N, H<:GATLayer, Bool} <: Lux.AbstractLuxLayer
    heads::NTuple{N, H}
    concat::Bool # true = concat, false = average
end

# Initial parameters
function Lux.initialparameters(rng::AbstractRNG, l::MultiHeadGAT)
    ps = NamedTuple{Tuple(Symbol("head$i") for i in 1:length(l.heads))}(
        ntuple(i -> Lux.initialparameters(rng, l.heads[i]), length(l.heads))
    )
    return ps
end

# Initial states
function Lux.initialstates(rng::AbstractRNG, l::MultiHeadGAT)
    st = NamedTuple{Tuple(Symbol("head$i") for i in 1:length(l.heads))}(
        ntuple(i -> Lux.initialstates(rng, l.heads[i]), length(l.heads))
    )
    return st
end

# Dispatcher to allow `model(x, ps, st; concat=true|false)`
(l::MultiHeadGAT)(
    x, ps, st; concat = l.concat) = l(x, ps, st, Val(concat))

# Forward pass: concatenation mode
function (l::MultiHeadGAT)(x, ps, st, ::Val{true})
    H, A = x
    outputs = Matrix{Float32}[]
    new_states = NamedTuple()

    for i in 1:length(l.heads)
        head = l.heads[i]
        pname = Symbol("head$i")
        head_ps = ps[pname]
        head_st = st[pname]

        (out, A), new_head_st = head((H, A), head_ps, head_st)
        push!(outputs, out)

        new_states = merge(new_states, NamedTuple{(pname,)}((new_head_st,)))
    end

    return (reduce(vcat, outputs)::Matrix{Float32}, A), new_states
end

# Forward pass: averaging mode
function (l::MultiHeadGAT)(x, ps, st, ::Val{false})
    H, A = x
    outputs = Matrix{Float32}[]
    new_states = NamedTuple()

    for i in 1:length(l.heads)
        head = l.heads[i]
        pname = Symbol("head$i")
        head_ps = ps[pname]
        head_st = st[pname]

        (out, A), new_head_st = head((H, A), head_ps, head_st)
        push!(outputs, out)

        new_states = merge(new_states, NamedTuple{(pname,)}((new_head_st,)))
    end

    stacked = cat(outputs...; dims=3)               # (F_out, N, num_heads)
    final = dropdims(mean(stacked; dims=3), dims=3) # (F_out, N)
    return (final::Matrix{Float32}, A), new_states
end

# ==== Try with Cora data ===== #
function group_by_col(mypairs::Vector{Tuple{Int, Int}}, N::Int)
    grouped = [Tuple{Int, Int}[] for _ in 1:N]
    for (i, j) in mypairs
        push!(grouped[j], (i, j))
    end
    return grouped  # Vector{Vector{Tuple{Int, Int}}} of length N
end

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
    AdjM = create_adj(edges, n; undirected = true, selfloops = true)
    mypairs = [(i, j) for i in axes(AdjM, 1), j in axes(AdjM, 2) if AdjM[i,j]]
    A = group_by_col(mypairs, n)

    train_idx, val_idx, test_idx = (1:140, 141:640, 1709:2708)

    return H, y, AdjM, A, (f_in, f_out), (train_idx, val_idx, test_idx)
end

function main(;
    num_heads = 2,
    hidden_dim = 64,
    learning_rate = 0.005f0,
    epochs = 100, 
    patience = 20
)
    H, y, _, A, (f_in, f_out), (train_idx, val_idx, test_idx) = cora_data()

    rng = Random.MersenneTwister(23)

    model = Lux.Chain(
        MultiHeadGAT(
            ntuple(_ -> GATLayer(
                f_in, hidden_dim; 
                N = size(H, 2)), num_heads), 
            false),
        x -> x[1],
        LayerNorm((f_in,)),
        Dense(f_in, f_out, swish)
    )
    ps, st = Lux.setup(rng, model)

    opt = AdamW(learning_rate)
    opt_state = setup(opt, ps)

    @printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps) / 1.0e6)

    function loss_function(ps, st, model::Chain, (x, y, adj, mask))
        y_pred, st = model((x, adj), ps, st)
        loss = CrossEntropyLoss(; agg=mean, logits=Val(true))(
            y_pred[:, mask], 
            y[:, mask]
        )
        return loss, st, (; y_pred)
    end

    accuracy(y_pred, y) = mean(onecold(y_pred) .== onecold(y)) * 100
    train_loss(ps, st, model::Chain, (X, y, adj, mask)) = loss_function(ps, st, model::Chain, (X, y, adj, mask))[1]

    best_loss_val = Inf
    cnt = 0

    #backend = DI.AutoMooncake(; config=nothing)
    backend = DI.AutoEnzyme(mode=set_runtime_activity(Enzyme.Reverse))

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
            )[1],
        Array(y),
    )
    @printf "Test Loss: %.6f\tTest Acc: %.4f%%\n" test_loss test_acc
    return test_loss, test_acc
end

Enzyme.API.strictAliasing!(false)
test_loss, test_acc = main(
    num_heads = 8,
    hidden_dim = 8,
    learning_rate = 0.005f0,
    epochs = 50, 
    patience = 10
)

# ==== tests ==== #
H, y, AdjM, A, (f_in, f_out), (train_idx, val_idx, test_idx) = cora_data()
N = size(H, 2)
rng = Random.MersenneTwister(23)

model = GATLayer(f_in, 2*f_out; N = N)
ps, st = Lux.setup(rng, model)

(y_pred), st = model((H, A), ps, st)

using BenchmarkTools
@btime model((H, A), ps, st)
# 6.373 ms (2 allocations: 96 bytes)

@code_warntype model((H, A), ps, st)

@btime attention!v2($st.E, $st.Q, $st.K, $A, $model.d)

model = Lux.Chain(
    MultiHeadGAT(
        ntuple(_ -> GATLayer(
            f_in, 64; N = size(H, 2)), 8), 
        false),
    x -> x[1],
    LayerNorm((f_in,)),
    Dense(f_in, f_out, leakyrelu)
)
ps, st = Lux.setup(rng, model)
