# Source: https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/tutorials/gnn_intro/

using Lux, GNNLux
using MLDatasets
using LinearAlgebra, Random, Statistics
import GraphMakie
import CairoMakie as Makie
using Enzyme, Optimisers, OneHotArrays

#ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
rng = Random.seed!(10) # for reproducibility

dataset = MLDatasets.KarateClub()

karate = dataset[1]

karate.node_data.labels_comm

g = mldataset2gnngraph(dataset)

x = zeros(Float32, g.num_nodes, g.num_nodes)
x[diagind(x)] .= 1

train_mask = [true, false, false, false, true, false, false, false, true,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, true, false, false, false, false, false,
    false, false, false, false]

labels = g.ndata.labels_comm
y = onehotbatch(labels, 0:3)

g = GNNGraph(g, ndata = (; x, y, train_mask))

println("Number of nodes: $(g.num_nodes)")
println("Number of edges: $(g.num_edges)")
println("Average node degree: $(g.num_edges / g.num_nodes)")
println("Number of training nodes: $(sum(g.ndata.train_mask))")
println("Training node label rate: $(mean(g.ndata.train_mask))")
println("Has self-loops: $(has_self_loops(g))")
println("Is undirected: $(is_bidirected(g))")

edge_index(g)

GraphMakie.graphplot(g |> to_unidirected, node_size = 20, node_color = labels, arrow_show = false)

Lux.@concrete struct GCN <: GNNContainerLayer{(:conv1, :conv2, :conv3, :dense)}
    nf::Int
    nc::Int
    hd1::Int
    hd2::Int
    conv1
    conv2
    conv3
    dense
    use_bias::Bool
    init_weight
    init_bias
end

function GCN(num_features, num_classes, hidden_dim1, hidden_dim2; use_bias = true, init_weight = glorot_uniform, init_bias = zeros32) # constructor
    conv1 = GCNConv(num_features => hidden_dim1)
    conv2 = GCNConv(hidden_dim1 => hidden_dim1)
    conv3 = GCNConv(hidden_dim1 => hidden_dim2)
    dense = Dense(hidden_dim2, num_classes)
    return GCN(num_features, num_classes, hidden_dim1, hidden_dim2, conv1, conv2, conv3, dense, use_bias, init_weight, init_bias)
end

function (gcn::GCN)(g::GNNGraph, x, ps, st) # forward pass
    dense = StatefulLuxLayer{true}(gcn.dense, ps.dense, GNNLux._getstate(st, :dense))
    x, stconv1 = gcn.conv1(g, x, ps.conv1, st.conv1)
    x = tanh.(x)
    x, stconv2 = gcn.conv2(g, x, ps.conv2, st.conv2)
    x = tanh.(x)
    x, stconv3 = gcn.conv3(g, x, ps.conv3, st.conv3)
    x = tanh.(x)
    out = dense(x)
    return (out, x), (conv1 = stconv1, conv2 = stconv2, conv3 = stconv3)
end


function LuxCore.initialparameters(rng::TaskLocalRNG, l::GCN) # initialize model parameters
    weight_c1 = l.init_weight(rng, l.hd1, l.nf)
    weight_c2 = l.init_weight(rng, l.hd1, l.hd1)
    weight_c3 = l.init_weight(rng, l.hd2, l.hd1)
    weight_d = l.init_weight(rng, l.nc, l.hd2)
    if l.use_bias
        bias_c1 = l.init_bias(rng, l.hd1)
        bias_c2 = l.init_bias(rng, l.hd1)
        bias_c3 = l.init_bias(rng, l.hd2)
        bias_d = l.init_bias(rng, l.nc)
        return (
            ; conv1 = ( weight = weight_c1, bias = bias_c1), 
            conv2 = ( weight = weight_c2, bias = bias_c2), 
            conv3 = ( weight = weight_c3, bias = bias_c3), 
            dense = ( weight = weight_d,bias =  bias_d))
    end
    return (
        ; conv1 = ( weight = weight_c1), 
        conv2 = ( weight = weight_c2), 
        conv3 = ( weight = weight_c3), 
        dense = ( weight_d))
end

num_features = 34
num_classes = 4
hidden_dim1 = 4
hidden_dim2 = 2

gcn = GCN(num_features, num_classes, hidden_dim1, hidden_dim2)
ps, st = LuxCore.setup(rng, gcn)

(ŷ, emb_init), st = gcn(g, g.x, ps, st)

function visualize_embeddings(h; colors = nothing)
    xs = h[1, :] |> vec
    ys = h[2, :] |> vec
    Makie.scatter(xs, ys, color = labels, markersize = 20)
end

visualize_embeddings(emb_init, colors = labels)

function custom_loss(gcn, ps, st, tuple)
    g, x, y = tuple
    logitcrossentropy = CrossEntropyLoss(; logits=Val(true))
    (ŷ, _) ,st = gcn(g, x, ps, st)
    return  logitcrossentropy(ŷ[:, train_mask], y[:, train_mask]), (st), 0
end

using Zygote

function train_model!(gcn, ps, st, g)
    train_state = Lux.Training.TrainState(gcn, ps, st, Adam(1e-2))
    for iter in 1:2000
            _, loss, _, train_state = Lux.Training.single_train_step!(
                AutoZygote(), 
                custom_loss,(g, g.x, g.y), 
                train_state)

        if iter % 100 == 0
            println("Epoch: $(iter) Loss: $(loss)")
        end
    end

    return gcn, ps, st
end

gcn, ps, st = train_model!(gcn, ps, st, g);

(ŷ, emb_final), st = gcn(g, g.x, ps, st)
mean(onecold(ŷ[:, train_mask]) .== onecold(g.y[:, train_mask]))
mean(onecold(ŷ[:, .!train_mask]) .== onecold(y[:, .!train_mask]))

visualize_embeddings(emb_final, colors = labels)
