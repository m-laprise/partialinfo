using EvoTrees
using MLJModelInterface
using Statistics

# RANDOM DATA
config = EvoTreeRegressor(
    loss=:mse, 
    nrounds=100, 
    max_depth=6,
    nbins=32,
    eta=0.1)

x_train = rand(1000, 10)
y_train = vec(randn(1000) .+ sum(x_train[:,1:4], dims=2) .+ sum(abs2, x_train[:,5:10], dims=2))
m = EvoTrees.fit_evotree(config; x_train, y_train)
preds = m(x_train)
mean((preds - y_train).^2)

using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], title="Residuals vs. Predictions",
          xlabel="Predictions", ylabel="Residuals")
scatter!(ax, preds, y_train .- preds, markersize=2)
ax2 = Axis(fig[2, 1], title="Residuals vs. True values",
          xlabel="True values", ylabel="Residuals")
scatter!(ax2, y_train, y_train .- preds, markersize=2)
ax3 = Axis(fig[1, 2], title="Residuals vs. True values",
          xlabel="True values", ylabel="Predictions")
scatter!(ax3, y_train, preds, markersize=2)
display(fig)


# BOSTON HOUSING DATA - REGRESSION
using EvoTrees
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random

df = MLDatasets.BostonHousing().dataframe

#Preprocessing
#split data according to train and eval indices, and separate features from the target variable.
Random.seed!(123)

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

train_data = df[train_indices, :]
eval_data = df[setdiff(1:nrow(df), train_indices), :]

x_train, y_train = Matrix(train_data[:, Not(:MEDV)]), train_data[:, :MEDV]
x_eval, y_eval = Matrix(eval_data[:, Not(:MEDV)]), eval_data[:, :MEDV]

#Training
#pass optional x_eval and y_eval arguments, which enable the usage of early stopping.

config = EvoTreeRegressor(
    nrounds=200, 
    early_stopping_rounds=10,
    eta=0.2, 
    max_depth=4, 
    lambda=1, 
    rowsample=0.7, 
    colsample=0.7)

model = EvoTrees.fit(config;
    x_train, y_train,
    x_eval, y_eval,
    print_every_n=10)

#get predictions by passing training and testing data to our model.
#apply various evaluation metric, such as the MAE (mean absolute error):

pred_train = model(x_train)
pred_eval = model(x_eval)

mean(abs, pred_train .- y_train)
mean(abs, pred_eval .- y_eval)

mean(abs2, pred_train .- y_train)
mean(abs2, pred_eval .- y_eval)

EvoTrees.importance(model)

using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], title="Residuals vs. Predictions",
          xlabel="Predictions", ylabel="Residuals")
scatter!(ax, pred_eval, y_eval .- pred_eval, markersize=4)
ax2 = Axis(fig[2, 1], title="Residuals vs. True values",
          xlabel="True values", ylabel="Residuals")
scatter!(ax2, y_eval, y_eval .- pred_eval, markersize=4)
ax3 = Axis(fig[1, 2], title="Predictions vs. True values",
          xlabel="True values", ylabel="Predictions")
scatter!(ax3, y_eval, pred_eval, markersize=4)
display(fig)

# XGBoost version

using XGBoost
using Term
using OrderedCollections: OrderedDict
# create and train a gradient boosted tree model
bst = xgboost(
    (x_train, y_train), 
    num_round=200, 
    early_stopping_rounds=10,
    watchlist=OrderedDict(
        "train" => XGBoost.DMatrix(x_train, y_train), 
        "eval" => XGBoost.DMatrix(x_eval, y_eval)),
    max_depth=4, 
    eta=0.2,
    objective="reg:squarederror",
    subsample=0.7,
    colsample_bytree=0.7,
    alpha=1,
)
# obtain model predictions
ŷ = predict(bst, x_train)
ŷeval = predict(bst, x_eval)

#mse
mean(abs2, ŷ - y_train)
mean(abs2, ŷeval - y_eval)


# display importance statistics retaining feature names
importancereport(bst)
importance(bst)
