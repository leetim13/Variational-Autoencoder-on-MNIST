using Flux
using MLDatasets
using Statistics
using Logging
using Test
using Random

# Load MNIST data, binarise it, split into train and test sets (10000 each) and partition train into mini-batches of M=100.
# You may use the utilities from A2, or dataloaders provided by a framework
function load_binarized_mnist(train_size=1000, test_size=1000)
  train_x, train_label = MNIST.traindata(1:train_size);
  test_x, test_label = MNIST.testdata(1:test_size);
  @info "Loaded MNIST digits with dimensionality $(size(train_x))"
  train_x = reshape(train_x, 28*28,:)
  test_x = reshape(test_x, 28*28,:)
  @info "Reshaped MNIST digits to vectors, dimensionality $(size(train_x))"
  train_x = train_x .> 0.5; #binarize
  test_x = test_x .> 0.5; #binarize
  @info "Binarized the pixels"
  return (train_x, train_label), (test_x, test_label)
end

function batch_data((x,label)::Tuple, batch_size=100)
  """
  Shuffle both data and image and put into batches
  """
  N = size(x)[end] # number of examples in set
  rand_idx = shuffle(1:N) # randomly shuffle batch elements
  batch_idx = Iterators.partition(rand_idx,batch_size) # split into batches
  batch_x = [x[:,i] for i in batch_idx]
  batch_label = [label[i] for i in batch_idx]
  return zip(batch_x, batch_label)
end
# if you only want to batch xs
batch_x(x::Array, batch_size=100) = batch_data((x,zeros(size(x)[end])),batch_size)[1]



### Implementing the model

## Load the Data
train_data, test_data = load_binarized_mnist();
train_x, train_label = train_data;
test_x, test_label = test_data;

## Model Dimensionality
Dh = 500
Ddata = 28^2


### TL;DR Flux Documentation
# find more docs here: https://fluxml.ai/Flux.jl/stable/models/basics/
#
# How to make a simple neural network
simple = Chain(Dense(Ddata,Dh, tanh), Dense(Dh, 10, σ))

# Try it on data
simple(train_x)

# write a loss function
classification_loss(x,label) = mean((simple(x) .- Flux.onehotbatch(label,0:9)).^2) 

# try it on data and one hot labels
classification_loss(train_x, train_label)

# Take gradients, this uses Zyogte under the hood
# But makes it convenient to take gradients wrt 
# the paramters of the model:
θ = Flux.params(simple)

# Note that the first argument still is a function,
# so we make it anonymous with no arguments ()->...
gs = Flux.gradient(
                   ()-> classification_loss(train_x, train_label),
                   θ)

# Equivalently, here's a slightly fancier syntax with "do" blocks:
#
gs = Flux.gradient(θ) do
  loss = classification_loss(train_x,train_label)
  return loss
end

 
# Gradient optimization:
# ADAM optimizer with default parameters
opt = ADAM()

# update the paramters with gradients
Flux.Optimise.update!(opt,θ,gs)

# Do these in a training loop
# over batches of the data
# model params
θ = Flux.params(simple)
# ADAM optimizer with default parameters
opt = ADAM()
# over batches of the data
for i in 1:10
  for d in batch_data(train_data)
    gs = Flux.gradient(θ) do
      batch_loss = classification_loss(d...)
      return batch_loss
    end
    # 
    # update the paramters with gradients
    Flux.Optimise.update!(opt,θ,gs)
  end
  if i%1 == 0 # change 1 to higher number to compute and print less frequently
    @info "Test loss at epoch $i: $(classification_loss(test_data...))"
  end
end
