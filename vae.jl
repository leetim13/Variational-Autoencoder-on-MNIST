# import Automatic Differentiation
# You may use Neural Network Framework, but only for building MLPs
# i.e. no fancy probabilistic implementations
using Flux
using MLDatasets
using Statistics
using Logging
using Test
using Random
using StatsFuns: log1pexp
Random.seed!(412414);

#### Probability Stuff
# Make sure you test these against a standard implementation!

# log-pdf of x under Factorized or Diagonal Gaussian N(x|μ,σI)
function factorized_gaussian_log_density(mu, logsig,xs)
  """
  mu and logsig either same size as x in batch or same as whole batch
  returns a 1 x batchsize array of likelihoods
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end

# log-pdf of x under Bernoulli
function bernoulli_log_density(logit_means,x)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
  b = x .* 2 .- 1 # {0,1} -> {-1,1}
  return - log1pexp.(-b .* logit_means)
end
## This is really bernoulli
@testset "test stable bernoulli" begin
  using Distributions
  x = rand(10,100) .> 0.5
  μ = rand(10)
  logit_μ = log.(μ./(1 .- μ))
  @test logpdf.(Bernoulli.(μ),x) ≈ bernoulli_log_density(logit_μ,x)
  # over i.i.d. batch
  @test sum(logpdf.(Bernoulli.(μ),x),dims=1) ≈ sum(bernoulli_log_density(logit_μ,x),dims=1)
end

# sample from Diagonal Gaussian x~N(μ,σI) (hint: use reparameterization trick here)
sample_diag_gaussian(μ,logσ) = (ϵ = randn(size(μ)); μ .+ exp.(logσ).*ϵ)
# sample from Bernoulli (this can just be supplied by library)
sample_bernoulli(θ) = rand.(Bernoulli.(θ))

# Load MNIST data, binarise it, split into train and test sets (10000 each) and partition train into mini-batches of M=100.
# You may use the utilities from A2, or dataloaders provided by a framework
function load_binarized_mnist(train_size=10000, test_size=1000)
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
batch_x(x::AbstractArray, batch_size=100) = first.(batch_data((x,zeros(size(x)[end])),batch_size))


### Implementing the model

## Load the Data
train_data, test_data = load_binarized_mnist(10000, 10000);#change train_size to 10000 (@Jesse in ProbML forum)
train_x, train_label = train_data;
test_x, test_label = test_data;

## Test the dimensions of loaded data
@testset "correct dimensions" begin
  @test size(train_x) == (784,10000)
  @test size(train_label) == (10000,)
  @test size(test_x) == (784,10000)
  @test size(test_label) == (10000,)
end

## Model Dimensionality
# #### Set up model according to Appendix C (using Bernoulli decoder for Binarized MNIST)
# Set latent dimensionality=2 and number of hidden units=500.
Dz, Dh = 2, 500
Ddata = 28^2

# ## Generative Model
# This will require implementing a simple MLP neural network
# See example_flux_model.jl for inspiration
# Further, you should read the Basics section of the Flux.jl documentation
# https://fluxml.ai/Flux.jl/stable/models/basics/
# that goes over the simple functions you will use.
# You will see that there's nothing magical going on inside these neural network libraries
# and when you implemented a neural network in previous assignments you did most of the work.
# If you want more information about how to use the functions from Flux, you can always reference
# the internal docs for each function by typing `?` into the REPL:
# ? Chain
# ? Dense
decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))

## Model Distributions
#note sig is logsig
log_prior(z) = factorized_gaussian_log_density(0,0,z)#TODO

function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
  θ = decoder(z) # TODO: parameters decoded from latent z
  return  sum(bernoulli_log_density(θ,x), dims=1)# return likelihood for each element in batch
end

joint_log_density(x,z) = log_prior(z) .+ log_likelihood(x,z) #TODO

## Amortized Inference
function unpack_gaussian_params(θ)
  μ, logσ = θ[1:2,:], θ[3:end,:]
  return  μ, logσ
end

encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, Dz*2), unpack_gaussian_params)

#TODO
# Hint: last "layer" in Chain can be 'unpack_gaussian_params'

log_q(q_μ, q_logσ, z) = factorized_gaussian_log_density(q_μ, q_logσ, z) #TODO: write log likelihood under variational distribution.

function elbo(x)
  q_μ, q_logσ  = encoder(x)# variational parameters from data
  z = sample_diag_gaussian(q_μ, q_logσ)# sample from variational distribution
  joint_ll = joint_log_density(x,z)# joint likelihood of z and x under model
  log_q_z = log_q(q_μ, q_logσ, z) #likelihood of z under variational distribution
  elbo_estimate = sum(joint_ll .- log_q_z)/size(q_μ)[2] #mean of elbo
  return elbo_estimate
end

function loss(x)
  return -1*elbo(x) #TODO: scalar value for the variational loss over elements in the batch
end

# Training with gradient optimization:
# See example_flux_model.jl for inspiration

function train_model_params!(loss, encoder, decoder, train_x, test_x; nepochs=10)
  # model params
  ps = Flux.params(encoder, decoder) # parameters to update with gradient descent
  # ADAM optimizer with default parameters
  opt = ADAM()
  # over batches of the data
  for i in 1:nepochs
    for d in batch_x(train_x)
      gs = Flux.gradient(ps) do
        batch_loss = loss(d)
        return batch_loss # compute gradients with respect to variational loss over batch
      end
      Flux.Optimise.update!(opt,ps,gs) # update the paramters with gradients
    end
    if i%1 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(batch_x(test_x)[1]))"
    end
  end
  @info "Parameters of encoder and decoder trained!"
end


## Train the model
train_model_params!(loss,encoder,decoder,train_x,test_x, nepochs=100)

### Save the trained model!
using BSON:@save
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
save_dir = "trained_models"
if !(isdir(save_dir))
  mkdir(save_dir)
  @info "Created save directory $save_dir"
end
@save joinpath(save_dir,"encoder_params.bson") encoder
@save joinpath(save_dir,"decoder_params.bson") decoder
@info "Saved model params in $save_dir"

## Load the trained model!
using BSON:@load
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
load_dir = "trained_models"
@load joinpath(load_dir,"encoder_params.bson") encoder
@load joinpath(load_dir,"decoder_params.bson") decoder
@info "Load model params from $load_dir"


# Visualization
using Images
using Plots
using StatsFuns #added package StatsFuns
# make vector of digits into images, works on batches also
mnist_img(x) = ndims(x)==2 ? Gray.(permutedims(reshape(x,28,28,:), [2, 1, 3])) : Gray.(transpose(reshape(x,28,28)))

## Example for how to use mnist_img to plot digit from training data
plot(mnist_img(train_x[:,1]))

########################################################
plot() #clear plot first

#Question 3a
num_samples = 10 #10 samples z

# save each of the plots in a list
plot_ber_means_lst = Any[] #plot list of bernoulli means
plot_binary_lst = Any[] #plot list of binary image from  product of Bernoullis

for i in 1:num_samples
  μ = Vector([0;0]) #init mu
  logσ = Vector([0;0]) #init log sig
  sample_z_from_prior = sample_diag_gaussian(μ,logσ) # sample a z from the prior.
  decoded_x = decoder(sample_z_from_prior)
  mean_vector = 1.0 ./ (1.0 .+ exp.(-1 * decoded_x)) #apply logistic
  push!(plot_ber_means_lst, plot(mnist_img(mean_vector))) #push to plot_ber_means list
  push!(plot_binary_lst, plot(mnist_img(sample_bernoulli(mean_vector)))) #push to plot_binary list
end
plot_3a = [plot_ber_means_lst; plot_binary_lst] #concat these two plot lists
display(plot(plot_3a..., layout=grid(2, 10), size=(2000, 500)))
savefig(joinpath("plots", "3a.png")) #save plots to png

#Question 3b
plot() #clear plot

plot_mean_1 = [[] for i = 1:10] #init x array
plot_mean_2 = [[] for i = 1:10] #init y array
# for i in 1:10
#   push!(plot_mean_1, Array{Any,2}())
#   push!(plot_mean_2, [])
# end
# plot_mean_1 = fill([], 10)
# plot_mean_2 = fill([], 10)

#create lables 0 to 9
lables_vec = ["$i" for i = 0:9] #(d) Colour each point according to the class label (0 to 9)
lables_lst = reshape(lables_vec, 1, length(lables_vec))
length_ = size(train_label)[1]

for i in 1:length_
  mu, logsig = encoder(train_x[:,i]) #(a) Encode each image in the training set
  push!(plot_mean_1[train_label[i]+1], mu[1])
  push!(plot_mean_2[train_label[i]+1], mu[2])
end
plot(plot_mean_1, plot_mean_2,seriestype = :scatter, xlabel="mean of z1 encoding", ylabel="mean of z2 encoding", label=lables_lst)
savefig(joinpath("plots","3b.png"))

# Q3 (c)
function interpolate(za, zb, alpha)
   alpha .* za .+ (1 .- alpha) .* zb
end

#plot random numbers to find different class
plot(mnist_img(train_x[:,1111])) #number 4
four_ = train_x[:,1111]
plot(mnist_img(train_x[:,2222])) #number 9
nine_ = train_x[:,2222]
plot(mnist_img(train_x[:,3333])) #number 3
three_ = train_x[:,3333]
plot(mnist_img(train_x[:,5549])) #number 8
eight_ = train_x[:,5549]
plot(mnist_img(train_x[:,6666])) #number 1
one_ = train_x[:,6666]
plot(mnist_img(train_x[:,2])) #number 0
zero_ = train_x[:,2]

# 3 pairs of different class
pair_eight_three = (eight_, three_)
pair_four_zero = (four_, zero_)
pair_one_nine = (one_, nine_)

#list of all pairs
all_pairs_lst = [pair_eight_three , pair_four_zero, pair_one_nine]

plots_lst = [] #init lst for plots
for i in 1:3
  pair = all_pairs_lst[i]
  class_a = pair[1]
  class_b = pair[2]

  class_a_encoder = encoder(class_a)
  class_a_mu = class_a_encoder[1]
  class_a_sig = exp.(class_a_encoder[2])

  class_b_encoder = encoder(class_b)
  class_b_mu = class_b_encoder[1]
  class_b_sig = exp.(class_b_encoder[2])

  for j in 1:10 #j from 1 to 10 = 10 images in total per row
    alpha = (j-1)/9 #equally space alpha to plot 10 images per row
    interpolate_mu_alpha = interpolate(class_a_mu, class_b_mu, alpha)
    decoded_mu_alpha = decoder(interpolate_mu_alpha)
    ber_logit = exp.(decoded_mu_alpha) ./ (1 .+ exp.(decoded_mu_alpha))
    push!(plots_lst, plot(mnist_img(ber_logit[:]))) #Concatenate these plots into one figure.
  end
end
print(size(plots_lst))
display(plot(plots_lst..., layout=grid(3,10), size =(2000, 600)))
savefig(joinpath("plots","3c.png"))
plot()




# Q4 a
#Check dim of train_x[:,i]
dim_train_x = size(train_x[:,3]) #Assume dim = 1; flattened
total_pixels = 28*28
half_pixels = Int(total_pixels/2)
function top_half_image_x(x)
  #Assume x is flattened alr.
  # returns only the top half of a 28x28 array.
    return x[1:half_pixels]
end

function log_p_given_z(top_half_x,z)
  # computes log p(top half of image x|z)
  decoded_z = decoder(z)
  top_half_decoded = top_half_image_x(decoded_z)
  return  sum(bernoulli_log_density(top_half_decoded, top_half_x), dims=1) #returns likelihood
end

function joint_log_top_half(x,z)
  # computes the log joint density log p(z,top half of image x) for each z in the array.
  top_half_x = top_half_image_x(x)
  log_prior(z) .+ log_p_given_z(top_half_x, z)
end

# Q4b
# Initialize variational parameters
phi_mu = randn(Dz)
phi_log_sig = randn(Dz)
v_params = (phi_mu, phi_log_sig) #init variational params

function elbo_estimate(x, v_params, k_num_samples)
#modified from previous question
  mu_vec = v_params[1]
  log_sig_vec = v_params[2]
  sig_vec = exp.(log_sig_vec)

  z_sample = sig_vec.*randn(Dz, k_num_samples) .+ mu_vec
  joint_ll = joint_log_top_half(x, z_sample)
  log_q = factorized_gaussian_log_density(mu_vec, log_sig_vec, z_sample)
  return mean(joint_ll - log_q)
end

function fit_variational_dist(init_params, x; num_itrs=200, lr= 1e-2, k = 10)
  #modified from A2
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(param_theta->elbo_estimate(x, param_theta, k), params_cur)
    params_cur =  params_cur .+ lr.*grad_params[1]
    @info "negative elbo estimate: $(elbo_estimate(x, params_cur, k))"
  end
  return params_cur
end

image_idx = rand((1:10000)) #randomly select an index in train_x
plot(mnist_img(train_x[:,image_idx]))
#optimize and train elbo
trained_params = fit_variational_dist(v_params, train_x[:,image_idx])

function skillcontour!(f; colour=nothing)
  #taken from A2_src.jl
  n = 100
  x = range(-3,stop=3,length=n)
  y = range(-3,stop=3,length=n)
  z_grid = Iterators.product(x,y) # meshgrid for contour
  z_grid = reshape.(collect.(z_grid),:,1) # add single batch dim
  z = f.(z_grid)
  z = getindex.(z,1)'
  max_z = maximum(z)
  levels = [.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2] .* max_z
  if colour==nothing
  p1 = contour!(x, y, z, fill=false, levels=levels)
  else
  p1 = contour!(x, y, z, fill=false, c=colour,levels=levels,colorbar=false)
  end
  plot!(p1)
end

plot(title="Plot of isocountours of the joint and approximate posterior")
gaussian_1(zs) = exp(joint_log_top_half(train_x[:,image_idx], zs))
skillcontour!(gaussian_1,colour=:red)
gaussian_2(zs) = exp(factorized_gaussian_log_density(trained_params[1], trained_params[2], zs))
display(skillcontour!(gaussian_2, colour=:blue))
savefig(joinpath("plots", "4bd_$image_idx.png"))



z_sample = sample_diag_gaussian(trained_params[1], trained_params[2])
decoded_z = decoder(z_sample)
mean_vector = 1.0 ./ (1.0 .+ exp.(-1 * decoded_z)) #apply logistic
concat_image_top_bottom = vcat(top_half_image_x(train_x[:,image_idx]), mean_vector[half_pixels+1:total_pixels])
final_merged_image = reshape(concat_image_top_bottom, total_pixels)

plot_lst = [] #init list for plots
push!(plot_lst, plot(mnist_img(final_merged_image)))
push!(plot_lst, plot(mnist_img(train_x[:,image_idx])))
display(plot(plot_lst..., size=(2000, 600)))
savefig(joinpath("plots", "4be_$image_idx.png"))
