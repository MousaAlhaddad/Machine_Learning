# PyTorch
Pytorch is a framework for building and training neural networks. The fundamental data structure for 
neural networks are tensors and PyTorch is built around tensors. PyTorch tensors can be added, multiplied, subtracted, etc,
just like Numpy arrays. In


## Coding 
### Basics 
  # import Pytorch 
  import torch
  # Set a random seed 
  torch.manual_seed(seed_int)
  # Return a tensor filled with random numbers
  torch.randn(size_int_tuple)
  # Return a tensor with the same size as input
  torch.rand_like(input_tensor)
  # 
  
## Basics
  import torch
  torch.manual_seed(7) # Set the random seed so things are predictable
  torch.randn((1, 5))
  torch.mm(features,weights.view((5,1))) # Remember that for matrix multiplications, the number of columns in the first tensor must
  equal to the number of rows in the second column.
  weights.view(a, b) # return a new tensor with the same data as weights with size (a, b).
  ### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

activation(torch.mm(activation(torch.mm(features,W1)+B1),W2)+B2)

torch.from_numpy(a)

b.numpy()

The memory is shared between the Numpy array and Torch tensor, 
so if you change the values in-place of one object, the other will change as well.
