# PyTorch
Pytorch is a framework for building and training neural networks. The fundamental data structure for 
neural networks are tensors and PyTorch is built around tensors. PyTorch tensors can be added, multiplied, subtracted, etc,
just like Numpy arrays. 


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
    
    # Perform a matrix multiplication
      '''
      If input is a (n×m) tensor, and mat2 is a (m×p) tensor, out will be a (n×p) tensor. Remember that
      for matrix multiplications, the number of columns in the first tensor must equal to the number of
      rows in the second column.
      '''
    torch.mm(input_tensor, mat2_tensor)
    
    # Return a new tensor with the same data as the self tensor but of a different shape
    torch.Tensor.view(*shape_int)
    
    # Convert a numpy array into a pytorch tensor 
      '''
      The memory is shared between the Numpy array and Torch tensor, 
      so if you change the values in-place of one object, the other will change as well.
      '''
    torch.from_numpy(array)
    
    # Convert a pytorch tensor into a numpy array 
      '''
      The memory is shared between the Numpy array and Torch tensor, 
      so if you change the values in-place of one object, the other will change as well.
      '''
    torch.Tensor.numpy(tensor)
  
### Torchvision
        # Import datasets and transfoms 
        from torchvision import datasets, transforms
        
        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
                              
        # Download and load the training MNIST data data
         '''
            root (string) - Root directory of dataset where MNIST/processed/training.pt and
                MNIST/processed/test.pt exist.
            train (bool, optional) - If True, creates dataset from training.pt, otherwise 
                from test.pt.
            download (bool, optional) – If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not downloaded again.
            transform (callable, optional) – A function/transform that takes in an PIL image
                and returns a transformed version.
            '''
        trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
           
            
        # Combine a dataset and a sampler, and provide an iterable over the given dataset
            '''
            dataset (Dataset): dataset from which to load the data
            batch_size (int, optional): how many samples per batch to load (default: 1)
            shuffle (bool, optional): set to True to have the data reshuffled at every 
                epoch (default: False)
            '''
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
        # Iterate inside a loader 
            # 1
        i = 0
        for images, labels in loader:
            i += len(labels)
        i
            # 2
        dataiter = iter(loader)
        images, labels = dataiter.next()
        print(type(images))
        print(images.shape)
        print(labels.shape)
        
        # Showing an image 
        plt.imshow(image_tensor.numpy().squeeze(), cmap='Greys_r')
        
        # Flatten the images of a loader 
        images = images.view(images.shape[0],-1)
   # Neural Networks
        # Import the nn module 
        from torch import nn
        
        # Building networks
            # 1
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                # Create a hidden layer of linear transformation
                self.hidden = nn.Linear(input_int, hidden_int)
                # Create an output layer of linear transformation
                self.output = nn.Linear(hidden_int, output_int)
                    '''
                    In practice, the ReLU function (nn.relu) is used almost exclusively as the activation
                        function for hidden layers.
                    '''
                # Define sigmoid activation and softmax output 
                self.sigmoid = nn.Sigmoid()
                self.softmax = nn.Softmax(dim=1)
                
            def forward(self, x):
                # Pass the input tensor through each operation
                x = self.hidden(x)
                x = self.sigmoid(x)
                x = self.output(x)
                x = self.softmax(x)
                return x
                
        # Create the network 
        model = Network()
        
            # 2 
        import torch.nn.functional as F
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                # Create a hidden layer of linear transformation
                self.hidden = nn.Linear(input_int, hidden_int)
                # Create an output layer of linear transformation
                self.output = nn.Linear(hidden_int, output_int)
       
            def forward(self, x):
                    '''
                    In practice, the ReLU function (F.relu) is used almost exclusively as the activation
                        function for hidden layers.
                    '''
                # Create a sigmoid activation to the hidden layer
                x = F.sigmoid(self.hidden(x))
                # Create a softmax activation to the output layer
                x = F.softmax(self.output(x), dim=1)
                return x
                
        # Create the network 
        model = Network()
        
        # Show the text representation of the model
        model 
        
        # Show the the weight and bias tensors of the hidden layer
        model.hidden.weight
        model.hidden.bias
        
        # Calculate probabilities 
            # 1 If there is no activation function at the end of the neural network: 
        def softmax(x):
            return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)
        probabilities = softmax(out)
        
        
        
        
            

