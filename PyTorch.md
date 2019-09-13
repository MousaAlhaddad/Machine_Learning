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
