import torch
from data_preparation import GetImages


with open('training_loader.pkl', 'rb') as rb:
    training_data = torch.load(rb)
    print(training_data)

