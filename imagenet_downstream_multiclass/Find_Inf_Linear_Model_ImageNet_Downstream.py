#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
import torchvision.transforms as transforms
# from torchvision.models import resnet50, ResNet50_Weight
from data_utils.cub2011 import Cub2011
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pytorch_influence_functions as ptif
import pandas as pd
import ast
import pytorch_influence_functions as ptif
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# In[2]:


data_path = "/tmp2/chhavi/"
dataset = "pet"
input_size=2048
num_classes=37


# In[3]:


loaded_df_train = pd.read_csv(f'{data_path}resnet50_features/train_features_labels.csv')

# Convert DataFrame columns back to tensors
train_feat_loaded_df = loaded_df_train['features'].apply(ast.literal_eval) 
train_feat_loaded_tensor = torch.tensor(train_feat_loaded_df.tolist())

train_labels_loaded_tensor = torch.tensor(loaded_df_train['gt_labels'].tolist())

# Reshape tensors to their original sizes
n_rows, n_cols = loaded_df_train.shape[0], len(loaded_df_train.iloc[0].features.split(','))
train_x_feat_tensor = train_feat_loaded_tensor.view(n_rows, n_cols)
train_labels_tensor = train_labels_loaded_tensor.view(n_rows)


# In[4]:


loaded_df_test = pd.read_csv(f'{data_path}resnet50_features/test_features_labels.csv')

# Convert DataFrame columns back to tensors
test_feat_loaded_df = loaded_df_test['features'].apply(ast.literal_eval) 
test_feat_loaded_tensor = torch.tensor(test_feat_loaded_df.tolist())

test_labels_loaded_tensor = torch.tensor(loaded_df_test['gt_labels'].tolist())

# Reshape tensors to their original sizes
n_rows, n_cols = loaded_df_test.shape[0], len(loaded_df_test.iloc[0].features.split(','))
test_x_feat_tensor = test_feat_loaded_tensor.view(n_rows, n_cols)
test_labels_tensor = test_labels_loaded_tensor.view(n_rows)


# In[5]:


# Define your model
class LinearClassifier(nn.Module):
    def __init__(self, input_size=2048, num_classes=37):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


# In[6]:


# Standardize the data
mean = train_x_feat_tensor.mean(dim=0)
std = train_x_feat_tensor.std(dim=0)
train_std_features = (train_x_feat_tensor - mean) / std

# Move data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_std_features = train_std_features.to(device)
train_labels = train_labels_tensor.to(device)

# Create a TensorDataset
train_dataset = TensorDataset(train_std_features, train_labels)

# Create a DataLoader for batch training
batch_size = 64  # You can adjust the batch size as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataloader_noshuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
model_path = f'{data_path}resnet50_features/linear_classifier_{dataset}.pt'

input_size = train_x_feat_tensor.shape[1]
num_classes = len(torch.unique(train_labels_tensor))
print(input_size, num_classes)
# Instantiate the model and move it to GPU if available
model = LinearClassifier(input_size, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[7]:


model = LinearClassifier()
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


# In[8]:


# After training, you can use the model for predictions
# For example, if you have a test feature tensor called test_features:
# Move test data to GPU if available
test_std_features = (test_x_feat_tensor - mean)/std
test_std_features = test_std_features.to(device)
test_labels = test_labels_tensor.to(device)

test_dataset = TensorDataset(test_std_features, test_labels)
test_dataloader_noshuffle = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_outputs = model(test_std_features)
predicted_labels = torch.argmax(test_outputs, dim=1)

# Calculate test accuracy
test_accuracy = torch.sum(predicted_labels == test_labels).item() / len(test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# In[9]:


for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Size: {param.size()}")
    print(f"Parameter values: {param}")


# In[10]:


test_labels.shape


# In[11]:


config = {"outdir": f"{data_path}resnet50_features/imagenet_{dataset}_inf_outdir",
        "seed": seed,
        "gpu": 0,
        "dataset": dataset,
        "num_classes":num_classes ,
        "test_sample_num": False,
        "test_start_index": 3400,
        "recursion_depth": 370,
        "r_averaging": 10,
        "scale": 25,
        "damp": 0.01,
        "calc_method": "img_wise",
        "log_filename": None}


# In[12]:


ptif.init_logging(f"{data_path}resnet50_features/imagenet_{dataset}_inf_outdir/logfile.log")
ptif.calc_img_wise(config, model, train_dataloader_noshuffle, test_dataloader_noshuffle)


# In[ ]:




