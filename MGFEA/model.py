import torch
from torch.utils.data import Dataset

from torch_geometric.nn import GCNConv, VGAE
import torch.nn.functional as F
from scipy.sparse import (vstack, coo_matrix, csr_matrix, csc_matrix)
from typing import Union
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###VGAE encoder
class VGAEncoder(torch.nn.Module):
    def __init__(self, In_channels, Out_channels):
        super(VGAEncoder, self).__init__()
        Mid_channels = min(int((In_channels + Out_channels)/2), int(2 * Out_channels))
        self.conv1 = GCNConv(In_channels, Mid_channels,cached=True)
        self.conv_mu = GCNConv(Mid_channels, Out_channels,cached=True)
        self.conv_logvar = GCNConv(Mid_channels, Out_channels,cached=True)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.nn.Dropout(p=0.2)(x)
        x = x.relu()
        vgae_mu = self.conv_mu(x, edge_index)
        vgae_logvar = self.conv_logvar(x, edge_index)
        return vgae_mu, vgae_logvar


###VAE encoder
class VAEncoder(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(VAEncoder, self).__init__()
        
        D_mid = min(int((D_in + D_out)/2),int(2* D_out))
        
        self.linear1 = torch.nn.Linear(D_in, D_mid)
        self.linear2 = torch.nn.Linear(D_mid, D_out)

        self.linear3 = torch.nn.Linear(D_out, D_mid)
        self.linear4 = torch.nn.Linear(D_mid, D_in)
        
    def encode(self, x):
        x = F.relu(self.linear1(x))
        x_u = self.linear2(x)
        x_logsigma = self.linear2(x)
        
        return x_u, x_logsigma
    
    def decode(self, z):
        z = F.relu(self.linear3(z))
        
        return torch.sigmoid(self.linear4(z))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        
        eps = torch.randn_like(std)
        
        return mu + eps*std
    
    
    def forward(self, x):
        x_u, x_logsigma = self.encode(x)
        z = self.reparameterize(x_u, x_logsigma)
        
        return self.decode(z), x_u, x_logsigma






###spatial model
class sgAE(torch.nn.Module):
    def __init__(self, In_channels, Out_channels, pc_channels):
        super(sgAE, self).__init__()
        self.cellvgae = VGAE(VGAEncoder(In_channels, pc_channels))
        self.genevgae = VGAE(VGAEncoder(int(2*pc_channels), pc_channels))
        
        mid_channels = min(int((In_channels + Out_channels)/2), int(2 * Out_channels))
        self.fc1 = torch.nn.Linear(In_channels, mid_channels)
        self.fc2 = torch.nn.Linear(mid_channels, Out_channels)
    
    
    def cos2weight(self, cemb, gemb):
        '''
        compute gene weights in specific cell
        '''
        return F.softmax(torch.mm(cemb, gemb.T)/torch.mul(torch.norm(cemb,dim=1).unsqueeze(0).T, torch.norm(gemb.T,dim=0)), dim = 1)
    
    def forward(self, x ,edge_index, x_pca, edge_index_pca):
        gemb =   self.genevgae.encode(x_pca, edge_index_pca)
        
        cemb = self.cellvgae.encode(x, edge_index)
        
        x =self.cos2weight(cemb, gemb) * x 
        
        
        z_exp = self.fc2(F.relu(self.fc1(x)))
        return x, z_exp, gemb , cemb






### single cell model
class scAE(torch.nn.Module):

    def __init__(self, CellIn_Dim, Latent_Dim, GeneIn_Dim, Gene_pc_Dim):
        super(scAE, self).__init__()
        self.vgae = VGAE(VGAEncoder(GeneIn_Dim, Gene_pc_Dim)) ###for gene embedding with VGAE
        
        self.vaencoder_1 = VAEncoder(CellIn_Dim, Gene_pc_Dim) ###VAE decoder
        
        mid_channels = min(int((CellIn_Dim + Latent_Dim)/2), int(2 * Latent_Dim))
        self.fc1 = torch.nn.Linear(CellIn_Dim, mid_channels)
        self.fc2 = torch.nn.Linear(mid_channels, Latent_Dim) 
    
    def cos2weight(self, cemb, gemb):
        '''
        compute gene weights in specific cell
        '''
        return F.softmax(torch.mm(cemb, gemb.T)/torch.mul(torch.norm(cemb,dim=1).unsqueeze(0).T, torch.norm(gemb.T,dim=0)), dim = 1)
    
       
    def forward(self, x, x_pca, edge_index):
        gemb = self.vgae.encode(x_pca, edge_index)
        
        _, x_u, x_logsigma = self.vaencoder_1(x) ###AE encoder
        
        x_z = self.vaencoder_1.reparameterize(x_u, x_logsigma) ###latent space
        
        x = self.cos2weight(x_z, gemb) * x ### multiply with gene embedding
        
        
        z_merge = self.fc2(F.relu(self.fc1(x)))
        
        return  x, x_z ,x_u, x_logsigma, z_merge, gemb


class SparseDataset(Dataset):
    
    def __init__(self, dataIn:Union[np.ndarray, coo_matrix, csr_matrix, csc_matrix], transform=None):
        if type(dataIn) == coo_matrix:
            self.dataIn = dataIn.tocsr()
        else:
            self.dataIn = dataIn
        self.transform = transform


    def __getitem__(self, index:int):
        x = self.dataIn[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.dataIn.shape[0]
    
    

def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)
    
def sparse_batch_collate(data_batch): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """

    if type(data_batch[0]) == csc_matrix or type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)


    return data_batch