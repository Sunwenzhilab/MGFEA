from MGFEA.model import scAE, sgAE, SparseDataset,sparse_batch_collate
from MGFEA import train, ModelPlot, preprocess
from tqdm import tqdm
import torch.optim as optim
import torch
import numpy as np
from scipy.sparse import csr_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphFEA(object):
    """
    --------------------
    model description:
    MGFEA model for single cell metabolism prediction

    --------------------
    parameter:
    dataIn_h5: An anndata object with single cell transcriptome, input genes should be consisted of metabolism genes and highly variable genes

    cMat: An anndata object with GSMM model. We offer several examples such as scFEA graph, Recon and IMM1865. The stohemistry matrix is the 'X' of anndata.

    batch_size: the number of input vectors in one batch

    train_epoch_num: How many times the model read the trained dataset.

    pc_N: The dimensions of the latent variable

    lr: learning rate

    loss_w: relative weight of latent loss (metabolism graph reconstruct loss).With larger weight, model learns gene relationships better.loss_w should be less than 1. default: 0.5

    save_lossfig: path of loss plot

    save_model: path of trained model parameter file

    coexp: Whether to add gene coexpression relationships in GSMM guided gene interaction graph

    """
    def __init__(self, 
                 dataIn_h5, 
                 cMat, 
                 batch_size=100, 
                 train_epoch_num=100,
                 pc_N=1024,
                 lr=0.001,
                 loss_w = 0.5,
                 save_lossfig='./model_loss.jpg',
                 save_model='./model.pth',coexp=True):
        self.dataIn_h5 = dataIn_h5
        if type(dataIn_h5.X) != csr_matrix:
            self.dataIn = dataIn_h5.X.tocsr()
        else:
            self.dataIn = dataIn_h5.X.tocoo()
        self.batch_size = batch_size
        self.cMat = cMat #stohemistry matrix from GSMM model
        self.train_epoch_num = train_epoch_num
        self.pc_N = pc_N
        self.lr = lr
        self.loss_w = loss_w #weight of latent loss
        self.save_lossfig = save_lossfig
        self.save_model = save_model
        self.coexp = coexp # logical variable input genes contains highly variable genes
    def run(self):
        
        dataloader = torch.utils.data.DataLoader(SparseDataset(self.dataIn.tocoo()),
                                                 batch_size=self.batch_size,shuffle=False, 
                                                 num_workers=0,collate_fn=sparse_batch_collate)

        INPUT_DIM = self.dataIn.shape[1]
        LATENT_DIM = self.cMat.shape[1]        

        ### get gene embedding
        dataIn_pca,dataIn_index = preprocess.geneEmbedding(self.dataIn.toarray(),
                                                           self.dataIn_h5.var_names.values,
                                                           self.cMat,pc_num=self.pc_N,coexp = self.coexp)
        
        dataIn_pca = torch.FloatTensor(dataIn_pca.T).to(device)
        dataIn_index = torch.LongTensor(dataIn_index).to(device)
        
        
        scae = scAE(INPUT_DIM,LATENT_DIM,dataIn_pca.shape[1],int(self.pc_N/2)).to(device)
        

        optimizer = optim.Adam(scae.parameters(),lr=self.lr)

        losse = []

        scae.train()
        for epoch in tqdm(range(self.train_epoch_num)):   
            for data in iter(dataloader):
                data_0  = data.to_dense().to(device)
                optimizer.zero_grad()##
                data_merge, data_z ,data_u, data_sigma, z_merge, gemb = scae(data_0,dataIn_pca,dataIn_index)
                model_train = train.scAEModel_loss(scae,data_0,data_merge,self.cMat,
                                                   data_z,data_u, data_sigma,
                                                   z_merge, gemb, dataIn_index)
                loss = model_train.Total_Loss(self.loss_w)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            
            l = loss.item()
            fl= model_train.Flux_Loss().item()
            ll= model_train.Latent_Loss().item()
            print((l,fl,ll))
            losse.append([l,fl,ll])

            totalloss = {'Total_loss':np.array(losse)[:,0].tolist(),
                     'Flux_loss':np.array(losse)[:,1].tolist(),
                     'latent_loss':np.array(losse)[:,2].tolist()}
            if epoch % 2 == 0:
                ModelPlot.modelplot.trainPlot(totalloss,self.save_lossfig)
        
            if epoch % int(self.train_epoch_num/2) == 0:
                torch.save(scae.state_dict(), self.save_model)
        
        with torch.no_grad():     
            x, _ , _, _, z_latent, _ = scae(torch.FloatTensor(self.dataIn.todense()).to(device),dataIn_pca,dataIn_index)
        
        return x.cpu().numpy(), z_latent.cpu().numpy()
                

class SpatialGraphFEA(object):
    """
    --------------------
    model description:
    MGFEA model for spatial metabolism prediction

    --------------------
    parameter:
    dataIn_h5: An anndata object with spatial transcriptome, input genes should be consisted of metabolism genes and highly variable genes

    edge_index: The spatial graph of spots in dataset. 

    cMat: An anndata object with GSMM model. We offer several examples such as scFEA graph, Recon and IMM1865. The stohemistry matrix is the 'X' of anndata.

    train_epoch_num: How many times the model read the trained dataset. default: 100 

    pc_N: The dimensions of the latent variable default: 1024

    lr: learning rate default: 0.001

    loss_w: relative weight of latent loss (metabolism graph reconstruct loss).With larger weight, model learns gene relationships better. loss_w should be less than 1. default: 0.5 

    save_lossfig: path of loss plot default: ./model_loss.jpg

    save_model: path of trained model parameter file default: ./model.pth

    coexp: Whether to add gene coexpression relationships in GSMM guided gene interaction graph default: True

    lambda_latent: The relative weights of gene latent loss and cell latent loss. With larger weight, model learns gene relationships better, otherwise model learns spatial spots relationships better. default: 0.5

    """
    def __init__(self, 
                 dataIn_h5,
                 edge_index, 
                 cMat, 
                 train_epoch_num=100,
                 pc_N=1024,
                 lr=0.001,
                 loss_w = 0.5,
                 save_lossfig='./model_loss.jpg',
                 save_model='./model.pth',coexp=True,lambda_latent=0.5):
        
        self.dataIn_h5 = dataIn_h5
        self.dataIn = dataIn_h5.X.todense()
        self.edge_index = edge_index 
        self.cMat = cMat
        self.train_epoch_num = train_epoch_num
        self.pc_N=pc_N
        self.lr = lr
        self.loss_w = loss_w
        self.save_lossfig = save_lossfig
        self.save_model = save_model
        self.coexp = coexp
        self.lamb = lambda_latent
    def run(self):
        
        dataloader_x=torch.FloatTensor(self.dataIn).to(device)
        dataloader_edge_index=torch.LongTensor(self.edge_index).to(device)
        INPUT_DIM = self.dataIn.shape[1]
        LATENT_DIM = self.cMat.shape[1]
              
        
        
        dataIn_pca, dataIn_index = preprocess.geneEmbedding(self.dataIn, 
                                                  self.dataIn_h5.var_names.values, 
                                                  self.cMat, 
                                                  pc_num=self.pc_N,
                                                  coexp = self.coexp )
        
        dataIn_index = torch.LongTensor(dataIn_index).to(device)
        dataIn_pca = torch.FloatTensor(dataIn_pca.T).to(device)
        
        sgae= sgAE(INPUT_DIM, LATENT_DIM, int(self.pc_N/2)).to(device)
        
        optimizer = optim.Adam(sgae.parameters(), lr=self.lr)

        losse = []
        sgae.train()
        for epoch in tqdm(range(self.train_epoch_num)):
            optimizer.zero_grad()
            _, z_exp, gemb, cemb = sgae(dataloader_x, dataloader_edge_index, 
                                        dataIn_pca, dataIn_index)
            gemb = gemb.cpu()
            cemb = cemb.cpu()
            z_exp = z_exp.cpu()
            
            
            
            
            model_train = train.VGAEModel_loss(sgae, dataloader_x, 
                                               cemb, dataloader_edge_index,
                                               gemb, dataIn_index,
                                               self.cMat, z_exp,
                                               self.lamb)

            loss = model_train.Total_Loss(self.loss_w)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            
            l = loss.item()
            fl= model_train.Flux_Loss().item()
            ll= model_train.Latent_Loss().item()
            
            losse.append([l,fl,ll])
            print((l,fl,ll))
            totalloss = {'Total_loss':np.array(losse)[:,0].tolist(),
                     'Flux_loss':np.array(losse)[:,1].tolist(),
                     'latent_loss':np.array(losse)[:,2].tolist()}
            
            if epoch % 2 == 0:
                ModelPlot.modelplot.trainPlot(totalloss,self.save_lossfig)
        
            if epoch % int(self.train_epoch_num/2) == 0:
                torch.save(sgae.state_dict(), self.save_model)
                
        with torch.no_grad():
            x, z_exp, _, _, = sgae(dataloader_x,dataloader_edge_index,
                                                          dataIn_pca,dataIn_index)
            
        return x.cpu().numpy(), z_exp.cpu().numpy()
class RefGraphFEA(object):
    """
    --------------------
    model description:
    MGFEA model for spatial metabolism prediction with maldi reference

    --------------------
    parameter:
    dataIn_h5: An anndata object with spatial transcriptome, input genes should be consisted of metabolism genes and highly variable genes

    edge_index: The spatial graph of spots in dataset. 

    cMat: An anndata object with GSMM model. We offer several examples such as scFEA graph, Recon and IMM1865. The stohemistry matrix is the 'X' of anndata.

    mb_df: DataFrame Maldi dataset with corresponding spots index and corresponding metabolite name with our gsmm model

    train_epoch_num: How many times the model read the trained dataset. default: 100 

    pc_N: The dimensions of the latent variable default: 1024

    lr: learning rate default: 0.00005

    loss_w: relative weight of latent loss (metabolism graph reconstruct loss).With larger weight, model learns gene relationships better. default: 0.5 the sum of loss_w and loss_ref should be less than 1

    loss_r: the weight of reference maldi dataset in total loss function default: 0.4 the sum of loss_w and loss_ref should be less than 1

    save_lossfig: path of loss plot default: ./model_loss.jpg

    save_model: path of trained model parameter file default: ./model.pth

    coexp: Whether to add gene coexpression relationships in GSMM guided gene interaction graph default: True

    lambda_latent: The relative weights of gene latent loss and cell latent loss. With larger weight, model learns gene relationships better, otherwise model learns spatial spots relationships better. default: 0.5

    """
    def __init__(self, 
                 dataIn_h5,
                 edge_index, 
                 cMat, 
                 mb_df,
                 train_epoch_num=100,
                 pc_N=1024,
                 lr=0.00005,
                 loss_w = 0.5,
                 loss_r = 0.4,
                 save_lossfig='./model_loss.jpg',
                 save_model='./model.pth',path_log='./log.csv',coexp = True,lambda_latent=0.5):
        
        self.dataIn_h5 = dataIn_h5
        self.dataIn = dataIn_h5.X.todense()
        self.edge_index = edge_index
        self.cMat = cMat
        self.train_epoch_num = train_epoch_num
        self.pc_N=pc_N
        self.lr = lr
        self.loss_w = loss_w
        self.loss_r = loss_r
        self.save_lossfig = save_lossfig
        self.save_model = save_model
        self.ref_mb_df = mb_df #should be a dataframe and its index is metabolites name in gsmm
        self.coexp = coexp
        self.lamb = lambda_latent
        self.path_log = path_log
    def run(self):
        
        dataloader_x=torch.FloatTensor(self.dataIn).to(device)
        dataloader_edge_index=torch.LongTensor(self.edge_index).to(device)
        INPUT_DIM = self.dataIn.shape[1]
        LATENT_DIM = self.cMat.shape[1]
              
        
        
        dataIn_pca, dataIn_index = preprocess.geneEmbedding(self.dataIn, 
                                                  self.dataIn_h5.var_names.values, 
                                                  self.cMat, pc_num=self.pc_N,coexp = self.coexp)
        
        dataIn_index = torch.LongTensor(dataIn_index).to(device)
        dataIn_pca = torch.FloatTensor(dataIn_pca.T).to(device)
        
        sgae= sgAE(INPUT_DIM, LATENT_DIM, int(self.pc_N/2)).to(device)
        
        optimizer = optim.Adam(sgae.parameters(), lr=self.lr)
        maldi_index = [self.cMat.obs.index.get_loc(i) for i in self.ref_mb_df.T.index]
        maldi_index = torch.LongTensor(maldi_index)
        maldi_m = torch.FloatTensor(self.ref_mb_df.values)

        losse = []
        sgae.train()
        for epoch in tqdm(range(self.train_epoch_num)):
            optimizer.zero_grad()
            _, z_exp, gemb, cemb = sgae(dataloader_x, dataloader_edge_index, 
                                        dataIn_pca, dataIn_index)
            gemb = gemb.cpu()
            cemb = cemb.cpu()
            z_exp = z_exp.cpu()
            
            
            
            
            model_train = train.RefVGAEModel_loss(sgae, dataloader_x, 
                                               cemb, dataloader_edge_index,
                                               gemb, dataIn_index,
                                               self.cMat, z_exp,maldi_index,maldi_m,lambda_latent=self.lamb)

            loss = model_train.Total_Loss(self.loss_w,self.loss_r)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            
            l = loss.item()
            fl= model_train.Flux_Loss().item()
            ll= model_train.Latent_Loss().item()
            rl= model_train.Ref_Loss().item()

            losse.append([l,fl,ll,rl])
            print((l,fl,ll,rl))
            totalloss = {'Total_loss':np.array(losse)[:,0].tolist(),
                     'Flux_loss':np.array(losse)[:,1].tolist(),
                     'latent_loss':np.array(losse)[:,2].tolist(),
                     'ref_loss':np.array(losse)[:,3].tolist()}
            
            if epoch % 2 == 0:
                ModelPlot.modelplot.trainPlot(totalloss,self.save_lossfig)
        
            if epoch % int(self.train_epoch_num/2) == 0:
                torch.save(sgae.state_dict(), self.save_model)
                
        with torch.no_grad():
            x, z_exp, _, _, = sgae(dataloader_x,dataloader_edge_index,
                                                          dataIn_pca,dataIn_index)
        import pandas as pd
        pd.DataFrame(totalloss).to_csv(self.path_log)    
        return x.cpu().numpy(), z_exp.cpu().numpy()