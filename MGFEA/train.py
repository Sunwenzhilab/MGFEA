import torch
import torch.nn.functional as F
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class scAEModel_loss(object):
    
    def __init__( self, scAE_model ,DataIn , 
                 Data_merge, scMat, 
                 Data_z, Data_u, Data_sigma , 
                 z_merge, gene_embeding, dataIn_index):
        
        self.scAE_model =scAE_model
        self.DataIn = DataIn
        self.Data_merge = Data_merge
        self.scMat = scMat
        self.C2R_mat = self.scMat.X.todense().T
        self.Data_u = Data_u
        self.Data_sigma = Data_sigma
        self.Data_z = Data_z 
        self.z_merge = z_merge
        self.gene_embeding = gene_embeding
        self.dataIn_index = dataIn_index
        self.node_num = self.DataIn.shape[1]
    
    
    def Flux_Loss(self):
        
        CompdBlanceLoss = torch.abs(torch.matmul(self.z_merge,torch.FloatTensor(self.C2R_mat).to(Device)))
        
        NonegCompLoss = torch.mean(torch.abs(self.z_merge)-self.z_merge,dim=0)
    
        return torch.mean(CompdBlanceLoss) + torch.mean(NonegCompLoss)    

    def Latent_Loss(self):
        mseloss = torch.nn.MSELoss()
        
        geneembedingLoss = self.scAE_model.vgae.recon_loss(self.gene_embeding,self.dataIn_index) + 1/self.node_num *self.scAE_model.vgae.kl_loss()
        
        data_recon_loss = mseloss(self.scAE_model.vaencoder_1.decode(self.Data_z),self.DataIn)
        
        data_KLloss = -0.5 * torch.sum(1 + self.Data_sigma - self.Data_u.pow(2) - self.Data_sigma.exp()) 
        
        return geneembedingLoss + data_recon_loss + data_KLloss 
                
    
    def Total_Loss(self,loss_w):
        
        return (1-loss_w) * self.Flux_Loss() + loss_w * self.Latent_Loss() 
class VGAEModel_loss(object):
    
    def __init__( self, sgAE_model, DataIn, 
                 cemb, edge_index_c, 
                 gemb, edge_index_g,
                 scMat, z_exp,lambda_latent):
        self.sgAE_model = sgAE_model
        self.DataIn = DataIn
        self.cemb = cemb
        self.edge_index_c = torch.Tensor(edge_index_c).cpu()
        self.gemb = gemb
        self.edge_index_g = torch.Tensor(edge_index_g).cpu()
        self.scMat = scMat
        self.z_exp = z_exp
        self.lamb = lambda_latent
        
        
        
    def Flux_Loss(self):
        
        CompdBlanceLoss = torch.abs(torch.mm(self.z_exp,torch.FloatTensor(self.scMat.X.todense().T)))
        FluxLabel = torch.sign(self.z_exp)
        Nonegcount = torch.abs(FluxLabel) - FluxLabel
        
        NonegCompLoss = torch.abs(self.z_exp)-self.z_exp
    
        return torch.mean(CompdBlanceLoss) + torch.mean(NonegCompLoss)   + torch.mean(Nonegcount)


    def Latent_Loss(self):
        cell_latent_loss = (1/self.DataIn.shape[0]) * self.sgAE_model.cellvgae.kl_loss()
        cell_recon_loss = self.sgAE_model.cellvgae.recon_loss(self.cemb, self.edge_index_c)
        
        gene_latent_loss = (1/self.DataIn.shape[1]) * self.sgAE_model.genevgae.kl_loss()
        gene_recon_loss = self.sgAE_model.genevgae.recon_loss(self.gemb, self.edge_index_g)
        
        
        return (1-self.lamb) * cell_latent_loss + (1-self.lamb) * cell_recon_loss + self.lamb * gene_latent_loss + self.lamb * gene_recon_loss
    
    
        
    def Total_Loss(self,loss_w):
        total_loss =(1-loss_w) * self.Flux_Loss() + loss_w * self.Latent_Loss() 
        
        return total_loss


class RefVGAEModel_loss(object):
    
    def __init__( self, sgAE_model, DataIn, 
                 cemb, edge_index_c, 
                 gemb, edge_index_g,
                 scMat, z_exp,maldi_index,maldi_m,lambda_latent):
        self.sgAE_model = sgAE_model
        self.DataIn = DataIn
        self.cemb = cemb
        self.edge_index_c = torch.Tensor(edge_index_c).cpu()
        self.gemb = gemb
        self.edge_index_g = torch.Tensor(edge_index_g).cpu()
        self.scMat = scMat
        self.z_exp = z_exp
        self.maldi_index = maldi_index
        self.maldi_m = maldi_m
        self.lamb = lambda_latent
        
    def Flux_Loss(self):
        
        CompdBlanceLoss = torch.abs(torch.mm(self.z_exp,torch.FloatTensor(self.scMat.X.todense().T)))
        FluxLabel = torch.sign(self.z_exp)
        Nonegcount = torch.abs(FluxLabel) - FluxLabel
        
        NonegCompLoss = torch.abs(self.z_exp)-self.z_exp
        zeropunish = torch.abs(1/(self.z_exp+1e-4))
        bound = torch.abs(self.z_exp-0.95)+torch.abs(self.z_exp-1)
    
        return torch.mean(CompdBlanceLoss) + torch.mean(zeropunish) +torch.mean(bound)#+ torch.mean(NonegCompLoss)   + torch.mean(Nonegcount) 


    def Latent_Loss(self):
        cell_latent_loss = (1/self.DataIn.shape[0]) * self.sgAE_model.cellvgae.kl_loss()
        cell_recon_loss = self.sgAE_model.cellvgae.recon_loss(self.cemb, self.edge_index_c)
        
        gene_latent_loss = (1/self.DataIn.shape[1]) * self.sgAE_model.genevgae.kl_loss()
        gene_recon_loss = self.sgAE_model.genevgae.recon_loss(self.gemb, self.edge_index_g)
        
        
        return (1-self.lamb)*cell_latent_loss + self.lamb*gene_latent_loss + (1-self.lamb)*cell_recon_loss + self.lamb*gene_recon_loss 
    def Ref_Loss(self):
        mseloss = torch.nn.MSELoss()
        CompdBlance = torch.mm(self.z_exp,torch.FloatTensor(self.scMat.X.todense().T))
        
        #Diffloss = mseloss(CompdBlance[:,self.maldi_index],self.maldi_m)
        Diffloss = 1-torch.mean(F.cosine_similarity(CompdBlance[:,self.maldi_index],self.maldi_m, dim=0))
        return torch.mean(Diffloss)
    
        
    def Total_Loss(self,loss_w,loss_r):
        total_loss =(1-loss_w-loss_r) * self.Flux_Loss() + loss_w * self.Latent_Loss() + loss_r * self.Ref_Loss()
        
        return total_loss
