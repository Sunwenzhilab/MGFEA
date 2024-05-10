import numpy as np
import pandas as pd
import scipy as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def define_genesffea(adata,graph,num_hvg=None):
    """
    simplify metabolism graph and define a label named "genes4fea" to determine the input gene set of model
    parameter:

    adata: anndata object The scRNA or stRNA dataset.

    graph: anndata object The metabolism model.

    num_hvg: number of input hvg With default parameter 'None', we choose the most variable non-metabolism genes to balance information variation. 
    """
    met_gs = list(set([j for i in graph.var.genes for j in i.split(';')]))
    met_genes = adata.var.loc[adata.var.index.isin(met_gs),]
    hvg = adata.var.loc[~adata.var.index.isin(met_gs),]
    
    if not num_hvg == None:
        if num_hvg == 0:
            merge_gs = list(set([j for i in graph.var.genes for j in i.split(';')]))
        else:
            hvg_l = hvg.sort_values('dispersions_norm',ascending=False).index[:num_hvg]
            merge_gs = list(set([j for i in graph.var.genes for j in i.split(';')]) | 
                    set(hvg_l))
            adata.var['fea_hvg'] = adata.var['symbol'].isin(hvg_l)
    else:
        
        cutoff = met_genes.loc[met_genes['dispersions_norm']>0,'dispersions_norm'].sum()
        l_index = np.max(np.where(np.cumsum(hvg.loc[hvg['dispersions_norm']>0,].sort_values(by='dispersions_norm',ascending=False)['dispersions_norm'].to_list()) < cutoff)[0])
        hvg_l = hvg.index[:l_index+1]
        merge_gs = list(set([j for i in graph.var.genes for j in i.split(';')]) | 
                    set(hvg_l))
        adata.var['fea_hvg'] = adata.var['symbol'].isin(hvg_l)
        #find the union of metabolism genes and hvg
        
    
    adata.var['genes4fea'] = adata.var['symbol'].isin(merge_gs)
    
    ##simplify graph
    nogs = list(set([j for i in graph.var.genes for j in i.split(';')]) - set(adata[:,adata.var['genes4fea']].var.symbol))
    #nogs member of metbolism genes but not expressed genes
    filter = [False if len(set(i.split(';')) - set(nogs)) ==0 else True for i in graph.var.genes ]
    #if genes related with single reaction are not expressed at all, remove them
    graph2 = graph[:,filter]
    return graph2
def feature2adj(featureMat, k=5):
    """
    :param featureMat: (N, M) can be spatial information or gene feature expression
    :param k: int
    :return: adj_m: (N, N)
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(featureMat)
    adj_m=nbrs.kneighbors_graph(featureMat, mode='connectivity').toarray() 
    return adj_m

def reconstruct_gnet(scMat, involved_genes_max=100):
    """
    scMat: h5py file including the reactions and its related genes
    """
    #scMat = scMat[scMat.obs['involved_path_c']<=involved_genes_max,:]
    sc_r  = scMat.X.todense()
    sc_r  = abs(sc_r) # absolute value
    
    scNet = np.dot(sc_r.T, sc_r) # reaction adjacent network
    
    #scNet = (scNet>0).astype(int) # binary matrix
    
    All_reaction_genes = scMat.var['genes'].values
    All_reaction_genes = [i.split(';') for i in All_reaction_genes]
    reactionGset = list(set([j for i in All_reaction_genes for j in i]))
    reactionG_Mat = pd.DataFrame(0,index = range(len(All_reaction_genes)), columns=reactionGset)
    
    for i in range(len(All_reaction_genes)):
        reactionG_Mat.loc[i,list(set(All_reaction_genes[i]))] = 1
    
    g2gNet = np.dot(scNet, reactionG_Mat.values) # use scNet to add the edges between genes
    g2gNet = np.dot(g2gNet.T, g2gNet) # gene adjacent network
    g2gNet = (g2gNet>0).astype(int) # binary matrix
    
    return reactionGset, g2gNet


def adj2adjindex(adj):
    """
    :param adj_m: (N, N)
    :return: adj_index: (N,2)
    """
    ###convert adj matirix to adj_index
    adj = sp.sparse.coo_matrix(adj)
    adj_index = np.vstack((adj.row, adj.col))
    
    return adj_index


def geneEmbedding(dataIn, Genes, scMat, pc_num=512,coexp = True):
    """
    dataIn: (N, M) M is the number of genes,N is the number of cells
    scMat: h5py file including the reactions and its related genes
    """
    
    ### PCA decomposition for cells and get the co-expression network of genes
    if dataIn.shape[1] > pc_num:
        dataIn_pca = PCA(n_components=pc_num).fit_transform(dataIn.T).T
    else:
        dataIn_pca = dataIn
    
    reactionGset, g2gNet = reconstruct_gnet(scMat, involved_genes_max=10)
    g2gNet_tab = pd.DataFrame(g2gNet, index=reactionGset, columns=reactionGset)
    ######
    
    ### finnal gene-gene network from co-expression network and reaction-gene network
    overlap_genes = list(set(Genes).intersection(set(reactionGset)))
    g2g_net  =  g2gNet_tab.loc[overlap_genes, overlap_genes]
    coexpadj_tab = pd.DataFrame(0, index=Genes, columns=Genes)

    ##add metabolism edge
    coexpadj_tab.loc[overlap_genes, overlap_genes] += g2g_net.loc[overlap_genes, overlap_genes]
    ##add coexp edge

    similarities = cosine_similarity(sparse.csr_matrix( np.array(dataIn_pca.T)))
    #print('pairwise dense output:\n {}\n'.format(similarities))
    
    sim_df = pd.DataFrame(similarities,index=Genes,columns = Genes)
    hvg = list(set(Genes).difference(set(overlap_genes)))
    abs_sim = sim_df.loc[~sim_df.index.isin(hvg),sim_df.index.isin(hvg)]
    edge_coexp = abs_sim.idxmax()
    for i in range(len(edge_coexp.index)):
        coexpadj_tab.loc[edge_coexp.index[i],edge_coexp[i]]+=1
        coexpadj_tab.loc[edge_coexp[i],edge_coexp.index[i]]+=1
    coexpadj_tab = (coexpadj_tab.values>0).astype(int)
    
    
    if coexp ==False:
        g2gNet_tab = g2gNet_tab.loc[Genes, Genes]
        gene_index = adj2adjindex(g2gNet_tab)
    else:
        gene_index = adj2adjindex(coexpadj_tab)
    
    
    return dataIn_pca, gene_index