# from @opeltre/revert

import torch
import torch.nn as nn
import sklearn

from sklearn.manifold import TSNE

class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp estimation of optimal transport plans.
    
    Compute optimal transport `T : [N, M]` between two point clouds
    `x : [N, d]` and `y : [M, d]`. 
    """

    def __init__(self, temp=.2, n_it=10, p=1):
        super().__init__()
        self.temp = temp
        self.n_it = n_it
        self.p = p

    def forward(self, cdist, A=None, B=None):
        """ 
        Optimal transport plan sending A to a neighbour of B.  

        The Gibbs kernel `Q : [N, M]` is defined by exponentiating the cross-
        distance matrix `cdist`, scaled to a standard deviation 
        proportional to the inverse temperature parameter.
        
            Q = torch.exp(-cdist / temp * cdist.std())

        In order to match the source and target single-point densities 
        `A : [N]` and `B : [M]`, Lagrange multipliers `U` and `V` 
        (left and right single-point densities) are iterated upon. 
        
        The returned transport plan `T = U[:,None] * Q * V` matches `A` 
        and almost `B` (both defaulting to uniform distributions on N and M).
        """
        temp, n_it = self.temp, self.n_it
        
        #--- Boundary conditions
        N, M = cdist.shape
        device = cdist.device
        if isinstance(A, type(None)): A = torch.ones(N, device=device) / N
        if isinstance(B, type(None)): B = torch.ones(M, device=device) / M

        #--- Gibbs density
        C = (cdist - cdist.min()) / (cdist.std())
        C = C ** self.p if not self.p == 1 else C
        Q = torch.exp(- C / temp)
        Q = Q / Q.sum()

        #--- Sinkhorn-Knopp --- 
        with torch.no_grad():
            T = 0. + Q.detach()
            U, V = torch.ones(N, device=device), torch.ones(M, device=device)
            for i in range(n_it):
                V *= (B / T.sum([0]))
                T = U[:,None] * Q * V
                U *= (A / T.sum([1]))
                T = U[:,None] * Q * V       
        return T

#--- Plot Transport plan --- 

def tsne (x, k=2, p=30, N=4000): 
    mytsne = TSNE(n_components=k, init='pca', perplexity=p,
                  n_iter=N)
    return mytsne.fit_transform(x.detach())

import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pylab

def plot_transport(x_gen, x_true, Pi):

    fig, ax = pylab.subplots(figsize=(20, 15))

    n1, n2 = x_gen.shape[0], x_true.shape[0]
    i, j = torch.arange(n1).repeat(n2), torch.arange(n2).repeat_interleave(n1)
    
    #--- transport density ---
    edges = torch.stack([x_gen[i], x_true[j]], 1)
    lw = 1 * Pi.T.flatten() * n1
    lc = mc.LineCollection(edges, colors='#ffc', linewidths=1.5 * lw)
    ax.add_collection(lc)

    #---  expected displacement ---
    A = Pi.sum([1])
    Pi_x = Pi / A[:,None]
    i_true = torch.arange(n2).repeat(n1)
    ys = x_true[i_true].view([n1, n2, 2])
    ym = (ys * Pi_x[:,:,None]).sum([1])
    vecs = torch.stack([x_gen, x_gen + .2 * (ym - x_gen)], 1)
    lc2 = mc.LineCollection(vecs, colors='#dfa', linewidths=.4)
    ax.add_collection(lc2)

    #--- point clouds ---
    plt.scatter(x_true[:,0], x_true[:,1], 4, color='#fa3')
    plt.scatter(x_gen[:,0], x_gen[:,1], 4, color='#2e8')
    ax.autoscale()