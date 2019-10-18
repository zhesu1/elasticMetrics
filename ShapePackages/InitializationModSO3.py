import scipy.io as sio
import torch
import numpy as np
from ShapePackages.OneFormMaps import f_to_df


def initialize_overSO3(f1, f2):
    
    df1 = f_to_df(f1)
    df2 = f_to_df(f2)
    
    n1 = oneForm_to_n(df1).view(3,-1)
    n1 = n1 - torch.sum(n1, 1).view(3,-1).expand(-1,n1.size(1))/n1.size(1)
    n2 = oneForm_to_n(df2).view(3,-1)
    n2 = n2 - torch.sum(n2, 1).view(3,-1).expand(-1,n2.size(1))/n2.size(1)
#     print(torch.sum(n1, 1)/n1.size(1), torch.sum(n2, 1)/n1.size(1))
    
    Rz = torch.tensor([[-1, 0., 0],[0,-1,0],[0,0,1]]) # z fixed rotate xy
    Rx = torch.tensor([[1, 0., 0],[0, -1, 0],[0,0,-1]]) # x fixed rotate yz
    Ry = torch.tensor([[-1, 0., 0],[0, 1, 0],[0,0,-1]]) # y fixed rotate xz
    
    n10, n20 = n1.view(3,-1), n2.view(3,-1)
    U1, D1, V1 = torch.svd(n10)
    U2, D2, V2 = torch.svd(n20)
    
    D0 = torch.eye(3)
    D0[2,2] = torch.sign(torch.det(U1@torch.inverse(U2)))
    R = U1@D0@torch.inverse(U2)
    
    allR = torch.stack((R, Rz@R, Rx@R, Ry@R))
    allf2 = torch.einsum("lij,jmn->limn",[allR, f2])
    
    L = np.zeros(4)
    for i in range(4):
        Diff = f1 - allf2[i]
        L[i] = torch.norm(Diff.flatten()).numpy()
        
    Ind = np.argmin(L)
    return allf2[Ind], L


def oneForm_to_n(df): # with respect to theta and phi
    m, n = df.shape[-2:]
    phi = torch.linspace(0, np.pi, m)
    theta = torch.linspace(0, 2 * np.pi, n)
    PHI, THETA = torch.meshgrid([phi, theta])
    
    df[0, 0], df[1, 0], df[2, 0] = df[0, 0] * torch.sin(PHI), df[1, 0] * torch.sin(PHI), df[2, 0] * torch.sin(PHI)
    n_f = torch.cross(df[:, 0], df[:, 1], dim=0)
    return n_f

