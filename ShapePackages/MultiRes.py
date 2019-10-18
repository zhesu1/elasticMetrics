import torch


# reduce resolution of surfaces
def reduce_resolution(f,m,n):
    t,x,y = f.size()
    f0 = torch.zeros(t,m,y)
    for i in range(m):
        f0[:,i,:] = f[:,round((x-1)*i/(m-1)),:]
    f_new = torch.zeros(t,m,n)
    for j in range(n):
        f_new[:,:,j] = f0[:,:,round((y-1)*j/(n-1))]
    return f_new


# here we use the bilinear interpolation
def up_sample(Coe, Tstps):
    t0 = torch.linspace(0,1, Coe.size(1))
    tv = torch.linspace(0,1,Tstps)
    Ind_tv = torch.floor((Coe.size(1)-1)*tv).long()
    
    # fix the boundary points
    Ind_tv1 = Ind_tv+1
    Ind_tv1[Ind_tv1>Coe.size(1)-1] = Coe.size(1)-1
    
    # use the linear interpolation method
    C = (tv - t0[Ind_tv])*(Coe.size(1)-1)
    
    return Coe[:,Ind_tv] + (Coe[:,Ind_tv1] - Coe[:,Ind_tv])*C