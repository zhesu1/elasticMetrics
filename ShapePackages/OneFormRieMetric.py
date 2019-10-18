import numpy as np
import torch
from math import pi
from ShapePackages.OneFormMaps import f_to_df, integral_over_s2


def multi_dim_inv_for_22mn(A, detA):  # compute the inverse of the first 2D slice of a 2*2*m*n matrix
    C = torch.zeros(A.size())
    C[0, 0] = A[1, 1]
    C[1, 1] = A[0, 0]
    C[0, 1] = -A[0, 1]
    C[1, 0] = -A[1, 0]
    return C / detA  # C: 2*2*m*n


def riemann_metric(df, w, v, a, b, c, d):  # df,w,v: 3*2*m*n
    A = torch.einsum("simn,sjmn->ijmn", [df, df])  # A = df^Tdf: 2*2*m*n
    detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0] + 1e-7  # detA = det(df^Tdf): m*n
    invA = torch.zeros(A.size())
    invA[:, :, 1:df.size(-2) - 1] = multi_dim_inv_for_22mn(A[:, :, 1:df.size(-2) - 1],
                                                           detA[1:df.size(-2) - 1])  # invA = (df^Tdf)^{-1} 2*2*m*n
    df_dfplus = torch.einsum("is...,st...,jt...->ij...", [df, invA, df])
    B1 = torch.einsum("ismn,stmn,ktmn,kimn->mn", [w, invA, v, df_dfplus])
    B2 = torch.einsum("ismn,tsmn,tkmn,klmn,jlmn,jimn->mn", [invA, df, w, invA, df, v])
    B3 = torch.einsum("ismn,stmn,itmn->mn", [w, invA, df]) * torch.einsum("ismn,stmn,itmn->mn", [v, invA, df])
    # 1: measures the change in metric
    part1 = (B1 + B2 - B3)/2
    # 2: measure the change in volume density
    part2 = B3/2
    # 3: measures the change in normal direction
    part3 = torch.einsum("ismn,stmn,itmn->mn", [w, invA, v]) - B1
    # 4: the additional term which measures ?
    part4 = (B1 - B2)/2

    inner_prod = (a * part1 + b * part2 + c * part3 + d * part4) * detA.sqrt()
    return integral_over_s2(inner_prod)


# calculate the length of a curve for differentials
def length_dfunc_Imm(curve, a, b, c, d):  # size of curve: T*3*2*m*n
    d_curve = (curve[1: curve.size(0)] - curve[0: curve.size(0)-1])*(curve.size(0)-1)
    L = torch.zeros(curve.size(0)-1)
    for i in range(curve.size(0)-1):
        L[i] = riemann_metric(curve[i], d_curve[i], d_curve[i], a, b, c, d).sqrt()
    return torch.sum(L)/(curve.size(0)-1)


# calculate the length of a curve for immersions
def length_func_surf_Imm(curve_f, a, b, c, d):  # size of curve: T*3*2*m*n
    curve = torch.zeros(curve_f.size(0), 3, 2, *curve_f.shape[-2:])
    for i in range(curve_f.size(0)):
        curve[i] = f_to_df(curve_f[i])
    d_curve = (curve[1: curve.size(0)] - curve[0: curve.size(0)-1])*(curve.size(0)-1)
    L = torch.zeros(curve.size(0)-1)
    for i in range(curve.size(0)-1):
        L[i] = riemann_metric(curve[i], d_curve[i], d_curve[i], a, b, c, d).sqrt()
    return torch.sum(L)/(curve.size(0)-1)





