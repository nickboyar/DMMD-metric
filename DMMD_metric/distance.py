import torch

SCALE = 1000
GAMMA = 1 / (2 * 10**2)

def mmd(gen_mat, real_mat):
    
    gen_mat = torch.from_numpy(gen_mat)
    real_mat = torch.from_numpy(real_mat)
    
    gen_mat_mul = torch.matmul(gen_mat, gen_mat.T)
    real_mat_mul = torch.matmul(real_mat, real_mat.T)
    gen_real_mat_mul = torch.matmul(gen_mat, real_mat.T)

    gen_mat_square_norm = torch.diag(gen_mat_mul)
    real_mat_square_norm = torch.diag(real_mat_mul)
    
    kernel_xx = (torch.unsqueeze(gen_mat_square_norm, 1) + torch.unsqueeze(gen_mat_square_norm, 0) - 2 * gen_mat_mul)
    expectation_kernel_xx = torch.mean(torch.exp(-GAMMA * kernel_xx))
    
    kernel_yy = (torch.unsqueeze(real_mat_square_norm, 1) + torch.unsqueeze(real_mat_square_norm, 0) - 2 * real_mat_mul)
    expectation_kernel_yy = torch.mean(torch.exp(-GAMMA * kernel_yy))
    
    kernel_xy = (torch.unsqueeze(gen_mat_square_norm, 1) + torch.unsqueeze(real_mat_square_norm, 0) - 2 * gen_real_mat_mul)
    expectation_kernel_xy = torch.mean(torch.exp(-GAMMA * kernel_xy))
    
    return SCALE * (expectation_kernel_xx + expectation_kernel_yy - 2 * expectation_kernel_xy)