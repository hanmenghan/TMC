import torch
import torch.nn as nn
from models.image import ImageEncoder
import torch.nn.functional as F


# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))


class TMC(nn.Module):
    def __init__(self, args):
        super(TMC, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        self.clf_depth = nn.ModuleList()
        self.clf_rgb = nn.ModuleList()
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(depth_last_size, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_depth.append(nn.Linear(depth_last_size, args.n_classes))

        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(rgb_last_size, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            rgb_last_size = hidden
        self.clf_rgb.append(nn.Linear(rgb_last_size, args.n_classes))

    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.args.n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        depth_evidence, rgb_evidence = F.softplus(depth_out), F.softplus(rgb_out)
        depth_alpha, rgb_alpha = depth_evidence+1, rgb_evidence+1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha


class ETMC(TMC):
    def __init__(self, args):
        super(ETMC, self).__init__(args)
        last_size = args.img_hidden_sz * args.num_image_embeds + args.img_hidden_sz * args.num_image_embeds
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden
        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        pseudo_out = torch.cat([rgb, depth], -1)
        for layer in self.clf:
            pseudo_out = layer(pseudo_out)

        depth_evidence, rgb_evidence, pseudo_evidence = F.softplus(depth_out), F.softplus(rgb_out), F.softplus(pseudo_out)
        depth_alpha, rgb_alpha, pseudo_alpha = depth_evidence+1, rgb_evidence+1, pseudo_evidence+1
        depth_rgb_alpha = self.DS_Combin_two(self.DS_Combin_two(depth_alpha, rgb_alpha), pseudo_alpha)
        return depth_alpha, rgb_alpha, pseudo_alpha, depth_rgb_alpha
