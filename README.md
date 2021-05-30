# Trusted Multi-View Classification

This repository contains the code of our ICLR'2021 paper [Trusted Multi-View Classification](https://arxiv.org/abs/2102.02051) [[中文介绍]](https://mp.weixin.qq.com/s/thx3WSqc64rcEJVOS3OY7A). We will gradually improve and enhance the code. Here we provide a demo and detailed instructions for constructing trustworthy multi-view/multi-modal classification algorithm.

## Requirment

- Pytorch 1.3.0
- Python 3
- sklearn
- numpy
- scipy

## Quick Start

To convert your networks into a trusted multimodal classification model, it is better to refer to the following steps:

- Step 1: The softmax layer of a conventional neural-network-based classifier is replaced with an activation function layer (e.g., RELU) to ensure that the network outputs non-negative values.
- Step 2: Use the method in the paper to construct a trusted classifier for each modality. (1) Treat the output of the neural network as evidence $\mathbf{e}$. (2) Construct Dirichlet distribution with $\mathbf{e}+1$. (3) Calculate subjective uncertainty $u$ and belief masses for each modality.
- Step 3: Use dempster’s combination rule rather than traditional fusion strategies to combine the uncertainty and belief masses from different modalities.
    <details>
    <summary>
    Code of dempster’s combination rule.
    </summary>

    ```python
    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a
    ```
    </details>
- Step 4: Use a multi-task strategy and overall loss function in the paper to optimize the model.
    <details>
    <summary>
    Code of overall loss function.
    </summary>

    ```python
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

        return (A + B)
    ```
    </details>

This method is also suitable for other scenarios that require trusted integration, such as Ensemble Learning, Multi-View Learning.

## Citation

If you find TMC helps your research, please cite our paper:

```
@inproceedings{
han2021trusted,
title={Trusted Multi-View Classification},
author={Zongbo Han and Changqing Zhang and Huazhu Fu and Joey Tianyi Zhou},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=OOsR8BzCnl5}
}
```

## Acknowledgement

We thank the authors of [EDL](https://muratsensoy.github.io/uncertainty.html). Other loss functions except for cross entropy to quantify classification uncertainty are also provided in [EDL](https://muratsensoy.github.io/uncertainty.html).

## Questions?

Please report any bugs and I will get to them ASAP. For any additional questions, feel free to email zongbo@tju.edu.cn.
