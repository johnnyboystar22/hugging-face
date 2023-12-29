# Based on https://github.com/RobertCsordas/moe_layer

import torch
import torch.distributed
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def dist_logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    # Calculate numerically stable distributed logsumexp
    xmax = x.max(dim=dim, keepdim=True).values
    torch.distributed.all_reduce(xmax, op=torch.distributed.ReduceOp.MAX)

    xe = (x - xmax).exp().sum(dim=dim, keepdim=True)
    torch.distributed.all_reduce(xe, op=torch.distributed.ReduceOp.SUM)

    res = xmax + xe.log()
    if not keepdim:
        res = res.squeeze(dim)

    return res


def log_mean(x: torch.Tensor, dim: int = 0):
    if torch.distributed.is_initialized():
        xlse = dist_logsumexp(x, dim=dim)

        # Normalize
        n = torch.tensor(x.shape[dim]).to(x.device)
        torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
        return xlse - n.log()
    else:
        return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return -(l * l.exp()).sum(-1)


class MoE(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        expert_size: int,
        k: int,
        dropout: float = 0,
        selection_mode: str = "sigmoid",
        activation_after_topk: bool = False,
        activation=F.relu,
        bias: bool = False,
        v_dim: Optional[int] = None,
        sinkhorn_n_iters: int = 3,
        expert_dropout: float = 0.0,
        weight_std_scale: float = 1.0,
    ):
        super().__init__()
        self.k_dim = d_model
        self.v_dim = v_dim if v_dim is not None else d_model
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.dropout = dropout
        self.selection_mode = selection_mode
        self.k_vec_dim = self.k_dim
        self.n_heads = k
        self.activation_after_topk = activation_after_topk
        self.activation = activation
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.expert_dropout = expert_dropout

        if self.selection_mode not in {"softmax", "sigmoid", "sinkmoid"}:
            raise ValueError(f"Unknown selection mode {self.selection_mode}")

        self.keys = torch.nn.Parameter(
            torch.empty(self.n_experts, self.k_vec_dim, self.expert_size)
        )
        self.values = torch.nn.Parameter(
            torch.empty(self.n_experts, self.expert_size, self.v_dim)
        )
        self.expert_sel = torch.nn.Linear(self.k_vec_dim, self.n_experts, bias=False)
        torch.nn.init.normal_(self.keys, std=d_model**-0.5 * weight_std_scale)
        torch.nn.init.normal_(self.values, std=self.size**-0.5 * weight_std_scale)
        torch.nn.init.normal_(
            self.expert_sel.weight, std=self.k_vec_dim**-0.5 * weight_std_scale
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(int(self.n_experts * self.expert_size))
            )
            # NOTE What is o_bias?
            self.o_bias = torch.nn.Parameter(torch.zeros(self.v_dim))
        else:
            self.bias = None
            self.o_bias = None

        self.renorm_keep_std(self.expert_sel.weight, dim=1)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def entropy_reg(self, sel: torch.Tensor) -> float:
        # Everything is done in log scale
        sel = sel.flatten(0, -2)
        sel = F.log_softmax(sel, dim=-1)
        sel = log_mean(sel, -2)
        return -entropy_l(sel).mean()

    def compute_scores(
        self, input: torch.Tensor, index, expert_scores: torch.Tensor
    ) -> torch.Tensor:
        scores = index  * F.linear(input, self.keys, None)
        if self.bias is not None:
            scores = scores + index * self.bias

        scores = self.activation(scores)
        scores = scores * expert_scores[..., None]

        if self.dropout > 0:
            # Standard dropout on the "up-projected scores"
            scores = F.dropout(scores, self.dropout, training=self.training)

        return scores

    def sel_activation(self, sel: torch.Tensor) -> torch.Tensor:
        if self.selection_mode == "sinkmoid":
            if self.training:
                with torch.no_grad():
                    sel = self.sinkhorn_unnorm(sel)
            else:
                sel = torch.sigmoid(sel)
        elif self.selection_mode == "sigmoid":
            sel = torch.sigmoid(sel)
        elif self.selection_mode == "softmax":
            sel = F.softmax(sel, dim=-1)
        else:
            assert False

        return sel

    def sinkhorn_unnorm(self, x: torch.Tensor) -> torch.Tensor:
        # Based on https://arxiv.org/abs/2202.01169. Unnormalized verison
        A, B = x.shape[-2:]

        a = torch.zeros_like(x[..., 0, :])
        b = torch.zeros_like(x[..., 0])

        for _ in range(self.sinkhorn_n_iters):
            b = math.log(A) - (x - a[..., None, :]).logsumexp(-1)
            if torch.distributed.is_initialized():
                a = math.log(B) - dist_logsumexp(x - b[..., None], -2)
            else:
                a = math.log(B) - (x - b[..., None]).logsumexp(-2)

        return (a[..., None, :] + b[..., None] + x).exp()
    
    def create_index(self, index: torch.Tensor) -> torch.Tensor:
        bs, seq_len = index.shape
        one_hot = torch.nn.functional.one_hot(index, num_classes=self.n_experts)
        return one_hot.unsqueeze(-1).expand(bs, seq_len, self.n_experts, self.expert_size).reshape((bs, seq_len, -1))

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape the keys and values into one big matrix
        if self.keys.ndim > 2:
            self.keys.data = torch.reshape(self.keys.transpose(1,2), (int(self.n_experts * self.expert_size), self.k_vec_dim))
        if self.values.ndim > 2:
            self.values.data = torch.reshape(self.values, (int(self.n_experts * self.expert_size), self.k_vec_dim)).T

        # Selection score calculation
        # sel = sel_raw = F.linear(input, self.expert_sel, None)
        sel = sel_raw = self.expert_sel(input)
        reg_loss = self.entropy_reg(sel_raw)

        # Selection activation and topk
        if (not self.activation_after_topk) or (self.selection_mode == "sinkmoid"):
            # Sinkhorn should be always applied before top-k
            sel = self.sel_activation(sel)

        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel) < self.expert_dropout
            sel = sel.masked_fill(mask, float("-inf"))

        # sel val and sel_index have shape (bs, seq_len, n_heads)
        # where n_heads is the number of experts we select
        # Example: sel_val[1,3,:] are the scores (ordered) of token 4 of sequence 2
        #     [0.69,0.42] are the scores
        # Example: sel_index[1,3,:] are the indices (ordered) of token 4 of sequence 2
        #     [2,1] are the indices

        sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)

        if self.activation_after_topk or (self.selection_mode == "sinkmoid"):
            sel_val = torch.gather(sel_raw, -1, sel_index)
            # for sinkmoid, the score is always calculated by a sigmoid
            sel_val = (
                torch.sigmoid(sel_val)
                if self.selection_mode == "sinkmoid"
                else self.sel_activation(sel_val)
            )

        # Preprocess the selection indices. They will be needed for both layers and save some time
        sel_indices = [
            self.create_index(sel_index[..., h].long())
            for h in range(sel_index.shape[-1])
        ]

        # "Up-projection" layer for each head
        scores_l = [
            self.compute_scores(input, sel_indices[h], sel_val[..., h])
            for h in range(sel_index.shape[-1])
        ]

        # Down projection layer for each head
        res = 0
        for scores in scores_l:
            # we don't need to mask with the indices here since the
            # hidden activations of the non-used experts are zero
            res = res + F.linear(scores, self.values, None)

        if self.o_bias is not None:
            res = res + self.o_bias
        return res, reg_loss


if __name__ == "__main__":
    import torch
    from transformers.models.sigma_moe.cpu_src.moe_layer import MoE as CPUMoE
    from transformers.models.sigma_moe.cuda_src.moe_layer import MoE as CUDAMoE

    torch.manual_seed(0)

    d_model = 128
    n_experts = 4
    expert_size = 64
    k = 2

    cpu_moe = CPUMoE(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    )

    cuda_moe = CUDAMoE(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).cuda()

    cuda_moe.load_state_dict(cpu_moe.state_dict())

    inp = torch.randn(1, 1, 128)
    cpu_x, _ = cpu_moe(inp)
    cuda_x, _ = cuda_moe(inp.cuda())

    cuda_x: torch.Tensor

    if torch.allclose(cpu_x, cuda_x.cpu(), atol=1e-5):
        print("✅ CPU and CUDA match")
    else:
        print("❌ CPU and CUDA differ")
