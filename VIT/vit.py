"""
Pytorch implementation of Vision Transformer
--------------------------------------------

1. Reshaping: HxWxC -> Nx(P²*C) -> P is resolution of image patch, N = H*W/P²
2. Flatten patches -> map to D dimensions with linear layer => Patch Embeddings
3. Prepend learnable embedding z0 -> represents image y after encoder
4. Add Positional Embeddings to Patch Embeddings to retain positional information
5. Run it through L layers of the Encoder
5. Classification head after encoder attached to z0 --> Pretraining: Linear, Linear, Linear --> Finetuning: Linear
(6.) Apply softmax to get probability distribution

Transformer Encoder
-------------------
for i in range(L)
    1. Inputs
    2. LayerNorm
    3. Multi Head Self Attention
    4. Residual connection from 1.
    5. LayerNorm
    6. MLP
        6.1 Linear
        6.2 GELU
        6.3 Linear
        6.4 GELU
    7. Residual connection from 4.
y = LayerNorm(z0)  <---- classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Standard Self Attention algorithm, which was also used in the original transformer paper.
    """
    def __init__(self, in_dim=1024, out_dim=128):
        super(SelfAttention, self).__init__()
        self.dh = out_dim
        self.q = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.k = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.v = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, z):
        q, k, v = self.q(z), self.k(z), self.v(z)
        z = torch.matmul(q, k.T)
        z = z.div(self.dh**0.5)
        A = F.softmax(z)
        z = torch.matmul(A, v)
        return z


class MultiHeadAttention(nn.Module):
    """
    Usual MultiHeadAttention.
    Essentially just takes a latent tensor in and runs it through multiple Self Attention Layers.
    After that, it concatenates all outputs and runs it through a linear layer
    """
    def __init__(self, dim=1024, heads=8):
        super(MultiHeadAttention, self).__init__()
        dim_h = dim // heads
        self.heads = heads
        self.self_attention_heads = nn.ModuleList([
            SelfAttention(in_dim=dim, out_dim=dim_h) for _ in range(heads)
        ])
        self.project = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, z):
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(z)
            else:
                attn = sa_head(z)
                out = torch.cat((out, attn), axis=1)
        out = self.project(out)
        return out


class Encoder(nn.Module):
    """
    Usual encoder of a transformer. Decoders are not needed in VisionTransformer.
    """
    def __init__(self, input_dim, dim=1024, heads=8):
        super(Encoder, self).__init__()
        self.LayerNorm_1 = nn.LayerNorm(input_dim)
        self.LayerNorm_2 = nn.LayerNorm(input_dim)
        self.MSA = MultiHeadAttention(dim, heads)
        self.MLP = nn.Sequential(*[
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
        ])

    def forward(self, x):
        norm = self.LayerNorm_1(x)
        msa = self.MSA(norm)
        x = x + msa
        norm = self.LayerNorm_2(x)
        mlp = self.MLP(norm)
        x = x + mlp
        return x


class VisionTransformer(nn.Module):
    def __init__(self, L=6, H=256, W=256, heads=8, dim=1024, patch_size=32):
        super(VisionTransformer, self).__init__()
        self.L = L
        self.P = patch_size
        self.W = W
        self.H = H
        self.N = self.H * self.W // self.P ** 2
        self.D = dim
        self.C = 3
        self.Classes = 10
        self.embed_dim = torch.Size((self.N+1, self.D))
        self.enc_layers = nn.Sequential(*[Encoder(self.embed_dim, heads=heads) for _ in range(self.L)])
        self.PatchEmbedding = nn.Linear(in_features=(self.P ** 2) * self.C, out_features=self.D)
        self.PositionalEmbedding = nn.Parameter(torch.zeros(self.embed_dim))
        self.LayerNorm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(in_features=self.D, out_features=self.Classes)
        self.cls_token = nn.Parameter(torch.zeros((1, self.D)))

    def reshape(self, image):
        """
        Reshaping: HxWxC -> Nx(P²*C) -> P is resolution of image patch, N = H*W/P² and flatten
        Example: 3x256x256 -> 64x(32*32*3) where P=32
        Code from: https://discuss.pytorch.org/t/extract-image-patches-from-image/133153
        :param image:
        :return: patched and flattened image
        """
        # -----------------------#
        # Gets the right shape but doesn't give correct values.
        # -----------------------#
        # return image.reshape((-1, self.N, (self.P**2)*3))

        # -----------------------#
        # Works, but is slower than actually used method.
        # -----------------------#
        # res = []
        # p = self.P
        # for i in range(1, (self.N**0.5)+1):
        #     for j in range(1, (self.N**0.5)+1):
        #         res.append(image[:, p*(i-1):p*i, p*(j-1):p*j].flatten())
        # return torch.stack(res)

        out = image.unfold(2, self.P, self.P).unfold(1, self.P, self.P)
        out = torch.transpose(out, 3, 4)
        out = out.permute(1, 2, 0, 3, 4)
        return out.contiguous().view(out.size(0) * out.size(1), -1)

    def forward(self, x):
        """
        Reshape to 64x(32*32*3) N = 256*256 / 32² = 64
        PatchEmbedding from 64x(1024*3) to 64x1024
        Prepend z0 (1x1024) --> 65x1024
        Add Positional Embedding
        :param x: image to be classified
        :return: one hot vector of classes 1x10
        """
        n_samples = x.shape[0]
        x = self.reshape(x)
        x = self.PatchEmbedding(x)
        x = torch.cat((self.cls_token, x))  # Insert cls token at the first position.
        x += self.PositionalEmbedding
        x = self.enc_layers(x)
        x = self.LayerNorm(x)
        x = x[0, :]  # Take only the feature from the cls token.
        x = self.head(x)  # Classification head --> turn x into one hot encoded vector.
        x = F.softmax(x)
        return x


if __name__ == '__main__':
    x = torch.randn((3, 256, 256))
    VIT = VisionTransformer()
    output = VIT(x)
    print(output)
