import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionGaussian(nn.Module):
    """
    Helper module using self-attention to generate Gaussian parameters (mean and log_variance).
    """
    def __init__(self, embed_dim, nhead, dropout=0.1):
        super(SelfAttentionGaussian, self).__init__()
        self.embed_dim = embed_dim
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.output_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x) 
        x = self.norm(x + self.dropout(attn_output))
        projected = self.output_proj(x)
        
        # Split into mean and log_variance
        mu = projected[..., :self.embed_dim]
        log_var = projected[..., self.embed_dim:]
        return mu, log_var

class UE(nn.Module):
    """
    Uncertainty Estimation (MGUE) module fusing multimodality features.
    """
    def __init__(self, embed_dim1=256, h1=13, w1=13, embed_dim2=32, seq_len2=5,
                 transformer_dim=512, nhead=8, transformer_layers=1, dropout=0.1):
        super(UE, self).__init__()
        self.embed_dim1 = embed_dim1
        self.h1 = h1
        self.w1 = w1
        self.seq_len1 = h1 * w1
        self.embed_dim2 = embed_dim2
        self.seq_len2 = seq_len2
        self.transformer_dim = transformer_dim

        # Initial Gaussian Distribution Generation
        self.sa_gauss_init1 = SelfAttentionGaussian(embed_dim1, nhead, dropout)
        self.sa_gauss_init2 = SelfAttentionGaussian(embed_dim2, nhead, dropout)

        # MLP Mapping to Transformer Dimension
        self.mlp1 = nn.Linear(embed_dim1, transformer_dim)
        self.mlp2 = nn.Linear(embed_dim2, transformer_dim)
        self.mlp_norm1 = nn.LayerNorm(transformer_dim)
        self.mlp_norm2 = nn.LayerNorm(transformer_dim)

        # Transformer Interaction
        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=transformer_dim, nhead=nhead, dim_feedforward=transformer_dim * 4,
            dropout=dropout, activation=F.relu, batch_first=True
        )
        self.transformer1 = nn.TransformerDecoder(decoder_layer1, num_layers=transformer_layers)

        decoder_layer2 = nn.TransformerDecoderLayer(
            d_model=transformer_dim, nhead=nhead, dim_feedforward=transformer_dim * 4,
            dropout=dropout, activation=F.relu, batch_first=True
        )
        self.transformer2 = nn.TransformerDecoder(decoder_layer2, num_layers=transformer_layers)

        # Posterior Gaussian Distribution Generation
        self.sa_gauss_post1 = SelfAttentionGaussian(transformer_dim, nhead, dropout)
        self.sa_gauss_post2 = SelfAttentionGaussian(transformer_dim, nhead, dropout)

        # Dimension Restoration
        self.restore_mlp1 = nn.Linear(transformer_dim, embed_dim1)
        self.restore_mlp2 = nn.Linear(transformer_dim, embed_dim2)

        # Loss Function (L1 Norm is robust to outliers as per the paper)
        self.l1_loss = nn.L1Loss()

    def _reshape_input1(self, feat1):
        n, h, w, c = feat1.shape
        return feat1.view(n, self.seq_len1, c)

    def _reshape_input2(self, feat2):
        return feat2

    def forward(self, feat1, feat2):
        # 1. Reshape Inputs
        feat1_flat = self._reshape_input1(feat1)
        feat2_flat = self._reshape_input2(feat2)

        # 2. MLP Mapping
        mapped_feat1 = self.mlp_norm1(self.mlp1(feat1_flat))
        mapped_feat2 = self.mlp_norm2(self.mlp2(feat2_flat))

        # 3. Transformer Interaction (Cross-Attention)
        fused_feat1 = self.transformer1(tgt=mapped_feat1, memory=mapped_feat2)
        fused_feat2 = self.transformer2(tgt=mapped_feat2, memory=mapped_feat1)

        # 4. Posterior Gaussian Distribution
        mu_post1, log_var_post1 = self.sa_gauss_post1(fused_feat1)
        mu_post2, log_var_post2 = self.sa_gauss_post2(fused_feat2)

        # 5. Noise Injection (Reparameterization Trick)
        sigma_post1 = torch.exp(0.5 * log_var_post1)
        sigma_post2 = torch.exp(0.5 * log_var_post2)
        
        eps1 = torch.randn_like(sigma_post1)
        eps2 = torch.randn_like(sigma_post2)

        noisy_feat1 = mu_post1 + eps1 * sigma_post1
        noisy_feat2 = mu_post2 + eps2 * sigma_post2

        # 6. Dimension Restoration
        restored_feat1_flat = self.restore_mlp1(noisy_feat1)
        restored_feat2_flat = self.restore_mlp2(noisy_feat2)

        # 7. Reconstruction Loss Calculation (L1)
        loss1 = self.l1_loss(restored_feat1_flat, feat1_flat)
        loss2 = self.l1_loss(restored_feat2_flat, feat2_flat)

        # 8. KL Divergence Loss (logsigma_sum equivalent)
        # Formula: KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss1 = -0.5 * torch.sum(1 + log_var_post1 - mu_post1.pow(2) - log_var_post1.exp())
        kl_loss2 = -0.5 * torch.sum(1 + log_var_post2 - mu_post2.pow(2) - log_var_post2.exp())
        
        # Average the KL divergence loss over the batch size
        batch_size = feat1_flat.size(0)
        kl_loss_total = (kl_loss1 + kl_loss2) / batch_size

        return loss1, loss2, kl_loss_total