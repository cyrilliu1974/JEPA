import torch
import torch.nn as nn
import torch.nn.functional as F

# 測試 projection 層的輸出範圍
# 確認 LayerNorm 是否正常工作

encoder_embed_dim = 1024
projection_dim = 256

projection = nn.Sequential(
    nn.Linear(encoder_embed_dim, projection_dim),
    nn.LayerNorm(projection_dim),
)

# 模擬真實 z 向量：norm ≈ 97，std ≈ 3
z_fake = torch.randn(4, encoder_embed_dim) * 3.0
z_fake = z_fake / z_fake.norm(dim=-1, keepdim=True) * 97.0

print(f"Input norm: {z_fake.norm(dim=-1).mean():.2f}")

z_proj = projection(z_fake)
print(f"After projection norm: {z_proj.norm(dim=-1).mean():.4f}")
print(f"After projection std:  {z_proj.std():.4f}")

z_proj_norm = F.normalize(z_proj, dim=-1)
print(f"After F.normalize norm: {z_proj_norm.norm(dim=-1).mean():.4f}")

# 計算兩個不同輸入之間的距離
z_fake2 = torch.randn(4, encoder_embed_dim) * 3.0
z_fake2 = z_fake2 / z_fake2.norm(dim=-1, keepdim=True) * 97.0
z_proj2 = projection(z_fake2)
z_proj_norm2 = F.normalize(z_proj2, dim=-1)

cos_sim = F.cosine_similarity(z_proj_norm, z_proj_norm2, dim=-1)
print(f"\nCosine similarity between different inputs: {cos_sim.mean():.4f}")
print(f"(Close to 1.0 = still collapsing, close to 0 = distinguishable)")