import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import random

# ===== 시드 고정 =====
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. Dataset 정의
# ============================================

class BeerPairDataset(Dataset):
    """
    같은 맥주 (chem, sensory) 쌍은 positive.
    배치 내에서 섞인 sensory를 negative로 사용하는 contrastive 학습용 Dataset.
    """
    def __init__(self, chem_data, sensory_data):
        assert chem_data.shape[0] == sensory_data.shape[0]
        self.chem = chem_data
        self.sensory = sensory_data

    def __len__(self):
        return self.chem.shape[0]

    def __getitem__(self, idx):
        return self.chem[idx], self.sensory[idx], idx


# 2. Two-Tower 모델 정의
# ============================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(512, 256, 128, 128, 64, 32)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoTowerModel(nn.Module):
    def __init__(self, chem_dim, sensory_dim, latent_dim=32):
        super().__init__()
        self.chem_encoder = MLPEncoder(chem_dim, latent_dim=latent_dim, hidden_dims=(32, 16))
        self.sensory_encoder = MLPEncoder(sensory_dim, latent_dim=latent_dim, hidden_dims=(128, 64))

    def forward(self, chem_input, sensory_input):
        chem_latent = self.chem_encoder(chem_input)
        sensory_latent = self.sensory_encoder(sensory_input)
        chem_latent = F.normalize(chem_latent, dim=1)
        sensory_latent = F.normalize(sensory_latent, dim=1)
        return chem_latent, sensory_latent


# 3. Contrastive Loss
# ============================================

def contrastive_batch_loss(chem_latent, sensory_latent, margin=0.3):
    """
    - 같은 인덱스 (i,i): positive pair
    - 배치 내에서 permute된 sensory_latent를 negative로 사용.
    """
    batch_size = chem_latent.size(0)

    pos_cos = F.cosine_similarity(chem_latent, sensory_latent, dim=1)
    perm = torch.randperm(batch_size, device=chem_latent.device)
    neg_sensory = sensory_latent[perm]
    neg_cos = F.cosine_similarity(chem_latent, neg_sensory, dim=1)

    pos_loss = 1.0 - pos_cos
    neg_loss = torch.relu(neg_cos - margin)

    loss = pos_loss.mean() + neg_loss.mean()
    return loss, pos_cos.mean().item(), neg_cos.mean().item()


# (옵션) InfoNCE 로스는 당분간 사용 안 하면 그대로 두고 주석 유지
def info_nce_loss(chem_latent, sensory_latent, tau=0.09):
    logits = (chem_latent @ sensory_latent.T) / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        pos = torch.diag(logits).mean()
        neg = (logits.sum(dim=1) - torch.diag(logits)) / (logits.size(1) - 1)
        neg = neg.mean()
    return loss, pos.item(), neg.item()


# 4. Training Loop
# ============================================

chem_cols = [
    'ethyl_acetate',
    'ethanol..v.v.',
    'ethyl_octanoate',
    'ethyl.phenylacetate',
    'lactic_acid.mg.L.',
    'protein.g.L.',
]


def train_model(
    chem_data,
    sensory_data,
    n_epochs=500,
    batch_size=32,
    lr=1e-3,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BeerPairDataset(chem_data, sensory_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TwoTowerModel(chem_dim=chem_data.shape[1],
                          sensory_dim=sensory_data.shape[1],
                          latent_dim=32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        epoch_pos_cos = 0.0
        epoch_neg_cos = 0.0
        n_batches = 0

        for chem_batch, sensory_batch, _ in dataloader:
            chem_batch = chem_batch.to(device)
            sensory_batch = sensory_batch.to(device)

            optimizer.zero_grad()
            chem_latent, sensory_latent = model(chem_batch, sensory_batch)

            loss, pos_cos_mean, neg_cos_mean = contrastive_batch_loss(chem_latent, sensory_latent)
            # loss, pos_cos_mean, neg_cos_mean = info_nce_loss(chem_latent, sensory_latent)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pos_cos += pos_cos_mean
            epoch_neg_cos += neg_cos_mean
            n_batches += 1

        scheduler.step()
        print(f"[Epoch {epoch:03d}] Loss={epoch_loss/n_batches:.4f} | "
              f"PosCos={epoch_pos_cos/n_batches:.3f} | NegCos={epoch_neg_cos/n_batches:.3f}")

    return model


# 5. 평가 및 간단 t-SNE
# ============================================

def evaluate_and_plot(model, chem_data_scaled, sensory_data_scaled, sample_size=500, n_perplexity=30):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        chem = torch.from_numpy(chem_data_scaled).to(device)
        sens = torch.from_numpy(sensory_data_scaled).to(device)
        zc, zs = model(chem, sens)
        zc = zc.cpu().numpy()
        zs = zs.cpu().numpy()

    pos_cos = np.sum(zc * zs, axis=1)
    perm = np.random.permutation(len(zc))
    neg_cos = np.sum(zc * zs[perm], axis=1)

    plt.figure(figsize=(8, 4))
    plt.hist(pos_cos, bins=30, alpha=0.7, label='Positive (same beer)')
    plt.hist(neg_cos, bins=30, alpha=0.7, label='Negative (shuffled)')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Positive vs Negative Cosine')
    plt.show()

    y_true = np.concatenate([np.ones_like(pos_cos), np.zeros_like(neg_cos)])
    y_score = np.concatenate([pos_cos, neg_cos])
    auc = roc_auc_score(y_true, y_score)
    print(f"AUC (pos vs neg cosine): {auc:.3f}")

    n = len(zc)
    idx = np.random.choice(n, size=min(sample_size, n), replace=False)
    emb = np.vstack([zc[idx], zs[idx]])
    tsne = TSNE(n_components=2, perplexity=min(n_perplexity, len(idx)-1),
                random_state=42, init='random')
    emb2d = tsne.fit_transform(emb)
    chem2d, sens2d = emb2d[:len(idx)], emb2d[len(idx):]

    plt.figure(figsize=(8, 6))
    plt.scatter(chem2d[:, 0], chem2d[:, 1], c='tomato', marker='o', alpha=0.6, label='Chem')
    plt.scatter(sens2d[:, 0], sens2d[:, 1], c='steelblue', marker='^', alpha=0.6, label='Sensory')
    plt.legend()
    plt.title('t-SNE of Joint Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

    K = 5
    sims = zc @ zs.T
    ranks = np.argsort(-sims, axis=1)
    hits = (ranks[:, :K] == np.arange(len(zc))[:, None]).any(axis=1)
    recall_at_k = hits.mean()
    print(f"Recall@{K}: {recall_at_k:.3f}")


# 6. 상세 시각화 (Pair Alignment + Attribute Gradient)
# ============================================

def build_pairs_to_compare(chem_df, sensory_df, name_pairs):
    """
    name_pairs: [('lactic_acid.mg.L.', 'taste_sour'), ...]
    반환: [(c_idx, s_idx, c_name, s_name), ...]
    """
    pairs = []
    for chem_col, sens_col in name_pairs:
        if chem_col not in chem_df.columns:
            raise ValueError(f"chem_df에 컬럼이 없습니다: {chem_col}")
        if sens_col not in sensory_df.columns:
            raise ValueError(f"sensory_df에 컬럼이 없습니다: {sens_col}")
        c_idx = chem_df.columns.get_loc(chem_col)
        s_idx = sensory_df.columns.get_loc(sens_col)
        pairs.append((c_idx, s_idx, chem_col, sens_col))
    return pairs


def visualize_detailed_analysis(model, chem_data, sensory_data,
                                n_perplexity=30, random_state=42):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        chem_tensor = torch.from_numpy(chem_data).to(device)
        sensory_tensor = torch.from_numpy(sensory_data).to(device)
        zc, zs = model(chem_tensor, sensory_tensor)
        zc = zc.cpu().numpy()
        zs = zs.cpu().numpy()

    n_samples = chem_data.shape[0]
    all_emb = np.vstack([zc, zs])

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=n_perplexity,
                random_state=random_state, init='pca', learning_rate='auto')
    all_2d = tsne.fit_transform(all_emb)

    chem_2d = all_2d[:n_samples]
    sens_2d = all_2d[n_samples:]

    # 6-1. Pair Alignment
    plt.figure(figsize=(10, 8))
    sample_indices = np.random.choice(n_samples, 50, replace=False)
    plt.scatter(chem_2d[:, 0], chem_2d[:, 1], c='red', alpha=0.2, s=30, label='Chem (All)')
    plt.scatter(sens_2d[:, 0], sens_2d[:, 1], c='blue', alpha=0.2, s=30, label='Sensory (All)')

    for idx in sample_indices:
        plt.plot([chem_2d[idx, 0], sens_2d[idx, 0]],
                 [chem_2d[idx, 1], sens_2d[idx, 1]],
                 color='gray', alpha=0.5, linewidth=1)
        plt.scatter(chem_2d[idx, 0], chem_2d[idx, 1], c='red', edgecolors='k', s=60)
        plt.scatter(sens_2d[idx, 0], sens_2d[idx, 1], c='blue', edgecolors='k', marker='^', s=60)

    plt.title("Pair Alignment Check (Top 50 Random Samples)\nShort lines = Better Joint Embedding")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # 6-2. Attribute Gradient
    name_pairs = [
        ('lactic_acid.mg.L.', 'taste_sour'),
    ]
    pairs_to_compare = build_pairs_to_compare(chemical_df[chem_cols], feature_df, name_pairs)

    fig, axes = plt.subplots(len(pairs_to_compare), 2, figsize=(14, 6 * len(pairs_to_compare)))
    if len(pairs_to_compare) == 1:
        axes = np.array([axes])

    for i, (c_idx, s_idx, c_name, s_name) in enumerate(pairs_to_compare):
        c_vals = chem_data[:, c_idx]
        s_vals = sensory_data[:, s_idx]

        sc1 = axes[i, 0].scatter(chem_2d[:, 0], chem_2d[:, 1], c=c_vals, cmap='Reds', alpha=0.8, s=40)
        axes[i, 0].set_title(f"{c_name} Distribution")
        plt.colorbar(sc1, ax=axes[i, 0])

        sc2 = axes[i, 1].scatter(sens_2d[:, 0], sens_2d[:, 1], c=s_vals, cmap='Blues', alpha=0.8, marker='^', s=40)
        axes[i, 1].set_title(f"{s_name} Distribution")
        plt.colorbar(sc2, ax=axes[i, 1])

    plt.tight_layout()
    plt.show()


# 7. 전체 실행 스크립트
# ============================================

def main():
    # numpy 배열로 변환
    chem_data = chemical_df[chem_cols].values.astype("float32")
    sensory_df = feature_df.drop(columns=['palate_acetaldehyde', 'palate_acetate',
                                          'palate_dentist', 'aroma_hops_sum'])
    sensory_data = sensory_df.values.astype("float32")

    chem_train, chem_val, sens_train, sens_val = train_test_split(
        chem_data, sensory_data, test_size=0.2, random_state=42, shuffle=True
    )

    chem_scaler = StandardScaler()
    sensory_scaler = StandardScaler()
    chem_train_scaled = chem_scaler.fit_transform(chem_train)
    sensory_train_scaled = sensory_scaler.fit_transform(sens_train)
    chem_val_scaled = chem_scaler.transform(chem_val)
    sensory_val_scaled = sensory_scaler.transform(sens_val)

    model = train_model(
        chem_train_scaled,
        sensory_train_scaled,
        n_epochs=200,
        batch_size=32,
        lr=1e-3
    )

    evaluate_and_plot(model, chem_val_scaled, sensory_val_scaled,
                      sample_size=500, n_perplexity=30)

    visualize_detailed_analysis(model, chem_val_scaled, sensory_val_scaled)


if __name__ == "__main__":
    main()
