# model_mm_joint_uav_lidar.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

def sinusoidal_position_encoding(L, D, device):
    pe = torch.zeros(L, D, device=device)
    position = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L,D)

class RAWImage(nn.Module):
    def __init__(self, emb_dim=256, drop_prob=0.3):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.img_backbone = backbone

        img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        img_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer("img_mean", img_mean, persistent=False)
        self.register_buffer("img_std", img_std, persistent=False)

        self.img_head = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        xn = (x - self.img_mean) / self.img_std
        h  = self.img_backbone(xn)
        h  = self.img_head(h)
        return h

class LidarMARCUS(nn.Module):
    def __init__(self, channel, fusion, emb_dim, dropout):
        super().__init__()
        channel = 32; 
        self.c1 = nn.Conv2d(20, channel, 3, padding=1)
        self.b1 = nn.BatchNorm2d(channel)
        self.c2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.b2 = nn.BatchNorm2d(channel)
        self.c3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.b3 = nn.BatchNorm2d(channel)
        self.c4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.b4 = nn.BatchNorm2d(channel)
        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1  = nn.Linear(3200, 1024)
        self.bn1  = nn.BatchNorm1d(1024)
        self.fc2  = nn.Linear(1024, emb_dim)
        self.bn2  = nn.BatchNorm1d(emb_dim)
        self.do1  = nn.Dropout(dropout)
        self.do2  = nn.Dropout(0.2)
        self.do3  = nn.Dropout(0.2)
        self.lidar_classifier = nn.Linear(emb_dim, 4)

        self.reg_l1 = 1e-5
        self.reg_l2 = 1e-4
        self.fusion = fusion

    def forward(self, x):
        x = x.squeeze(1)  # (B,20,20,20)
        a = self.b1(F.relu(self.c1(x)))
        z = self.b2(F.relu(self.c2(a))); z = a + z
        z = self.pool(z); z = self.do1(z)
        z = self.b3(F.relu(self.c3(z)))
        z = self.b4(F.relu(self.c4(z)))
        z = self.flat(z)
        z = self.bn1(F.relu(self.fc1(z))); z = self.do2(z)
        z = self.bn2(F.relu(self.fc2(z))); z = self.do3(z)
        if self.fusion:
            return z  # (B,512)
        else:
            return self.lidar_classifier(z)  # (B,20)
            
    
    def compute_regularization_loss(self):
        l1_loss = 0
        l2_loss = 0
        for name, param in self.named_parameters():
            if any(x in name for x in ["fc1.weight", "fc2.weight",
                                       "fc1.bias", "fc2.bias"]) and param.requires_grad:
                l1_loss += torch.sum(torch.abs(param)) 
                l2_loss += torch.sum(param ** 2)
        return self.reg_l1 * l1_loss + self.reg_l2 * l2_loss

class AggIMAGEConv(nn.Module):
    def __init__(self, d, ksize, dropout):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv1 = nn.Conv1d(d, d, kernel_size=ksize, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(d)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d, d, kernel_size=ksize, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(d)

    def forward(self, f5):
        x = f5.transpose(1, 2)          # (B,256,5)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop(y)
        y2 = self.conv2(y)
        y2 = self.bn2(y2)
        y2 = self.act(y2)
        y = y + y2

        y_mean = y.mean(dim=-1)          # (B,256)
        return y_mean

class UAViamgeAgg(nn.Module):
    def __init__(self, emb_dim, ksize, uav_dropout, detection = False):
        super().__init__()
        self.img_enc = RAWImage(emb_dim=emb_dim)
        self.view_agg = AggIMAGEConv(d=emb_dim, ksize=ksize, dropout=uav_dropout)
        # self.detector = nn.Linear(emb_dim, 11)
        self.detector = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=uav_dropout),
            nn.Linear(128, 11)
        )
        self.detection = detection
    def forward(self, imgs_uav):
        """
        imgs_uav: (B, K, 5, 3, H, W)
        """
        B, K, V, C, H, W = imgs_uav.shape
        x = imgs_uav.reshape(B*K*V, C, H, W)              # (B*K*5, C, H, W)
        f = self.img_enc(x)                                # (B*K*5, 256)
        f = f.view(B*K, V, -1).contiguous()               # (B*K, 5, 256)
        f = self.view_agg(f)                               # (B*K, 256)

        if self.detection:
            return torch.relu(self.detector(f))  # (B,11)
        else:
            return f.view(B, K, -1).contiguous()           # (B, K, 256)
            

class HETEattenFUSION(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model

        self.fuse_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.uav_type  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.lidar_type = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.fuse_type, std=0.02)
        nn.init.trunc_normal_(self.uav_type,  std=0.02)
        nn.init.trunc_normal_(self.lidar_type, std=0.02)
        self.fuse_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.fuse_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, f_uav, f_lidar):
        B = f_uav.size(0)
        device = f_uav.device
        fuse = self.fuse_token.expand(B, 1, -1)   # (B,1,D)
        f_lidar = f_lidar.unsqueeze(1)            # (B,1,D)
        x = torch.cat([fuse, f_lidar, f_uav], dim=1)  # (B, 1+1+K, D)
        type_embed = torch.cat([
            self.fuse_type.expand(B, 1, -1),
            self.lidar_type.expand(B, 1, -1),
            self.uav_type.expand(B, f_uav.size(1), -1)
        ], dim=1)  # (B, 1+1+K, D)
        x = x + type_embed
        L = x.size(1)
        pe = sinusoidal_position_encoding(L, self.d_model, device).unsqueeze(0)  # (1,L,D)
        x = x + pe
        h = self.encoder(x, src_key_padding_mask=None)   # (B,L,D)
        fused = h[:, 0, :]                                # (B,D)
        out = self.mlp_head(fused)                        # (B,D)
        return out

class MultiModalFusionModel(nn.Module):
    def __init__(self,
                 img_in_shape=(3,108,192),
                 uav_emb_dim=256,
                 car_emb_dim=256, 
                 num_classes=28,
                 attn_d_model=256,
                 attn_nhead=4,
                 attn_num_layers=2,
                 attn_ff=768,
                 uav_dropout=0.1,
                 car_dropout=0.3,
                 fusion_dropout=0.2,
                 ):
        super().__init__()

        self.uav_enc = UAViamgeAgg(emb_dim=uav_emb_dim,
            ksize=3, uav_dropout=uav_dropout
        )

        self.joint_fusion = HETEattenFUSION(
            d_model=attn_d_model, nhead=attn_nhead, num_layers=attn_num_layers, dim_feedforward=attn_ff, dropout=fusion_dropout)

        self.handsoff = nn.Sequential(
            nn.LayerNorm(attn_d_model),
            nn.Linear(attn_d_model, num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar, uav_images):
        f_uav = self.uav_enc(uav_images)       # (B,K,256)
        f_lidar = self.lidar_enc(lidar)        # (B,512)
        f_fused = self.joint_fusion(f_uav, f_lidar)  # (B,256)
        logits = self.handsoff(f_fused)      # (B,num_classes)

        return logits