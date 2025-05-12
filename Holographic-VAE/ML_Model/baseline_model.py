import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EarMeasureEncoder(nn.Module):
    def __init__(self, ear_anthro_dim, ear_emb_dim):
        super(EarMeasureEncoder, self).__init__()
        self.ear_anthro_dim = ear_anthro_dim
        self.fc = nn.Sequential(
            nn.Linear(ear_anthro_dim, ear_emb_dim),
        )

    def forward(self, ear_anthro):
        assert ear_anthro.shape[1] == self.ear_anthro_dim
        return self.fc(ear_anthro)

class HeadMeasureEncoder(nn.Module):
    def __init__(self, head_anthro_dim, head_emb_dim):
        super(HeadMeasureEncoder, self).__init__()
        self.head_anthro_dim = head_anthro_dim
        self.fc = nn.Sequential(
            nn.Linear(head_anthro_dim, head_emb_dim),
        )

    def forward(self, head_anthro):
        assert head_anthro.shape[1] == self.head_anthro_dim
        return self.fc(head_anthro)

class ConvNNHrtfSht(nn.Module):
    def __init__(self, args):
        super(ConvNNHrtfSht, self).__init__()
        self.ear_enc = EarMeasureEncoder(args.ear_anthro_dim, args.ear_emb_dim)
        self.head_enc = HeadMeasureEncoder(args.head_anthro_dim, args.head_emb_dim)
        self.lr_enc = nn.Embedding(2, args.lr_emb_dim)
        self.freq_enc = nn.Embedding(args.freq_bin, args.freq_emb_dim)
        self.condition_dim = args.condition_dim
        emb_concat_dim = args.ear_emb_dim + args.head_emb_dim + args.freq_emb_dim + args.lr_emb_dim
        self.fc = nn.Linear(emb_concat_dim, args.condition_dim)

        # select normalization mode
        self.norm = args.norm

        # generator blocks
        self.conv1 = self.make_gen_block(1, 4, kernel_size=7, stride=3)
        self.conv2 = self.make_gen_block(4, 16, kernel_size=5, stride=2)
        self.conv3 = self.make_gen_block(16, 32, kernel_size=5, stride=2)
        self.conv4 = self.make_gen_block(32, 32, kernel_size=5, stride=3)
        out_channels = 440 if args.target == "hrtf" else 64
        self.conv5 = self.make_gen_block(32, out_channels, kernel_size=5, stride=2, final_layer=True)

    def make_gen_block(self, in_ch, out_ch, kernel_size=3, stride=2, final_layer=False):
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size, stride)]
        if not final_layer:
            if self.norm == "batch":
                layers.append(nn.BatchNorm1d(out_ch))
            elif self.norm == "layer":
                # single-group layer norm over channels
                layers.append(nn.GroupNorm(1, out_ch))
            elif self.norm == "instance":
                layers.append(nn.InstanceNorm1d(out_ch))
            else:
                raise ValueError(f"Unknown norm: {self.norm}")
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def unsqueeze_condition(self, latent):
        return latent.view(len(latent), 1, self.condition_dim)

    def forward(self, ear_anthro, head_anthro, frequency, left_or_right):
        # encode anthropometry & condition
        ear_emb = self.ear_enc(ear_anthro)
        head_emb = self.head_enc(head_anthro)
        freq_emb = self.freq_enc(frequency)
        lr_emb = self.lr_enc(left_or_right)

        latent = torch.cat((ear_emb, head_emb, freq_emb, lr_emb), dim=1)
        latent = self.unsqueeze_condition(self.fc(latent))

        # pass through conv blocks
        x = self.conv1(latent)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.conv5(x)

        # permanent fix: squeeze spatial dimension
        out = out.squeeze(-1)  # [B, C]
        return out
