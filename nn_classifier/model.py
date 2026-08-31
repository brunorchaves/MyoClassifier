"""CNN 1D pequena para classificar janelas de EMG bruto (8 canais x 100
amostras). Arquitetura simples de proposito: e o que a maior parte da
literatura sobre EPN612/NinaPro usa como baseline de deep learning, e treina
rapido em GPU nos 612 sujeitos do EPN612.
"""

import torch
import torch.nn as nn

IN_CHANNELS = 8


def _conv_block(in_ch, out_ch, kernel_size=5):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
    )


class EMGConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            _conv_block(IN_CHANNELS, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 8, WINDOW_SIZE)
        features = self.backbone(x).flatten(1)  # (batch, 128)
        return self.head(features)

    def backbone_state_dict(self):
        return self.backbone.state_dict()

    def load_backbone_state(self, state_dict, strict: bool = True):
        self.backbone.load_state_dict(state_dict, strict=strict)

    def freeze_backbone(self, freeze: bool = True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


def build_model(num_classes: int, pretrained_backbone: str | None = None, freeze_backbone: bool = False) -> EMGConvNet:
    model = EMGConvNet(num_classes)
    if pretrained_backbone:
        checkpoint = torch.load(pretrained_backbone, map_location="cpu")
        model.load_backbone_state(checkpoint["backbone"])
        if freeze_backbone:
            model.freeze_backbone(True)
    return model
