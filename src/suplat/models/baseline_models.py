"""
Baseline classifier models for 89×89 single-channel radio images.
Adapted from p2_DCRECLASS/src/dcreclass/models/classifiers.py.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class CNN(nn.Module):
    """6-block CNN with progressive dropout. AdaptiveAvgPool handles variable spatial size."""
    def __init__(self, input_shape, num_classes=2):
        super(CNN, self).__init__()

        in_channels = input_shape[0]

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.15),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.20),

            # Block 3
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.30),

            # Block 5
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.35),

            # Block 6
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.40),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ScatterNet(nn.Module):
    """
    CNN-based classifier for scattering coefficients.
    Accepts scat_shape=(C_scat, H_scat, W_scat) determined at runtime.
    For 89×89 images with J=2, L=8: scat_shape = (81, 23, 23).
    """
    def __init__(self, scat_shape, num_classes=2, hidden_dim=16, dropout_rate=0.5, J=2):
        super(ScatterNet, self).__init__()

        C_scat, H_scat, W_scat = scat_shape

        def conv_block(in_ch, out_ch, k, s, p, dropout_p=0.2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_p)
            )

        if J == 4:
            downsample_blocks = 1
        elif J == 3:
            downsample_blocks = 2
        elif J == 2:
            downsample_blocks = 3
        elif J == 1:
            downsample_blocks = 4
        else:
            downsample_blocks = 3

        scat_blocks = []
        scat_blocks.append(conv_block(C_scat, hidden_dim, 3, 1, 1, dropout_p=0.2))
        scat_blocks.append(SEBlock(hidden_dim))

        for i in range(downsample_blocks):
            dropout_p = 0.2 + i * 0.1
            scat_blocks.append(conv_block(hidden_dim, hidden_dim, 3, 1, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim))
            scat_blocks.append(conv_block(hidden_dim, hidden_dim, 3, 2, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim))

        self.scat_encoder = nn.Sequential(*scat_blocks)

        with torch.no_grad():
            dummy_scat = torch.zeros(1, C_scat, H_scat, W_scat)
            scat_f = self.scat_encoder(dummy_scat)
            feature_dim = scat_f.view(1, -1).size(1)

        self.fc1 = nn.Linear(feature_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.scat_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


class SimpleScatterNet(nn.Module):
    """Simple MLP: Flatten → Dense(120) → Dense(84) → Dropout(0.5) → output."""
    def __init__(self, input_shape, num_classes=2):
        super(SimpleScatterNet, self).__init__()
        flat_dim = 1
        for d in input_shape:
            flat_dim *= d
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DualScatterSqueezeNet(nn.Module):
    """
    Dual-branch network: raw image CNN + scattering coefficient CNN.
    Both branches' feature dims are computed dynamically via dummy forward.
    """
    def __init__(self, img_shape, scat_shape, num_classes,
                 hidden_dim1=32, hidden_dim2=16, classifier_hidden_dim=32,
                 dropout_rate=0.5, J=2):
        super(DualScatterSqueezeNet, self).__init__()
        C_img, H_img, W_img = img_shape
        C_scat, H_scat, W_scat = scat_shape

        # Image branch
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(C_img, 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )

        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4)
        )

        def conv_block(in_ch, out_ch, k, s, p, dropout_p=0.2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_p)
            )

        if J == 4:
            downsample_blocks = 1
        elif J == 3:
            downsample_blocks = 2
        elif J == 2:
            downsample_blocks = 3
        elif J == 1:
            downsample_blocks = 4
        else:
            raise ValueError("J must be 1, 2, 3, or 4")

        scat_blocks = []
        scat_blocks.append(conv_block(C_scat, hidden_dim2, 3, 1, 1, dropout_p=0.2))
        scat_blocks.append(SEBlock(hidden_dim2))
        for i in range(downsample_blocks):
            dropout_p = 0.2 + i * 0.1
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 1, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim2))
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 2, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim2))

        self.conv_to_latent_scat = nn.Sequential(*scat_blocks)

        with torch.no_grad():
            dummy_img  = torch.zeros(1, C_img, H_img, W_img)
            dummy_scat = torch.zeros(1, C_scat, H_scat, W_scat)
            img_f  = self.conv_to_latent_img(self.cnn_encoder(dummy_img))
            scat_f = self.conv_to_latent_scat(dummy_scat)
            combined_dim = img_f.view(1, -1).size(1) + scat_f.view(1, -1).size(1)

        self.FC_input      = nn.Linear(combined_dim, hidden_dim1)
        self.bn1           = nn.BatchNorm1d(hidden_dim1)
        self.dropout1      = nn.Dropout(0.5)
        self.FC_hidden     = nn.Linear(hidden_dim1, classifier_hidden_dim)
        self.bn2           = nn.BatchNorm1d(classifier_hidden_dim)
        self.dropout2      = nn.Dropout(0.5)
        self.FC_classifier = nn.Linear(classifier_hidden_dim, num_classes)
        self.act           = nn.LeakyReLU(0.2)

    def forward(self, img, scat):
        x_img  = self.conv_to_latent_img(self.cnn_encoder(img))
        x_scat = self.conv_to_latent_scat(scat)
        x_img  = x_img.view(x_img.size(0), -1)
        x_scat = x_scat.view(x_scat.size(0), -1)
        x = torch.cat([x_img, x_scat], dim=1)
        x = self.act(self.bn1(self.FC_input(x)))
        x = self.dropout1(x)
        x = self.act(self.bn2(self.FC_hidden(x)))
        x = self.dropout2(x)
        return self.FC_classifier(x)
