
# target_mask_network.py
#
# V1 — label-guided target-source extraction branch (mask -> apply -> classify).
#
# This branch is independent of the main SELD backbone. It operates on the same
# (B, nb_ch, T, F) input produced by the current DataGenerator, but only takes
# the first 4 raw Log-Mel channels (channels 0:4). It learns a time-frequency
# mask, applies the mask to the Mel features (BSS-Mel), and uses a small
# classifier to predict the presence probability of the target class over the
# entire chunk (weak/chunk-level, not frame-by-frame).
#
# The computational graph is intentionally ordered as mask -> apply -> classify
# (rather than parallel branches), because otherwise the BCE loss would not
# provide pressure on the shape of the mask.
#
# This module intentionally does not use CST_conformer.py or Encoder_module.py.
# Those modules are designed for 10-channel inputs and heavier attention-based
# processing, which is unnecessary for the V1 question:
# "Can a label-guided mask learn to produce useful features?"


import torch
import torch.nn as nn


class MaskNetwork(nn.Module):
    """
    Input:  x_mel  (B, 4, T, F)   -- 4 raw Log-Mel channels (channels 0:4
                                    of the current feature tensor)
    Output: mask   (B, T, F)      -- values in the range [0, 1] after sigmoid

    No frequency pooling is performed in the convolutional path
    (padding='same'), so there is no need to upsample F again at the end.
    This directly avoids the issue in the initial design where pooling required
    a subsequent upsampling step, which has been removed in the final design.
    """

    def __init__(self, params):
        super().__init__()
        nb_mel_bins = params['nb_mel_bins']
        conv1_out = params.get('mask_conv1_out', 16)
        conv2_out = params.get('mask_conv2_out', 32)
        gru_hidden = params.get('mask_gru_hidden', 32)
        dropout = params.get('mask_dropout', 0.05)

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, conv1_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_out),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_out),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )

        # GRU input = (channels from conv2) * (nb_mel_bins).
        # The C and F dimensions are flattened for each time frame.
        gru_input_dim = conv2_out * nb_mel_bins
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Since the GRU is bidirectional, its output size is 2 * gru_hidden.
        # This Linear layer independently maps that representation to exactly
        # nb_mel_bins. This makes the design more general and less dependent
        # on a numerical relationship between gru_hidden and nb_mel_bins.
        self.proj = nn.Linear(2 * gru_hidden, nb_mel_bins)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_mel):
        # x_mel: (B, 4, T, F)
        h = self.conv1(x_mel)              # (B, conv1_out, T, F)
        h = self.conv2(h)                  # (B, conv2_out, T, F)

        B, C, T, F = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, T, C * F)   # (B, T, C*F)

        h, _ = self.gru(h)                 # (B, T, 2*gru_hidden)
        h = self.proj(h)                   # (B, T, nb_mel_bins)

        mask = torch.sigmoid(h)            # (B, T, F)
        return mask


class TargetClassifier(nn.Module):
    """
    Input:  bss_mel  (B, 4, T, F)  -- original Mel features multiplied by mask
    Output: logit    (B,)          -- raw logits (without sigmoid), for use with
                                      BCEWithLogitsLoss

    Intentionally lightweight: one Conv2D + global average pooling + one Linear.
    The goal of V1 is not to find the best classifier, but to answer the question:
    "Can the gradient path from this classifier improve the shape of the mask?"
    """

    def __init__(self, params):
        super().__init__()
        hidden_ch = params.get('cls_conv_out', 16)
        dropout = params.get('mask_dropout', 0.05)

        self.conv = nn.Sequential(
            nn.Conv2d(4, hidden_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # Global average pooling over (T, F)
        self.fc = nn.Linear(hidden_ch, 1)

        self.apply(MaskNetwork._init_weights)

    def forward(self, bss_mel):
        h = self.conv(bss_mel)             # (B, hidden_ch, T, F)
        h = self.pool(h).flatten(1)        # (B, hidden_ch)
        logit = self.fc(h).squeeze(-1)     # (B,)
        return logit


class TargetMaskBranch(nn.Module):
    """
    Complete wrapper: mask -> apply -> classify

    The forward method accepts the complete SELD feature tensor
    (B, nb_ch, T, F), exactly as produced by the existing DataGenerator,
    and internally extracts the first 4 Mel channels.

    This means V1 does not require any changes to cls_data_generator.py
    or batch_feature_extraction.py.
    """

    def __init__(self, params):
        super().__init__()
        self.mask_net = MaskNetwork(params)
        self.classifier = TargetClassifier(params)

    def forward(self, x_full):
        # x_full: (B, nb_ch=10, T, F)
        x_mel = x_full[:, :4]                       # (B, 4, T, F) -- channels 0:4

        mask = self.mask_net(x_mel)                 # (B, T, F)
        bss_mel = x_mel * mask.unsqueeze(1)          # Broadcast over all 4 Mel channels

        logit = self.classifier(bss_mel)             # (B,)
        return logit, mask, bss_mel


def extract_target_chunk_label(label_adpit, target_class_idx):
    """
    label_adpit: (B, T_label, 6, 4, nb_classes) -- ADPIT label tensor produced
        by the DataGenerator.
        Dimension 6 = combined ADPIT slots (a0, b0, b1, c0, c1, c2).
        Dimension 4 = (activation, x, y, z), where index 0 is the activation.
        Last dimension = class.

    Output: (B,) binary tensor -- 1 if the target class is active at any point
    in the chunk (any label frame and any ADPIT slot).

    This is a weak/chunk-level label, not a frame-level label. It is therefore
    aligned with the TargetClassifier, which also produces only one output for
    the entire chunk after global average pooling.
    """
    activity = label_adpit[:, :, :, 0, target_class_idx]      # (B, T_label, 6)
    chunk_label = (activity.sum(dim=(1, 2)) > 0).float()       # (B,)
    return chunk_label

