"""
env_conditioning.py

Blind (label-free) environment estimation + FiLM (Feature-wise Linear
Modulation) conditioning, used to let the CST-former backbone adapt its
internal processing to the current acoustic conditions (noise / reverberation)
in a fully differentiable way -- as a complement to the discrete RL-based
window/filterbank selector in rl_feature_selector.py.

Design notes
------------
- estimate_environment() is a blind, heuristic estimator. It reuses the same
  kind of lightweight statistics already computed in
  rl_feature_selector.compute_state() (energy ratio, spectral flux, temporal
  variance) and adds two new heuristics: a coarse SNR estimate and a coarse
  inter-channel coherence proxy used as a reverberation indicator. None of
  these are validated acoustic parameters (a proper T60/DRR estimate would
  need access to the raw multichannel waveform, ideally computed upstream in
  cls_feature_class.py) -- they are cheap proxies meant to give the FiLM
  generator *some* signal about "how noisy/reverberant is this segment",
  which the network can then learn to exploit or ignore during training.

- FiLM is applied as: x' = (1 + gamma(env)) * x + beta(env)
  The generator's last linear layer is zero-initialized, so gamma == 0 and
  beta == 0 at the start of training -> FiLM starts as an identity mapping.
  This matters for finetune_mode: even with use_env_film=True, a freshly
  added FiLM layer will not disrupt a pretrained backbone at epoch 0; it only
  starts to deviate from identity as training updates its weights.

- Every FiLM submodule is created *only* if use_film=True is passed in at
  construction time. When use_env_film=False (the default in parameters.py),
  ConvBlock / CST_attention keep their exact original structure and
  state_dict keys, so existing checkpoints (params['pretrained_model_weights'])
  keep loading without any changes.
"""

import torch
import torch.nn as nn

ENV_DIM = 5  # [energy_ratio, spectral_flux, temporal_var, snr_estimate, coherence_proxy]


def estimate_environment(x, eps=1e-6):
    """
    Blind environment descriptor from a multichannel feature block.

    Args:
        x: (B, ch, T, F) -- multichannel feature tensor, e.g. the mel+GCC
           (or FOA+IV) block that CST_former receives as input, *before*
           the (b m) 1 t f channel-collapsing rearrange.
        eps: numerical floor.

    Returns:
        env_vector: (B, ENV_DIM) tensor. Computed under no_grad -- gradients
        never flow back into the estimate itself, only into the FiLM
        generator's own weights that consume it.
    """
    with torch.no_grad():
        B, C, T, F = x.shape

        energy = x.pow(2).mean(dim=(1, 2))                       # (B, F)
        low = energy[:, :F // 2].mean(dim=-1)
        high = energy[:, F // 2:].mean(dim=-1) + eps
        energy_ratio = torch.log1p((low / high).clamp(min=0))

        flux = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        flux = torch.log1p(flux.clamp(min=0))

        frame_energy = x.pow(2).mean(dim=(1, 3))                  # (B, T)
        temporal_var = torch.log1p(frame_energy.var(dim=-1).clamp(min=0))

        # --- crude blind SNR proxy ---
        # bottom ~20% energy frames ~ noise floor, top ~20% ~ signal+noise
        sorted_energy, _ = frame_energy.sort(dim=-1)
        k = max(1, T // 5)
        noise_floor = sorted_energy[:, :k].mean(dim=-1) + eps
        signal_peak = sorted_energy[:, -k:].mean(dim=-1) + eps
        snr_db = 10.0 * torch.log10(signal_peak / noise_floor)
        snr_norm = torch.tanh(snr_db / 20.0)                      # squashed to (-1, 1)

        # --- crude reverberation proxy: coherence between the first two
        #     feature channels (typically the first two mic-channel mel
        #     planes). Low coherence often correlates with diffuse /
        #     reverberant energy, but this is a heuristic, not a validated
        #     DRR/T60 estimate.
        if C >= 2:
            ch_a = x[:, 0]
            ch_b = x[:, 1]
            num = (ch_a * ch_b).mean(dim=(1, 2))
            denom = (ch_a.pow(2).mean(dim=(1, 2)).sqrt() *
                     ch_b.pow(2).mean(dim=(1, 2)).sqrt() + eps)
            coherence = (num / denom).clamp(-1.0, 1.0)
        else:
            coherence = torch.zeros(B, device=x.device, dtype=x.dtype)

        env_vector = torch.stack(
            [energy_ratio, flux, temporal_var, snr_norm, coherence], dim=-1
        )  # (B, ENV_DIM)

    return env_vector


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    x' = (1 + gamma(env)) * x + beta(env)

    channel_dim: which dimension of x holds the `num_features` channels
        that gamma/beta modulate (e.g. 1 for (B, C, T, F) conv features,
        2 for (B, T, C) attention features).
    """
    def __init__(self, env_dim, num_features, hidden_dim=32, channel_dim=1):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_features = num_features
        self.net = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features * 2),
        )
        # Zero-init the last layer -> gamma=0, beta=0 at t=0 -> identity mapping.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, env_vector):
        if env_vector is None:
            return x
        gamma_beta = self.net(env_vector)                 # (B, 2*num_features)
        gamma, beta = gamma_beta.chunk(2, dim=-1)          # (B, num_features) each
        gamma = 1.0 + gamma

        shape = [1] * x.dim()
        shape[0] = x.shape[0]
        shape[self.channel_dim] = self.num_features
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)

        return gamma * x + beta
