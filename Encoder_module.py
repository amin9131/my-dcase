import torch.nn 
import torch.nn as nn

import torch.nn.functional as F

from env_conditioning import FiLM, ENV_DIM


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 use_film=False, env_dim=ENV_DIM, film_hidden=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

        # Registered only when explicitly enabled, so state_dict keys stay
        # unchanged for existing checkpoints when use_film=False.
        self.use_film = use_film
        if self.use_film:
            self.film = FiLM(env_dim=env_dim, num_features=out_channels,
                              hidden_dim=film_hidden, channel_dim=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, env_vector=None):
        x = self.bn(self.conv(x))
        if self.use_film:
            x = self.film(x, env_vector)
        x = F.relu(x)
        return x

class conv_encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        self.params = params
        self.t_pooling_loc = params["t_pooling_loc"]
        assert(len(params['f_pool_size']))

        self.use_film = params.get('use_env_film', False)
        env_dim = params.get('env_vector_dim', ENV_DIM)
        film_hidden = params.get('film_hidden_dim', 32)

        self.conv_block_list = nn.ModuleList()

        if self.params['ChAtten_DCA']: in_channels = 1
        else: in_channels = in_feat_shape[1]

        for conv_cnt in range(len(params['f_pool_size'])):
            self.conv_block_list.append(nn.Sequential(
                ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_channels,
                          out_channels=params['nb_cnn2d_filt'],
                          use_film=self.use_film, env_dim=env_dim, film_hidden=film_hidden),
                nn.MaxPool2d((params['t_pool_size'][conv_cnt] if self.t_pooling_loc == 'front' else 1,
                              params['f_pool_size'][conv_cnt])),
                nn.Dropout2d(p=params['dropout_rate']),
            ))

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, env_vector=None):
        # Each entry of conv_block_list is nn.Sequential(ConvBlock, MaxPool2d, Dropout2d).
        # We index into it manually (instead of calling the Sequential as a whole) purely so
        # we can pass env_vector to ConvBlock only -- this does NOT change any state_dict keys,
        # since keys come from module registration, not from how forward() is invoked.
        for conv_cnt in range(len(self.conv_block_list)):
            block = self.conv_block_list[conv_cnt]
            x = block[0](x, env_vector)  # ConvBlock
            x = block[1](x)              # MaxPool2d
            x = block[2](x)              # Dropout2d
        return x

class Encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()

        self.encoder = conv_encoder(in_feat_shape, params)

    def forward(self, x, env_vector=None):
        x = self.encoder(x, env_vector)
        return x
