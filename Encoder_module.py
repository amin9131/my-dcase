import torch.nn 
import torch.nn as nn

import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

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

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class conv_encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()
        self.params = params
        self.t_pooling_loc = params["t_pooling_loc"]
        assert(len(params['f_pool_size']))

        self.conv_block_list = nn.ModuleList()

        if self.params['ChAtten_DCA']: in_channels = 1
        else: in_channels = in_feat_shape[1]

        for conv_cnt in range(len(params['f_pool_size'])):
            self.conv_block_list.append(nn.Sequential(
                ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_channels,
                          out_channels=params['nb_cnn2d_filt']),
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

    def forward(self, x):
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)  # out: B,C,T,F
        return x

class Encoder(torch.nn.Module):
    def __init__(self, in_feat_shape, params):
        super().__init__()

        self.encoder = conv_encoder(in_feat_shape, params)

    def forward(self, x):
        x = self.encoder(x)
        return x