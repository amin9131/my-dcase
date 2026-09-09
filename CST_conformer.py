import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import Encoder_module 

from env_conditioning import FiLM, ENV_DIM, estimate_environment


class CST_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.nb_mel_bins = params['nb_mel_bins']
        self.ChAtten_dca = params['ChAtten_DCA']
        self.linear_layer = params['LinearLayer']
        self.FreqAtten = params['FreqAtten']
        self.cmt_block = params['CMT_block']
        self.dropout_rate = params['dropout_rate']
        self.temp_embed_dim = temp_embed_dim
        self.nb_ch = 10

        self.use_film = params.get('use_env_film', False)
        env_dim = params.get('env_vector_dim', ENV_DIM)
        film_hidden = params.get('film_hidden_dim', 32)


        # Channel attention w. Divided Channel Attention (DCA) ---------------------------------------------#
        if self.ChAtten_dca:
            self.ch_attn_embed_dim = params['nb_cnn2d_filt']  # 64
            self.ch_mhsa = nn.MultiheadAttention(embed_dim=self.ch_attn_embed_dim, num_heads=params['nb_heads'],
                                      dropout=self.dropout_rate, batch_first=True)
            self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.ch_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # Environment-conditioned FiLM after channel attention. Registered only when enabled,
        # so state_dict keys are unchanged (and old checkpoints keep loading) when use_env_film=False.
        if self.use_film:
            self.ch_film = FiLM(env_dim=env_dim, num_features=self.temp_embed_dim,
                                 hidden_dim=film_hidden, channel_dim=2)

        # temporal attention -----------------------------------------------------------------------------------#
        self.embed_dim_4_freq_attn = params['nb_cnn2d_filt'] # Update the temp embedding if freq attention is applied
        # self.temp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim_4_freq_attn if params['FreqAtten'] else self.embed_dim,
        #                           num_heads= 8,
        #                           dropout=self.dropout_rate, batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if self.linear_layer:
            self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # Environment-conditioned FiLM after temporal attention (same opt-in rule as above).
        if self.use_film:
            self.temp_film = FiLM(env_dim=env_dim, num_features=self.temp_embed_dim,
                                   hidden_dim=film_hidden, channel_dim=2)

        self.temp_gru = torch.nn.GRU(input_size= self.embed_dim_4_freq_attn, hidden_size=self.embed_dim_4_freq_attn,
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate if self.dropout_rate > 0. else nn.Identity())

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

    def forward(self, x, M, C, T, F, env_vector=None):
        # x shape = 32(batch_size) * 50 * 1280
        

        #CST-attention (DCA)
        # channel attention
        # print('line_67  x shape in CST_attention class = ', x.shape)
        x_init = x.clone() # x shape = 32*50*1280
        xc = rearrange(x_init, 'b t (m f c)-> (b t f) m c', c=C, f=F).contiguous() # xc = 3200*10*64

        xc, _ = self.ch_mhsa(xc, xc, xc)
        xc = rearrange(xc, ' (b t f) m c -> b t (f m c)', t=T, f=F).contiguous() # 32*50*1280

        
        if self.linear_layer:
            xc = self.activation(self.ch_linear(xc))
        xc = xc + x_init
        if self.dropout_rate:
            xc = self.drop_out(xc)
        xc = self.ch_layer_norm(xc)
        if self.use_film:
            xc = self.ch_film(xc, env_vector)

        # temporal attention
        xt = rearrange(xc, ' b t (f m c) -> (b f m) t c', m=M, f=F).contiguous() # 640*50*64
        
        (xt, _) = self.temp_gru(xt)
        xt = torch.tanh(xt)
        xt = xt[:, :, xt.shape[-1]//2:] * xt[:, :, :xt.shape[-1]//2]

        # xt, _ = self.temp_mhsa(xt, xt, xt)
        xt = rearrange(xt, ' (b f m) t c -> b t (f m c)', m=M, f=F).contiguous() # 32*50*1280
        
        if self.linear_layer:
            xt = self.activation(self.temp_linear(xt))
        xt = xt + xc
        # xt = xt + xc
        if self.dropout_rate:
            xt = self.drop_out(xt)
        x = self.temp_layer_norm(xt)
        if self.use_film:
            x = self.temp_film(x, env_vector)

        return x

class CST_encoder(torch.nn.Module):
    def __init__(self, temp_embed_dim, params): # temp_embed_dim = 128
        super().__init__()
        self.freq_atten = params['FreqAtten']
        self.ch_atten_dca = params['ChAtten_DCA']
        self.ch_atten_ule = params['ChAtten_ULE']
        self.nb_ch = 10
        n_layers = params['nb_self_attn_layers']

        self.block_list = nn.ModuleList([CST_attention(
            temp_embed_dim = temp_embed_dim,
            params=params
        ) for _ in range(n_layers)] # n_layers = 2
        )

    def forward(self, x, env_vector=None):
        B, C, T, F = x.size() # 320(= batch_size * nb_ch) * 64 * 50 * 2
        # print('B C T F =', B, C, T, F)
        M = self.nb_ch # Number of Microphone Channels

        # CST-attention
        if self.ch_atten_dca:
            B = B // M  # Real Batch
            x = rearrange(x, '(b m) c t f -> b t (m f c)', b=B, m=M).contiguous() # 32(= batch_size) * 50 * 1280

        # DST-attention
        # if self.ch_atten_ule or self.freq_atten:
        #     x = rearrange(x, 'b c t f -> b t (f c)').contiguous()

        for block in self.block_list:
            x = block(x, M, C, T, F, env_vector)

        return x





class FC_layer(torch.nn.Module):
    """
    Fully Connected layer for baseline

    Args:
        out_shape (int): output shape for SLED
                         ex. 39 for single-ACCDOA, 117 for multi-ACCDOA
        temp_embed_dim (int): the input size
        params : parameters from parameter.py
    """
    def __init__(self, out_shape,temp_embed_dim, params):
        super().__init__()

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    nn.Linear(params['fnn_size'] if fc_cnt else temp_embed_dim, params['fnn_size'], bias=True))
        self.fnn_list.append(
            nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else temp_embed_dim, out_shape[-1],
                      bias=True))

    def forward(self, x:torch.Tensor):
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa


class CST_former(torch.nn.Module):
    """
    CST_former : Channel-Spectral-Temporal Transformer for SELD task
    """
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ch_attn_dca = params['ChAtten_DCA']
        self.use_env_film = params.get('use_env_film', False)
        self.encoder = Encoder_module.Encoder(in_feat_shape, params)

        self.conv_block_freq_dim = int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size']))) # 64 / 4*4*2 = 2
        self.input_nb_ch = 10
        self.temp_embed_dim = self.conv_block_freq_dim * params['nb_cnn2d_filt'] * self.input_nb_ch if self.ch_attn_dca \
            else self.conv_block_freq_dim * params['nb_cnn2d_filt']
        # self.temp_embed_dim =  self.conv_block_freq_dim * params['nb_cnn2d_filt'] # 16 * 64 = 1024

        ## Attention Layer===========================================================================================#
        # self.cmt_block = params['CMT_block']
        # if not self.cmt_block:
        #     self.attention_stage = CST_encoder(self.temp_embed_dim, params)
        # else:
        #     self.attention_stage = CMT_block(params, self.temp_embed_dim)
        self.attention_stage = CST_encoder(self.temp_embed_dim, params)

        ## Fully Connected Layer ======================================================================================#
        self.fc_layer = FC_layer(out_shape, self.temp_embed_dim, params)

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

    def forward(self, x, video=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        B, M, T, F = x.size() # B = 16, M = 10, T = 250, F = 64
        # print(' B M T F = ', B, M, T, F)

        # Environment vector is estimated once, on the *original* (un-collapsed) multichannel
        # block, so it reflects the whole recording rather than a single mic-channel slice.
        env_vector = None
        env_vector_for_encoder = None
        if self.use_env_film:
            env_vector = estimate_environment(x)  # (B, ENV_DIM)

        if self.ch_attn_dca:
            x = rearrange(x, 'b m t f -> (b m) 1 t f', b=B, m=M, t=T, f=F).contiguous() # 160 * 1 * 250 * 64
            if env_vector is not None:
                # The encoder operates on the collapsed (b*m) batch. Repeat each recording's
                # env vector M times so index (b*M + m) lines up with the rearrange above
                # (row-major: for each b, M consecutive rows -> repeat_interleave, not repeat).
                env_vector_for_encoder = env_vector.repeat_interleave(M, dim=0)
        else:
            env_vector_for_encoder = env_vector

        # print('x shape = ', x.shape)
        x = self.encoder(x, env_vector_for_encoder) # OUT : [(b m) c t f] if ch_attn_dca else [b c t f]
        # print('x shape after conv encoder = ', x.shape)
        x = self.attention_stage(x, env_vector) # attention operates at the true batch size B

        doa = self.fc_layer(x)

        return doa
    