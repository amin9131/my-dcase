import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import Encoder_module 


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


        # Channel attention w. Divided Channel Attention (DCA) ---------------------------------------------#
        if self.ChAtten_dca:
            self.ch_attn_embed_dim = params['nb_cnn2d_filt']  # 64
            self.ch_mhsa = nn.MultiheadAttention(embed_dim=self.ch_attn_embed_dim, num_heads=params['nb_heads'],
                                      dropout=self.dropout_rate, batch_first=True)
            self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.ch_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

            # self.ch_gru = torch.nn.GRU(input_size= self.ch_attn_embed_dim, hidden_size=self.ch_attn_embed_dim,
            #                     num_layers=params['nb_rnn_layers'], batch_first=True,
            #                     dropout=params['dropout_rate'], bidirectional=True)

        # Spectral attention -------------------------------------------------------------------------------#
        # if self.FreqAtten:
        #     self.sp_attn_embed_dim = params['nb_cnn2d_filt']  # 64
        #     self.embed_dim_4_freq_attn = params['nb_cnn2d_filt'] # Update the temp embedding if freq attention is applied
        #     self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim, num_heads= 8,
        #                               dropout=self.dropout_rate, batch_first=True)
        #     self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        #     if self.linear_layer:
        #         self.sp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        #     self.sp_gru = torch.nn.GRU(input_size= self.sp_attn_embed_dim, hidden_size=self.sp_attn_embed_dim,
        #                         num_layers=params['nb_rnn_layers'], batch_first=True,
        #                         dropout=params['dropout_rate'], bidirectional=True)

        # temporal attention -----------------------------------------------------------------------------------#
        self.embed_dim_4_freq_attn = params['nb_cnn2d_filt'] # Update the temp embedding if freq attention is applied
        # self.temp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim_4_freq_attn if params['FreqAtten'] else self.embed_dim,
        #                           num_heads= 8,
        #                           dropout=self.dropout_rate, batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if self.linear_layer:
            self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

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

    def forward(self,x, M, C, T, F):
        # x shape = 32(batch_size) * 50 * 1280
        

        #CST-attention (DCA)
        # channel attention
        # print('line_67  x shape in CST_attention class = ', x.shape)
        x_init = x.clone() # x shape = 32*50*1280
        xc = rearrange(x_init, 'b t (m f c)-> (b t f) m c', c=C, f=F).contiguous() # xc = 3200*10*64
        
        # (xc, _) = self.ch_gru(xc)
        # xc = torch.tanh(xc)
        # xc = xc[:, :, xc.shape[-1]//2:] * xc[:, :, :xc.shape[-1]//2]

        xc, _ = self.ch_mhsa(xc, xc, xc)
        xc = rearrange(xc, ' (b t f) m c -> b t (f m c)', t=T, f=F).contiguous() # 32*50*1280

        
        if self.linear_layer:
            xc = self.activation(self.ch_linear(xc))
        xc = xc + x_init
        if self.dropout_rate:
            xc = self.drop_out(xc)
        xc = self.ch_layer_norm(xc)

        # spectral attention
        # xs = rearrange(xc, ' b t (f m c) -> (b t m) f c', c=C, t=T, f=F).contiguous() # 16000*2*64


        
        # (xs, _) = self.sp_gru(xs)
        # xs = torch.tanh(xs)
        # xs = xs[:, :, xs.shape[-1]//2:] * xs[:, :, :xs.shape[-1]//2]

        # xs, _ = self.sp_mhsa(xs, xs, xs)

        # xs = rearrange(xs, ' (b t m) f c -> b t (f m c)', m=M, t=T).contiguous() # 32*50*1280

        # if self.linear_layer:
        #     xs = self.activation(self.sp_linear(xs))
        # xs = xs + xc
        # if self.dropout_rate:
        #     xs = self.drop_out(xs)
        # xs = self.sp_layer_norm(xs)

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

    def forward(self, x):
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
            x = block(x, M, C, T, F)

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

# def make_pairs(x):
#     """make the int -> tuple
#     """
#     return x if isinstance(x, tuple) else (x, x)

# class ConvDW3x3(nn.Module):
#     def __init__(self, dim, kernel_size=3):
#         super(ConvDW3x3, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels=dim,
#             out_channels=dim,
#             kernel_size=make_pairs(kernel_size),
#             padding=make_pairs(1),
#             groups=dim)

#         self.initialize_weights()

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')

#     def forward(self, x):
#         x = self.conv(x)
#         return x

# class LocalPerceptionUint(torch.nn.Module):
#     def __init__(self, dim, act=False):
#         super(LocalPerceptionUint, self).__init__()
#         self.act = act
#         self.conv_3x3_dw = ConvDW3x3(dim)
#         if self.act:
#             self.actation = nn.Sequential(
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#             )

#         self.initialize_weights()

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         if self.act:
#             out = self.actation(self.conv_3x3_dw(x))
#             return out
#         else:
#             out = self.conv_3x3_dw(x)
#             return out

# class ConvGeluBN(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride_size, padding=1):
#         """build the conv3x3 + gelu + bn module
#         """
#         super(ConvGeluBN, self).__init__()
#         self.kernel_size = make_pairs(kernel_size)
#         self.stride_size = make_pairs(stride_size)
#         self.padding_size = make_pairs(padding)
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.conv3x3_gelu_bn = nn.Sequential(
#             nn.Conv2d(in_channels=self.in_channel,
#                       out_channels=self.out_channel,
#                       kernel_size=self.kernel_size,
#                       stride=self.stride_size,
#                       padding=self.padding_size),
#             nn.GELU(),
#             nn.BatchNorm2d(self.out_channel)
#         )

#         self.initialize_weights()

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         x = self.conv3x3_gelu_bn(x)
#         return x

# class InvertedResidualFeedForward(torch.nn.Module):
#     def __init__(self, dim, dim_ratio=4.):
#         super(InvertedResidualFeedForward, self).__init__()
#         output_dim = int(dim_ratio * dim)
#         self.conv1x1_gelu_bn = ConvGeluBN(
#             in_channel=dim,
#             out_channel=output_dim,
#             kernel_size=1,
#             stride_size=1,
#             padding=0
#         )
#         self.conv3x3_dw = ConvDW3x3(dim=output_dim)
#         self.act = nn.Sequential(
#             nn.GELU(),
#             nn.BatchNorm2d(output_dim)
#         )
#         self.conv1x1_pw = nn.Sequential(
#             nn.Conv2d(output_dim, dim, 1, 1, 0),
#             nn.BatchNorm2d(dim)
#         )

#         self.initialize_weights()

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         x = self.conv1x1_gelu_bn(x)
#         out = x + self.act(self.conv3x3_dw(x))
#         out = self.conv1x1_pw(out)
#         return out

# class Spec_attention(torch.nn.Module):
#     def __init__(self, temp_embed_dim, params):
#         super().__init__()
#         # self.params = params
#         self.dropout_rate = params['dropout_rate']
#         self.linear_layer = params['linear_layer']
#         self.temp_embed_dim = temp_embed_dim

#         # Spectral attention -------------------------------------------------------------------------------#
#         self.sp_attn_embed_dim = params['nb_cnn2d_filt']  # 64
#         self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim, num_heads=params['nb_heads'],
#                                   dropout=params['dropout_rate'], batch_first=True)
#         self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
#         if self.params['LinearLayer']:
#             self.sp_linear = nn.Linear(self.sp_attn_embed_dim, self.sp_attn_embed_dim)

#         self.activation = nn.GELU()
#         self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()

#         self.initialize_weights()

#     def initialize_weights(self):
#         # initialization
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Conv2d):
#             nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self,x, C, T, F):
#         # spectral attention
#         x_init = x
#         x_attn_in = rearrange(x_init, ' b t (f c) -> (b t) f c', c=C,f=F).contiguous()
#         xs, _ = self.sp_mhsa(x_attn_in, x_attn_in, x_attn_in)
#         xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
#         if self.linear_layer:
#             xs = self.activation(self.sp_linear(xs))
#         xs = xs + x_init
#         if self.dropout_rate:
#             xs = self.drop_out(xs)
#         x_out = self.sp_layer_norm(xs)
#         return x_out

# class Temp_attention(torch.nn.Module):
#     def __init__(self, temp_embed_dim, params):
#         super().__init__()
#         # self.params = params
#         self.dropout_rate = params['dropout_rate']
#         self.linear_layer = params['linear_layer']
#         self.temp_embed_dim = temp_embed_dim
#         self.embed_dim_4_freq_attn = params['nb_cnn2d_filt']  # Update the temp embedding if freq attention is applied
#         # temporal attention -----------------------------------------------------------------------------------#
#         self.temp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim_4_freq_attn if params['FreqAtten'] else self.temp_embed_dim,
#                                   num_heads=params['nb_heads'],
#                                   dropout=params['dropout_rate'], batch_first=True)
#         self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
#         if self.params['LinearLayer']:
#             self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

#         self.activation = nn.GELU()
#         self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()

#         self.initialize_weights()

#     def initialize_weights(self):
#         # initialization
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Conv2d):
#             nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self,x, C, T, F):
#         # temporal attention
#         x_init = x
#         xt = rearrange(x_init, ' b t (f c) -> (b f) t c', c=C).contiguous()
#         xt, _ = self.temp_mhsa(xt, xt, xt)
#         xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
#         if self.linear_layer:
#             xt = self.activation(self.temp_linear(xt))
#         xt = xt + x_init
#         if self.dropout_rate:
#             xt = self.drop_out(xt)
#         x_out = self.temp_layer_norm(xt)
#         return x_out

# class CMT_Layers(torch.nn.Module):
    # def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.):
    #     super().__init__()
    #     self.cmt_split = params['CMT_split']
    #     self.ch_attn_dca = params['ChAtten_DCA']
    #     self.ch_attn_ule = params['ChAtten_ULE']
    #     self.temp_embed_dim = temp_embed_dim
    #     self.ffn_ratio = ffn_ratio
    #     self.dim = params['nb_cnn2d_filt']

    #     self.norm1 = nn.LayerNorm(self.dim)
    #     self.LPU = LocalPerceptionUint(self.dim)
    #     self.IRFFN = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

    #     if not self.cmt_split:
    #         # print('temp_embed_dim = ', temp_embed_dim)
    #         self.cst_attention = CST_attention(temp_embed_dim=self.temp_embed_dim,params=params)
    #     # elif self.cmt_split:
    #     #     self.spectral_atten = Spec_attention(temp_embed_dim=self.temp_embed_dim, params=params)
    #     #     self.temporal_atten = Temp_attention(temp_embed_dim=self.temp_embed_dim, params=params)
    #     #     self.norm2 = nn.LayerNorm(self.dim)
    #     #     self.LPU2 = LocalPerceptionUint(self.dim)
    #     #     self.IRFFN2 = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

    #     self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    #     self.initialize_weights()

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    # def forward(self, x):
    #     if not self.cmt_split:
    #         # if not self.ch_attn_dca: # ch_attn (ULE) or dst_attn
    #         #     lpu = self.LPU(x)
    #         #     x = x + lpu

    #         #     B, C, T, F = x.size()
    #         #     x = rearrange(x, 'b c t f -> b t (f c)')
    #         #     if not self.ch_attn_dca: # channel attention with unfolding
    #         #         x = self.cst_attention(x,10,C, T, F)
    #         #     else:   # dst attention
    #         #         x = self.cst_attention(x,C,T,F)

    #         #     x_2 = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
    #         #     x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
    #         #     norm1 = self.norm1(x_2)
    #         #     norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
    #         #     ffn = self.IRFFN(norm1)
    #         #     x = x_res + self.drop_path(ffn)

    #         if self.ch_attn_dca: # ch_attn (DCA)
    #             B, C, T, F = x.size()
    #             M = 10

    #             lpu = self.LPU(x)
    #             x = x + lpu

    #             x = rearrange(x, '(b m) c t f -> b t (m f c)', m=M).contiguous()
    #             x = self.cst_attention(x, M, C, T, F)

    #             x_2 = rearrange(x, 'b t (m f c) -> b (t m f) c', m=M, f=F).contiguous()
    #             x_res = rearrange(x, 'b t (m f c) -> (b m) c t f', m=M, f=F).contiguous()
    #             norm1 = self.norm1(x_2)
    #             norm1 = rearrange(norm1, 'b (t m f) c -> (b m) c t f', f=F, c=C, t=T).contiguous()
    #             ffn = self.IRFFN(norm1)
    #             x = x_res + self.drop_path(ffn)

    #     # else: # CMT Split
    #     #     if not self.ch_attn_dca and not self.ch_attn_ule:
    #     #         # Spectral Conformer
    #     #         lpu = self.LPU(x)
    #     #         x = x + lpu

    #     #         B, C, T, F = x.size()
    #     #         x = rearrange(x, 'b c t f -> b t (f c)').contiguous()
    #     #         x = self.spectral_atten(x, C, T, F)

    #     #         x_s = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
    #     #         x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
    #     #         norm1 = self.norm1(x_s)
    #     #         norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
    #     #         ffn_s = self.IRFFN(norm1)
    #     #         xs = x_res + self.drop_path(ffn_s)

    #     #         # Temporal Conformer
    #     #         lpu2 = self.LPU2(xs)
    #     #         xs = xs + lpu2

    #     #         B, C, T, F = xs.size()
    #     #         x2 = rearrange(xs, 'b c t f -> b t (f c)').contiguous()
    #     #         x2 = self.temporal_atten(x2, C, T, F)

    #     #         x_t = rearrange(x2, 'b t (f c) -> b (t f) c', f=F).contiguous()
    #     #         x_res_t = rearrange(x2, 'b t (f c) -> b c t f', f=F).contiguous()
    #     #         norm2 = self.norm2(x_t)
    #     #         norm2 = rearrange(norm2, 'b (t f) c -> b c t f', f=F).contiguous()
    #     #         ffn_t = self.IRFFN2(norm2)
    #     #         x = x_res_t + self.drop_path(ffn_t)
    #     #     else:
    #     #         print("CST attention with split cmt block is not implemented yet.")
    #     #         raise()
    #     return x

# class CMT_block(torch.nn.Module):
#     def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.1):
#         super().__init__()
#         self.temp_embed_dim = temp_embed_dim
#         self.num_layers = params['nb_self_attn_layers']
#         self.ch_atten_dca = params['ChAtten_DCA']
#         self.ffn_ratio = ffn_ratio
#         self.nb_ch = 10

#         self.block_list = nn.ModuleList([CMT_Layers(
#             params=params,
#             temp_embed_dim=self.temp_embed_dim,
#             ffn_ratio=self.ffn_ratio,
#             drop_path_rate=drop_path_rate
#         ) for i in range(self.num_layers)]
#         )

#     def forward(self, x):
#         B, C, T, F = x.size()
#         M = self.nb_ch

#         for block in self.block_list:
#             x = block(x)

#         if self.ch_atten_dca: # CST (DCA)
#             B = B // M
#             x = rearrange(x, '(b m) c t f -> b t (m f c)', b=B,m=M).contiguous()
#         else: # CST (ULE) & DST
#             x = rearrange(x, 'b c t f -> b t (f c)', c=C, t=T, f=F).contiguous()

#         return x


class CST_former(torch.nn.Module):
    """
    CST_former : Channel-Spectral-Temporal Transformer for SELD task
    """
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ch_attn_dca = params['ChAtten_DCA']
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

        if self.ch_attn_dca:
            x = rearrange(x, 'b m t f -> (b m) 1 t f', b=B, m=M, t=T, f=F).contiguous() # 160 * 1 * 250 * 64

        
        # print('x shape = ', x.shape)
        x = self.encoder(x) # OUT : [(b m) c t f] if ch_attn_dca else [b c t f]
        # print('x shape after conv encoder = ', x.shape)
        x = self.attention_stage(x)

        doa = self.fc_layer(x)

        return doa
    



params = dict(
    nb_cnn2d_filt=64,
    nb_cnn2d_filts_in = [10, 32, 64],
    nb_cnn2d_filts_out = [32, 64, 128],
    # nb_cnn2d_filts_in = [10, 128, 128],
    # nb_cnn2d_filts_out = [128, 128, 128],
    unique_classes = 13,
    f_pool_size = [4, 4, 2],
    t_pool_size = [5, 1, 1],
    dropout_rate = 0.05,
    rnn_size = 128,
    nb_heads = 8,
    fnn_size = 128,
    nb_fnn_layers = 1,
    fs=24000,
    hop_len_s=0.02,
    label_hop_len_s=0.1,
    max_audio_len_s=60,
    nb_mel_bins=64,

    use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
    fmin_doa_salsalite = 50,
    fmax_doa_salsalite = 2000,
    fmax_spectra_salsalite = 9000,

    # MODEL TYPE
    multi_accdoa=True,  # False - Single-ACCDOA or True - Multi-ACCDOA
    thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

    # added params:
    encoder = 'conv',           # ['conv', 'ResNet', 'SENet']
    LinearLayer = False,        # Linear Layer right after attention layers (usually not used/employed in baseline model)
    FreqAtten = True,          # Use of Divided Spectro-Temporal Attention (DST Attention)
    ChAtten_DCA = True,        # Use of Divided Channel-S-T Attention (CST Attention)
    ChAtten_ULE = False,        # Use of Divided C-S-T attention with Unfold (Unfolded CST attention)
    CMT_block = False,          # Use of LPU & IRFNN
    CMT_split = False,          # Apply LPU & IRFNN on S, T attention layers independently


    # DNN MODEL PARAMETERS
    label_sequence_length=50,    # Feature sequence length
    batch_size=16,              # Batch size
    # added param:
    t_pooling_loc = 'front',
    # f_unpool_size = [2, 4, 4],
    # t_unpool_size = [1, 1, 5],

    self_attn=True,
    nb_self_attn_layers=2,
    TE_dim_ff = 512,
    TE_num_layers = 2,
    nb_rnn_layers = 2)

model = CST_former(in_feat_shape = [16, 10, 250, 64], out_shape=[16, 50, 126],params= params )
# model = SeldModel()

# # محاسبه تعداد پارامترها
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable Parameters: {num_params}')