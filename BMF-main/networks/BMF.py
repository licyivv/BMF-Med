import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import math  
from functools import partial  


from networks.segformer import *



class EnhancedGlobalExtraction(nn.Module):
    def __init__(self, edge_enhance=True):
        super().__init__()
        self.edge_enhance = edge_enhance
        
        self.proj = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        if self.edge_enhance:
            self.edge_conv = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
    
    def forward(self, x, H, W):
        B, N, D = x.shape
        x_2d = x.reshape(B, D, H, W)
        
        avg_pool = x_2d.mean(dim=1, keepdim=True)
        max_pool = x_2d.max(dim=1, keepdim=True)[0]
        
        if self.edge_enhance:
            sobel_x = F.conv2d(avg_pool, 
                              torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                          dtype=torch.float32, device=avg_pool.device),
                              padding=1)
            sobel_y = F.conv2d(avg_pool, 
                              torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                          dtype=torch.float32, device=avg_pool.device),
                              padding=1)
            edge_map = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-6)
            edge_enhanced = self.edge_conv(edge_map)
            avg_pool = avg_pool * (1 + edge_enhanced)

        
        cat = torch.cat([avg_pool, max_pool], dim=1)
        proj = self.proj(cat)
        
        return proj.reshape(B, N, 1)

class EnhancedContextExtraction(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        
        self.dconv3x3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
        self.dconv5x5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
        self.dconv7x7 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(3*dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // self.reduction, kernel_size=1),
            nn.BatchNorm2d(dim // self.reduction)
        )
    
    def forward(self, x, H, W):
        B, N, D = x.shape
        x_2d = x.reshape(B, D, H, W)
        
        feat3x3 = self.dconv3x3(x_2d)
        feat5x5 = self.dconv5x5(x_2d)
        feat7x7 = self.dconv7x7(x_2d)
        
        fused = torch.cat([feat3x3, feat5x5, feat7x7], dim=1)
        fused = self.fusion(fused)
        
        proj_feat = self.proj(fused)
        
        return proj_feat.reshape(B, N, -1)

class EnhancedMultiscaleFusion(nn.Module):
    def __init__(self, dim, reduction=2, use_boundary_attention=True):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.use_boundary_attention = use_boundary_attention
        
        self.local = EnhancedContextExtraction(dim, reduction)
        self.global_ = EnhancedGlobalExtraction(edge_enhance=True)
        
        if self.use_boundary_attention:
            self.boundary_attention = nn.Sequential(
                nn.Conv2d(dim//reduction, dim//reduction, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim//reduction),
                nn.ReLU(),
                nn.Conv2d(dim//reduction, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        self.global_proj = nn.Linear(1, dim//reduction)
        self.bn = nn.BatchNorm1d(dim//reduction)
        
    def forward(self, x, g, H, W):
        B, N, D = x.shape
        
        local_feat = self.local(x, H, W)
        global_feat = self.global_(g, H, W)
        global_feat = self.global_proj(global_feat)
        
        if self.use_boundary_attention:
            local_2d = local_feat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            boundary_att = self.boundary_attention(local_2d)
            local_2d = local_2d * boundary_att
            local_feat = local_2d.permute(0, 2, 3, 1).reshape(B, N, -1)
        
        fused = local_feat + global_feat
        
        return self.bn(fused.permute(0, 2, 1)).permute(0, 2, 1)

class MASAG(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        self.reduction = reduction
        self.multi = EnhancedMultiscaleFusion(dim, reduction)
        
        self.selection = nn.Conv1d(
            in_channels=dim//reduction,
            out_channels=2,
            kernel_size=1
        )
        
        self.proj = nn.Linear(dim, 2*dim)
        self.bn_final = nn.BatchNorm1d(2*dim)
    
    def forward(self, x, g, H, W):
        B, N, D = x.shape
        x_, g_ = x, g
        
        multi = self.multi(x, g, H, W)
        
        selection = self.selection(multi.permute(0, 2, 1))
        selection = selection.permute(0, 2, 1)
        A, B = F.softmax(selection, dim=-1).unbind(dim=-1)
        
        x_att = (A.unsqueeze(-1) * x_) + x_
        g_att = (B.unsqueeze(-1) * g_) + g_
        
        x_sig = torch.sigmoid(x_att)
        g_att = x_sig * g_att
        g_sig = torch.sigmoid(g_att)
        x_att = g_sig * x_att
        
        interaction = x_att * g_att
        output = self.proj(interaction)
        return self.bn_final(output.permute(0, 2, 1)).permute(0, 2, 1)



class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, 2D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,h,w) -> torch.Tensor:
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)
        
        print(h)
        print(w)
        attn = self.attn(norm_1, norm_2)
        # attn = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(attn)

        # residual1 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x1)
        # residual2 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x2)
        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class BiFormerCNN(nn.Module):
    def __init__(self, dim, n_win=7, side_dwconv=5, before_attn_dwconv=3):
        super().__init__()
        # 位置嵌入 CNN
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, 
                                       padding=before_attn_dwconv//2, groups=dim)
        else:
            self.pos_embed = nn.Identity()
        
        # 侧边深度可分离卷积
        self.side_dwconv = nn.Conv2d(dim, dim, kernel_size=side_dwconv, 
                                     padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else nn.Identity()

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        # 应用位置嵌入卷积
        x = x + self.pos_embed(x)
        
        # 应用侧边深度可分离卷积
        x = x + self.side_dwconv(x)
        
        return x

class DualTransformerBlock(nn.Module):
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix", 
                 use_biformer_cnn=True, n_win=7, side_dwconv=5, before_attn_dwconv=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        
        # 原有的空间和通道注意力
        self.spatial_attn = EfficientAttention(
            in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=head_count
        )
        self.channel_attn = ChannelAttention(in_dim)
        
        # 可选的 BiFormer CNN 模块
        self.use_biformer_cnn = use_biformer_cnn
        if use_biformer_cnn:
            self.biformer_cnn = BiFormerCNN(
                dim=in_dim, 
                n_win=n_win, 
                side_dwconv=side_dwconv, 
                before_attn_dwconv=before_attn_dwconv
            )
        
        self.spatial_weight = nn.Parameter(torch.ones(1))
        self.channel_weight = nn.Parameter(torch.ones(1))
        
        # MLP 保持不变
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))
        
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # 归一化
        x_norm = self.norm1(x)

        # 准备空间注意力输入
        spatial_in = rearrange(x_norm, "b (h w) d -> b d h w", h=H, w=W)
        
        # 可选的 BiFormer CNN 处理
        if self.use_biformer_cnn:
            spatial_in = self.biformer_cnn(spatial_in)

        # 空间注意力
        spatial_out = self.spatial_attn(spatial_in)
        spatial_out = rearrange(spatial_out, "b d h w -> b (h w) d")

        # 通道注意力
        channel_out = self.channel_attn(x_norm)

        # 加权融合
        fusion = x + self.spatial_weight * spatial_out + self.channel_weight * channel_out

        # MLP+残差
        norm2 = self.norm2(fusion)
        mlp1 = self.mlp1(norm2, H, W)
        fusion2 = fusion + mlp1
        norm3 = self.norm3(fusion2)
        mlp2 = self.mlp2(norm3, H, W)
        out = fusion2 + mlp2

        return out
########################
# Encoder
class MiT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        aa_filt=3
        edge_scale=1.5

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 =OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 =OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )
        #  self.patch_embed3 = OverlapPatchEmbeddings(
        #     image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2],
        # )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []
        
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


# Decoder
  
def _init_weights(module, scheme=''):  
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):  
        if scheme == 'normal':  
            nn.init.normal_(module.weight, std=.02)  
            if module.bias is not None:  
                nn.init.zeros_(module.bias)  
        elif scheme == 'trunc_normal':  
            # Placeholder for actual implementation  
            pass  
        elif scheme == 'xavier_normal':  
            nn.init.xavier_normal_(module.weight)  
            if module.bias is not None:  
                nn.init.zeros_(module.bias)  
        elif scheme == 'kaiming_normal':  
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  
            if module.bias is not None:  
                nn.init.zeros_(module.bias)  
        else:  
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels  
            fan_out //= module.groups  
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))  
            if module.bias is not None:  
                nn.init.zeros_(module.bias)  
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):  
        nn.init.constant_(module.weight, 1)  
        nn.init.constant_(module.bias, 0)  
    elif isinstance(module, nn.LayerNorm):  
        nn.init.constant_(module.weight, 1)  
        nn.init.constant_(module.bias, 0)  


class SCEU(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):  
        super(SCEU, self).__init__()  
        self.in_channels = in_channels  
        self.out_channels = out_channels  
        self.up_dwc = nn.Sequential(  
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,  
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),  
            nn.BatchNorm2d(self.in_channels),  
            nn.ReLU(inplace=True)  
        )  
        # self.SCM = Shift_channel_mix(shift_size=1)
        
        # Ensure the output channels is consistent with what we want  
        self.pwc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)  
        self.init_weights('normal')  

    def init_weights(self, scheme=''):  
        for m in self.modules():  
            _init_weights(m, scheme)  

    def forward(self, x):  
        x = self.up_dwc(x)  
        # x = self.SCM(x)  
        x = self.pwc(x)  # This should output `out_channels`  
        return x  

class PatchExpand(nn.Module):  
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):  
        super().__init__()  
        self.input_resolution = input_resolution  
        self.dim = dim  
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()  
        self.norm = norm_layer(dim // dim_scale)  

        # Adjust SCEU’s input channels based on dim * 2, output should be dim // 2  
        self.sceu = SCEU(in_channels=2 * dim, out_channels=dim // 2)  

    def forward(self, x):  
        H, W = self.input_resolution  
        x = self.expand(x)  

        B, L, C = x.shape  
        assert L == H * W, "input feature has wrong size"  

        # Reshape into [B, H, W, C]  
        x = x.view(B, H, W, C)  
        
        # Rearrange the tensor to shape expected by SCEU  
        x = rearrange(x, "b h w c -> b c h w")  

        # Pass through SCEU  
        x = self.sceu(x)  # Now x has shape [B, out_channels, H*2, W*2]  
        
        # Reshape and rearrange for output  
        B, out_channels, H_out, W_out = x.shape  # SCEU's output  
        assert H_out == 2 * H and W_out == 2 * W, "SCEU output size mismatch"  

        # Prepare output with shape [B, 4*H*W, C/2]  
        x = rearrange(x, "b c h w -> b (h w) c")  
        
        # Final shape is [B, 4*H*W, C/2]  
        x = x.reshape(B, -1, out_channels)  # Reduce channels by half  
        
        return x  


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # self.cross_attn = CrossAttentionBlock(
            #     dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            self.cross_attn = MASAG(
                dims , reduction=2
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # self.cross_attn = CrossAttentionBlock(
            #     dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            self.cross_attn =  MASAG(
                dims*2 , reduction=2
            )
            
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2,h,w))
            
            cat_linear_x =torch.add(x1_expand,cat_linear_x)
            
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class BMF(nn.Module):
    def __init__(self, num_classes=4, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        # dims, key_dim, value_dim, layers = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2]]
        #light
        dims, key_dim, value_dim, layers = [[64, 160, 256], [64, 160, 256], [64, 160, 256], [2, 2, 2]]
        self.backbone = MiT(
            image_size=224,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        # in_out_chan = [
        #     [64, 128, 128, 128, 160],
        #     [320, 320, 320, 320, 256],
        #     [512, 512, 512, 512, 512],
        # ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        #light
        in_out_chan = [
            [32, 64, 64, 64, 80],
            [160, 160, 160, 160, 128],
            [256, 256, 256, 256, 256],
        ]
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.backbone(x)

        b, c, _, _ = output_enc[2].shape

        # ---------------Decoder-------------------------
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))

        return tmp_0
