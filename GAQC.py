import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from MyDataset import MyDataset
from MyDataset import MyDataset
# from Dataset_all import MyDataset
from torchvision import transforms
import math
from thop import profile
from einops import rearrange

# ===================== GA 模块 =====================
def build_erp_coord_map(h, w, device):
    """生成 ERP 球面坐标编码 [sinθ,cosθ,sinφ,cosφ]"""
    xs = torch.linspace(-math.pi, math.pi, w, device=device)
    ys = torch.linspace(-math.pi / 2, math.pi / 2, h, device=device)
    theta, phi = torch.meshgrid(xs, ys, indexing="xy")
    theta, phi = theta.t(), phi.t()
    coord = torch.stack(
        [torch.sin(theta), torch.cos(theta),
         torch.sin(phi), torch.cos(phi)], dim=0
    ).unsqueeze(0)
    return coord  # [1,4,H,W]

def compute_freq_energy(x, out_size=None):
    """简单频域能量（通道均值后的幅度）"""
    b, c, h, w = x.shape
    mag = torch.fft.rfft2(x, norm="ortho").abs().mean(1, keepdim=True)
    if out_size is not None:
        mag = F.interpolate(mag, size=out_size, mode="bilinear", align_corners=False)
    return mag  # [B,1,H',W']

class FreqMSLargeKernelAttention(nn.Module):
    def __init__(self, channels, reduction=4,
                 freq_kernel=7, spatial_kernel=15):
        super().__init__()
        self.c = channels
        self.reduction = max(1, reduction)

        pad_f = freq_kernel // 2
        self.freq_dw_h = nn.Conv2d(channels, channels, (freq_kernel, 1),
                                   padding=(pad_f, 0), groups=channels)
        self.freq_dw_w = nn.Conv2d(channels, channels, (1, freq_kernel),
                                   padding=(0, pad_f), groups=channels)
        mid = max(1, channels // self.reduction)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(3 * channels, mid, 1), nn.ReLU(True),
            nn.Conv2d(mid, channels, 1)
        )

        pad_s = spatial_kernel // 2
        self.spatial_dw_h = nn.Conv2d(3, 3, (spatial_kernel, 1),
                                      padding=(pad_s, 0), groups=3)
        self.spatial_dw_w = nn.Conv2d(3, 3, (1, spatial_kernel),
                                      padding=(0, pad_s), groups=3)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, groups=3),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1)
        )

    def _channel_att(self, mag):
        x = self.freq_dw_w(self.freq_dw_h(mag))
        g1 = F.adaptive_avg_pool2d(x, 1)
        g2 = F.adaptive_avg_pool2d(x, 2).mean((2, 3), keepdim=True)
        g3 = F.adaptive_avg_pool2d(x, 4).mean((2, 3), keepdim=True)
        cat = torch.cat([g1, g2, g3], 1)
        return torch.sigmoid(self.channel_proj(cat))

    def _spatial_att(self, x, fe):
        mean = x.mean(1, keepdim=True)
        l2 = torch.sqrt((x ** 2).mean(1, keepdim=True) + 1e-6)
        fe_up = F.interpolate(fe, size=x.shape[2:], mode="bilinear",
                              align_corners=False)
        s = torch.cat([mean, l2, fe_up], 1)
        s = self.spatial_dw_w(self.spatial_dw_h(s))
        return torch.sigmoid(self.spatial_refine(s))

    def forward(self, x):
        mag = torch.fft.rfft2(x, norm="ortho").abs()
        fe = mag.mean(1, keepdim=True)
        ca = self._channel_att(mag)
        sa = self._spatial_att(x, fe)
        return x * ca * sa

class DeformConv2dLite(nn.Module):
    """简单版 Deformable Conv：offset 由 conv 预测，用 grid_sample 实现"""
    def __init__(self, in_c, out_c, k=3, stride=1, pad=1):
        super().__init__()
        self.k, self.stride, self.pad = k, stride, pad
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k))
        nn.init.kaiming_uniform_(self.weight, a=1)
        self.bias = nn.Parameter(torch.zeros(out_c))

    def forward(self, x, offset):
        b, c, h, w = x.shape
        device = x.device
        k = self.k
        ph, pw = self.pad, self.pad
        # base grid
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device),
            indexing='ij'
        )
        base = torch.stack((xx, yy), 0).float()  # [2,H,W]
        base = base.unsqueeze(0).repeat(b, 1, 1, 1)  # [B,2,H,W]

        offset = offset.view(b, 2 * k * k, h, w)
        offset = offset.reshape(b, k * k, 2, h, w)
        outs = []
        for i in range(k * k):
            off = offset[:, i]  # [B,2,H,W]
            y = base[:, 1] + off[:, 1]
            x_ = base[:, 0] + off[:, 0]
            gx = (x_ / max(w - 1, 1)) * 2 - 1
            gy = (y / max(h - 1, 1)) * 2 - 1
            grid = torch.stack((gx, gy), dim=-1)
            sample = F.grid_sample(x, grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
            outs.append(sample)
        sampled = torch.stack(outs, 2).reshape(b, c * k * k, h, w)
        weight = self.weight.view(self.weight.size(0), -1, 1, 1)
        out = F.conv2d(sampled, weight, self.bias)
        return out

class ERPDistortionAwareDCN(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        aux_c = c + 5
        self.offset_conv = nn.Conv2d(aux_c, 2 * k * k, 3, padding=1)
        self.dist_gate = nn.Sequential(
            nn.Conv2d(aux_c, aux_c // 2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(aux_c // 2, 1, 1),
            nn.Sigmoid()
        )
        self.base_conv = nn.Conv2d(c, c, k, padding=k // 2)
        self.defconv = DeformConv2dLite(c, c, k, pad=k // 2)

    def forward(self, x):
        b, c, h, w = x.shape
        coord = build_erp_coord_map(h, w, x.device).expand(b, -1, -1, -1)
        freq_e = compute_freq_energy(x, (h, w))
        aux = torch.cat([x, coord, freq_e], 1)
        offset = self.offset_conv(aux)
        gate = self.dist_gate(aux)
        y_base = self.base_conv(x)
        y_def = self.defconv(x, offset)
        return (1 - gate) * y_base + gate * y_def

class ERPBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = FreqMSLargeKernelAttention(c)
        self.dcn = ERPDistortionAwareDCN(c)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.attn(x)
        y = self.dcn(y)
        y = self.bn(y)
        return self.relu(y + x)

# ===================== QCR 模块 =====================
class LocalGlobalAttention(nn.Module):
    def __init__(self, channels, reduction=4, num_kernels=3, dilations=(1, 2, 3)):
        super().__init__()
        assert num_kernels == len(dilations)
        self.c = channels
        mid = max(4, channels // reduction)
        self.num_kernels = num_kernels

        self.global_reduce = nn.Conv2d(channels, mid, 1, bias=False)
        self.row_ctx = nn.Conv2d(mid, mid, kernel_size=(7, 1),
                                 padding=(3, 0), groups=mid, bias=False)
        self.col_ctx = nn.Conv2d(mid, mid, kernel_size=(1, 7),
                                 padding=(0, 3), groups=mid, bias=False)

        self.global_fuse = nn.Sequential(
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()   # 输出 [B,C,H,W] 的全局失真 gating
        )

        self.local_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                      groups=channels, bias=False)
            for d in dilations
        ])
        self.local_weight = nn.Conv2d(channels, num_kernels, 1, bias=True)

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape

        # 全局分支
        g = self.global_reduce(x)              # [B,mid,H,W]
        row = g.mean(dim=3, keepdim=True)
        row = self.row_ctx(row)               # [B,mid,H,1]
        row = row.expand(-1, -1, -1, w)       # broadcast 到整幅图

        col = g.mean(dim=2, keepdim=True)
        col = self.col_ctx(col)               # [B,mid,1,W]
        col = col.expand(-1, -1, h, -1)

        global_ctx = row + col                # [B,mid,H,W]
        global_gate = self.global_fuse(global_ctx)  # [B,C,H,W]

        # 局部分支
        local_feats = [conv(x) for conv in self.local_convs]  # K * [B,C,H,W]
        local_stack = torch.stack(local_feats, dim=1)         # [B,K,C,H,W]

        weight = self.local_weight(x)                         # [B,K,H,W]
        weight = torch.softmax(weight, dim=1).unsqueeze(2)    # [B,K,1,H,W]

        local_out = (local_stack * weight).sum(dim=1)         # [B,C,H,W]

        # 特征融合
        y = local_out * global_gate                          
        y = self.out_proj(y)

        return self.act(x + y)

# ===================== 级联融合模块 =====================
class GAQC_CascadeFusion(nn.Module):
    
    def __init__(self, feature_dims):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_scales = len(feature_dims)
        
    
        
        # 为每个尺度创建GA模块
        self.erp_blocks = nn.ModuleList()
        for dim in feature_dims:
            self.erp_blocks.append(ERPBlock(dim))
        
        # 为每个尺度创建QCR模块
        self.lga_blocks = nn.ModuleList()
        for i, dim in enumerate(feature_dims):
            lga_input_dim = dim * 2  # 原始特征 + ERP特征
            self.lga_blocks.append(LocalGlobalAttention(lga_input_dim))
        
        # LGA输出通道数（与输入相同）
        lga_output_dims = [dim * 2 for dim in feature_dims]
        
        # 通道对齐和尺度对齐的卷积层
        # 从尺度1到尺度2：将96通道下采样并调整到192通道（与尺度2的LGA输出匹配）
        self.align_1_to_2 = nn.Sequential(
            nn.Conv2d(lga_output_dims[0], lga_output_dims[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(lga_output_dims[1]),
            nn.ReLU(inplace=True)
        )
        
        # 从尺度2到尺度3：将192通道下采样并调整到384通道（与尺度3的LGA输出匹配）
        self.align_2_to_3 = nn.Sequential(
            nn.Conv2d(lga_output_dims[1], lga_output_dims[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(lga_output_dims[2]),
            nn.ReLU(inplace=True)
        )
        
        # 最终融合层 - 将384通道转换为192通道
        self.final_fusion = nn.Sequential(
            nn.Conv2d(lga_output_dims[2], feature_dims[2], 1),
            nn.BatchNorm2d(feature_dims[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # print(f"LGA输出维度: {lga_output_dims}")
        
    def forward(self, features):
        """
        features: 多尺度特征列表 [feat1, feat2, feat3]，分辨率从高到低
        每个feat的形状: [B, C, H, W]
        """
        # 确保features顺序是从高分辨率到低分辨率
        feat1, feat2, feat3 = features  # feat1分辨率最高，feat3分辨率最低
        
        # 打印输入特征信息
        # print(f"输入特征形状: feat1={feat1.shape}, feat2={feat2.shape}, feat3={feat3.shape}")
        
        # 1. 处理尺度1（最高分辨率）
        erp_feat1 = self.erp_blocks[0](feat1)  # [B, 48, 128, 128]
        combined_feat1 = torch.cat([feat1, erp_feat1], dim=1)  # [B, 96, 128, 128]
        lga_feat1 = self.lga_blocks[0](combined_feat1)  # [B, 96, 128, 128]
        
        # 2. 将尺度1的特征下采样并与尺度2融合
        # 对齐到尺度2的通道和分辨率
        aligned_feat1 = self.align_1_to_2(lga_feat1)  # [B, 192, 64, 64]
        
        # 处理尺度2
        erp_feat2 = self.erp_blocks[1](feat2)  # [B, 96, 64, 64]
        combined_feat2 = torch.cat([feat2, erp_feat2], dim=1)  # [B, 192, 64, 64]
        lga_feat2 = self.lga_blocks[1](combined_feat2)  # [B, 192, 64, 64]
        
        # 融合尺度1和尺度2的特征
        # print(f"融合前: lga_feat2={lga_feat2.shape}, aligned_feat1={aligned_feat1.shape}")
        fused_feat2 = lga_feat2 + aligned_feat1  # [B, 192, 64, 64]
        
        # 3. 将融合后的特征下采样并与尺度3融合
        aligned_feat2 = self.align_2_to_3(fused_feat2)  # [B, 384, 32, 32]
        
        # 处理尺度3
        erp_feat3 = self.erp_blocks[2](feat3)  # [B, 192, 32, 32]
        combined_feat3 = torch.cat([feat3, erp_feat3], dim=1)  # [B, 384, 32, 32]
        lga_feat3 = self.lga_blocks[2](combined_feat3)  # [B, 384, 32, 32]
        
        # 融合尺度2和尺度3的特征
        # print(f"融合前: lga_feat3={lga_feat3.shape}, aligned_feat2={aligned_feat2.shape}")
        fused_feat3 = lga_feat3 + aligned_feat2  # [B, 384, 32, 32]
        
        # 4. 最终融合和池化
        final_feat = self.final_fusion(fused_feat3)  # [B, 192, 1, 1]
        
        return final_feat.flatten(1)  # [B, 192]

# ===================== 主模型 =====================
class GAQC_Cascade_QualityAssessment(nn.Module):
    def __init__(self, backbone_name='repvit_m1', pretrained=True, dropout_rate=0.2):
        super().__init__()
        
        # 创建backbone，使用正确的out_indices
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained,
            num_classes=0,
            features_only=True,
            out_indices=[0, 1, 2]  # RepVIT有3个特征层
        )
        
        # 获取多尺度特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.backbone(dummy)
            self.feature_dims = [feat.shape[1] for feat in features]
            self.feature_shapes = [feat.shape[2:] for feat in features]
            # print(f"多尺度特征维度: {self.feature_dims}")
            # print(f"多尺度特征形状: {self.feature_shapes}")
        
        # ERP+LGA级联特征融合
        self.feature_fusion = GAQC_CascadeFusion(
            feature_dims=self.feature_dims
        )
        
        # 获取最终特征维度
        final_feat_dim = self.feature_dims[2]  # 使用尺度3的维度
        
        # 简化回归头
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(final_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        
        # print(f"最终特征维度: {final_feat_dim}")
    
    def forward(self, x):
        # 1. 主干网络提取多尺度特征
        features = self.backbone(x)
        
        # 2. ERP+LGA级联特征融合
        fused_features = self.feature_fusion(features)  # [B, C3]
        
        # 3. 回归预测
        scores = self.regressor(fused_features)  # [B, 1]
        return scores.squeeze(-1)  # [B]

if __name__ == "__main__":
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    
    # 创建级联融合模型
    net = GAQC_Cascade_QualityAssessment().to(device=device)

    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(net, inputs=(dummy_input, ))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")   
    print(f"参数量: {params / 1e6:.2f} M")
    
    
    # 打印模型结构信息
    print(f"回归头第一层输入维度: {net.regressor[1].in_features}")
    print(f"回归头第一层输出维度: {net.regressor[1].out_features}")
    
    # 测试输入
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    print(f"测试输入形状: {dummy_input.shape}")
    
    # 测试前向传播
    net.eval()
    with torch.no_grad():
        try:
            output = net(dummy_input)
            print(f"模型输出形状: {output.shape}")
            print("前向传播测试成功！")
            
            # 测试数据集
            test_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            # 创建数据集和加载器
            test_dataset = MyDataset('/mnt/10T/liziyi/LargeKernel/JUFE-10K/resized_dis_1024', 
                                   '/mnt/10T/liziyi/LargeKernel/JUFE-10K/JUFE-10k_mos.csv', 
                                   mode='test', transform=test_transform)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=2,
                num_workers=0,
                shuffle=False,
            )
            
            # 测试模型
            net.eval()
            with torch.no_grad():
                for imgs, mos in test_loader:
                    imgs = imgs.to(device=device)
                    out = net(imgs)
                    print(f"模型输出形状: {out.shape}")
                    print(f"模型输出: {out}")
                    print(f"真实MOS: {mos}")
                    print(f"真实MOS形状: {mos.shape}")
                    break
                    
        except Exception as e:
            print(f"前向传播错误: {e}")
            import traceback
            traceback.print_exc()