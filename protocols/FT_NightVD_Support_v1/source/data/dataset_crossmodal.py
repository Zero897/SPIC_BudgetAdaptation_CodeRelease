import os
import hashlib
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
}


# ================= 补充：FBM噪声生成  =================
def fbm_noise(width, height, octaves=4, base_scale=2):
    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 0.5
    frequency = base_scale
    for _ in range(octaves):
        # 假设 perlin_noise 已在文件中定义
        noise += amplitude * perlin_noise(width, height, scale=frequency)
        amplitude *= 0.5
        frequency *= 2
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise

# ================= 1. 柏林噪声生成 (保持不变) =================
def perlin_noise(width, height, scale=5):
    grid_w = int(scale)
    grid_h = int(scale)
    linx = np.linspace(0, grid_w, width, endpoint=False)
    liny = np.linspace(0, grid_h, height, endpoint=False)
    x, y = np.meshgrid(linx, liny)

    def f(t): return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    p = np.random.rand(grid_h + 1, grid_w + 1)
    xf = x % 1
    yf = y % 1
    xi = x.astype(int) % grid_w
    yi = y.astype(int) % grid_h

    n0 = p[yi, xi] * (1 - xf) + p[yi, xi + 1] * xf
    n1 = p[yi + 1, xi] * (1 - xf) + p[yi + 1, xi + 1] * xf
    noise = n0 * (1 - f(yf)) + n1 * f(yf)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise

# ================= 修改后的增强版烟雾合成 (不使用深度图) =================
def add_synthetic_smoke(img, is_night=False, return_params=False, severity=None, support_expanded=False):
    """
    基于流体动力学模拟的增强版烟雾合成（无深度图版）
    """
    # 确保输入是 float32 [0, 1]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
        
    h, w = img.shape[:2]

    # 1. 基础流体纹理 (使用 FBM 制造多尺度细节)
    base_texture = fbm_noise(w, h, octaves=5, base_scale=random.uniform(2.0, 4.5))
    
    # 2. 🔥 核心创新：流体扭曲 (制造风吹的卷曲感)
    # 使用低频噪声生成 X 和 Y 方向的位移场
    dx = fbm_noise(w, h, octaves=2, base_scale=10) * 40 - 20
    dy = fbm_noise(w, h, octaves=2, base_scale=10) * 40 - 20
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    map_x = np.clip(x_grid + dx, 0, w-1).astype(np.float32)
    map_y = np.clip(y_grid + dy, 0, h-1).astype(np.float32)
    # 对基础纹理进行重采样扭曲
    smoke_texture = cv2.remap(base_texture, map_x, map_y, cv2.INTER_LINEAR)
    
    # 3. 动态分配透射率 (Transmission)
    # 由于没有深度图，我们通过混合不同频率的噪声来模拟烟雾的全局浓度分布
    global_density = fbm_noise(w, h, octaves=2, base_scale=random.uniform(1.0, 2.0))
    if severity is None:
        density_strength = 0.5
        constant_bias = random.uniform(0.0, 0.05) if support_expanded else random.uniform(0.1, 0.4)
        transmission_power = random.uniform(2.8, 4.0) if support_expanded else random.uniform(1.2, 2.5)
    elif severity == "light":
        density_strength = random.uniform(0.15, 0.35)
        constant_bias = random.uniform(0.55, 0.75)
        transmission_power = random.uniform(0.7, 1.1)
    elif severity == "dense":
        density_strength = random.uniform(0.35, 0.55)
        constant_bias = random.uniform(0.25, 0.45)
        transmission_power = random.uniform(1.2, 1.8)
    elif severity == "very_dense":
        density_strength = random.uniform(0.50, 0.75)
        constant_bias = random.uniform(0.05, 0.20)
        transmission_power = random.uniform(2.0, 3.2)
    else:
        raise ValueError(
            "severity must be one of None, 'light', 'dense', or 'very_dense'"
        )
    
    # 计算初步透射率 t
    # 结合了全局浓度分布和局部流体细节
    gt_t = (1.0 - global_density * density_strength) * (constant_bias + (1.0 - constant_bias) * smoke_texture)
    
    # 4. 非线性对比度增强 (制造烟雾撕裂感和极浓区域)
    # 这里的幂次运算会让浓的地方更浓，淡的地方更透
    real_t = np.power(gt_t, transmission_power)
    real_t = np.clip(real_t, 0.01, 1.0)[..., None] # 压低底限，允许彻底遮挡

    # 5. 🔥 大气光 A 的空间多样化 (模拟光照不均)
    is_black_smoke = random.random() < 0.25 # 25% 概率生成黑烟
    if is_night:
        a_base_val = random.uniform(0.15, 0.35) if is_black_smoke else random.uniform(0.5, 0.8)
    else:
        a_base_val = random.uniform(0.1, 0.3) if is_black_smoke else random.uniform(0.8, 0.95)
    
    A_base = np.array([a_base_val, a_base_val, a_base_val], dtype=np.float32)
    # 增加一个局部增益图，模拟由于烟雾散射导致的局部亮度不均
    light_variance = cv2.resize(perlin_noise(8, 8, 2), (w, h))
    light_variance = 0.9 + 0.2 * light_variance 
    A_map = A_base[None, None, :] * light_variance[:, :, None]

    # 6. 光学合成公式: I = J * t + A * (1 - t)
    degraded = img * real_t + A_map * (1 - real_t)
    
    # 7. 夜间特殊处理 (模拟辉光)
    if is_night:
        gray_img = np.mean(img, axis=2)
        bright_mask = np.clip((gray_img - 0.3) * 2, 0, 1)
        bright_boost = 1 + 0.4 * bright_mask[..., None] * (1 - real_t)
        degraded = degraded * bright_boost

    degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)
    
    if return_params:
        # real_t: [H, W, 1], atmos_light: 取 A_map 的均值以适配原有接口
        return degraded, real_t, np.mean(A_map, axis=(0, 1))
        
    return degraded
    
# ================= 3. Dataset 类 =================
class CrossModalUAVDataset(Dataset):
    def __init__(
        self,
        root,
        size=256,
        use_depth=False,
        augment=True,
        second_smoke=False,
        severity_mix=False,
        consensus_smokes=0,
        consensus_severity="very_dense",
    ):
        self.root = root
        self.size = size
        self.use_depth = use_depth
        self.augment = augment
        self.second_smoke = bool(second_smoke)
        self.severity_mix = bool(severity_mix)
        self.consensus_smokes = int(consensus_smokes)
        self.consensus_severity = consensus_severity
        self.support_arm = os.environ.get("FT_NIGHT_VD_ARM", "default")
        self.support_seed = os.environ.get("FT_NIGHT_VD_SEED", "")
        if self.support_arm not in ("default", "expanded"):
            raise ValueError("FT_NIGHT_VD_ARM must be default or expanded")
        if self.support_arm == "expanded" and not self.support_seed:
            raise ValueError("FT_NIGHT_VD_SEED is required for expanded arm")
        if self.consensus_smokes < 0:
            raise ValueError("consensus_smokes must be non-negative")
        if self.consensus_severity not in ("light", "dense", "very_dense"):
            raise ValueError(
                "consensus_severity must be 'light', 'dense', or 'very_dense'"
            )

        self.rgb_dir = os.path.join(root, "gt") 
        self.ir_dir = os.path.join(root, "ir")
        # self.depth_dir = os.path.join(root, "depth") if use_depth else None

        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"GT folder not found: {self.rgb_dir}")
        if not os.path.isdir(self.ir_dir):
            raise FileNotFoundError(f"IR folder not found: {self.ir_dir}")

        self.train_list = sorted(
            (
                name
                for name in os.listdir(self.rgb_dir)
                if os.path.isfile(os.path.join(self.rgb_dir, name))
                and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            ),
            key=str.lower,
        )
        if not self.train_list:
            raise RuntimeError(f"No GT images found in: {self.rgb_dir}")

        missing_ir = []
        for raw_name in self.train_list:
            name_id = os.path.splitext(raw_name)[0]
            ir_path = self.find_image_path(self.ir_dir, name_id)
            if ir_path is None:
                ir_path = self.find_image_path(self.ir_dir, name_id + "_ir")
            if ir_path is None:
                missing_ir.append(name_id)
        if missing_ir:
            examples = ", ".join(missing_ir[:10])
            raise FileNotFoundError(
                f"Missing paired IR images for {len(missing_ir)} GT samples "
                f"in {self.ir_dir}. First examples: {examples}"
            )

    def __len__(self):
        return len(self.train_list)

    def find_image_path(self, folder, name_without_ext):
        # 完善后缀支持
        extensions = [
            '.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff',
            '.JPG', '.PNG', '.JPEG', '.BMP', '.TIF', '.TIFF',
        ]
        for ext in extensions:
            full_path = os.path.join(folder, name_without_ext + ext)
            if os.path.exists(full_path):
                return full_path
        return None

    def load_img(self, path):
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at: {path}")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = cv2.resize(
            img,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA,
        )
        return img.astype(np.float32)

    # 🔥 16-bit TIFF 安全读取函数 (修复了 None 检查逻辑)
    def load_ir_img(self, path):
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"IR image not found at: {path}")
        
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Failed to load IR image: {path}")

        # 16-bit 归一化
        if img.dtype == np.uint16 or img.dtype == np.int32:
            img_min, img_max = img.min(), img.max()
            img = (img.astype(np.float32) - img_min) / (img_max - img_min + 1e-8)
        else:
            img = img.astype(np.float32) / 255.0

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.mean(img, axis=2)

        img = cv2.resize(
            img,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA,
        )
        
        # 强制补回维度 (H, W) -> (H, W, 1)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        return img.astype(np.float32)

    def apply_sync_transform(self, gt, lq, ir, gt_t):
        if random.random() > 0.5:
            gt, lq, ir = cv2.flip(gt, 1), cv2.flip(lq, 1), cv2.flip(ir, 1)
            gt_t = cv2.flip(gt_t, 1)
        if random.random() > 0.5:
            gt, lq, ir = cv2.flip(gt, 0), cv2.flip(lq, 0), cv2.flip(ir, 0)
            gt_t = cv2.flip(gt_t, 0)
        k = random.randint(0, 3)
        if k > 0:
            gt = np.rot90(gt, k).copy()
            lq = np.rot90(lq, k).copy()
            ir = np.rot90(ir, k).copy()
            gt_t = np.rot90(gt_t, k).copy()
        return gt, lq, ir, gt_t

    def __getitem__(self, idx):
        raw_name = self.train_list[idx]
        name_id = os.path.splitext(raw_name)[0]

        # 1. 加载图像
        gt_path = self.find_image_path(self.rgb_dir, name_id)
        gt = self.load_img(gt_path)

        # 2. 生成合成烟雾
        brightness = np.mean(gt) * 255.0
        is_night = brightness < 70 
        severity = (
            random.choices(
                ["light", "dense", "very_dense"],
                weights=[5, 3, 2],
                k=1,
            )[0]
            if self.severity_mix
            else None
        )
        lq, gt_t_numpy, gt_A_numpy = add_synthetic_smoke(
            gt,
            is_night=is_night,
            return_params=True,
            severity=severity,
            support_expanded=(self.support_arm == "expanded" and is_night and
                              int(hashlib.sha256((self.support_seed + "|" + name_id).encode()).hexdigest(), 16) % 5 == 0),
        )
        
        # 3. 加载 IR
        ir_path = self.find_image_path(self.ir_dir, name_id)
        if ir_path is None:
             ir_path = self.find_image_path(self.ir_dir, name_id + "_ir")
        ir = self.load_ir_img(ir_path)
            
        # 4. 加载 Depth
        # depth_path = self.find_image_path(self.depth_dir, name_id) if self.depth_dir else None
        # if depth_path:
        #     depth = self.load_img(depth_path)
        #     if depth.ndim == 3: depth = np.mean(depth, axis=2, keepdims=True)
        # else:
        #     depth = np.zeros((self.size, self.size, 1), dtype=np.float32)

        # 5. 数据增强
        if self.augment:
            gt, lq, ir, gt_t_numpy = self.apply_sync_transform(gt, lq, ir, gt_t_numpy)

        # Experiment C: independently synthesize a second smoke observation
        # from the same transformed GT. This preserves exact spatial alignment.
        if self.second_smoke:
            brightness_2 = np.mean(gt) * 255.0
            lq2, gt_t2_numpy, _ = add_synthetic_smoke(
                gt,
                is_night=brightness_2 < 70,
                return_params=True,
                severity=None,
            )

        # 6. 🔥 统一 Tensor 转换 (移除了之前多余的手动转换块)
        def to_tensor(img_np):
            t = torch.from_numpy(img_np).float()
            if t.ndim == 2:
                t = t.unsqueeze(-1)
            return t.permute(2, 0, 1) * 2.0 - 1.0

        gt_t = to_tensor(gt)
        lq_t = to_tensor(lq)
        ir_t = to_tensor(ir) 
        # depth_t = to_tensor(depth)
        
        # 7. 物理参数转换
        phys_t = torch.from_numpy(gt_t_numpy).unsqueeze(0) if gt_t_numpy.ndim==2 else torch.from_numpy(gt_t_numpy).permute(2,0,1)
        phys_A = torch.from_numpy(gt_A_numpy).view(-1)

        # 8. NaN 安全清理
        ir_t = torch.nan_to_num(ir_t, 0.0)
        gt_t = torch.nan_to_num(gt_t, 0.0)

        sample = {
            "gt": gt_t, "lq": lq_t, "ir": ir_t, 
            "gt_t": phys_t, "gt_A": phys_A, "txt": "", "name": name_id
        }
        if self.second_smoke:
            sample["lq2"] = to_tensor(lq2)
            sample["gt_t2"] = (
                torch.from_numpy(gt_t2_numpy).unsqueeze(0)
                if gt_t2_numpy.ndim == 2
                else torch.from_numpy(gt_t2_numpy).permute(2, 0, 1)
            )
        if self.consensus_smokes > 0:
            consensus_lq = []
            consensus_gt_t = []
            brightness_consensus = np.mean(gt) * 255.0
            for _ in range(self.consensus_smokes):
                smoke, smoke_t, _ = add_synthetic_smoke(
                    gt,
                    is_night=brightness_consensus < 70,
                    return_params=True,
                    severity=self.consensus_severity,
                )
                consensus_lq.append(to_tensor(smoke))
                smoke_t = (
                    torch.from_numpy(smoke_t).unsqueeze(0)
                    if smoke_t.ndim == 2
                    else torch.from_numpy(smoke_t).permute(2, 0, 1)
                )
                consensus_gt_t.append(smoke_t.float())
            sample["consensus_lq"] = torch.stack(consensus_lq, dim=0)
            sample["consensus_gt_t"] = torch.stack(consensus_gt_t, dim=0)
        return sample
