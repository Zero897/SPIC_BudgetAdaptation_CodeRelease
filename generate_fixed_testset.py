import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import shutil # 🌟 确认已导入

# ================= 补充：FBM噪声生成  =================
def fbm_noise(width, height, octaves=4, base_scale=2):
    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 0.5
    frequency = base_scale
    for _ in range(octaves):
        noise += amplitude * perlin_noise(width, height, scale=frequency)
        amplitude *= 0.5
        frequency *= 2
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise

# ================= 1. 柏林噪声生成 =================
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

# ================= 修改后：支持浓度控制的烟雾合成 =================
def add_synthetic_smoke_controlled(img, level="light", is_night=False):
    """
    根据 level 控制烟雾浓度: "light", "dense", "very_dense"
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
        
    h, w = img.shape[:2]

    base_texture = fbm_noise(w, h, octaves=5, base_scale=random.uniform(2.0, 4.5))
    
    dx = fbm_noise(w, h, octaves=2, base_scale=10) * 40 - 20
    dy = fbm_noise(w, h, octaves=2, base_scale=10) * 40 - 20
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    map_x = np.clip(x_grid + dx, 0, w-1).astype(np.float32)
    map_y = np.clip(y_grid + dy, 0, h-1).astype(np.float32)
    smoke_texture = cv2.remap(base_texture, map_x, map_y, cv2.INTER_LINEAR)
    
    global_density = fbm_noise(w, h, octaves=2, base_scale=random.uniform(1.0, 2.0))
    
    # 🔥 核心控制逻辑：根据浓度等级调整透射率参数
    if level == "light":
        # 淡烟：保留较高的透射率底线，非线性衰减较弱
        constant_bias = random.uniform(0.4, 0.6)
        power_val = random.uniform(1.0, 1.3)
    elif level == "dense":
        # 浓烟：透射率底线降低，非线性衰减增强
        constant_bias = random.uniform(0.15, 0.3)
        power_val = random.uniform(1.8, 2.5)
    elif level == "very_dense":
        # 极浓烟雾：几乎没有透射率底线，强烈的非线性压制使得局部完全遮挡
        constant_bias = random.uniform(0.0, 0.05)
        power_val = random.uniform(2.8, 4.0)
    else:
        constant_bias = random.uniform(0.1, 0.4)
        power_val = random.uniform(1.2, 2.5)
        
    gt_t = (1.0 - global_density * 0.5) * (constant_bias + (1.0 - constant_bias) * smoke_texture)
    
    # 幂次越大，t 越小，烟越浓
    real_t = np.power(gt_t, power_val)
    real_t = np.clip(real_t, 0.01, 1.0)[..., None] 

    is_black_smoke = random.random() < 0.25
    if is_night:
        a_base_val = random.uniform(0.15, 0.35) if is_black_smoke else random.uniform(0.5, 0.8)
    else:
        a_base_val = random.uniform(0.1, 0.3) if is_black_smoke else random.uniform(0.8, 0.95)
    
    A_base = np.array([a_base_val, a_base_val, a_base_val], dtype=np.float32)
    light_variance = cv2.resize(perlin_noise(8, 8, 2), (w, h))
    light_variance = 0.9 + 0.2 * light_variance 
    A_map = A_base[None, None, :] * light_variance[:, :, None]

    degraded = img * real_t + A_map * (1 - real_t)
    
    if is_night:
        gray_img = np.mean(img, axis=2)
        bright_mask = np.clip((gray_img - 0.3) * 2, 0, 1)
        bright_boost = 1 + 0.4 * bright_mask[..., None] * (1 - real_t)
        degraded = degraded * bright_boost

    degraded = np.clip(degraded, 0.0, 1.0)
    return (degraded * 255.0).astype(np.uint8)

# ================= 批量生成执行函数 =================
def generate_dataset():
    input_gt_dir = "data/dataset/uav/test/gt"
    input_ir_dir = "data/dataset/uav/test/ir"
    
    # 构建严谨的评测集目录结构
    base_out_dir = "data/dataset/uav/synthetic_smoke_eval_1000"
    out_lq_dir = os.path.join(base_out_dir, "lq")  # 放合成烟雾
    out_gt_dir = os.path.join(base_out_dir, "gt")  # 放清晰原图
    out_ir_dir = os.path.join(base_out_dir, "ir")  # 放对应红外图
    
    for d in [out_lq_dir, out_gt_dir, out_ir_dir]:
        os.makedirs(d, exist_ok=True)
        
    # 辅助函数：寻找对应的 IR 图
    def find_ir_path(base_name):
        extensions = ['.tif', '.png', '.jpg', '.bmp', '.TIF']
        suffixes = ['', '_ir']
        for suff in suffixes:
            for ext in extensions:
                p = os.path.join(input_ir_dir, base_name + suff + ext)
                if os.path.exists(p): return p
        return None

    # 1. 读取所有有效图像，并确保它有对应的 IR 图
    valid_exts = {'.jpg', '.png', '.jpeg', '.bmp', '.tif'}
    all_images = [f for f in os.listdir(input_gt_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not all_images:
        print("❌ 错误：未找到任何有效图像，请检查 input_gt_dir 路径！")
        return

    # 2. 预扫描图像，按亮度分类白天和黑夜，同时剔除没有 IR 图的废片
    print("🔍 正在预扫描图像 (过滤缺失IR的数据，并区分昼夜)...")
    day_images = []
    night_images = []
    
    for img_name in tqdm(all_images, desc="扫描数据集"):
        base_name = os.path.splitext(img_name)[0]
        
        # 检查 IR 是否存在
        ir_path = find_ir_path(base_name)
        if ir_path is None: 
            continue # 没有对应IR图，直接跳过
            
        img_path = os.path.join(input_gt_dir, img_name)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: continue
        
        brightness = np.mean(img_gray)
        if brightness < 70:
            night_images.append((img_name, ir_path))
        else:
            day_images.append((img_name, ir_path))
            
    print(f"✅ 扫描完成: 有效白天图像 {len(day_images)} 张, 有效黑夜图像 {len(night_images)} 张。")

    # =========================================================================
    # 🌟 修改 1：按 2:1 的比例精确抽取 1000 张图像
    # =========================================================================
    total_target = 1000
    target_day = int(total_target * (2 / 3))  # 约 666 张
    target_night = total_target - target_day  # 约 334 张

    # 防止原图数量不够导致随机抽样报错
    # 如果原图不够 1000 张，使用 random.choices 会自动进行有放回抽样（重复使用图片生成不同烟雾）
    if not night_images:
        target_day, target_night = total_target, 0
    elif not day_images:
        target_day, target_night = 0, total_target

    selected_day = random.choices(day_images, k=target_day) if day_images else []
    selected_night = random.choices(night_images, k=target_night) if night_images else []
    
    selected_images = selected_day + selected_night
    random.shuffle(selected_images) 

    # =========================================================================
    # 🌟 修改 2：按照 5:3:2 分配浓度任务 (总数 1000 张)
    # =========================================================================
    # 淡烟 (50%): 500 张
    # 浓烟 (30%): 300 张
    # 极浓烟 (20%): 200 张
    num_light = int(total_target * 0.5)      # 500
    num_dense = int(total_target * 0.3)      # 300
    num_very_dense = total_target - num_light - num_dense # 200

    tasks = (
        [("light", i) for i in range(num_light)] +
        [("dense", i) for i in range(num_light, num_light + num_dense)] +
        [("very_dense", i) for i in range(num_light + num_dense, total_target)]
    )
    random.shuffle(tasks) 

    print(f"🚀 开始合成并对齐 {total_target} 张图像 (白天 {target_day}, 黑夜 {target_night} | 5:3:2 浓度)...")
    
    # 5. 正式合成与文件分发
    for task_idx, (level, idx) in enumerate(tqdm(tasks, desc="构建评测集")):
        img_name, ir_path = selected_images[task_idx]
        gt_path = os.path.join(input_gt_dir, img_name)
        
        # 读取 GT
        img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        brightness = np.mean(img_rgb)
        is_night = brightness < 70 
        
        # 合成烟雾图 (LQ)
        smoke_img = add_synthetic_smoke_controlled(img_rgb, level=level, is_night=is_night)
        smoke_img_bgr = cv2.cvtColor(smoke_img, cv2.COLOR_RGB2BGR)
        
        # 构造统一的前缀名：方便 DataLoader 无脑读取
        time_tag = "night" if is_night else "day"
        base_name = os.path.splitext(img_name)[0]
        # 🌟 修改 3：命名格式扩展为 4 位数，以支持 1000 张以上的命名 (例如: 0899_imageXYZ_day_dense)
        unified_name = f"{idx:04d}_{base_name}_{time_tag}_{level}"
        
        # 保存合成的烟雾图 (PNG格式)
        cv2.imwrite(os.path.join(out_lq_dir, unified_name + ".png"), smoke_img_bgr)
        
        # 拷贝真实的 GT 和 IR 原文件 (保留原有后缀和数据深度)
        gt_ext = os.path.splitext(gt_path)[1]
        ir_ext = os.path.splitext(ir_path)[1]
        
        shutil.copy(gt_path, os.path.join(out_gt_dir, unified_name + gt_ext))
        shutil.copy(ir_path, os.path.join(out_ir_dir, unified_name + ir_ext))

    print(f"\n🎉 评测集构建完美收工！")
    print(f"📁 烟雾图(LQ): {out_lq_dir}")
    print(f"📁 清晰图(GT): {out_gt_dir}")
    print(f"📁 红外图(IR): {out_ir_dir}")

if __name__ == "__main__":
    generate_dataset()