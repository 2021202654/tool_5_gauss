# graphene_features.py (物理升级版)
import numpy as np
import pandas as pd

def calculate_theoretical_k(df):
    """
    基于 Klemens-Callaway 简化模型的物理特征计算。
    【升级】调整了基准参数，使其更能反映悬空(Suspended)石墨烯的高热导率特性。
    """
    # 1. 获取参数 (使用 get 增强鲁棒性)
    T = df.get('temperature', 300.0)
    L = df.get('length_um', 10.0) 
    defect = df.get('defect_ratio', 0.0) 
    
    # 2. 归一化缺陷 (Defect Penalty)
    # 缺陷对热导率是毁灭性打击，保持指数级惩罚
    log_D = np.log10(defect + 1e-12)
    norm_D = (log_D - (-8)) / 6.0
    defect_factor = (1.0 - 0.85 * norm_D) # 稍微加强一点缺陷的敏感度
    
    # 3. 温度因子 (Umklapp Scattering)
    # 纯净石墨烯遵循 ~1/T 规律
    temp_factor = (300.0 / (T + 1.0)) ** 1.0 
    
    # 4. 尺寸因子 (Ballistic Transport)
    # 修正点：对于大尺寸(>5um)，提升增益上限
    # 物理逻辑：L=10um 时，声子平均自由程并未完全被边界截断
    size_factor = 1.0 + 0.6 * np.log10(L + 0.1)
    size_factor = np.clip(size_factor, 0.5, 5.0) # 提高上限到 5倍
    
    # 5. 🔥 核心修正：基准常数 (Base Constant)
    # 旧值: 2000.0 (过于保守，像是有基底的情况)
    # 新值: 3200.0 (更接近悬空石墨烯的本征基准)
    # 当 L 很大且无缺陷时，3200 * size_factor 可以达到 5000+，符合 Balandin 实验结果
    base_constant = 3200.0 
    
    # 理论估算值
    k_theory = base_constant * temp_factor * size_factor * defect_factor
    return np.maximum(k_theory, 10.0) 

def enhance_features(df):
    """
    特征工程管道：原始数据 -> 机器学习可用特征
    """
    df_out = df.copy()
    
    # 1. 基础对数变换
    if 'temperature' in df_out.columns:
        df_out['log_temp'] = np.log10(df_out['temperature'] + 1.0)
    if 'length_um' in df_out.columns:
        df_out['log_length'] = np.log10(df_out['length_um'] + 0.001)
    if 'defect_ratio' in df_out.columns:
        df_out['log_defect'] = np.log10(df_out['defect_ratio'] + 1e-9)
        
    # 2. 处理基底因子
    if 'substrate_type' in df_out.columns:
        sub_map = {
            'Suspended': 1.0, 
            'hBN': 0.8, 
            'SiO2': 0.5, 
            'Au': 0.2, 
            'Cu': 0.2
        }
        df_out['substrate_factor'] = df_out['substrate_type'].map(sub_map).fillna(0.5)
    else:
        df_out['substrate_factor'] = 0.5

    # 3. 注入物理灵魂
    raw_theory_k = calculate_theoretical_k(df_out)
    
    # 最终理论特征 = 修正后的物理上限 * 基底衰减
    # 这样 Suspended 就能跑到 4000-5000，而 SiO2 依然会被拉回 2000 以下
    df_out['log_theory_k'] = np.log10(raw_theory_k * df_out['substrate_factor'])
    
    return df_out