#!/usr/bin/env python3
"""
完整的數據預處理腳本
一次性處理所有數據並生成預處理後的數據集
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from collections import Counter
import pickle
import json
import warnings
import time  # 🔧 優化：用於進度顯示
warnings.filterwarnings('ignore')

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config_fixed as config

class DataPreprocessor:
    """完整的數據預處理器"""
    
    def __init__(self, config_data=None):
        self.config_data = config_data or config.DATA_CONFIG
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.all_labels = []
        self.feature_columns = None
        self.target_column = 'label'
        
    def _normalize_label(self, value: str) -> str:
        """標籤清洗：保留所有DoS/DDoS相關攻擊和正常流量"""
        txt = str(value).strip()
        if not txt or txt.lower() in ['nan', 'none', 'null', '']:
            return 'BENIGN'
        low = txt.lower()
        
        # 🔧 處理 CICIDS2017 數值標籤映射（常見映射）
        # 注意：這些映射可能需要根據實際數據集調整
        # 通常 0 或某個特定值代表 BENIGN，其他值代表不同攻擊類型
        # 由於無法確定確切映射，我們先嘗試字符串匹配，如果失敗則根據數值判斷
        
        # 先嘗試字符串匹配（適用於 CIC-DDoS2019 等）
        if 'drdos' in low or 'dns' in low:
            return 'DDoS'  # DrDoS_DNS 映射為 DDoS
        elif 'ddos' in low or 'distributed denial of service' in low or 'distributed dos' in low:
            return 'DDoS'
        elif 'dos hulk' in low or 'hulk' in low:
            return 'DoS_Hulk'
        elif 'dos goldeneye' in low or 'goldeneye' in low:
            return 'DoS_GoldenEye'
        elif 'dos slowloris' in low or 'slowloris' in low:
            return 'DoS_slowloris'
        elif 'dos slowhttptest' in low or 'slowhttptest' in low:
            return 'DoS_Slowhttptest'
        elif 'benign' in low or 'normal' in low or 'legitimate' in low or 'clean' in low or 'safe' in low or 'regular' in low:
            return 'BENIGN'
        
        # 🔧 處理 CICIDS2017 數值標籤映射（根據 preprocess.py）
        # 映射關係：1=BENIGN, 3=DoS_Slowhttptest, 5=DoS_slowloris, 6=DoS_Hulk, 13=DDoS, 15=DoS_GoldenEye
        # 只保留 DDoS 相關和 BENIGN，其他標籤過濾掉
        try:
            label_num = int(float(txt))
            if label_num == 1:
                return 'BENIGN'
            elif label_num == 3:
                return 'DoS_Slowhttptest'
            elif label_num == 5:
                return 'DoS_slowloris'
            elif label_num == 6:
                return 'DoS_Hulk'
            elif label_num == 13:
                return 'DDoS'
            elif label_num == 15:
                return 'DoS_GoldenEye'
            else:
                # 其他數值標籤（非 DDoS 相關）都過濾掉
                return 'UNKNOWN'
        except (ValueError, TypeError):
            # 無法轉換為數值，視為字符串標籤
            pass
        
        # 其他攻擊類型都歸類為UNKNOWN，後續會被過濾掉
        return 'UNKNOWN'
    
    def _create_scaler(self, scaler_type: str):
        """創建標準化器"""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def _looks_standard_scaled(self, df: pd.DataFrame, feature_cols) -> bool:
        """簡單判斷資料是否已接近 StandardScaler 後分布。"""
        try:
            X = df[feature_cols]
            mean_abs_avg = X.mean().abs().mean()
            std_avg = X.std().mean()
            return mean_abs_avg < 0.5 and 0.5 < std_avg < 2.5
        except Exception:
            return False
    
    def _detect_outliers(self, X, method='isolation_forest', **kwargs):
        """異常值檢測"""
        if not self.config_data.get('preprocessing', {}).get('outlier_detection', False):
            return np.ones(len(X), dtype=bool)  # 不檢測，保留所有樣本
        
        print(f"🔍 應用 {method.upper()} 異常值檢測...")
        
        contamination = kwargs.get('contamination', 0.1)
        threshold = kwargs.get('threshold', 0.1)
        
        try:
            if method == 'isolation_forest':
                # 使用 Isolation Forest
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=kwargs.get('random_state', 42),
                    n_estimators=100,
                    max_samples='auto'
                )
                outlier_labels = iso_forest.fit_predict(X)
                # 1 表示正常樣本，-1 表示異常樣本
                inlier_mask = outlier_labels == 1
                
            elif method == 'iqr':
                # 使用 IQR 方法（四分位距）
                inlier_mask = np.ones(len(X), dtype=bool)
                for i in range(X.shape[1]):
                    Q1 = np.percentile(X[:, i], 25)
                    Q3 = np.percentile(X[:, i], 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    feature_mask = (X[:, i] >= lower_bound) & (X[:, i] <= upper_bound)
                    inlier_mask = inlier_mask & feature_mask
                    
            elif method == 'zscore':
                # 使用 Z-score 方法
                z_scores = np.abs((X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6))
                # 保留 Z-score < 3 的樣本
                inlier_mask = np.all(z_scores < 3, axis=1)
                
            else:
                print(f"⚠️ 未知異常值檢測方法: {method}，跳過異常值檢測")
                return np.ones(len(X), dtype=bool)
            
            outlier_count = len(X) - inlier_mask.sum()
            outlier_ratio = outlier_count / len(X)
            
            print(f"  - 檢測到異常值: {outlier_count} 個 ({outlier_ratio*100:.2f}%)")
            
            # 如果異常值比例超過閾值，只移除超過閾值的部分
            if outlier_ratio > threshold:
                # 計算需要保留的樣本數
                max_outliers = int(len(X) * threshold)
                # 如果檢測到的異常值太多，只移除最多 threshold 比例的樣本
                if outlier_count > max_outliers:
                    # 對於 Isolation Forest，可以根據異常分數排序
                    if method == 'isolation_forest':
                        iso_forest = IsolationForest(
                            contamination=threshold,
                            random_state=kwargs.get('random_state', 42),
                            n_estimators=100
                        )
                        outlier_labels = iso_forest.fit_predict(X)
                        inlier_mask = outlier_labels == 1
                        print(f"  - 調整後移除異常值: {(~inlier_mask).sum()} 個 ({threshold*100:.1f}%)")
            
            return inlier_mask
            
        except Exception as e:
            print(f"❌ 異常值檢測失敗 ({method}): {e}")
            return np.ones(len(X), dtype=bool)  # 失敗時保留所有樣本
    
    def _balance_data(self, X, y, method='smote', **kwargs):
        """數據平衡"""
        if not self.config_data.get('balancing', {}).get('enabled', False):
            return X, y
        
        print(f"🚀 應用 {method.upper()} 數據平衡...")
        
        # 分析原始數據不平衡程度
        distribution = Counter(y)
        total_samples = len(y)
        max_count = max(distribution.values())
        min_count = min(distribution.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"  - 原始數據: {dict(distribution)}")
        print(f"  - 不平衡比例: {imbalance_ratio:.2f}:1")
        
        try:
            if method == 'smote':
                sampler = SMOTE(
                    random_state=kwargs.get('random_state', 42),
                    k_neighbors=kwargs.get('k_neighbors', 5),
                    sampling_strategy=kwargs.get('sampling_strategy', 'auto')
                )
            elif method == 'adasyn':
                sampler = ADASYN(
                    random_state=kwargs.get('random_state', 42),
                    n_neighbors=kwargs.get('k_neighbors', 5),
                    sampling_strategy=kwargs.get('sampling_strategy', 'auto')
                )
            elif method == 'borderline_smote':
                sampler = BorderlineSMOTE(
                    random_state=kwargs.get('random_state', 42),
                    k_neighbors=kwargs.get('k_neighbors', 5),
                    sampling_strategy=kwargs.get('sampling_strategy', 'auto')
                )
            elif method == 'random_oversampler':
                sampler = RandomOverSampler(
                    random_state=kwargs.get('random_state', 42),
                    sampling_strategy=kwargs.get('sampling_strategy', 'auto')
                )
            else:
                print(f"⚠️ 未知平衡方法: {method}，跳過數據平衡")
                return X, y
            
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            
            # 分析平衡後數據
            balanced_distribution = Counter(y_balanced)
            balanced_imbalance_ratio = max(balanced_distribution.values()) / min(balanced_distribution.values())
            
            print(f"  - 平衡後數據: {dict(balanced_distribution)}")
            print(f"  - 新不平衡比例: {balanced_imbalance_ratio:.2f}:1")
            print(f"✅ {method.upper()} 平衡完成")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"❌ 數據平衡失敗 ({method}): {e}")
            return X, y

    def _write_preprocess_report(self, output_dir: str, metadata: dict) -> None:
        """輸出預處理流程文字摘要報告（Markdown）。"""
        try:
            preprocessing_config = self.config_data.get('preprocessing', {})
            balancing_config = self.config_data.get('balancing', {})
            train_ratio = self.config_data.get('train_ratio', 0.8)
            test_ratio = self.config_data.get('test_ratio', 0.2)
            validation_ratio = self.config_data.get('validation_ratio', 0.1)

            lines = []
            lines.append("# 預處理流程摘要")
            lines.append("")
            lines.append("## 流程概述")
            lines.append("1. 載入原始 CSV，清理欄位名稱與標籤欄位。")
            lines.append("2. 移除缺失值與 Inf，並對極端值做分位數裁剪。")
            lines.append("3. 標籤正規化：只保留 DoS/DDoS + BENIGN，其餘過濾。")
            lines.append("4. 固定 LabelEncoder：只在首次 fit，之後遇到新標籤直接過濾。")
            lines.append("5. 特徵標準化：以單一 scaler fit 於全局資料，後續重複使用。")
            lines.append("6. 選配：異常值檢測與資料平衡（SMOTE/ADASYN 等）。")
            lines.append("7. 分割訓練/測試，再由訓練集分割驗證集。")
            lines.append("8. 產生全域測試集與已標準化版本。")
            lines.append("9. 以 Dirichlet 產生 Non-IID client 資料。")
            lines.append("10. 輸出 metadata、scaler、label_encoder、feature_cols。")
            lines.append("")
            lines.append("## 重要參數")
            lines.append(f"- scaler_type: {preprocessing_config.get('scaler_type', 'standard')}")
            lines.append(f"- outlier_detection: {preprocessing_config.get('outlier_detection', False)}")
            lines.append(f"- outlier_method: {preprocessing_config.get('outlier_method', 'isolation_forest')}")
            lines.append(f"- outlier_threshold: {preprocessing_config.get('outlier_threshold', 0.1)}")
            lines.append(f"- balancing_enabled: {balancing_config.get('enabled', False)}")
            lines.append(f"- balancing_method: {balancing_config.get('method', 'smote')}")
            lines.append(f"- train_ratio: {train_ratio}")
            lines.append(f"- test_ratio: {test_ratio}")
            lines.append(f"- validation_ratio: {validation_ratio}")
            lines.append(f"- num_clients: {metadata.get('num_clients')}")
            lines.append(f"- num_features: {metadata.get('num_features')}")
            lines.append(f"- num_classes: {metadata.get('num_classes')}")
            lines.append(f"- class_labels: {metadata.get('class_labels')}")
            lines.append(f"- global_test_path: {metadata.get('global_test_path')}")
            lines.append("")
            lines.append("## Non-IID 設定")
            lines.append("- Dirichlet alpha: 0.5")
            lines.append("- client_samples_per_round: 500~2000")
            lines.append("- min_samples_per_class: 20")
            lines.append("")
            lines.append("## 輸出檔案")
            lines.append("- preprocessor.pkl (scaler + label_encoder + feature_columns)")
            lines.append("- scaler.pkl")
            lines.append("- label_encoder.pkl")
            lines.append("- feature_cols.json")
            lines.append("- metadata.json")
            lines.append("- global_test.csv / global_test_scaled.csv")

            report_path = os.path.join(output_dir, "preprocess_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"  - 預處理摘要已保存至: {report_path}")
        except Exception as e:
            print(f"  - ⚠️ 生成預處理摘要失敗: {e}")
    
    def preprocess_single_file(self, input_path: str, output_dir: str, client_id: int = None):
        """預處理單個文件"""
        print(f"\n📁 處理文件: {input_path}")
        
        # 讀取數據
        try:
            df = pd.read_csv(input_path)
            print(f"  - 原始數據形狀: {df.shape}")
        except Exception as e:
            print(f"❌ 讀取文件失敗: {e}")
            return None
        
        # 數據清洗
        # 🔧 處理 CIC-DDoS2019 數據集的標籤列（可能是 " Label" 帶空格）
        df.columns = df.columns.str.strip()  # 先去除前後空格
        if ' Label' in df.columns and 'label' not in df.columns:
            df.rename(columns={' Label': 'label'}, inplace=True)
        df.columns = df.columns.str.lower()  # 再轉為小寫
        
        # 檢查標籤列
        if self.target_column not in df.columns:
            candidates = [c for c in df.columns if c.lower() in ("label", "labels", "target", "class", "attack_label", "target label")]
            if candidates:
                df.rename(columns={candidates[0]: self.target_column}, inplace=True)
                print(f"  - 重命名標籤列: {candidates[0]} → {self.target_column}")
            else:
                # 如果找不到，嘗試使用最後一列作為標籤
                last_col = df.columns[-1]
                if df[last_col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    # 數值標籤，可能是已編碼的標籤
                    df.rename(columns={last_col: self.target_column}, inplace=True)
                    print(f"  - 使用最後一列作為標籤: {last_col} → {self.target_column}")
                else:
                    print(f"❌ 找不到標籤列")
                    return None
        
        # 處理缺失值和異常值
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"  - 處理缺失值: {missing_before} 個")
            df = df.dropna()
            print(f"  - 清理後數據形狀: {df.shape}")
        
        # 處理無窮大值和過大值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            print(f"  - 處理無窮大值: {inf_count} 個")
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            print(f"  - 清理後數據形狀: {df.shape}")
        
        # 🚀 破局配置 3.6：對重尾特徵進行 log1p 轉換（解決類別 3 特徵量級不對稱問題）
        # 診斷：類別 3 (DoS_Slowhttptest) 可能包含極大的 Flow Duration 或 Packet Count，導致特徵量級不對稱
        # 解決：對重尾特徵先做 log1p 轉換，再進行標準化
        # 🔧 優化：可通過環境變量控制，如果數據已經處理過可以跳過
        enable_log_transform = os.environ.get("ENABLE_LOG1P_TRANSFORM", "1").strip().lower() in ("1", "true", "yes", "on")
        
        if enable_log_transform:
            heavy_tailed_features = [
                'flow duration', 'total fwd packets', 'total backward packets',
                'total length of fwd packets', 'total length of bwd packets',
                'fwd iat total', 'bwd iat total', 'flow bytes/s', 'flow packets/s',
                'fwd packets/s', 'bwd packets/s', 'subflow fwd bytes', 'subflow bwd bytes'
            ]
            
            # 🔧 優化：使用向量化操作，一次性處理所有重尾特徵
            log_transformed_cols = []
            for col in numeric_cols:
                if col != self.target_column:
                    col_lower = col.lower().strip()
                    # 檢查是否為重尾特徵
                    if any(htf in col_lower for htf in heavy_tailed_features):
                        log_transformed_cols.append(col)
            
            if log_transformed_cols:
                # 向量化處理：一次性處理所有重尾特徵
                df[log_transformed_cols] = df[log_transformed_cols].clip(lower=0)
                df[log_transformed_cols] = np.log1p(df[log_transformed_cols])
                print(f"  - 🚀 破局配置 3.6：對 {len(log_transformed_cols)} 個重尾特徵應用 log1p 轉換（向量化處理）")
                if len(log_transformed_cols) <= 5:
                    print(f"  - 轉換的特徵: {log_transformed_cols}")
        else:
            print(f"  - ⚠️ log1p 轉換已禁用（ENABLE_LOG1P_TRANSFORM=0），跳過重尾特徵轉換")
        
        # 處理過大值（超過float64範圍）
        for col in numeric_cols:
            if col != self.target_column:
                max_val = df[col].max()
                min_val = df[col].min()
                if max_val > 1e15 or min_val < -1e15:
                    print(f"  - 處理過大值列 {col}: max={max_val:.2e}, min={min_val:.2e}")
                    # 使用分位數裁剪
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    df[col] = df[col].clip(lower=q01, upper=q99)
        
        # 標籤標準化（只保留 DDoS 相關和 BENIGN）
        # 檢查標籤是否為數值（已編碼）
        if df[self.target_column].dtype in ['int64', 'int32', 'float64', 'float32']:
            # 數值標籤，需要映射為類別名稱（只保留 DDoS 相關）
            print(f"  - 檢測到數值標籤，唯一值: {sorted(df[self.target_column].unique())}")
            print(f"  - 將數值標籤映射為類別名稱（只保留 DDoS 相關和 BENIGN）")
            original_count = len(df)
            # 先轉換為字符串，然後應用標籤映射
            df[self.target_column] = df[self.target_column].astype(str).apply(self._normalize_label)
            
            # 只保留 DDoS 相關攻擊和正常流量
            valid_labels = ['DDoS', 'DoS_Hulk', 'DoS_GoldenEye', 'DoS_slowloris', 'DoS_Slowhttptest', 'BENIGN']
            df = df[df[self.target_column].isin(valid_labels)]
            filtered_count = len(df)
            if original_count != filtered_count:
                print(f"  - 過濾非 DDoS 相關攻擊: {original_count} → {filtered_count} 樣本（保留 {filtered_count/original_count*100:.1f}%）")
        else:
            # 字符串標籤，需要標準化
            df[self.target_column] = df[self.target_column].apply(self._normalize_label)
            
            # 只保留DoS/DDoS相關攻擊和正常流量
            original_count = len(df)
            valid_labels = ['DDoS', 'DoS_Hulk', 'DoS_GoldenEye', 'DoS_slowloris', 'DoS_Slowhttptest', 'BENIGN']
            df = df[df[self.target_column].isin(valid_labels)]
            filtered_count = len(df)
            if original_count != filtered_count:
                print(f"  - 過濾非DoS/DDoS攻擊: {original_count} → {filtered_count} 樣本")
        
        # 檢查數據分布
        label_dist = df[self.target_column].value_counts()
        print(f"  - 最終標籤分布: {dict(label_dist)}")
        
        # 檢查是否有有效數據
        if len(df) == 0:
            print(f"  - ⚠️ 文件過濾後無有效數據，跳過此文件")
            return None
        
        # 分離特徵和標籤
        feature_cols = [col for col in df.columns if col != self.target_column]
        X = df[feature_cols]
        y = df[self.target_column]
        
        # 標籤編碼
        # 🔧 修復客戶端類別數不匹配問題：確保 label_encoder 包含所有可能的類別
        unique_labels = sorted(y.unique())
        
        # 🔧 修復：只在首次擬合 label_encoder，之後不再重 fit
        if not self.all_labels or len(self.all_labels) == 0:
            # 第一次處理，擬合 label_encoder
            if df[self.target_column].dtype in ['int64', 'int32', 'float64', 'float32']:
                y_str = y.astype(str)
                self.label_encoder.fit(y_str.unique())
            else:
                self.label_encoder.fit(y.unique())
            self.all_labels = list(self.label_encoder.classes_)
            print(f"  - 首次擬合 label_encoder，學習到的標籤: {self.all_labels}")
            print(f"  - 類別數: {len(self.all_labels)}")
        else:
            # 檢查是否有新的類別（不再重 fit，直接過濾）
            if df[self.target_column].dtype in ['int64', 'int32', 'float64', 'float32']:
                y_str = y.astype(str)
                new_labels = set(y_str.unique()) - set(self.all_labels)
            else:
                new_labels = set(y.unique()) - set(self.all_labels)
            
            if new_labels:
                print(f"  - ⚠️ 發現新類別: {new_labels}，已過濾這些樣本以保持編碼固定")
                if df[self.target_column].dtype in ['int64', 'int32', 'float64', 'float32']:
                    df = df[~y_str.isin(new_labels)].copy()
                else:
                    df = df[~y.isin(new_labels)].copy()
                y = df[self.target_column]
        
        # 執行標籤編碼
        if df[self.target_column].dtype in ['int64', 'int32', 'float64', 'float32']:
            # 數值標籤，先轉換為字符串，然後使用 label_encoder
            y_str = y.astype(str)
            y_encoded = self.label_encoder.transform(y_str)
        else:
            # 字符串標籤，直接使用 label_encoder
            y_encoded = self.label_encoder.transform(y)
        
        # 特徵標準化
        scaler_type = self.config_data.get('preprocessing', {}).get('scaler_type', 'standard')
        if client_id is None:  # 只有第一次處理時才擬合標準化器
            self.scaler = self._create_scaler(scaler_type)
            X_scaled = self.scaler.fit_transform(X)
            print(f"  - 使用 {scaler_type} 標準化器")
        else:
            X_scaled = self.scaler.transform(X)
        
        # 異常值檢測（在標準化之後、平衡之前）
        preprocessing_config = self.config_data.get('preprocessing', {})
        if preprocessing_config.get('outlier_detection', False):
            outlier_method = preprocessing_config.get('outlier_method', 'isolation_forest')
            outlier_contamination = preprocessing_config.get('outlier_contamination', 0.1)
            outlier_threshold = preprocessing_config.get('outlier_threshold', 0.1)
            
            inlier_mask = self._detect_outliers(
                X_scaled,
                method=outlier_method,
                contamination=outlier_contamination,
                threshold=outlier_threshold,
                random_state=self.config_data.get('random_state', 42)
            )
            
            # 移除異常值
            original_count = len(X_scaled)
            X_scaled = X_scaled[inlier_mask]
            y_encoded = y_encoded[inlier_mask]
            filtered_count = len(X_scaled)
            
            if original_count != filtered_count:
                print(f"  - 移除異常值後: {original_count} → {filtered_count} 樣本 (移除 {original_count - filtered_count} 個)")
        
        # 🔧 修改：先分割數據（在 SMOTE 之前），以保存真正的標準化版本
        train_ratio = self.config_data.get('train_ratio', 0.8)
        test_ratio = self.config_data.get('test_ratio', 0.2)
        validation_ratio = self.config_data.get('validation_ratio', 0.1)
        
        # 先分割訓練集和測試集（使用標準化後的數據，但尚未經過 SMOTE）
        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = train_test_split(
            X_scaled, y_encoded,
            test_size=test_ratio,
            random_state=self.config_data.get('random_state', 42),
            stratify=y_encoded
        )
        
        # 再從訓練集中分割出驗證集
        if validation_ratio > 0:
            val_size = validation_ratio / train_ratio  # 相對於訓練集的比例
            X_train_scaled, X_val_scaled, y_train_encoded, y_val_encoded = train_test_split(
                X_train_scaled, y_train_encoded,
                test_size=val_size,
                random_state=self.config_data.get('random_state', 42),
                stratify=y_train_encoded
            )
        else:
            X_val_scaled, y_val_encoded = None, None
        
        # 🔧 修改：在 SMOTE 之前先保存標準化版本（保持 StandardScaler 的統計特性）
        client_suffix = f"_client_{client_id}" if client_id is not None else ""
        
        # 保存標準化版本的訓練集
        train_scaled_path = os.path.join(output_dir, f"train{client_suffix}_scaled.csv")
        train_scaled_data = pd.DataFrame(X_train_scaled, columns=feature_cols)
        train_scaled_data[self.target_column] = y_train_encoded
        train_scaled_data.to_csv(train_scaled_path, index=False)
        
        # 保存標準化版本的測試集
        test_scaled_path = os.path.join(output_dir, f"test{client_suffix}_scaled.csv")
        test_scaled_data = pd.DataFrame(X_test_scaled, columns=feature_cols)
        test_scaled_data[self.target_column] = y_test_encoded
        test_scaled_data.to_csv(test_scaled_path, index=False)
        
        # 保存標準化版本的驗證集
        if X_val_scaled is not None:
            val_scaled_path = os.path.join(output_dir, f"val{client_suffix}_scaled.csv")
            val_scaled_data = pd.DataFrame(X_val_scaled, columns=feature_cols)
            val_scaled_data[self.target_column] = y_val_encoded
            val_scaled_data.to_csv(val_scaled_path, index=False)
        
        # 🔧 修改：現在對訓練集進行 SMOTE（如果需要），然後保存 raw 版本
        balancing_config = self.config_data.get('balancing', {})
        if balancing_config.get('enabled', False):
            # 只對訓練集進行 SMOTE
            X_train_balanced, y_train_balanced = self._balance_data(
                X_train_scaled, y_train_encoded,
                method=balancing_config.get('method', 'smote'),
                random_state=balancing_config.get('random_state', 42),
                k_neighbors=balancing_config.get('k_neighbors', 5),
                sampling_strategy=balancing_config.get('sampling_strategy', 'auto')
            )
            # 測試集和驗證集不經過 SMOTE
            X_test_balanced, y_test_balanced = X_test_scaled, y_test_encoded
            if X_val_scaled is not None:
                X_val_balanced, y_val_balanced = X_val_scaled, y_val_encoded
            else:
                X_val_balanced, y_val_balanced = None, None
        else:
            # 沒有 SMOTE，直接使用標準化後的數據
            X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
            X_test_balanced, y_test_balanced = X_test_scaled, y_test_encoded
            if X_val_scaled is not None:
                X_val_balanced, y_val_balanced = X_val_scaled, y_val_encoded
            else:
                X_val_balanced, y_val_balanced = None, None
        
        # 保存 raw 版本（從 SMOTE 後的數據 inverse_transform）
        # 訓練集
        if self.scaler is not None:
            X_train_raw = self.scaler.inverse_transform(X_train_balanced)
        else:
            X_train_raw = X_train_balanced
        train_data = pd.DataFrame(X_train_raw, columns=feature_cols)
        train_data[self.target_column] = y_train_balanced
        train_path = os.path.join(output_dir, f"train{client_suffix}.csv")
        train_data.to_csv(train_path, index=False)
        
        # 測試集
        if self.scaler is not None:
            X_test_raw = self.scaler.inverse_transform(X_test_balanced)
        else:
            X_test_raw = X_test_balanced
        test_data = pd.DataFrame(X_test_raw, columns=feature_cols)
        test_data[self.target_column] = y_test_balanced
        test_path = os.path.join(output_dir, f"test{client_suffix}.csv")
        test_data.to_csv(test_path, index=False)
        
        # 驗證集
        if X_val_balanced is not None:
            if self.scaler is not None:
                X_val_raw = self.scaler.inverse_transform(X_val_balanced)
            else:
                X_val_raw = X_val_balanced
            val_data = pd.DataFrame(X_val_raw, columns=feature_cols)
            val_data[self.target_column] = y_val_balanced
            val_path = os.path.join(output_dir, f"val{client_suffix}.csv")
            val_data.to_csv(val_path, index=False)
        else:
            val_path = None
        
        print(f"  - 訓練集: {len(X_train_balanced)} 樣本 → {train_path}")
        print(f"  - 測試集: {len(X_test_balanced)} 樣本 → {test_path}")
        if X_val_balanced is not None:
            print(f"  - 驗證集: {len(X_val_balanced)} 樣本 → {val_path}")
        
        return {
            'train_path': train_path,
            'test_path': test_path,
            'val_path': val_path if X_val_balanced is not None else None,
            'train_scaled_path': train_scaled_path,
            'test_scaled_path': test_scaled_path,
            'val_scaled_path': val_scaled_path if X_val_balanced is not None else None,
            'feature_columns': feature_cols,
            'num_features': len(feature_cols),
            'num_classes': len(self.all_labels),
            'class_labels': self.all_labels
        }
    
    def preprocess_all_data(self, input_dir: str, output_dir: str, num_clients: int = 60):
        """預處理所有數據並生成客戶端數據集"""
        print("🚀 開始完整數據預處理...")
        print(f"  - 輸入目錄: {input_dir}")
        print(f"  - 輸出目錄: {output_dir}")
        print(f"  - 客戶端數量: {num_clients}")
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 🔧 優先使用配置中的 RAW_DATA_PATH
        raw_data_path = getattr(config, 'RAW_DATA_PATH', None)
        if raw_data_path and os.path.exists(raw_data_path):
            merged_file = Path(raw_data_path)
            csv_files = [merged_file]
            print(f"  - 使用配置的數據文件: {merged_file.name}")
        else:
            # ✅ 若已存在上次合併的暫存檔，直接復用（可跳過重新合併）
            temp_merged = Path(output_dir) / "temp_merged_all.csv"
            if temp_merged.exists():
                csv_files = [temp_merged]
                print(f"  - 使用既有暫存合併檔: {temp_merged}")
            else:
                # 查找合併的 CSV 文件或原始數據文件
                merged_file = Path(input_dir) / "CIC-IDS2017_merged.csv"
                if merged_file.exists():
                    csv_files = [merged_file]
                    print(f"  - 使用文件: {merged_file.name}")
                else:
                    # 嘗試查找其他可能的合併文件
                    possible_files = [
                        Path(input_dir) / "CICIDS2017_merged.csv",
                        Path(input_dir) / "merged.csv",
                        Path(input_dir) / "all_data.csv",
                    ]
                    csv_files = []
                    for f in possible_files:
                        if f.exists():
                            csv_files = [f]
                            print(f"  - 使用文件: {f.name}")
                            break
            
            if not csv_files:
                # 🔧 如果沒有合併文件，嘗試合併 train 和 test 目錄的數據
                train_dir = Path(input_dir) / "CICIDS2017" / "train"
                test_dir = Path(input_dir) / "CICIDS2017" / "test"
                
                if train_dir.exists():
                    train_files = list(train_dir.glob("*.csv"))
                    test_files = list(test_dir.glob("*.csv")) if test_dir.exists() else []
                    
                    if train_files:
                        print(f"  - 找到 {len(train_files)} 個訓練文件，{len(test_files)} 個測試文件，開始合併...")
                        # 合併所有訓練文件
                        merged_data = []
                        for train_file in train_files:
                            try:
                                df = pd.read_csv(train_file, low_memory=False)
                                merged_data.append(df)
                                print(f"    - 已讀取訓練: {train_file.name} ({len(df)} 樣本)")
                            except Exception as e:
                                print(f"    ⚠️ 讀取 {train_file.name} 失敗: {e}")
                        
                        # 合併所有測試文件（可選，用於全局測試集）
                        if test_files:
                            for test_file in test_files:
                                try:
                                    df = pd.read_csv(test_file, low_memory=False)
                                    merged_data.append(df)
                                    print(f"    - 已讀取測試: {test_file.name} ({len(df)} 樣本)")
                                except Exception as e:
                                    print(f"    ⚠️ 讀取 {test_file.name} 失敗: {e}")
                        
                        if merged_data:
                            merged_df = pd.concat(merged_data, ignore_index=True)
                            print(f"  - 合併完成: 總共 {len(merged_df)} 樣本")
                            # 保存臨時合併文件
                            temp_merged = Path(output_dir) / "temp_merged_all.csv"
                            merged_df.to_csv(temp_merged, index=False)
                            csv_files = [temp_merged]
                            print(f"  - 臨時合併文件已保存: {temp_merged}")
                        else:
                            print(f"❌ 無法合併客戶端數據文件")
                            return None
                    else:
                        print(f"❌ 在 {train_dir} 中找不到 CSV 文件")
                        return None
                else:
                    # 如果沒有合併文件，嘗試從現有數據重新生成
                    # 檢查是否有已處理的數據可以直接使用
                    existing_processed = Path(output_dir) / "train.csv"
                    if existing_processed.exists():
                        print(f"  - 找到已處理的數據: {existing_processed}")
                        csv_files = [existing_processed]
                    else:
                        print(f"❌ 找不到合併的 CSV 文件")
                        print(f"   請確保輸入目錄包含 CIC-IDS2017_merged.csv 或類似的合併文件")
                        return None
        
        # 處理每個文件
        results = []
        for i, csv_file in enumerate(csv_files):
            result = self.preprocess_single_file(str(csv_file), output_dir, client_id=None)
            if result:
                results.append(result)
        
        if not results:
            print("❌ 沒有成功處理任何文件")
            return None
        
        # 生成客戶端數據集
        print(f"\n🔄 生成 {num_clients} 個客戶端數據集...")
        
        # 讀取全局數據
        global_train = pd.read_csv(results[0]['train_path'])
        global_test = pd.read_csv(results[0]['test_path'])
        
        # 保存全局測試集（未額外處理）
        global_test_path = os.path.join(output_dir, "global_test.csv")
        global_test.to_csv(global_test_path, index=False)
        print(f"  - 全局測試集: {len(global_test)} 樣本 → {global_test_path}")

        # 🔧 修改：global_test_scaled 直接從 test_scaled.csv 讀取（保持 StandardScaler 特性）
        global_test_scaled_path = os.path.join(output_dir, "global_test_scaled.csv")
        try:
            # 直接從 test_scaled.csv 讀取（在 SMOTE 之前保存的標準化版本）
            if 'test_scaled_path' in results[0] and results[0]['test_scaled_path']:
                test_scaled_df = pd.read_csv(results[0]['test_scaled_path'])
                test_scaled_df.to_csv(global_test_scaled_path, index=False)
                print(f"  - 全局測試集（已標準化）: {len(test_scaled_df)} 樣本 → {global_test_scaled_path}")
            else:
                # 回退：從 raw 版本重新轉換
                feature_cols = results[0]['feature_columns']
                X_gt = global_test[feature_cols].copy()
                X_gt_scaled = self.scaler.transform(X_gt) if self.scaler is not None else X_gt.values
                scaled_df = pd.DataFrame(X_gt_scaled, columns=feature_cols)
                scaled_df['label'] = global_test['label'].values
                scaled_df.to_csv(global_test_scaled_path, index=False)
                print(f"  - 全局測試集（已標準化，回退方法）: {len(scaled_df)} 樣本 → {global_test_scaled_path}")
        except Exception as e:
            print(f"  - ⚠️ 產生 global_test_scaled 失敗: {e}，回退使用未標準化版本")
            global_test.to_csv(global_test_scaled_path, index=False)
        
        # 為每個客戶端生成數據 (改進的Non-IID分布)
        client_data_info = []
        
        # 🔧 修復客戶端類別數不匹配問題：確保所有客戶端都有所有類別的數據
        # 首先，確保 label_encoder 已經擬合了所有可能的類別
        all_unique_labels = sorted(global_train['label'].unique())
        if not self.all_labels or len(self.all_labels) != len(all_unique_labels):
            # 重新擬合 label_encoder，確保包含所有類別
            print(f"  - 🔧 重新擬合 label_encoder，確保包含所有類別: {all_unique_labels}")
            self.label_encoder.fit(all_unique_labels)
            self.all_labels = list(self.label_encoder.classes_)
            print(f"  - 學習到的所有標籤: {self.all_labels}")
            print(f"  - 總類別數: {len(self.all_labels)}")
        
        # 使用Dirichlet分佈創建Non-IID數據分布
        # alpha 越大，客戶端分布越接近 IID；這裡改為 1.0 減少極端不平衡
        alpha = 0.5
        # 控制Non-IID程度，越小越不均勻
        unique_labels = all_unique_labels  # 使用所有唯一標籤
        num_labels = len(unique_labels)
        
        print(f"  - 使用Dirichlet分佈 (α={alpha}) 創建Non-IID數據分布")
        print(f"  - 標籤類別: {unique_labels}")
        print(f"  - 總類別數: {num_labels}")
        
        # 🔧 修復：確保每個類別都有最小樣本數，避免出現 0 的情況
        # 增加最小樣本數，確保每個客戶端都有所有類別的數據
        min_samples_per_class = 50  # 🚀 破局配置 3.6：從 20 提高到 50（確保 Batch 有效，解決樣本稀缺下的梯度淹沒）
        
        # 🔧 優化：添加進度顯示
        print(f"  - 開始生成 {num_clients} 個客戶端數據集...")
        start_time = time.time() if 'time' in dir() else None
        try:
            import time
            start_time = time.time()
        except:
            pass
        
        for client_id in range(num_clients):
            # 🔧 優化：每10個客戶端顯示一次進度
            if (client_id + 1) % 10 == 0 or client_id == 0:
                if start_time:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (client_id + 1)
                    remaining = avg_time * (num_clients - client_id - 1)
                    print(f"  - 進度: {client_id + 1}/{num_clients} ({100*(client_id+1)/num_clients:.1f}%), 已用時: {elapsed:.1f}s, 預計剩餘: {remaining:.1f}s")
                else:
                    print(f"  - 進度: {client_id + 1}/{num_clients} ({100*(client_id+1)/num_clients:.1f}%)")
            
            # 為每個客戶端生成標籤分布權重
            np.random.seed(42 + client_id)
            label_weights = np.random.dirichlet([alpha] * num_labels)
            
            # 根據權重採樣數據
            client_samples = []
            total_samples = np.random.randint(500, 2000)  # 每個客戶端500-2000樣本
            
            # 先為每個類別分配最小樣本數
            min_samples_total = min_samples_per_class * num_labels
            remaining_samples = max(0, total_samples - min_samples_total)
            
            for i, label in enumerate(unique_labels):
                label_data = global_train[global_train['label'] == label]
                if len(label_data) > 0:
                    # 🔧 修復：先確保最小樣本數，確保每個客戶端都有所有類別
                    label_count_min = min(min_samples_per_class, len(label_data))
                    
                    # 然後根據權重分配剩餘樣本
                    if remaining_samples > 0:
                        label_count_extra = int(remaining_samples * label_weights[i])
                        label_count_extra = min(label_count_extra, len(label_data) - label_count_min)
                    else:
                        label_count_extra = 0
                    
                    label_count = label_count_min + label_count_extra
                    label_count = min(label_count, len(label_data))
                    
                    # 🔧 修復：確保每個類別至少有1個樣本（如果數據足夠）
                    if label_count == 0 and len(label_data) > 0:
                        label_count = 1  # 至少取1個樣本
                    
                    if label_count > 0:
                        sampled = label_data.sample(n=label_count, random_state=42 + client_id + i)
                        client_samples.append(sampled)
                else:
                    # 🔧 修復：如果某個類別在全局數據中不存在，記錄警告
                    print(f"  ⚠️ 警告：類別 {label} 在全局訓練數據中不存在")
            
            if client_samples:
                client_train = pd.concat(client_samples, ignore_index=True)
                client_train = client_train.sample(frac=1, random_state=42 + client_id).reset_index(drop=True)
            else:
                # 如果沒有採樣到數據，隨機採樣
                client_train = global_train.sample(n=min(1000, len(global_train)), random_state=42 + client_id)
            
            # 保存客戶端數據
            client_dir = os.path.join(output_dir, f"uav{client_id}")
            os.makedirs(client_dir, exist_ok=True)
            
            client_train_path = os.path.join(client_dir, "train.csv")
            client_test_path = os.path.join(client_dir, "test.csv")
            client_train_scaled_path = os.path.join(client_dir, "train_scaled.csv")
            client_test_scaled_path = os.path.join(client_dir, "test_scaled.csv")
            
            # 保存 raw 版本
            client_train.to_csv(client_train_path, index=False)
            # 🔧 可選：避免為每個 client 複製完整測試集（節省磁碟）
            # PREPROCESS_CLIENT_TEST_MODE: full | sample | none
            test_mode = os.environ.get("PREPROCESS_CLIENT_TEST_MODE", "full").strip().lower()
            if test_mode == "none":
                client_test_path = None
                client_test = None
            elif test_mode == "sample":
                sample_n = int(os.environ.get("PREPROCESS_CLIENT_TEST_SAMPLES", "2000"))
                sample_n = max(1, min(sample_n, len(global_test)))
                client_test = global_test.sample(n=sample_n, random_state=42 + client_id)
                client_test.to_csv(client_test_path, index=False)
            else:
                client_test = global_test.copy()
                global_test.to_csv(client_test_path, index=False)
            
            # 🔧 新增：生成標準化版本（使用 scaler transform）
            # 🔧 優化：批量處理標準化，減少重複操作
            try:
                feature_cols = results[0]['feature_columns']
                # 標準化訓練集（向量化操作，直接使用 .values）
                X_train_raw = client_train[feature_cols].values
                X_train_scaled = self.scaler.transform(X_train_raw) if self.scaler is not None else X_train_raw
                train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
                train_scaled_df['label'] = client_train['label'].values
                train_scaled_df.to_csv(client_train_scaled_path, index=False)
                
                # 標準化測試集（如果存在）
                if client_test is not None:
                    X_test_raw = client_test[feature_cols].values
                    X_test_scaled = self.scaler.transform(X_test_raw) if self.scaler is not None else X_test_raw
                    test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
                    test_scaled_df['label'] = client_test['label'].values
                    test_scaled_df.to_csv(client_test_scaled_path, index=False)
            except Exception as e:
                print(f"  ⚠️ 客戶端 {client_id} 生成標準化版本失敗: {e}")
                # 如果標準化失敗，至少確保 raw 版本已保存
            
            # 統計客戶端數據分布
            client_label_dist = client_train['label'].value_counts().to_dict()
            
            client_data_info.append({
                'client_id': client_id,
                'train_path': client_train_path,
                'test_path': client_test_path,
                'train_samples': len(client_train),
                'test_samples': 0 if client_test_path is None else (
                    sample_n if test_mode == "sample" else len(global_test)
                ),
                'label_distribution': client_label_dist,
                'label_weights': label_weights.tolist()
            })
            
            # 進度顯示已移到循環開始處
        
        # 保存預處理器
        preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
        with open(preprocessor_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'all_labels': self.all_labels,
                'feature_columns': results[0]['feature_columns'],
                'config': self.config_data
            }, f)
        
        # 🔧 修復：從實際數據中驗證類別數，確保metadata與實際數據一致
        # 檢查所有客戶端數據中的實際類別
        actual_labels = set()
        for info in client_data_info:
            train_path = info['train_path']
            if os.path.exists(train_path):
                try:
                    df_check = pd.read_csv(train_path, nrows=1000)
                    if 'label' in df_check.columns:
                        actual_labels.update(df_check['label'].unique())
                except:
                    pass
        
        # 如果實際類別數與metadata不一致，使用實際類別數
        actual_num_classes = len(actual_labels) if actual_labels else len(self.all_labels)
        if actual_num_classes != len(self.all_labels):
            print(f"  ⚠️ 警告：實際數據中的類別數 ({actual_num_classes}) 與編碼器類別數 ({len(self.all_labels)}) 不一致")
            print(f"  - 實際標籤: {sorted(actual_labels)}")
            print(f"  - 編碼器標籤: {self.all_labels}")
            # 使用實際類別數
            actual_num_classes = max(actual_num_classes, len(self.all_labels))
        else:
            actual_num_classes = len(self.all_labels)
        
        # 保存元數據
        metadata = {
            'num_clients': num_clients,
            'num_features': results[0]['num_features'],
            'num_classes': actual_num_classes,  # 🔧 使用實際類別數
            'class_labels': self.all_labels,
            'feature_columns': results[0]['feature_columns'],
            'client_data_info': client_data_info,
            'global_test_path': global_test_path,
            'preprocessing_config': self.config_data
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 🔧 新增：為雲端與聚合器寫出統一的特徵欄位與 scaler / label_encoder
        try:
            feature_cols_path = os.path.join(output_dir, "feature_cols.json")
            with open(feature_cols_path, 'w', encoding='utf-8') as f:
                json.dump({'feature_cols': results[0]['feature_columns']}, f, indent=2, ensure_ascii=False)
            print(f"  - 特徵欄位已保存至: {feature_cols_path}")
        except Exception as e:
            print(f"  - ⚠️ 保存 feature_cols.json 失敗: {e}")

        try:
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  - 標準化器已保存至: {scaler_path}")
        except Exception as e:
            print(f"  - ⚠️ 保存 scaler.pkl 失敗: {e}")

        try:
            le_path = os.path.join(output_dir, "label_encoder.pkl")
            with open(le_path, 'wb') as f:
                pickle.dump({'classes_': getattr(self.label_encoder, 'classes_', None)}, f)
            print(f"  - 標籤編碼器已保存至: {le_path}")
        except Exception as e:
            print(f"  - ⚠️ 保存 label_encoder.pkl 失敗: {e}")
        
        print(f"\n✅ 數據預處理完成!")
        print(f"  - 預處理器: {preprocessor_path}")
        print(f"  - 元數據: {metadata_path}")
        print(f"  - 全局測試集: {global_test_path}")
        print(f"  - 客戶端數據集: {num_clients} 個")

        # 🔧 輸出流程摘要報告
        self._write_preprocess_report(output_dir, metadata)
        
        return metadata

def main():
    """主函數"""
    print("🚀 聯邦學習數據預處理工具")
    print("=" * 50)
    
    # 配置路徑
    # 🔧 支援多種數據集格式
    raw_data_path = getattr(config, 'RAW_DATA_PATH', None)
    
    # 允許透過環境變數覆蓋輸入/輸出路徑
    env_input_dir = os.environ.get("PREPROCESS_INPUT_DIR")
    env_output_dir = os.environ.get("PREPROCESS_OUTPUT_DIR")
    
    # 如果 RAW_DATA_PATH 為 None，使用 CICIDS2017_generate_dataset 目錄
    if env_input_dir:
        input_dir = os.path.expanduser(env_input_dir)
        print(f"📂 使用環境變數輸入目錄: {input_dir}")
    elif raw_data_path is None:
        input_dir = "/home/ubuntuv100/uav/CICIDS2017_generate_dataset"
        print(f"📂 使用 CICIDS2017_generate_dataset 目錄（將自動合併客戶端分割文件）")
    elif os.path.exists(raw_data_path):
        input_dir = os.path.dirname(raw_data_path)  # 數據集所在目錄
        print(f"📂 使用配置的數據文件: {raw_data_path}")
    else:
        print(f"❌ 輸入文件不存在: {raw_data_path}")
        print(f"   請檢查 RAW_DATA_PATH 配置或文件路徑")
        # 嘗試使用 CICIDS2017_generate_dataset 作為備選
        input_dir = "/home/ubuntuv100/uav/CICIDS2017_generate_dataset"
        if os.path.exists(input_dir):
            print(f"   將嘗試使用備選目錄: {input_dir}")
        else:
            return
    
    if env_output_dir:
        output_dir = os.path.expanduser(env_output_dir)
    else:
        output_dir = getattr(config, "DATA_PATH", "/home/ubuntuv100/uav/newp1/processed_data")
    print(f"📁 輸出目錄: {output_dir}")
    num_clients = 60  # 客戶端數量（改為60以支持擴展性實驗）
    
    # 創建預處理器
    preprocessor = DataPreprocessor()
    
    # 執行預處理
    metadata = preprocessor.preprocess_all_data(input_dir, output_dir, num_clients)
    
    if metadata:
        print(f"\n📊 預處理統計:")
        print(f"  - 特徵數量: {metadata['num_features']}")
        print(f"  - 類別數量: {metadata['num_classes']}")
        print(f"  - 類別標籤: {metadata['class_labels']}")
        print(f"  - 客戶端數量: {metadata['num_clients']}")
        
        # 顯示客戶端數據分布
        print(f"\n📈 客戶端數據分布:")
        for info in metadata['client_data_info'][:5]:  # 只顯示前5個
            print(f"  - 客戶端 {info['client_id']}: {info['train_samples']} 訓練樣本")
        if len(metadata['client_data_info']) > 5:
            print(f"  - ... 還有 {len(metadata['client_data_info']) - 5} 個客戶端")

if __name__ == "__main__":
    main()
