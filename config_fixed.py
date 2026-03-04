
import os
import math
import logging
from typing import Dict, Any, Optional

def _env_flag(name: str, default: str = "0") -> bool:
    try:
        return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False

def _env_int_or_none(name: str, default: Optional[int] = None) -> Optional[int]:
    try:
        raw = os.environ.get(name, None)
        if raw is None:
            return default
        raw = raw.strip().lower()
        if raw in ("", "none", "null"):
            return None
        val = int(raw)
        return None if val <= 0 else val
    except Exception:
        return default

IS_FEDAVG_BASELINE: bool = os.environ.get("FEDAVG_BASELINE", "0") == "1"


ROUND_CLIENT_LIMIT: Optional[int] = None
SMALL_SCALE_SUMMARY: Dict[str, Any] = {}
SMALL_SCALE_MODE = _env_flag("SMALL_SCALE_MODE", "0")

# 基礎配置
def set_log_level(level="INFO"):
    """設置日誌級別"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

LEARNING_RATE = 1e-3
CLIENT_LR = LEARNING_RATE
SERVER_LR = 2e-4

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "128"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", str(max(TRAIN_BATCH_SIZE, 512))))
BATCH_SIZE = TRAIN_BATCH_SIZE
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "20"))
WEIGHT_DECAY = 1e-4

MODE = "virtual"
NUM_CLIENTS = 60
TOTAL_CLIENTS = 60
NUM_AGGREGATORS = int(os.environ.get("NUM_AGGREGATORS", "6"))
NUM_CLOUD_SERVERS = 1
TRAIN_RATIO = 0.8

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
RAW_DATA_PATH = None
MODEL_PATH = "./model"
LOG_DIR = "result"
IP_FILE = "./ip_list.txt"

GLOBAL_EVAL_MAX_SAMPLES: Optional[int] = _env_int_or_none("GLOBAL_EVAL_MAX_SAMPLES", None)
MODEL_TYPE = "federated"
CLASS_WEIGHTS = None

# 模型配置

MODEL_CONFIG = {
    'type': 'dnn',
    'input_dim': 84,
    'num_classes': 5,
    'output_dim': 5,
    'dropout_rate': 0.3,
    'd_model': 128,
    'num_layers': 2,
    'num_heads': 4,
    'd_ff': None,
    'max_seq_len': 84,
    'use_positional_encoding': True,
    'hidden_dims': [256, 128, 64],
    'use_batch_norm': True,
    'use_residual': True,
    'activation': 'gelu',
    'channels': [32, 64, 128],
    'kernel_sizes': [5, 3, 3],
    'pools': [2, 2, 2],
    'fc_hidden': 128,
    'knowledge_distillation': {
        'enabled': not IS_FEDAVG_BASELINE and not _env_flag("DISABLE_KD", "0"),
        'temperature': 4.0,
        'alpha': float(os.environ.get("KD_ALPHA", "0.1")),
        'loss_type': 'kl_div',
        'use_temperature': True,
        'kd_warmup_rounds': int(os.environ.get("KD_WARMUP_ROUNDS", "10")),
        'diversity_penalty': True,
        'diversity_weight': 1.5,
    }
}

# Client 端 GKD 配置
CLIENT_KD_CONFIG = {
    "enabled": (not IS_FEDAVG_BASELINE) and _env_flag("CLIENT_KD_ENABLED", "1"),
    "alpha": float(os.environ.get("CLIENT_KD_ALPHA", "0.1")),
    "temperature": float(os.environ.get("CLIENT_KD_TEMPERATURE", "4.0")),
    "loss_type": os.environ.get("CLIENT_KD_LOSS_TYPE", "kl_div"),
    "use_temperature": _env_flag("CLIENT_KD_USE_T", "1"),
    "warmup_rounds": int(os.environ.get("CLIENT_KD_WARMUP_ROUNDS", "5")),
}

MODEL_HEAD_PREFIXES = ["output_layer", "classifier"]

# 學習率調度配置
LEARNING_RATE_SCHEDULER = {
    "enabled": True,
    "mode": "manual",
    "scheduler_type": "cosine",
    "cycle_rounds": 25,
    "min_lr": 1e-4,
    "max_lr": 1e-3,
    "warmup_rounds": 3,
    "warmup_factor": 0.5,
    "step_size": 10,
    "gamma": 0.95,
    "f1_based_decay": True,
    "f1_threshold": 0.95,
    "f1_decay_factor": 0.5,
    "f1_high_threshold": 0.97,
    "f1_high_decay_factor": 0.5,
    "f1_ultra_threshold": 0.98,
    "f1_ultra_decay_factor": 0.3,
}

# 聯邦學習配置
FEDERATED_CONFIG = {
    'rounds': MAX_ROUNDS,
    'max_rounds': MAX_ROUNDS,
    'training_flow': 'select_then_train',
    'participation_strategy': {
        'early_rounds': {'threshold': 20, 'ratio': 0.8},
        'mid_rounds': {'threshold': 100, 'ratio': 0.7},
        'late_rounds': {'threshold': 200, 'ratio': 0.6},
        'final_rounds': {'ratio': 0.5}
    },
    'round_sync_interval': 15,
    'round_sync_retry_interval': 15,
    'force_participation_threshold': 3,
    'random_seed_base': 1000,
    'client_wait_time': 0.5,
    'client_check_interval': 3,
    'warmup_full_participation_rounds': 10,
    'upload_max_retries': 30,
    'upload_retry_delay': 2,
    'round_start_max_retries': 3,
    'class_balance': {
        'enabled': False,
        'focus_classes': [],
        'boost_factor': 1.0,
        'min_samples_per_class': 0,
    },
    'aggregation': {
        'smoothing_factor': 0.6
    }
}

# 聚合配置
FEDPROX_CONFIG = {
    "enabled": True,
    "mu": 0.05,
    "adaptive_mu": False,
    "warmup_rounds": 3,
    "mu_schedule": {
        "initial": 0.05,
        "final": 0.02,
        "decay_type": "cosine"
    },
    "high_performance_mu": 0.02,
    "high_performance_threshold": 0.97,
}

REGIONAL_AGGREGATION_CONFIG = {
    "enabled": True,
    "confshield": {
        "enabled": True,
        "norm_threshold": 1.5,
        "cosine_threshold": 0.5,
        "drift_threshold": 0.4,
        "enable_filtering": True,
    },
    "regional_alignment": {
        "enabled": True,
        "alignment_method": "mean",
        "alignment_strength": 0.2,
    },
    "fallback_to_fedavg": True,
}

AGGREGATION_STRATEGY = {
    "type": "hybrid",
    "data_weight": 1.0,
    "performance_weight": 0.0,
    "min_performance": 0.0,
    "warmup_min_performance": 0.0,
    "fallback_min_performance": 0.0,
    "performance_metric": "f1_score",
    "normalize_weights": True,
    "use_adaptive_weights": True,
    "apply_server_lr": True,
    "stability_factor": 0.9,
    "staleness_aware_weighting": {
        "enabled": True,
        "decay_factor": 0.05,
        "max_staleness": 5,
        "min_staleness_weight": 0.3,
    },
    "winner_take_most": {
        "enabled": False,
        "performance_gap_threshold": 0.05,
        "elite_weight_ratio": 0.75,
        "min_elite_weight": 0.60,
    },
    "top_k_selection": {
        "enabled": False,
        "top_k_ratio": 0.2,
        "min_clients": 3,
        "apply_after_round": 30,
    },
    "gradient_consistency": {
        "enabled": False,
        "cosine_similarity_threshold": 0.3,
        "exclude_opposite": True,
        "weight_penalty_factor": 0.1,
    },
    "quality_gate": {
        "enabled": False,
        "f1_drop_threshold": 0.15,
        "quick_eval_samples": 5000,
        "fallback_to_best_agg": True,
        "fusion_ratio": 0.7,
    }
}

AGGREGATION_CONFIG = {
    "weight_norm_regularization": {
        "enabled": True,
        "max_global_l2_norm": 150.0,
        "hard_limit": 200.0,
        "use_dynamic_hard_limit": True,
        "stable_norm_window": 10,
        "stable_norm_multiplier": 1.5,
        "scaling_factor": 0.90,
        "warn_threshold": 120.0,
        "strict_enforcement": True,
    },
    "elite_weight_projection": {
        "enabled": True,
        "max_distance": 0.2,
        "method": "cosine",
        "strength": 0.7,
        "min_best_f1": 0.2,
        "enable_f1_based_protection": True,
        "f1_drop_threshold": 0.20,
        "observation_period": 3,
    },
    "aggregation_method": {
        "type": "mean",
        "trim_ratio": 0.2,
        "use_performance_weighted_mean": False,
    },
    "weight_update_condition": {
        "use_numerical_comparison": True,
        "l2_distance_threshold": 1e-4,
        "key_layers": ["output_layer.weight", "layers.0.weight", "input_reshape.weight"],
    },
    'min_clients_for_aggregation': 3,
    'min_clients_absolute': 3,
    'max_wait_time': 300,
    'min_training_duration': 30,
    'min_participation_ratio': 0.4,
    'force_aggregation_after_timeout': True,
    'partial_aggregation_enabled': True,
    'min_partial_ratio': 0.4,
    'aggregator_quorum': 3,
    'min_aggregators_for_global': 2,
    'stale_policy': 'decay_then_drop',
    'max_staleness': 5,
    'staleness_decay_lambda': 1.1,
    'robust_aggregator': 'clip_then_avg',
    'clip_norm_max': 5.0,
    'clip_norm_eps': 1e-6,
    'trim_ratio': 0.0,
    'avg_by': 'data_size',
    'class_balance_factor': 1.0,
    'bn_aggregation_mode': 'affine_with_running',
    'allow_late_upload_buffer': True,
    'late_upload_max_round_lag': 3,
    'future_round_tolerance': 30,
    'fedprox': {
        'enabled': not _env_flag("DISABLE_FEDPROX", "0"),
        'mu': 0.01,
        'adaptive_mu': False,
        'target_drift': 0.1,
        'aggregation_method': 'weighted_avg'
    },
    'fednova': {
        'enabled': False,
        'aggregation_method': 'weighted_avg'
    },
    'server_ema': {
        'enabled': not _env_flag("DISABLE_SERVER_EMA", "0"),
        'decay': 0.95
    },
    'cloud_async': {
        'enabled': True
    },
    'quality_quorum': {
        'enabled': False,
        'non_blocking': True,
        'metric': 'delta_macro_f1',
        'min_delta': 0.0,
        'min_pass': 0,
        'weight_strength': 0.0
    },
    'dual_weighting': {
        'enabled': False,
        'beta_data': 1.0,
        'beta_pred': 1.0,
        'alpha_clip': 0.5
    },
    'server_momentum': {
        'enabled': False,
        'momentum': 0.9,
        'nesterov': False
    },
    'fedbn': {
        'enabled': False,
        'layers_to_exclude': [],
    }
}

SERVER_TRAINING_CONFIG = {
    "enabled": not IS_FEDAVG_BASELINE,
}

# 個人化 / 聯合預測配置
JOINT_PREDICTION_ALPHA_LOCAL = 0.5
JOINT_PREDICTION_ALPHA_AGGREGATOR = 0.25
JOINT_PREDICTION_ALPHA_GLOBAL = 0.25
PERSONALIZATION_ALPHA = 0.5

# 優化器配置
OPTIMIZER_CONFIG = {
    "type": "adam",
    "lr": LEARNING_RATE,
    "weight_decay": 2e-4,
    "momentum": 0.9,
    "beta1": 0.8,
    "beta2": 0.99,
    "eps": 1e-6
}

# 損失函數配置
LOSS_CONFIG = {
    "type": "cross_entropy",
    "reduction": "mean",
    "label_smoothing": 0.1,
    "label_smoothing_majority": 0.2,
    "focal_loss": True,
    "focal_alpha": 0.25,
    "focal_gamma": 2.5,
    "class_weights_enabled": True,
    "dynamic_class_weights": True,
    "class_weight_strategy": "adaptive_enhanced",
    "force_all_classes": True,
    "min_class_weight": 1.5,
    "max_class_weight": 1000.0,
    "zero_f1_boost": 3.0,
    "client_diversity_penalty_enabled": True,
    "client_diversity_weight": 0.1,
    "rare_class_threshold": 0.1,
    "rare_class_boost": 2.0,
    "late_ce_enabled": False,
    "late_ce_round": 0,
    "late_ce_temperature": 1.0
}

# 動態 Class Weight（Cloud Server 每輪評估後寫出，Client 下輪讀取）
DYNAMIC_CLASS_WEIGHTING = {
    "enabled": True,
    "filename": "dynamic_class_weights.json",
    "eps": 1e-3,
    "normalize_mean1": False,
    "ema_beta": 0.85,
    "min_weight": 0.5,
    "max_weight": 5.0,
    "extra_boost_top_k": 2,
    "extra_boost_factor": 1.15,
    "extra_boost_min_f1": 0.95,
    "use_sliding_window": True,
    "sliding_window_size": 3,
    "update_frequency": 1,
}

# 客戶端訓練配置
CLIENT_TRAINING_CONFIG = {
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "gradient_clipping": 2.0,
    "early_stopping": False,
    "patience": 0,
    "min_delta": 0.0
}

# 梯度噪聲配置
GRAD_NOISE_CONFIG = {
    "enabled": False,
    "base_sigma": 0.01,
    "decay": 0.0,
    "min_sigma": 0.0,
    "max_sigma": 0.05,
    "mode": "adaptive",
    "ref_grad_norm": 1.0,
    "log_interval": 50
}

# 客戶端選擇配置
CLIENT_SELECTION_CONFIG = {
    "strategy": "random",
    "base_participation_ratio": 0.6,
    "min_participation_ratio": 0.5,
    "max_participation_ratio": 0.7,
    "performance_threshold": 0.001,
    "improvement_threshold": 0.0001,
    "consecutive_failure_limit": 2,
    "health_check_interval": 2,
    "fairness_weight": 0.5,
    "performance_weight": 0.5,
    "health_skip": {
        "enabled": True,
        "min_accuracy": 0.1,
        "max_loss": 3.0,
        "cooldown_rounds": 3
    }
}

# 數據配置
DATA_CONFIG = {
    "train_ratio": 0.8,
    "test_ratio": 0.2,
    "validation_ratio": 0.1,
    "random_state": 42,
    "shuffle": True,
    "stratify": True,
    "augmentation": True,
    "normalization": "standard",
    "feature_selection": False,
    "data_usage_ratio": 1.0,
    "max_train_samples": 500,
    "max_test_samples": 1000,
    "balancing": {
        "enabled": True,
        "method": "borderline_smote",
        "k_neighbors": 3,
        "sampling_strategy": "auto",
        "random_state": 42
    },
    "preprocessing": {
        "scaler_type": "standard",
        "handle_missing": "drop",
        "outlier_detection": True,
        "outlier_method": "isolation_forest",
        "outlier_contamination": 0.1,
        "outlier_threshold": 0.1,
        "feature_scaling": True
    }
}

# GAN 生成式增強
GAN_AUGMENTATION_CONFIG = {
    "enabled": False,
    "target_label": 1,
    "ratio": 0.1,
    "target_class_ratio": 0.1,
    "min_class_ratio": 0.05,
    "max_new_samples": 0,
    "latent_dim": 32,
    "generator_path": "",
    "device": "auto"
}

# 網絡配置
NETWORK_CONFIG = {
    'cloud_server': {
        'host': '127.0.0.1',
        'port': 8083,
        'url': 'http://127.0.0.1:8083'
    },
    'aggregators': {
        'base_port': 8000,
        'host': '127.0.0.1',
        'ports': [8000, 8001, 8002, 8003, 8004, 8005]
    },
    'clients': {
        'host': '127.0.0.1',
        'base_port': 9000
    },
    'registration_delay': 3,
}

# 標籤配置
LABEL_COL = "label"
ALL_LABELS = [
    "BENIGN",
    "DDoS",
    "DoS_Hulk",
    "DoS_Slowhttptest",
    "DoS_slowloris"
]
ALL_TYPE_NAMES = ALL_LABELS
LABEL_MAPPING = {
    0: "BENIGN",
    1: "DDoS",
    2: "DoS_Hulk",
    3: "DoS_Slowhttptest",
    4: "DoS_slowloris",
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# 日誌配置
LOG_CONFIG = {
    "result_log_format": "csv",
    "unified_naming": True,
    "result_dir": "result",
    "show_client_details": True,
    "show_evaluation_details": True,
    "show_error_details": True,
    "show_round_summary": True,
    "show_aggregator_status": True
}

# 攻擊模擬配置
ATTACK_CONFIG = {
    "enabled": _env_flag("ATTACK_ENABLED", "0"),
    "malicious_ratio": float(os.environ.get("MALICIOUS_RATIO", "0.0")),
    "malicious_clients": os.environ.get("MALICIOUS_CLIENTS", "").strip(),
    "seed": int(os.environ.get("MALICIOUS_SEED", "42")),
    "label_flipping": {
        "enabled": _env_flag("LABEL_FLIP_ENABLED", "0"),
        "source_label": os.environ.get("LABEL_FLIP_SOURCE", "DDoS"),
        "target_label": os.environ.get("LABEL_FLIP_TARGET", "BENIGN"),
    },
    "model_poisoning": {
        "enabled": _env_flag("MODEL_POISON_ENABLED", "0"),
        "method": os.environ.get("MODEL_POISON_METHOD", "gaussian"),
        "sigma": float(os.environ.get("MODEL_POISON_SIGMA", "0.5")),
        "replace_prob": float(os.environ.get("MODEL_POISON_REPLACE_PROB", "1.0"))
    }
}

# ConfShield 權重異常偵測（action: monitor | soft | hard）
SECURITY_CONFIG = {
    "dbi_weight_anomaly": {
        "enabled": True,
        "pca_dim": 32,
        "cluster_k": 2,
        "min_cluster_ratio": 0.1,
        "distance_threshold": 1.5,
        "action": "monitor",
        "soft_factor": 0.3,
        "log_top_k": 5,
    }
}

# 梯度配置
GRADIENT_CLIPPING = {
    "enabled": True,
    "max_norm": 1.0,
    "norm_type": 2,
    "adaptive": True,
}

# 評估配置
EVAL_CONFIG = {
    'benign_logit_bias': 0.0,
    'eval_every_rounds': 2
}
EVAL_EVERY_ROUNDS = EVAL_CONFIG['eval_every_rounds']

# 收斂配置
CONVERGENCE_CONFIG = {
    'max_rounds': 25,
    'patience': 10,
    'min_improvement': 0.0005,
    'min_rounds': 15
}

# 大幅下滑保護
DROP_GUARD_CONFIG = {
    "enabled": True,
    "min_rounds": 15,
    "drop_threshold": 0.08,
    "drop_patience": 2
}

# Prototype Loss 配置（DISABLE_PROTOTYPE=1 / ENABLE_PROTOTYPE_ONLY=1 可覆寫）
PROTOTYPE_LOSS_CONFIG = {
    "enabled": _env_flag("ENABLE_PROTOTYPE_ONLY", "0") and not _env_flag("DISABLE_PROTOTYPE", "0"),
    "lambda_proto": 0.02,
    "path": os.path.join("model", "global_prototypes.npy"),
    "log_every_n_batches": 100,
    "schedule": {
        "enabled": True,
        "mode": "cosine",
        "max_rounds": 20,
        "start_lambda": 0.05,
        "end_lambda": 0.0
    }
}

# 全域類別權重配置
CLASS_WEIGHT_CONFIG = {
    "enabled": False,
    "path": os.path.join("model", "global_class_weights.npy"),
    "normalize": True
}

# 性能配置
PERFORMANCE_CONFIG = {
    "use_gpu": False,
    "num_workers": 8,
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "async_loading": False,
    "compression": False,
    "quantization": False
}

# Split Learning 配置
SPLIT_LEARNING_CONFIG = {
    'enabled': False,
    'compression': {'enabled': False},
    'disable_cloud_training': True
}

# 實驗配置
EXPERIMENT_CONFIG = {
    "name": "optimized_federated_learning",
    "description": "優化的聯邦學習配置",
    "version": "4.0",
    "tags": ["optimized", "cleaned", "efficient"],
    "save_config": True,
    "track_metrics": True,
    "early_stopping": False,
    "convergence_check": False
}


def _apply_small_scale_overrides():
    """啟用小規模模式時調整系統參數"""
    global NUM_AGGREGATORS, NUM_CLIENTS, TOTAL_CLIENTS, MAX_ROUNDS, ROUND_CLIENT_LIMIT, SMALL_SCALE_SUMMARY
    if not SMALL_SCALE_MODE:
        SMALL_SCALE_SUMMARY = {}
        ROUND_CLIENT_LIMIT = None
        return
    try:
        small_num_aggs = max(1, int(os.environ.get("SMALL_SCALE_NUM_AGGREGATORS", "2")))
        small_num_clients = max(small_num_aggs, int(os.environ.get("SMALL_SCALE_NUM_CLIENTS", "9")))
        small_rounds = max(1, int(os.environ.get("SMALL_SCALE_MAX_ROUNDS", "10")))
        small_clients_per_round = max(1, int(os.environ.get("SMALL_SCALE_CLIENTS_PER_ROUND", "3")))
        small_wait_time = max(60, int(os.environ.get("SMALL_SCALE_MAX_WAIT", "180")))
        
        agg_cfg = NETWORK_CONFIG.get('aggregators', {})
        ports = list(agg_cfg.get('ports', []))
        if ports:
            NUM_AGGREGATORS = max(1, min(NUM_AGGREGATORS, small_num_aggs, len(ports)))
            NETWORK_CONFIG['aggregators']['ports'] = ports[:NUM_AGGREGATORS]
        else:
            NUM_AGGREGATORS = max(1, min(NUM_AGGREGATORS, small_num_aggs))
        
        NUM_CLIENTS = max(NUM_AGGREGATORS, min(NUM_CLIENTS, small_num_clients))
        TOTAL_CLIENTS = NUM_CLIENTS
        MAX_ROUNDS = min(MAX_ROUNDS, small_rounds)
        FEDERATED_CONFIG['rounds'] = min(FEDERATED_CONFIG.get('rounds', MAX_ROUNDS), MAX_ROUNDS)
        FEDERATED_CONFIG['max_rounds'] = FEDERATED_CONFIG['rounds']
        
        clients_per_aggregator = max(1, math.ceil(NUM_CLIENTS / NUM_AGGREGATORS))
        ROUND_CLIENT_LIMIT = min(small_clients_per_round, clients_per_aggregator)
        target_ratio = min(1.0, ROUND_CLIENT_LIMIT / clients_per_aggregator)
        
        AGGREGATION_CONFIG['min_clients_for_aggregation'] = ROUND_CLIENT_LIMIT
        AGGREGATION_CONFIG['min_clients_absolute'] = ROUND_CLIENT_LIMIT
        AGGREGATION_CONFIG['min_participation_ratio'] = target_ratio
        AGGREGATION_CONFIG['max_wait_time'] = min(AGGREGATION_CONFIG.get('max_wait_time', 420), small_wait_time)
        AGGREGATION_CONFIG['min_training_duration'] = min(AGGREGATION_CONFIG.get('min_training_duration', 45), 30)
        
        CLIENT_SELECTION_CONFIG['base_participation_ratio'] = target_ratio
        CLIENT_SELECTION_CONFIG['min_participation_ratio'] = target_ratio
        CLIENT_SELECTION_CONFIG['max_participation_ratio'] = target_ratio
        
        total_rounds = FEDERATED_CONFIG['rounds']
        early_threshold = min(total_rounds, 3)
        mid_threshold = min(total_rounds, 6)
        late_threshold = min(total_rounds, 9)
        FEDERATED_CONFIG['participation_strategy'] = {
            'early_rounds': {'threshold': early_threshold, 'ratio': target_ratio},
            'mid_rounds': {'threshold': mid_threshold, 'ratio': target_ratio},
            'late_rounds': {'threshold': late_threshold, 'ratio': target_ratio},
            'final_rounds': {'ratio': target_ratio}
        }
        
        SMALL_SCALE_SUMMARY = {
            "enabled": True,
            "num_aggregators": NUM_AGGREGATORS,
            "num_clients": NUM_CLIENTS,
            "max_rounds": FEDERATED_CONFIG['rounds'],
            "clients_per_round_limit": ROUND_CLIENT_LIMIT
        }
    except Exception as exc:
        SMALL_SCALE_SUMMARY = {"error": str(exc)}
        ROUND_CLIENT_LIMIT = None


_apply_small_scale_overrides()

def validate_config():
    """驗證配置一致性"""
    assert LEARNING_RATE > 0, "學習率必須大於0"
    assert BATCH_SIZE > 0, "批次大小必須大於0"
    assert LOCAL_EPOCHS > 0, "本地訓練輪數必須大於0"
    assert MAX_ROUNDS > 0, "最大輪數必須大於0"
    assert NUM_CLIENTS > 0, "客戶端數量必須大於0"
    assert NUM_AGGREGATORS > 0, "聚合器數量必須大於0"
    print("配置驗證通過")

print("優化後的聯邦學習配置已加載")
print(f"核心參數: 學習率={LEARNING_RATE}, 服務器LR={SERVER_LR}, batch={BATCH_SIZE}, "
      f"local_epochs={LOCAL_EPOCHS}, max_rounds={MAX_ROUNDS}, clients={NUM_CLIENTS}, aggregators={NUM_AGGREGATORS}")
if SMALL_SCALE_MODE:
    print(f"小規模模式啟用: {SMALL_SCALE_SUMMARY}")

validate_config()