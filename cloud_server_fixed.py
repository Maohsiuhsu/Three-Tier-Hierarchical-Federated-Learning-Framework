from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import datetime
import config_fixed as config
import uvicorn
from fastapi.responses import Response, JSONResponse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import hashlib
import concurrent.futures
import asyncio
import json
import numpy as np
import time
import threading
import pickle
import os
import math
import traceback

try:
    import psutil
except ImportError:
    print("[Cloud Server] 警告：psutil 未安裝，系統資源監控功能將不可用")
    psutil = None

try:
    import aiohttp
except ImportError:
    print("[Cloud Server] 錯誤：aiohttp 未安裝，廣播功能將不可用")
    aiohttp = None

try:
    import pandas as pd
except ImportError:
    print("[Cloud Server] 錯誤：pandas 未安裝，無法讀取 CSV 文件")
    pd = None

try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("[Cloud Server] 錯誤：sklearn 未安裝，評估功能將不可用")
    accuracy_score = f1_score = classification_report = confusion_matrix = None
    StandardScaler = None

try:
    import requests
except ImportError:
    print("[Cloud Server] 錯誤：requests 未安裝，聚合器狀態輪詢功能將不可用")
    requests = None

try:
    import torch
except ImportError:
    print("[Cloud Server] 錯誤：torch 未安裝，權重處理功能將不可用")
    torch = None

_GLOBAL_TEST_CACHE = {"path": None, "df": None}
_GLOBAL_TEST_CACHE_LOCK = threading.Lock()

def perform_federated_averaging():
    global aggregator_weights, global_weights, PREVIOUS_GLOBAL_WEIGHTS
    global needs_rollback_flag, rollback_reason_str, BEST_GLOBAL_WEIGHTS, BEST_ROUND_ID, BEST_GLOBAL_F1
    global STABLE_GLOBAL_WEIGHTS, STABLE_ROUND_ID
    global STABILITY_CHECK_FAILURE_COUNT, STABILITY_CHECK_FAILURE_THRESHOLD
    global last_peak_round, PEAK_PROTECTION_ENABLED, PEAK_PROTECTION_ROUNDS
    global ROLLBACK_COUNT, LAST_ROLLBACK_ROUND, POST_ROLLBACK_TRUST_ALPHA, POST_ROLLBACK_ROUNDS
    global CURRENT_SERVER_LR_MULTIPLIER, CURRENT_FEDPROX_MU_MULTIPLIER
    global MIN_SERVER_LR_MULTIPLIER, MAX_FEDPROX_MU_MULTIPLIER, ROLLBACK_STABLE_ROUNDS
    global BEST_MODELS_HISTORY, TOP_N_BEST_MODELS
    global COSINE_SIMILARITY_THRESHOLD, WEIGHT_NORM_EXPLOSION_THRESHOLD
    global SOFT_ROLLBACK_F1_DROP_THRESHOLD, HARD_ROLLBACK_F1_DROP_THRESHOLD
    global HIGH_F1_STABLE_ROUNDS
    global ENABLE_ROLLBACK_MECHANISM
    
    round_upload_bytes = 0
    round_upload_mb = 0.0

    current_round = None
    try:
        import cloud_server_fixed
        if hasattr(cloud_server_fixed, 'app'):
            app_obj = cloud_server_fixed.app
            if hasattr(app_obj, 'state') and hasattr(app_obj.state, 'current_round'):
                current_round = app_obj.state.current_round
            elif hasattr(app_obj, 'state') and hasattr(app_obj.state, 'last_aggregation_round'):
                current_round = app_obj.state.last_aggregation_round
    except:
        pass

    soft_rollback_triggered = False
    if BEST_GLOBAL_F1 > 0 and rollback_reason_str:
        pass
    
    if needs_rollback_flag and BEST_GLOBAL_WEIGHTS is not None and ENABLE_ROLLBACK_MECHANISM:
        if LAST_ROLLBACK_ROUND is None or (current_round is not None and current_round > LAST_ROLLBACK_ROUND + POST_ROLLBACK_ROUNDS):
            LAST_ROLLBACK_ROUND = current_round
            ROLLBACK_STABLE_ROUNDS = 0
        else:
            ROLLBACK_STABLE_ROUNDS = 0
        ROLLBACK_COUNT += 1
        print(
            f"[Cloud Server] 聚合前回退：檢測到回退標記，回退到最佳模型 "
            f"(round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f}, 原因: {rollback_reason_str})"
        )
        print(f"[Cloud Server] 回退計數：連續回退 {ROLLBACK_COUNT}/{MAX_CONSECUTIVE_ROLLBACKS} 次")
        global BEST_MODELS_HISTORY, TOP_N_BEST_MODELS
        if len(BEST_MODELS_HISTORY) > 1:
            import random
            if random.random() < 0.7:
                selected_model = BEST_MODELS_HISTORY[0]
                print(f"[Cloud Server] 多樣性緩衝：使用最佳模型 (round={selected_model[0]}, f1={selected_model[1]:.4f})")
            else:
                selected_model = random.choice(BEST_MODELS_HISTORY[:min(3, len(BEST_MODELS_HISTORY))])
                print(f"[Cloud Server] 多樣性緩衝：隨機選擇模型 (round={selected_model[0]}, f1={selected_model[1]:.4f})")
            global_weights = {k: _coerce_tensor(v).clone() for k, v in selected_model[2].items()}
        else:
            global_weights = {k: _coerce_tensor(
                v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
        
        global MIN_SERVER_LR_MULTIPLIER, MAX_FEDPROX_MU_MULTIPLIER
    
        if ROLLBACK_COUNT >= MAX_CONSECUTIVE_ROLLBACKS:
            print(f"[Cloud Server] 連續回退 {ROLLBACK_COUNT} 次，調整策略：")
            CURRENT_SERVER_LR_MULTIPLIER *= 0.1
            CURRENT_FEDPROX_MU_MULTIPLIER *= 2.0
            CURRENT_SERVER_LR_MULTIPLIER = max(MIN_SERVER_LR_MULTIPLIER, CURRENT_SERVER_LR_MULTIPLIER)
            CURRENT_FEDPROX_MU_MULTIPLIER = min(MAX_FEDPROX_MU_MULTIPLIER, CURRENT_FEDPROX_MU_MULTIPLIER)
            print(f"[Cloud Server]   - SERVER_LR 乘數: {CURRENT_SERVER_LR_MULTIPLIER:.4f} (下限保護: {MIN_SERVER_LR_MULTIPLIER:.4f})")
            print(f"[Cloud Server]   - FedProx μ 乘數: {CURRENT_FEDPROX_MU_MULTIPLIER:.4f} (上限保護: {MAX_FEDPROX_MU_MULTIPLIER:.4f})")
        else:
            CURRENT_SERVER_LR_MULTIPLIER *= 0.2
            CURRENT_FEDPROX_MU_MULTIPLIER *= 1.5
            CURRENT_SERVER_LR_MULTIPLIER = max(MIN_SERVER_LR_MULTIPLIER, CURRENT_SERVER_LR_MULTIPLIER)
            CURRENT_FEDPROX_MU_MULTIPLIER = min(MAX_FEDPROX_MU_MULTIPLIER, CURRENT_FEDPROX_MU_MULTIPLIER)
            print(f"[Cloud Server]   - SERVER_LR 乘數: {CURRENT_SERVER_LR_MULTIPLIER:.4f} (回退後降低，下限保護: {MIN_SERVER_LR_MULTIPLIER:.4f})")
            print(f"[Cloud Server]   - FedProx μ 乘數: {CURRENT_FEDPROX_MU_MULTIPLIER:.4f} (回退後增加，上限保護: {MAX_FEDPROX_MU_MULTIPLIER:.4f})")

        needs_rollback_flag = False
        rollback_reason_str = ""
        HIGH_F1_STABLE_ROUNDS = 0
        print(f"[Cloud Server] 已回退到最佳模型權重 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})，將使用此權重進行聚合（信任比例: {POST_ROLLBACK_TRUST_ALPHA:.1%}）")

    if torch is None:
        print("[Cloud Server] 錯誤：torch 未安裝，無法執行聚合")
        return {}
    
    if not aggregator_weights:
        print("[Cloud Server] 沒有聚合器權重，返回空權重")
        try:
            if hasattr(app, 'state'):
                app.state.round_upload_mb = 0.0
                app.state.round_download_mb = 0.0
                app.state.round_total_mb = 0.0
            import os
            os.environ['ROUND_UPLOAD_MB'] = '0.0'
            os.environ['ROUND_DOWNLOAD_MB'] = '0.0'
            os.environ['ROUND_TOTAL_MB'] = '0.0'
        except Exception:
            pass
        return {}
    
    all_weights = []
    aggregator_weights_list = []
    total_data_size = 0
    performance_scores = []
    
    for agg_id, weights_list in aggregator_weights.items():
        if not weights_list:
            continue
        candidate_weights = None
        round_tolerance = 2 if ROLLBACK_COUNT >= MAX_CONSECUTIVE_ROLLBACKS else 1
        round_gap = 0
        if current_round is not None:
            round_candidates = [w for w in weights_list if w.get('round_id') == current_round]
            if round_candidates:
                candidate_weights = round_candidates[-1]
                round_gap = 0
            else:
                nearby_candidates = [w for w in weights_list 
                                   if abs(w.get('round_id', 0) - current_round) <= round_tolerance]
                if nearby_candidates:
                    nearby_candidates.sort(key=lambda w: abs(w.get('round_id', 0) - current_round))
                    candidate_weights = nearby_candidates[0]
                    candidate_round = candidate_weights.get('round_id', None)
                    round_gap = abs(candidate_round - current_round)
                    print(f"[Cloud Server] 聚合器 {agg_id} 使用相近輪次 {candidate_round} 的權重（目標輪次: {current_round}，誤差: {round_gap}）")
        
        if candidate_weights is None:
            candidate_weights = weights_list[-1]
            if current_round is not None:
                candidate_round = candidate_weights.get('round_id', None)
                round_gap = abs(candidate_round - current_round)
                if round_gap > round_tolerance:
                    print(f"[Cloud Server] 警告：聚合器 {agg_id} 沒有輪次 {current_round} 或相近輪次的權重，使用輪次 {candidate_round} 的權重（誤差: {round_gap}）")

        if current_round is not None and round_gap > round_tolerance:
            print(f"[Cloud Server] 跳過聚合器 {agg_id}：輪次差距 {round_gap} > 容忍 {round_tolerance}（避免舊權重污染）")
            continue
        
        aggregated_weights = candidate_weights.get('aggregated_weights', {})
        data_size = candidate_weights.get('data_size', 1000)
        aggregation_stats = candidate_weights.get('aggregation_stats', {})
        aggregator_weights_list.append(
            (agg_id, aggregated_weights, aggregation_stats, data_size, round_gap))
        accuracy = aggregation_stats.get('accuracy', 0.5)
        f1_score = aggregation_stats.get('f1_score', 0.5)
        performance_score = (accuracy + f1_score) / 2
        max_pred_ratio = aggregation_stats.get('max_pred_ratio', None)
        base_min_perf = getattr(config, "AGGREGATION_STRATEGY", {}).get(
            "min_performance", 0.0)
        warmup_min_perf = getattr(config, "AGGREGATION_STRATEGY", {}).get(
            "warmup_min_performance", 0.20)
        current_round = getattr(app.state, "current_round", None)
        if current_round is None and hasattr(app.state, "last_aggregation_round"):
            current_round = app.state.last_aggregation_round
        effective_min_perf = (
            warmup_min_perf if (current_round is not None and current_round <= 5) else base_min_perf
        )
        if performance_score < effective_min_perf:
            print(
                f"[Cloud Server] 跳過聚合器 {agg_id}: 性能分數 "
                f"{performance_score:.4f} < 門檻 {effective_min_perf:.2f}"
            )
            continue
    
        if aggregated_weights:
            freshness_factor = max(0.5, 1.0 - 0.15 * round_gap)
            all_weights.append({
                'weights': aggregated_weights,
                'data_size': data_size,
                'performance_score': performance_score,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'agg_id': agg_id,
                'round_gap': round_gap,
                'freshness_factor': freshness_factor,
                'max_pred_ratio': max_pred_ratio,
            })
            total_data_size += data_size
            performance_scores.append(performance_score)
    
    if not all_weights:
        fallback_min_perf = getattr(config, "AGGREGATION_STRATEGY", {}).get(
            "fallback_min_performance", 0.3)
        print(f"[Cloud Server] 全部聚合器被過濾，降門檻到 {fallback_min_perf:.2f} 重試")
        for item in aggregator_weights_list:
            if len(item) == 5:
                agg_id, aggregated_weights, aggregation_stats, data_size, round_gap = item
            else:
                agg_id, aggregated_weights, aggregation_stats, data_size = item[:4]
                round_gap = 0
            accuracy = aggregation_stats.get('accuracy', 0.5)
            f1_score = aggregation_stats.get('f1_score', 0.5)
            performance_score = (accuracy + f1_score) / 2
            max_pred_ratio = aggregation_stats.get('max_pred_ratio', None)
            if performance_score < fallback_min_perf:
                continue
            if aggregated_weights:
                freshness_factor = max(0.5, 1.0 - 0.15 * round_gap)
                all_weights.append({
                    'weights': aggregated_weights,
                    'data_size': data_size,
                    'performance_score': performance_score,
                    'accuracy': accuracy,
                    'f1_score': f1_score,
                    'agg_id': agg_id,
                    'round_gap': round_gap,
                    'freshness_factor': freshness_factor,
                    'max_pred_ratio': max_pred_ratio,
                })
                total_data_size += data_size
                performance_scores.append(performance_score)
        if not all_weights:
            print("[Cloud Server] 沒有有效的聚合器權重（含降門檻重試）")
        return {}

    print(f"[Cloud Server] 執行權重異常檢測過濾器（事前防禦）...")
    valid_weights = []
    rejected_count = 0
    
    reference_weights = None
    if BEST_GLOBAL_WEIGHTS is not None:
        reference_weights = BEST_GLOBAL_WEIGHTS
    elif global_weights is not None and len(global_weights) > 0:
        reference_weights = global_weights
    if reference_weights is not None:
        ref_norm = _compute_global_l2_norm(reference_weights)
        print(f"[Cloud Server] 參考權重範數: {ref_norm:.4f}")
        for agg_data in all_weights:
            agg_id = agg_data['agg_id']
            weights = agg_data['weights']
            cosine_sim = _compute_weight_vector_cosine_similarity(weights, reference_weights)
            if cosine_sim < COSINE_SIMILARITY_THRESHOLD:
                print(f"[Cloud Server] 拒絕 Aggregator {agg_id}: 權重方向偏離過大 (餘弦相似度: {cosine_sim:.4f} < {COSINE_SIMILARITY_THRESHOLD:.2f})")
                rejected_count += 1
                continue
            agg_norm = _compute_global_l2_norm(weights)
            if ref_norm > 0:
                norm_ratio = agg_norm / ref_norm
                if norm_ratio > WEIGHT_NORM_EXPLOSION_THRESHOLD:
                    print(f"[Cloud Server] 拒絕 Aggregator {agg_id}: 權重範數異常 (範數比: {norm_ratio:.2f} > {WEIGHT_NORM_EXPLOSION_THRESHOLD:.2f})")
                    rejected_count += 1
                    continue
                elif norm_ratio > 1.5:
                    scale_factor = 1.5 / norm_ratio
                    print(f"[Cloud Server] Aggregator {agg_id} 權重範數過大 ({norm_ratio:.2f}x)，縮放 {scale_factor:.2f} 倍")
                    for key in weights.keys():
                        w = _coerce_tensor(weights[key])
                        if isinstance(w, torch.Tensor):
                            weights[key] = w * scale_factor
                    agg_data['weights'] = weights
            
            agg_data['cosine_similarity'] = cosine_sim
            valid_weights.append(agg_data)
            print(f"[Cloud Server] Aggregator {agg_id} 通過檢查: 餘弦相似度={cosine_sim:.4f}, 性能={agg_data['performance_score']:.4f}")
    else:
        print(f"[Cloud Server] 警告：沒有參考權重，跳過異常檢測過濾")
        valid_weights = all_weights
    
    if not valid_weights:
        print(f"[Cloud Server] 所有聚合器都被過濾（拒絕 {rejected_count}/{len(all_weights)} 個），觸發回退")
        needs_rollback_flag = True
        rollback_reason_str = "all_aggregators_rejected_by_filter"
        if BEST_GLOBAL_WEIGHTS is not None:
            return {k: _coerce_tensor(v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
        return {}
    
    all_weights = valid_weights
    performance_scores = [w['performance_score'] for w in all_weights]
    total_data_size = sum(w['data_size'] for w in all_weights)
    
    if rejected_count > 0:
        print(f"[Cloud Server] 過濾結果: {rejected_count} 個聚合器被拒絕，{len(all_weights)} 個通過檢查")

    try:
        cr = getattr(app.state, "current_round", None)
        if cr is None and hasattr(app.state, "last_aggregation_round"):
            cr = getattr(app.state, "last_aggregation_round", None)
    except Exception:
        cr = None
    dbi_suspicious_ids, dbi_action, dbi_soft_factor = _analyze_aggregator_weights_with_dbi(
        all_weights, current_round=cr
    )
    dbi_suspicious_ids = set(dbi_suspicious_ids or [])
    if dbi_action == "hard" and dbi_suspicious_ids:
        before = len(all_weights)
        all_weights = [w for w in all_weights if w.get("agg_id") not in dbi_suspicious_ids]
        after = len(all_weights)
        print(
            f"[Cloud Server] ConfShield/DBI 硬剔除：從 {before} 個聚合器中移除 "
            f"{before - after} 個可疑聚合器（剩餘 {after} 個）"
        )
        if not all_weights:

            print("[Cloud Server] ConfShield/DBI 硬剔除後無剩餘聚合器，回退到未過濾狀態（僅做監測）")
            all_weights = valid_weights
        performance_scores = [w['performance_score'] for w in all_weights]
        total_data_size = sum(w['data_size'] for w in all_weights)
    elif dbi_action == "soft" and dbi_suspicious_ids:
        print(
            f"[Cloud Server] ConfShield/DBI 軟降權：標記 {len(dbi_suspicious_ids)} 個可疑聚合器，"
            f"稍後在聚合權重上乘以 soft_factor={dbi_soft_factor:.3f}"
        )
    else:
        dbi_action = "monitor"
        dbi_soft_factor = 1.0
    round_upload_bytes = 0
    for agg_data in all_weights:
        agg_weights = agg_data.get('weights', {})
        if agg_weights:
            agg_size = _compute_model_size_bytes(agg_weights)
            round_upload_bytes += agg_size
    round_upload_mb = round_upload_bytes / (1024 * 1024)
    print(f"[Cloud Server] 本輪通訊量（上傳）: {round_upload_mb:.4f} MB ({round_upload_bytes:,} bytes, {len(all_weights)} 個聚合器)")
    avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.5
    print(f"[Cloud Server] 開始Enhanced FedAvg聚合: {len(all_weights)}個聚合器")
    print(f"[Cloud Server] 性能統計: 平均={avg_performance:.4f}, 數據大小={total_data_size}")

    for idx, agg_data in enumerate(all_weights):
        agg_id = agg_data['agg_id']
        weights = agg_data['weights']
        data_size = agg_data['data_size']
        performance_score = agg_data['performance_score']
        total_norm = 0
        for layer_name, layer_weights in list(weights.items())[:5]:
            if isinstance(layer_weights, torch.Tensor):
                total_norm += layer_weights.norm().item()
            elif isinstance(layer_weights, np.ndarray):
                total_norm += np.linalg.norm(layer_weights)
        print(
            f"[Cloud Server] 聚合器 {agg_id}: 數據大小={data_size}, 性能分數="
            f"{performance_score:.4f}, 前5層範數={total_norm:.4f}"
        )
    
    try:
        strat = getattr(config, "AGGREGATION_STRATEGY", {}) or {}
        data_weight_ratio = float(strat.get("data_weight", 0.7))
        perf_weight_ratio = float(strat.get("performance_weight", 0.3))
    except Exception:
        data_weight_ratio = 0.7
        perf_weight_ratio = 0.3
    topk = max(1, min(2, len(all_weights)))
    sorted_by_perf = sorted(all_weights, key=lambda x: x.get("performance_score", 0.0), reverse=True)
    top_ids = {w["agg_id"] for w in sorted_by_perf[:topk]}
    weights_scalars = []
    total_scalar = 0.0

    try:
        bias_cfg = getattr(config, "BIAS_AWARE_AGGREGATION", {}) or {}
        bias_enabled = bool(bias_cfg.get("enabled", True))
        bias_threshold = float(bias_cfg.get("max_pred_ratio_threshold", 0.85))
        bias_penalty = float(bias_cfg.get("bias_penalty_factor", 0.6))
    except Exception:
        bias_enabled = True
        bias_threshold = 0.85
        bias_penalty = 0.6

    for agg_data in all_weights:
        ds = float(max(1.0, agg_data.get("data_size", 1.0)))
        perf = float(agg_data.get("performance_score", 0.5))
        fresh = float(agg_data.get("freshness_factor", 1.0))
        s_data = ds
        s_perf = perf
        scalar = data_weight_ratio * s_data + perf_weight_ratio * s_perf
        scalar *= fresh

        if bias_enabled:
            max_pred_ratio = agg_data.get("max_pred_ratio", None)
            if max_pred_ratio is not None:
                try:
                    max_pred_ratio = float(max_pred_ratio)
                except Exception:
                    max_pred_ratio = None

            if max_pred_ratio is not None and max_pred_ratio > bias_threshold:
                before = scalar
                scalar *= bias_penalty
                print(
                    f"[Cloud Server] Bias-aware 降權：Aggregator {agg_data.get('agg_id')} "
                    f"max_pred_ratio={max_pred_ratio:.3f} > {bias_threshold:.2f}，"
                    f"權重 {before:.3f} → {scalar:.3f}"
                )

        apply_dbi_soft = False
        dbi_soft_factor_to_apply = 1.0
        if dbi_action == "soft" and agg_data.get("agg_id") in dbi_suspicious_ids:
            apply_dbi_soft = True
            dbi_soft_factor_to_apply = dbi_soft_factor
        else:
            try:
                cr = getattr(app.state, "current_round", None)
                if cr is None and hasattr(app.state, "last_aggregation_round"):
                    cr = getattr(app.state, "last_aggregation_round", None)
                if cr is not None and hasattr(app.state, 'dbi_soft_weights') and cr in app.state.dbi_soft_weights:
                    dbi_soft_info = app.state.dbi_soft_weights[cr]
                    if agg_data.get("agg_id") in dbi_soft_info.get('suspicious_ids', set()):
                        apply_dbi_soft = True
                        dbi_soft_factor_to_apply = dbi_soft_info.get('soft_factor', 1.0)
            except Exception:
                pass
        
        if apply_dbi_soft:
            before_scalar = scalar
            scalar *= dbi_soft_factor_to_apply
            print(
                f"[Cloud Server] ConfShield/DBI 軟降權：Aggregator {agg_data.get('agg_id')} "
                f"scalar {before_scalar:.3f} → {scalar:.3f} (soft_factor={dbi_soft_factor_to_apply:.3f})"
            )

        if agg_data.get("agg_id") in top_ids:
            scalar *= 2.0
        weights_scalars.append(scalar)
        total_scalar += scalar

    if total_scalar <= 0:
        norm_scalars = [1.0 / len(all_weights)] * len(all_weights)
    else:
        norm_scalars = [s / total_scalar for s in weights_scalars]
    staleness_config = getattr(config, 'AGGREGATION_STRATEGY', {}).get('staleness_aware_weighting', {})
    if staleness_config.get('enabled', False):
        import math
        decay_factor = float(staleness_config.get('decay_factor', 0.05))
        max_staleness = int(staleness_config.get('max_staleness', 5))
        min_staleness_weight = float(staleness_config.get('min_staleness_weight', 0.3))
        staleness_weights = []
        for agg_data in all_weights:
            round_gap = int(agg_data.get('round_gap', 0))
            if round_gap > max_staleness:
                staleness_weight = min_staleness_weight
            else:
                staleness_weight = max(min_staleness_weight, math.exp(-round_gap * decay_factor))
            staleness_weights.append(staleness_weight)
        norm_scalars = [wf * sw for wf, sw in zip(norm_scalars, staleness_weights)]
        total_weight = sum(norm_scalars)
        if total_weight > 0:
            norm_scalars = [w / total_weight for w in norm_scalars]
        else:
            norm_scalars = [1.0 / len(norm_scalars)] * len(norm_scalars)
        
        print(f"[Cloud Server] Staleness-aware Weighting 應用後權重因子: {[f'{w:.3f}' for w in norm_scalars]}")
        round_gaps = [agg_data.get('round_gap', 0) for agg_data in all_weights]
        print(f"[Cloud Server] Round Gaps: {round_gaps}, Staleness 權重: {[f'{sw:.3f}' for sw in staleness_weights]}")
    print(f"[Cloud Server] 聚合權重 (data+performance+freshness+topk+staleness): {[f'{w:.3f}' for w in norm_scalars]}")
    
    try:
        strat = getattr(config, "AGGREGATION_STRATEGY", {}) or {}
        winner_config = strat.get("winner_take_most", {})
        if winner_config.get("enabled", False):
            max_perf = max(performance_scores) if performance_scores else 0.5
            gap_threshold = float(winner_config.get("performance_gap_threshold", 0.2))
            elite_ratio = float(winner_config.get("elite_weight_ratio", 0.85))
            min_elite = float(winner_config.get("min_elite_weight", 0.7))
            performance_gap = max_perf - avg_performance
            best_idx = performance_scores.index(max_perf) if performance_scores else 0
            
            if performance_gap > gap_threshold:
                best_agg_id = all_weights[best_idx].get('agg_id', 'unknown')  
                print(f"[Cloud Server] 🏆 啟動精英模式：Aggregator {best_agg_id} 領先平均值 {performance_gap:.4f} (閾值: {gap_threshold:.2f})")
                print(f"[Cloud Server]   精英權重比例: {elite_ratio:.1%}")
                new_norm_scalars = [0.0] * len(norm_scalars)
                new_norm_scalars[best_idx] = elite_ratio
                remaining = 1.0 - elite_ratio
                for i in range(len(norm_scalars)):
                    if i != best_idx:
                        new_norm_scalars[i] = remaining / (len(norm_scalars) - 1)
                norm_scalars = new_norm_scalars
                print(f"[Cloud Server] 🏆 精英模式權重分配: {[f'{w:.3f}' for w in norm_scalars]}")
            elif max_perf > 0.5 and best_idx < len(norm_scalars) and norm_scalars[best_idx] < min_elite:
                best_idx = performance_scores.index(max_perf)
                best_agg_id = all_weights[best_idx].get('agg_id', 'unknown')
                if norm_scalars[best_idx] < min_elite:
                    print(f"[Cloud Server] 🏆 次精英模式：Aggregator {best_agg_id} 表現優秀 (F1={max_perf:.4f})，提升權重至 {min_elite:.1%}")
                    scale_factor = (1.0 - min_elite) / (1.0 - norm_scalars[best_idx])
                    for i in range(len(norm_scalars)):
                        if i == best_idx:
                            norm_scalars[i] = min_elite
                        else:
                            norm_scalars[i] *= scale_factor
                    total = sum(norm_scalars)
                    if total > 0:
                        norm_scalars = [w / total for w in norm_scalars]
    except Exception as e:
        print(f"[Cloud Server] Winner-Take-Most 機制執行失敗: {e}")
    
    try:
        strat = getattr(config, "AGGREGATION_STRATEGY", {}) or {}
        grad_config = strat.get("gradient_consistency", {})
        if grad_config.get("enabled", False) and global_weights is not None and len(global_weights) > 0:
            cosine_threshold = float(grad_config.get("cosine_similarity_threshold", 0.3))
            exclude_opposite = grad_config.get("exclude_opposite", True)
            penalty_factor = float(grad_config.get("weight_penalty_factor", 0.1))
            best_idx = performance_scores.index(max(performance_scores)) if performance_scores else 0
            best_weights = all_weights[best_idx]['weights'] 
            print(f"[Cloud Server] 梯度一致性檢查：以 Aggregator {all_weights[best_idx].get('agg_id', 'unknown')} (F1={max(performance_scores):.4f}) 為參考方向")
            cosine_similarities = []
            for i, agg_data in enumerate(all_weights):
                agg_weights = agg_data['weights']
                cosine_sim = _compute_weight_vector_cosine_similarity(best_weights, agg_weights)
                cosine_similarities.append(cosine_sim)
                agg_id = agg_data.get('agg_id', 'unknown')
                print(f"[Cloud Server]   Aggregator {agg_id}: 餘弦相似度 = {cosine_sim:.4f}")
            adjusted_norm_scalars = []
            excluded_count = 0
            for i, (cosine_sim, orig_weight) in enumerate(zip(cosine_similarities, norm_scalars)):
                if exclude_opposite and cosine_sim < 0:
                    adjusted_norm_scalars.append(0.0)
                    excluded_count += 1
                    print(f"[Cloud Server]   排除 Aggregator {all_weights[i].get('agg_id', 'unknown')} (相似度 < 0)")
                elif cosine_sim < cosine_threshold:
                    new_weight = orig_weight * penalty_factor
                    adjusted_norm_scalars.append(new_weight)
                    print(f"[Cloud Server]   降權 Aggregator {all_weights[i].get('agg_id', 'unknown')} (相似度 {cosine_sim:.4f} < {cosine_threshold:.2f}): {orig_weight:.3f} → {new_weight:.3f}")
                else:
                    adjusted_norm_scalars.append(orig_weight)
            
            total_adjusted = sum(adjusted_norm_scalars)
            if total_adjusted > 0 and excluded_count < len(all_weights):
                norm_scalars = [w / total_adjusted for w in adjusted_norm_scalars]
                print(f"[Cloud Server] 梯度一致性檢查後權重: {[f'{w:.3f}' for w in norm_scalars]} (排除 {excluded_count} 個)")
            elif excluded_count >= len(all_weights):
                print(f"[Cloud Server] 警告：所有 Aggregator 都被排除，使用原始權重")

    except Exception as e:
        print(f"[Cloud Server] 梯度一致性檢查失敗: {e}")
    agg_method_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('aggregation_method', {})
    agg_method_type = agg_method_cfg.get('type', 'weighted')
    if agg_method_type in ['median', 'trimmed']:
        print(f"[Cloud Server] 檢測到聚合方法配置: type={agg_method_type}")
        weights_list = [agg_data['weights'] for agg_data in all_weights]
        data_sizes_list = [agg_data['data_size'] for agg_data in all_weights]
        performance_scores_list = [agg_data['performance_score'] for agg_data in all_weights]
        
        if agg_method_type == 'median' and len(weights_list) >= 2:
            print(f"[Cloud Server]  使用性能加權聚合（考慮 Aggregator 性能）")
            try:
                import sys
                current_module = sys.modules[__name__]
                if hasattr(current_module, 'app') and hasattr(current_module.app, 'state'):
                    current_round = getattr(current_module.app.state, 'last_aggregation_round', None)
                    if current_round is None:
                        current_round = 0
                    else:
                        current_round = int(current_round)
                else:
                    current_round = 0
            except:
                current_round = 0
            
            sorted_perfs = sorted(performance_scores_list, reverse=True)
            performance_gap = (sorted_perfs[0] - sorted_perfs[1]) if len(sorted_perfs) >= 2 else 0.0
        
            if current_round < 100:
                k = max(3, (len(weights_list) + 1) // 2)
            else:

                k = max(2, (len(weights_list) + 2) // 3)
            if performance_gap > 0.2:
                print(f"[Cloud Server] 性能差距過大 ({performance_gap:.3f} > 0.2)，只選擇性能最好的 1 個 Aggregator")
                k = 1
            sorted_indices = sorted(range(len(weights_list)), 
                                  key=lambda i: performance_scores_list[i], 
                                  reverse=True)
            top_k_indices = sorted_indices[:k]
            top_k_weights = [weights_list[i] for i in top_k_indices]
            top_k_perfs = [performance_scores_list[i] for i in top_k_indices]
            top_k_data_sizes = [data_sizes_list[i] for i in top_k_indices]
            
            if len(top_k_weights) >= 2:
                print(f"[Cloud Server] 權重兼容性檢查：檢查 {len(top_k_weights)} 個精英 Aggregator 的權重兼容性")
                def compute_weight_cosine_similarity(w1, w2):
                    import torch
                    similarities = []
                    for key in w1.keys():
                        if key in w2:
                            v1 = _coerce_tensor(w1[key]).flatten()
                            v2 = _coerce_tensor(w2[key]).flatten()
                            if v1.norm() > 0 and v2.norm() > 0:
                                cos_sim = torch.nn.functional.cosine_similarity(
                                    v1.unsqueeze(0), v2.unsqueeze(0), dim=1
                                ).item()
                                similarities.append(cos_sim)
                    return np.mean(similarities) if similarities else 0.0

                similarity_matrix = {}
                min_similarity = 1.0
                incompatible_pairs = []
                for i in range(len(top_k_weights)):
                    for j in range(i + 1, len(top_k_weights)):
                        sim = compute_weight_cosine_similarity(top_k_weights[i], top_k_weights[j])
                        similarity_matrix[(i, j)] = sim
                        similarity_matrix[(j, i)] = sim
                        min_similarity = min(min_similarity, sim)
                        if sim < 0.5:
                            incompatible_pairs.append((i, j, sim))
                print(f"[Cloud Server] 最小權重相似度: {min_similarity:.4f}")
                if min_similarity < 0.6:
                    print(f"[Cloud Server] 警告：檢測到權重不兼容（相似度 < 0.6），互斥對數: {len(incompatible_pairs)}")
                    for i, j, sim in incompatible_pairs:
                        print(f"[Cloud Server]   - Aggregator {top_k_indices[i]} (F1={top_k_perfs[i]:.3f}) vs Aggregator {top_k_indices[j]} (F1={top_k_perfs[j]:.3f}): 相似度={sim:.4f}")
                    is_peak_protection_active = False
                    if PEAK_PROTECTION_ENABLED and last_peak_round is not None:
                        try:
                            import sys
                            current_module = sys.modules[__name__]
                            if hasattr(current_module, 'app') and hasattr(current_module.app, 'state'):
                                current_round = getattr(current_module.app.state, 'last_aggregation_round', None)
                                if current_round is not None:
                                    rounds_since_peak = int(current_round) - last_peak_round
                                    if rounds_since_peak <= PEAK_PROTECTION_ROUNDS:
                                        is_peak_protection_active = True
                        except:
                            pass
                    
                    if is_peak_protection_active:
                        print(f"[Cloud Server] 峰值保護期間檢測到權重不兼容，直接回退到峰值模型 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})")
                        if BEST_GLOBAL_WEIGHTS is not None:
                            needs_rollback_flag = True
                            rollback_reason_str = f"peak_protection_weight_incompatibility_min_sim_{min_similarity:.4f}"
                            print(f"[Cloud Server] 已設置回退標記，將在聚合前回退到最佳模型")
                            global_weights = {k: _coerce_tensor(v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
                            print(f"[Cloud Server] 峰值保護期間：已直接使用最佳模型權重，跳過不兼容權重的聚合")
                            return global_weights
                        else:
                            print(f"[Cloud Server] 警告：峰值保護期間但無法回退（BEST_GLOBAL_WEIGHTS 為 None），降級為選擇最兼容的聚合器")
                    if not is_peak_protection_active:
                        avg_similarities = []
                        for i in range(len(top_k_weights)):
                            similarities_with_others = []
                            for j in range(len(top_k_weights)):
                                if i != j:
                                    sim = similarity_matrix.get((i, j), 0.0)
                                    similarities_with_others.append(sim)
                            avg_sim = np.mean(similarities_with_others) if similarities_with_others else 0.0
                            avg_similarities.append((i, avg_sim, top_k_perfs[i]))
                        
                        def combined_score(item):
                            idx, avg_sim, perf = item
                            max_avg_sim = max([s[1] for s in avg_similarities]) if avg_similarities else 1.0
                            min_avg_sim = min([s[1] for s in avg_similarities]) if avg_similarities else 0.0
                            max_perf = max([s[2] for s in avg_similarities]) if avg_similarities else 1.0
                            min_perf = min([s[2] for s in avg_similarities]) if avg_similarities else 0.0       
                            norm_sim = (avg_sim - min_avg_sim) / (max_avg_sim - min_avg_sim) if max_avg_sim > min_avg_sim else 0.5
                            norm_perf = (perf - min_perf) / (max_perf - min_perf) if max_perf > min_perf else 0.5 
                            return 0.6 * norm_sim + 0.4 * norm_perf
                        
                        avg_similarities.sort(key=combined_score, reverse=True)
                        selected_count = min(3, max(2, len(top_k_weights)))
                        selected_indices = [item[0] for item in avg_similarities[:selected_count]] 
                        print(f"[Cloud Server] 權重不兼容，選擇最兼容的 {selected_count} 個聚合器（基於相似度和性能）:")
                        for idx in selected_indices:
                            avg_sim = avg_similarities[selected_indices.index(idx)][1] if idx in [item[0] for item in avg_similarities[:selected_count]] else 0.0
                            print(f"[Cloud Server]   - Aggregator {top_k_indices[idx]} (F1={top_k_perfs[idx]:.3f}, 平均相似度={avg_sim:.4f})")
                        
                        top_k_indices = [top_k_indices[i] for i in selected_indices]
                        top_k_weights = [top_k_weights[i] for i in selected_indices]
                        top_k_perfs = [top_k_perfs[i] for i in selected_indices]
                        top_k_data_sizes = [top_k_data_sizes[i] for i in selected_indices]
                        k = len(selected_indices)
                    else:
                        best_idx = 0
                        print(f"[Cloud Server] 權重不兼容，只選取性能最好的 Aggregator {top_k_indices[best_idx]} (F1={top_k_perfs[best_idx]:.3f})")
                        top_k_indices = [top_k_indices[best_idx]]
                        top_k_weights = [top_k_weights[best_idx]]
                        top_k_perfs = [top_k_perfs[best_idx]]
                        top_k_data_sizes = [top_k_data_sizes[best_idx]]
                        k = 1
                else:
                    print(f"[Cloud Server] 權重兼容性檢查通過（最小相似度: {min_similarity:.4f} >= 0.6）")

            agg_method_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('aggregation_method', {})
            use_weighted_mean = agg_method_cfg.get('use_performance_weighted_mean', False)
        
            if use_weighted_mean and len(top_k_weights) >= 2:
                print(f"[Cloud Server] 性能加權平均：從 {len(weights_list)} 個權重中選擇性能最好的 {k} 個（F1: {[f'{p:.3f}' for p in top_k_perfs]}），然後進行性能加權平均")
                max_perf = max(top_k_perfs) if top_k_perfs else 0.5
                min_perf = min(top_k_perfs) if top_k_perfs else 0.0
                perf_range = max_perf - min_perf if max_perf > min_perf else 1.0
                performance_weights = []
                for perf in top_k_perfs:
                    if perf_range > 0:
                        normalized_perf = (perf - min_perf) / perf_range
                        weight = normalized_perf ** 2
                    else:
                        weight = 1.0
                    performance_weights.append(weight)

                total_weight = sum(performance_weights)
                if total_weight > 0:
                    performance_weights = [w / total_weight for w in performance_weights]
                else:
                    performance_weights = [1.0 / len(performance_weights)] * len(performance_weights)
                aggregated_result = _aggregate_weights_weighted_mean(top_k_weights, performance_weights)
                
                if aggregated_result:
                    print(f"[Cloud Server] 使用性能加權平均聚合完成: {len(aggregated_result)} 層")
                    return aggregated_result
            else:
                print(f"[Cloud Server] 性能加權中位數：從 {len(weights_list)} 個權重中選擇性能最好的 {k} 個（F1: {[f'{p:.3f}' for p in top_k_perfs]}）")
                aggregated_result = _aggregate_weights_median(top_k_weights)
                if aggregated_result:
                    print(f"[Cloud Server] 使用性能加權中位數聚合完成: {len(aggregated_result)} 層")
                    return aggregated_result
        
            print(f"[Cloud Server] 性能加權聚合返回空結果，降級為標準中位數")

        aggregated_result = aggregate_client_weights(weights_list, data_sizes_list, performance_scores_list)
        if aggregated_result:
            print(f"[Cloud Server] 使用 {agg_method_type} 聚合完成: {len(aggregated_result)} 層")
            return aggregated_result
        else:
            print(f"[Cloud Server] {agg_method_type} 聚合返回空結果，降級為 Enhanced FedAvg")

    try:
        template_weights = all_weights[0]['weights']
        global_weights = {}
        fedbn_config = getattr(
            config, 'AGGREGATION_CONFIG', {}).get('fedbn', {})
        fedbn_enabled = fedbn_config.get('enabled', False)
        layers_to_exclude = fedbn_config.get('layers_to_exclude', [])
        if fedbn_enabled:
            import re
            exclude_patterns = [re.compile(pattern)
                                           for pattern in layers_to_exclude]
            print(f"[Cloud Server] FedBN 已啟用，將排除以下層的聚合: {layers_to_exclude}")
        
        for idx, layer_name in enumerate(template_weights.keys()):
            if fedbn_enabled:
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern.search(layer_name):
                        should_exclude = True
                        break
                if should_exclude:
                    if layer_name in template_weights:
                        global_weights[layer_name] = _coerce_tensor(
                            template_weights[layer_name]).clone()
                        print(
                            f"[Cloud Server] FedBN: 跳過聚合層 {layer_name}（保留本地差異）"
                        )
                    continue

            if layer_name == 'output_layer.bias':
                best_bias = None
                best_performance = -1.0
                best_agg_id = None

                for agg_data in all_weights:
                    weights = agg_data['weights']
                    performance_score = agg_data['performance_score']
                    agg_id = agg_data.get('agg_id', 'unknown')

                    if layer_name in weights:
                        bias_tensor = weights[layer_name]
                        if isinstance(bias_tensor, np.ndarray):
                            bias_tensor = torch.from_numpy(bias_tensor).float()
                        elif not isinstance(bias_tensor, torch.Tensor):
                            bias_tensor = torch.tensor(
                                bias_tensor, dtype=torch.float32)


                        if performance_score > best_performance:
                            best_performance = performance_score
                            best_bias = _coerce_tensor(bias_tensor).clone()
                            best_agg_id = agg_id

                if best_bias is not None:
                    global_weights[layer_name] = best_bias
                    print(f"[Cloud Server] 🏆 輸出層 bias 使用最佳聚合器 (agg_id={best_agg_id}, performance={best_performance:.4f})")
                else:

                    if layer_name in template_weights:
                        global_weights[layer_name] = _coerce_tensor(
                            template_weights[layer_name]).clone()
                        print(f"[Cloud Server] 輸出層 bias 未找到最佳值，使用第一個聚合器的值")
                continue

            weighted_sum = None
            total_weight = 0
            valid_weights_count = 0
            
            for agg_data in all_weights:
                weights = agg_data['weights']
                data_size = agg_data['data_size']
                performance_score = agg_data['performance_score']
                
                if layer_name in weights:
                    layer_weights = weights[layer_name]
                    

                    if layer_weights is None:
                        print(f"[Cloud Server] 警告：聚合器 {agg_data.get('agg_id', 'unknown')} 的層 {layer_name} 權重為 None，跳過")
                        continue
                    
                    if isinstance(layer_weights, np.ndarray):
                        layer_weights = torch.from_numpy(layer_weights).float()
                    elif not isinstance(layer_weights, torch.Tensor):
                        layer_weights = torch.tensor(
                            layer_weights, dtype=torch.float32)
                    
                    if isinstance(layer_weights, torch.Tensor):
                        if torch.isnan(layer_weights).any() or torch.isinf(layer_weights).any():
                            print(f"[Cloud Server] 警告：聚合器 {agg_data.get('agg_id', 'unknown')} 的層 {layer_name} 包含 NaN/Inf，跳過")
                            continue

                    strategy_config = getattr(
                        config, 'AGGREGATION_STRATEGY', {})
                    strategy_type = strategy_config.get('type', 'weighted')
                    data_weight_ratio = strategy_config.get('data_weight', 0.7)
                    performance_weight_ratio = strategy_config.get(
                        'performance_weight', 0.3)
                    min_performance = strategy_config.get(
                        'min_performance', 0.1)
                    normalize_weights = strategy_config.get(
                        'normalize_weights', True)
                    use_adaptive = strategy_config.get(
                        'use_adaptive_weights', True)
                    current_round = 0
                    try:
                        if hasattr(app, 'state') and hasattr(app.state, 'current_round'):
                            current_round = app.state.current_round
                    except:
                        pass
                    max_rounds = getattr(config, 'MAX_ROUNDS', 500)

                    if strategy_type == 'weighted':
                        combined_weight = data_size
                        freshness_factor = agg_data.get('freshness_factor', 1.0)
                        combined_weight *= freshness_factor
                    elif strategy_type == 'performance':
                        performance_weight = max(
                            min_performance, performance_score)
                        combined_weight = performance_weight
                        freshness_factor = agg_data.get('freshness_factor', 1.0)
                        combined_weight *= freshness_factor
                    else:
                        normalized_data_size = data_size / \
                            total_data_size if total_data_size > 0 else 1.0 / \
                                len(all_weights)
                        if performance_scores:
                            max_perf = max(performance_scores)
                            min_perf = min(performance_scores)
                            if max_perf > min_perf:
                                normalized_performance = (
                                    performance_score - min_perf) / (max_perf - min_perf)
                            else:
                                normalized_performance = 1.0
                        else:
                            normalized_performance = 1.0

                        normalized_performance = max(
                            min_performance, normalized_performance)
                        if use_adaptive and max_rounds > 0:
                            progress = min(1.0, current_round / max_rounds)
                            adaptive_data_ratio = data_weight_ratio * \
                                (1.0 - 0.3 * progress)
                            adaptive_perf_ratio = performance_weight_ratio * \
                                (1.0 + 0.5 * progress)

                            total_ratio = adaptive_data_ratio + adaptive_perf_ratio
                            adaptive_data_ratio /= total_ratio
                            adaptive_perf_ratio /= total_ratio
                        else:
                            adaptive_data_ratio = data_weight_ratio
                            adaptive_perf_ratio = performance_weight_ratio


                        combined_weight = (adaptive_data_ratio * normalized_data_size +
                                         adaptive_perf_ratio * normalized_performance)


                        combined_weight *= data_size
                        

                        freshness_factor = agg_data.get('freshness_factor', 1.0)
                        combined_weight *= freshness_factor
                        if freshness_factor < 1.0:
                            round_gap_val = agg_data.get('round_gap', 0)
                            print(f"[Cloud Server] 聚合器 {agg_data.get('agg_id', 'unknown')} 權重降權：輪次差距={round_gap_val}，新鮮度因子={freshness_factor:.3f}")

                    if normalize_weights and total_weight > 0:
                        pass
                    weight_norm = layer_weights.norm().item()
                    if 'output_layer' in layer_name:
                        stability_factor = min(
                            1.0, max(0.8, weight_norm / 0.5))
                    elif 'layers' in layer_name and 'weight' in layer_name:
                        stability_factor = min(
                            1.0, max(0.5, weight_norm / 1.0))
                    else:
                        stability_factor = min(
                            1.0, max(0.1, weight_norm / 10.0))
                    
                    if weighted_sum is None:
                        weighted_sum = layer_weights * combined_weight * stability_factor
                    else:
                        weighted_sum += layer_weights * combined_weight * stability_factor
                    
                    total_weight += combined_weight * stability_factor
            
            if weighted_sum is not None and total_weight > 0:
                new_layer_weight = weighted_sum / total_weight
                
                apply_server_lr = getattr(config, "AGGREGATION_STRATEGY", {}).get("apply_server_lr", False)
                if apply_server_lr and layer_name in global_weights:
                    base_server_lr = getattr(config, "SERVER_LR", 1.0)
                    server_lr = base_server_lr * CURRENT_SERVER_LR_MULTIPLIER
                    
                    if PEAK_PROTECTION_ENABLED and last_peak_round is not None:
                        try:
                            import sys
                            current_module = sys.modules[__name__]
                            if hasattr(current_module, 'app') and hasattr(current_module.app, 'state'):
                                current_round = getattr(current_module.app.state, 'last_aggregation_round', None)
                                if current_round is not None:
                                    current_round = int(current_round)
                                    rounds_since_peak = current_round - last_peak_round
                                    if rounds_since_peak <= PEAK_PROTECTION_ROUNDS:

                                        server_lr = server_lr * 0.5
                                        if layer_name in ["output_layer.weight", "layers.0.weight"]:
                                            print(f"[Cloud Server] 峰值保護期間：SERVER_LR 進一步降低 50% (最終={server_lr:.6f})")
                        except:
                            pass
                    prev_weight = _coerce_tensor(global_weights[layer_name])
                    new_weight_t = _coerce_tensor(new_layer_weight)

                    if prev_weight.shape == new_weight_t.shape:
                        delta = new_weight_t - prev_weight
                        new_layer_weight = prev_weight + server_lr * delta
                        if layer_name in ["output_layer.weight", "layers.0.weight"]:
                            print(f"[Cloud Server] 應用 Server LR ({server_lr:.4f}) 到層 {layer_name}: delta_norm={delta.norm().item():.6f}")
                
                if isinstance(new_layer_weight, torch.Tensor):
                    if torch.isnan(new_layer_weight).any() or torch.isinf(new_layer_weight).any():
                        print(f"[Cloud Server] 錯誤：層 {layer_name} 聚合後包含 NaN/Inf，跳過此層")
                        continue
                
                if layer_name in global_weights:
                    prev_weight = _coerce_tensor(global_weights[layer_name])
                    new_weight_t = _coerce_tensor(new_layer_weight)
                    if prev_weight.shape == new_weight_t.shape:
                        weight_diff = (new_weight_t - prev_weight).abs().mean().item()
                        weight_diff_norm = (new_weight_t - prev_weight).norm().item()
                        if weight_diff < 1e-8:
                            print(f"[Cloud Server] 警告：層 {layer_name} 權重未更新 (mean_diff={weight_diff:.2e}, norm_diff={weight_diff_norm:.2e}, 有效權重數={valid_weights_count})")
                        elif valid_weights_count > 0:

                            if layer_name in ["output_layer.weight", "layers.0.weight", "input_reshape.weight"]:
                                print(f"[Cloud Server] 關鍵層 {layer_name} 已更新: mean_diff={weight_diff:.6f}, norm_diff={weight_diff_norm:.6f}, 有效權重數={valid_weights_count}")
                else:
                    if valid_weights_count > 0:
                        print(f"[Cloud Server] 新層 {layer_name} 已添加: 有效權重數={valid_weights_count}")
                
                global_weights[layer_name] = new_layer_weight
                layer_norm = global_weights[layer_name].norm().item()
                layer_mean = global_weights[layer_name].mean().item()
                layer_tensor = global_weights[layer_name]
                if layer_tensor.numel() > 1:
                    layer_std = layer_tensor.std().item()
                else:
                    layer_std = 0.0
                if 'layers' in layer_name and 'weight' in layer_name and 'output_layer' not in layer_name:
                    if layer_norm < 10.0:
                        if 'layers.0' in layer_name:
                            expected_norm = 18.0
                        elif 'layers.1' in layer_name:
                            expected_norm = 22.0
                        elif 'layers.2' in layer_name:
                            expected_norm = 16.0
                        else:
                            expected_norm = 10.0
                        if layer_norm < expected_norm * 0.5:
                            curr = _coerce_tensor(global_weights[layer_name])
                            scale_factor = max(
                                1.5, (expected_norm * 0.5) / (layer_norm + 1e-6))
                            amplified = curr * scale_factor
                            w_mean = amplified.mean().item()
                            if w_mean < OUTPUT_LAYER_MEAN_THRESHOLD:
                                offset = abs(w_mean) + 0.01
                                amplified = amplified + offset
                                print(
                                    f"[Cloud Server] 放大隱藏層 {layer_name}: 原範數={layer_norm:.4f}, 預期範數={expected_norm:.1f}, 放大倍數="
                                    f"{scale_factor:.2f}, 新範數={amplified.norm().item():.6f}, 添加偏移={offset:.4f}（原均值={w_mean:.4f}）"
                                )
                            else:
                                print(
                                    f"[Cloud Server] 放大隱藏層 {layer_name}: 原範數={layer_norm:.4f}, 預期範數="
                                    f"{expected_norm:.1f}, 放大倍數={scale_factor:.2f}, 新範數={amplified.norm().item():.6f}"
                                )

                            global_weights[layer_name] = amplified
                        elif layer_norm < expected_norm * 0.7:
                            curr = _coerce_tensor(global_weights[layer_name])
                            scale_factor = max(
                                1.2, (expected_norm * 0.7) / (layer_norm + 1e-6))
                            amplified = curr * scale_factor
                            print(
                                f"[Cloud Server] 輕微放大隱藏層 {layer_name}: 原範數={layer_norm:.4f}, 預期範數="
                                f"{expected_norm:.1f}, 放大倍數={scale_factor:.2f}, 新範數={amplified.norm().item():.6f}"
                            )
                            global_weights[layer_name] = amplified

                if 'output_layer' in layer_name:
                    curr = _coerce_tensor(global_weights[layer_name])
                    w_mean = curr.mean().item()
                    w_std = curr.std().item()
                    MIN_OUTPUT_LAYER_STD = 0.1
                    if w_std < MIN_OUTPUT_LAYER_STD:
                        scale_factor = MIN_OUTPUT_LAYER_STD / (w_std + 1e-6)
                        curr_centered = curr - w_mean
                        curr_scaled = curr_centered * scale_factor + w_mean
                        global_weights[layer_name] = curr_scaled
                        new_std = curr_scaled.std().item()
                        print(
                            f"[Cloud Server] 修復輸出層 {layer_name} 標準差: 原標準差={w_std:.6f}, 目標標準差="
                            f"{MIN_OUTPUT_LAYER_STD:.6f}, 縮放倍數={scale_factor:.2f}, 新標準差={new_std:.6f}"
                        )

                        curr = curr_scaled
                        w_std = new_std

                    if w_mean < OUTPUT_LAYER_MEAN_THRESHOLD:
                        offset = abs(w_mean) + 0.01
                        curr = curr + offset
                        global_weights[layer_name] = curr
                        print(
                            f"[Cloud Server] 修正輸出層 {layer_name} 權重均值: 原均值="
                            f"{w_mean:.4f}, 添加偏移={offset:.4f}, 新均值={curr.mean().item():.4f}"
                        )
                    layer_norm = global_weights[layer_name].norm().item()
                    if layer_norm < 3.0:
                        backup_tensor, backup_norm, best_agg = _select_best_backup_tensor(
                            layer_name, all_weights)
                        if backup_tensor is not None and backup_norm > layer_norm:
                            backup_tensor = _coerce_tensor(backup_tensor)
                            curr = _coerce_tensor(global_weights[layer_name])
                            if backup_norm > layer_norm * 2:
                                mix_ratio = 0.7
                            elif backup_norm > layer_norm * 1.5:
                                mix_ratio = 0.5
                            else:
                                mix_ratio = 0.3
                            repaired = (1.0 - mix_ratio) * curr + \
                                        mix_ratio * backup_tensor
                            if repaired.norm().item() < 3.0:

                                scale_factor = max(
                                    1.5, 3.0 / (repaired.norm().item() + 1e-6))
                                repaired = repaired * scale_factor
                                print(
                                    f"[Cloud Server] 修復並放大輸出層 {layer_name}: 來源聚合器={best_agg}, 混合比例="
                                    f"{mix_ratio:.1%}, 放大倍數={scale_factor:.2f}, 新範數={repaired.norm().item():.6f}"
                                )
                            else:
                                print(
                                    f"[Cloud Server] 修復輸出層 {layer_name}: 來源聚合器={best_agg}, 混合比例="
                                    f"{mix_ratio:.1%}, 新範數={repaired.norm().item():.6f}"
                                )
                            global_weights[layer_name] = repaired
                        elif ENABLE_WEIGHT_AMPLIFICATION and layer_norm < 1.0:

                            curr = _coerce_tensor(global_weights[layer_name])
                            scale_factor = max(

                                3.0, 3.0 / (layer_norm + 1e-6))
                            amplified = curr * scale_factor
                            global_weights[layer_name] = amplified
                            print(
                                f"[Cloud Server] 放大輸出層 {layer_name}: 原範數={layer_norm:.6f}, 放大倍數="
                                f"{scale_factor:.2f}, 新範數={amplified.norm().item():.6f}"
                            )
                        else:

                            curr = _coerce_tensor(global_weights[layer_name])
                            scale_factor = max(
                                1.5, 3.0 / (layer_norm + 1e-6))
                            amplified = curr * scale_factor
                            global_weights[layer_name] = amplified
                            print(
                                f"[Cloud Server] 輕微放大輸出層 {layer_name}: 原範數={layer_norm:.6f}, 放大倍數="
                                f"{scale_factor:.2f}, 新範數={amplified.norm().item():.6f}"
                            )
                    elif ENABLE_WEIGHT_AMPLIFICATION and layer_norm < 4.0:

                        curr = _coerce_tensor(global_weights[layer_name])
                        scale_factor = 1.1
                        amplified = curr * scale_factor
                        global_weights[layer_name] = amplified
                        print(
                            f"[Cloud Server] 輕微放大輸出層 {layer_name}: 原範數={layer_norm:.6f}, 新範數={amplified.norm().item():.6f}"
                        )

                    if 'bias' in layer_name:
                        curr = _coerce_tensor(global_weights[layer_name])
                        if curr.numel() > 0:

                            bias_values = curr.cpu().numpy()
                            max_bias_idx = np.argmax(bias_values)
                            min_bias_idx = np.argmin(bias_values)
                            bias_range = bias_values[max_bias_idx] - \
                                bias_values[min_bias_idx]
                            bias_std = float(bias_values.std())
                            MAX_BIAS_RANGE = 1.5
                            if bias_range > MAX_BIAS_RANGE:
                                bias_mean = float(bias_values.mean())
                                if bias_range > 3.0:
                                    target_range = MAX_BIAS_RANGE
                                else:
                                    target_range = 1.2
                                scale_factor = target_range / \
                                    (bias_range + 1e-6)
                                normalized_bias = (
                                    bias_values - bias_mean) * scale_factor + bias_mean
                                global_weights[layer_name] = torch.tensor(
                                    normalized_bias, dtype=curr.dtype, device=curr.device)
                                new_range = normalized_bias.max() - normalized_bias.min()
                                new_std = float(np.std(normalized_bias))
                                print(
                                    f"[Cloud Server] 輸出層 bias 不均勻（範圍={bias_range:.4f}, 標準差="
                                    f"{bias_std:.4f}），已調整為更均勻分佈（新範圍={new_range:.4f}, 新標準差={new_std:.4f}, 縮放倍數={scale_factor:.2f}）"
                                )
                            elif bias_std > 0.4:

                                bias_mean = float(bias_values.mean())

                                target_std = 0.25
                                scale_factor = target_std / (bias_std + 1e-6)
                                normalized_bias = (
                                    bias_values - bias_mean) * scale_factor + bias_mean
                                global_weights[layer_name] = torch.tensor(
                                    normalized_bias, dtype=curr.dtype, device=curr.device)
                                new_range = normalized_bias.max() - normalized_bias.min()
                                new_std = float(np.std(normalized_bias))
                                print(
                                    f"[Cloud Server] 輸出層 bias 標準差過大（標準差="
                                    f"{bias_std:.4f}），已調整（新標準差={new_std:.4f}, 新範圍={new_range:.4f}）"
                                )


                            if (curr <= 0).all().item():

                                offset = abs(curr.min().item()) + \
                                             0.1
                                global_weights[layer_name] = curr + offset
                                print(
                                    f"[Cloud Server] 輸出層 bias 全為負值，已添加偏移 {offset:.4f} 確保有正值"
                                )
                            elif (curr < 0).all().item():

                                offset = abs(curr.min().item()) + 0.05
                                global_weights[layer_name] = curr + offset
                                print(
                                    f"[Cloud Server] 輸出層 bias 全為嚴格負值，已添加偏移 {offset:.4f}"
                                )
                elif 'residual_layers' in layer_name and 'bias' in layer_name:
                    if layer_norm < RESIDUAL_BIAS_THRESHOLD:
                        backup_tensor, backup_norm, best_agg = _select_best_backup_tensor(
                            layer_name, all_weights)
                        if backup_tensor is not None and backup_norm > layer_norm:
                            curr = _coerce_tensor(global_weights[layer_name])
                            backup_tensor = _coerce_tensor(backup_tensor)

                            if backup_norm > layer_norm * 2:
                                mix_ratio = 0.7
                            elif backup_norm > layer_norm * 1.5:
                                mix_ratio = 0.6
                            else:
                                mix_ratio = 0.5
                            repaired = (1.0 - mix_ratio) * curr + \
                                        mix_ratio * backup_tensor
                            if repaired.norm().item() < RESIDUAL_BIAS_THRESHOLD:
                                repaired = backup_tensor.clone()
                            global_weights[layer_name] = repaired
                            print(
                                f"[Cloud Server] 修復殘差 bias {layer_name}: 來源聚合器={best_agg}, 混合比例={mix_ratio:.1%}"
                            )
                        else:

                            curr = _coerce_tensor(global_weights[layer_name])
                            if curr.numel() > 0:

                                fan_in = curr.size(0) if len(
                                    curr.shape) > 0 else 1

                                bound = max(0.1, 1.0 / math.sqrt(fan_in)
                                            if fan_in > 0 else 0.15)
                                torch.nn.init.uniform_(curr, -bound, bound)
                                global_weights[layer_name] = curr
                                new_norm = curr.norm().item()
                                print(
                                    f"[Cloud Server] 殘差 bias {layer_name} 無可用備援，已重新初始化: 新範數={new_norm:.6f} (bound={bound:.4f})"
                                )
                            else:
                                print(
                                    f"[Cloud Server] 殘差 bias {layer_name} 無可用備援且無法重新初始化"
                                )
                elif 'batch_norms' in layer_name and 'running_var' in layer_name:
                    tensor = _coerce_tensor(global_weights[layer_name])
                    if not torch.isfinite(tensor).all() or float(torch.min(tensor).item()) < BN_MIN_VARIANCE:
                        backup_tensor, _, best_agg = _select_best_backup_tensor(
                            layer_name, all_weights)
                        if backup_tensor is not None:
                            global_weights[layer_name] = _coerce_tensor(
                                backup_tensor)
                            print(
                                f"[Cloud Server] 修復 BatchNorm var {layer_name} 來自聚合器 {best_agg}"
                            )
                        else:
                            global_weights[layer_name] = torch.clamp(
                                tensor, min=BN_MIN_VARIANCE)
                            print(
                                f"[Cloud Server] BatchNorm var {layer_name} 夾取到最小值 {BN_MIN_VARIANCE}"
                            )
                    global_weights[layer_name] = _apply_bn_ema(
                        layer_name, _coerce_tensor(global_weights[layer_name]))
                elif 'batch_norms' in layer_name and 'running_mean' in layer_name:
                    tensor = _coerce_tensor(global_weights[layer_name])
                    if not torch.isfinite(tensor).all() or float(torch.abs(tensor).max().item()) > 50:
                        backup_tensor, _, best_agg = _select_best_backup_tensor(
                            layer_name, all_weights)
                        if backup_tensor is not None:
                            global_weights[layer_name] = _coerce_tensor(
                                backup_tensor)
                            print(
                                f"[Cloud Server] 修復 BatchNorm mean {layer_name} 來自聚合器 {best_agg}"
                            )
                        else:
                            global_weights[layer_name] = torch.clamp(
                                tensor, min=-50, max=50)
                            print(
                                f"[Cloud Server] BatchNorm mean {layer_name} 已限制在 [-50, 50]"
                            )
                    global_weights[layer_name] = _apply_bn_ema(
                        layer_name, _coerce_tensor(global_weights[layer_name]))
                elif layer_name.endswith('num_batches_tracked'):
                    tensor = _coerce_tensor(global_weights[layer_name])
                    val = float(tensor.item())
                    if (not math.isfinite(val)) or val < BN_MIN_BATCHES or val > BN_MAX_BATCHES:
                        backup_tensor, _, best_agg = _select_best_backup_tensor(
                            layer_name, all_weights)
                        if backup_tensor is not None:
                            sanitized = _clamp_bn_tracker_tensor(backup_tensor)
                            global_weights[layer_name] = sanitized
                            print(
                                f"[Cloud Server] 修復 BatchNorm {layer_name} 來自聚合器 {best_agg} (夾取範圍 [{BN_MIN_BATCHES}, {BN_MAX_BATCHES}])"
                            )
                        else:
                            clamped_val = max(BN_MIN_BATCHES, min(
                                BN_MAX_BATCHES, val if math.isfinite(val) else BN_MIN_BATCHES))
                            global_weights[layer_name] = torch.tensor(
                                float(clamped_val))
                            print(
                                f"[Cloud Server] BatchNorm {layer_name} 設為 {clamped_val}"
                            )
                    sanitized = _clamp_bn_tracker_tensor(
                        global_weights[layer_name])
                    ema_tracker = _apply_bn_ema(layer_name, sanitized)
                    global_weights[layer_name] = torch.round(
                        ema_tracker).long()

                if layer_norm < 1e-6:
                    print(
                        f"[Cloud Server] 警告：聚合層 {layer_name} 範數極小 ({layer_norm:.6f})"
                    )
                if abs(layer_mean) > 10 or layer_std > 10:
                    print(
                        f"[Cloud Server] 警告：聚合層 {layer_name} 值異常 (mean={layer_mean:.4f}, std={layer_std:.4f})"
                    )
                print(
                    f"[Cloud Server] 聚合層 {layer_name}: 形狀={global_weights[layer_name].shape}, 範數={layer_norm:.4f}, mean={layer_mean:.4f}, std={layer_std:.4f}"
                )
        try:
            all_params = []
            output_layer_params = []
            for key, value in global_weights.items():
                if isinstance(value, torch.Tensor):
                    flat = value.detach().cpu().flatten().numpy()
                    all_params.extend(flat.tolist())
                    if 'output_layer' in key or 'output' in key.lower():
                        output_layer_params.extend(flat.tolist())
                elif isinstance(value, np.ndarray):
                    flat = value.flatten()
                    all_params.extend(flat.tolist())
                    if 'output_layer' in key or 'output' in key.lower():
                        output_layer_params.extend(flat.tolist())

            if all_params:
                all_params = np.array(all_params)
                mean_val = float(np.mean(all_params))
                std_val = float(np.std(all_params))
                norm_val = float(np.linalg.norm(all_params))
                min_val = float(np.min(all_params))
                max_val = float(np.max(all_params))

                print(f"[Cloud Server] 聚合後權重統計:")
                print(f"  - 總參數數: {len(all_params)}")
                print(f"  - 均值: {mean_val:.6f}")
                print(f"  - 標準差: {std_val:.6f}")
                print(f"  - L2範數: {norm_val:.6f}")
                print(f"  - 最小值: {min_val:.6f}, 最大值: {max_val:.6f}")

                if output_layer_params:
                    output_params = np.array(output_layer_params)
                    output_mean = float(np.mean(output_params))
                    output_std = float(np.std(output_params))
                    output_norm = float(np.linalg.norm(output_params))
                    print(
                        f"  - 輸出層參數: 均值={output_mean:.6f}, 標準差={output_std:.6f}, L2範數={output_norm:.6f}")
                try:
                    if 'output_layer.weight' in global_weights:
                        output_weight = _coerce_tensor(
                            global_weights['output_layer.weight'])
                        per_class_weight_norms = [output_weight[i].norm(
                        ).item() for i in range(output_weight.shape[0])]
                        per_class_weight_means = [output_weight[i].mean(
                        ).item() for i in range(output_weight.shape[0])]
                        print(
                            f"[Cloud Server] 各類別輸出層權重範數: {[f'{n:.4f}' for n in per_class_weight_norms]}"
                        )
                        print(
                            f"[Cloud Server] 各類別輸出層權重均值: {[f'{m:.4f}' for m in per_class_weight_means]}"
                        )
                        max_norm_idx = np.argmax(per_class_weight_norms)
                        print(
                            f"[Cloud Server] 輸出層權重範數最大的類別: {max_norm_idx} (範數={per_class_weight_norms[max_norm_idx]:.4f})"
                        )

                    if 'output_layer.bias' in global_weights:
                        output_bias = _coerce_tensor(
                            global_weights['output_layer.bias'])

                        bias_values = output_bias.cpu().numpy()
                        print(
                            f"[Cloud Server] 各類別輸出層偏置: {[f'{b:.4f}' for b in bias_values]}"
                        )
                        max_bias_idx = np.argmax(bias_values)
                        min_bias_idx = np.argmin(bias_values)
                        print(
                            f"[Cloud Server] 輸出層偏置最大的類別: {max_bias_idx} (偏置={bias_values[max_bias_idx]:.4f})"
                        )
                        print(
                            f"[Cloud Server] 輸出層偏置最小的類別: {min_bias_idx} (偏置={bias_values[min_bias_idx]:.4f})"
                        )
                        bias_range = bias_values[max_bias_idx] - \
                            bias_values[min_bias_idx]
                        print(f"[Cloud Server] 輸出層偏置範圍: {bias_range:.4f}")
                except Exception as e:
                    print(f"[Cloud Server] 各類別輸出層權重/偏置診斷失敗: {e}")
        except Exception as diag_e:
            print(f"[Cloud Server] 聚合後權重統計診斷失敗: {diag_e}")
        current_round = getattr(app.state, 'current_round', None) or getattr(
            app.state, 'last_aggregation_round', None) or 0

        stability_check_passed = _check_weight_stability(
            global_weights, PREVIOUS_GLOBAL_WEIGHTS)
        if not stability_check_passed:
            STABILITY_CHECK_FAILURE_COUNT += 1
            print(
                f"[Cloud Server] 權重穩定性檢查失敗，檢測到退化跡象（連續失敗次數: {STABILITY_CHECK_FAILURE_COUNT}/{STABILITY_CHECK_FAILURE_THRESHOLD}）"
            )
            if STABILITY_CHECK_FAILURE_COUNT >= STABILITY_CHECK_FAILURE_THRESHOLD:
                print(
                    f"[Cloud Server] 連續 {STABILITY_CHECK_FAILURE_COUNT} 輪穩定性檢查失敗，觸發回退"
                )
                if BEST_GLOBAL_WEIGHTS is not None:
                    print(
                        f"[Cloud Server] 回退到最佳評估權重（round={BEST_ROUND_ID}，f1={BEST_GLOBAL_F1:.4f}，穩定性檢查觸發）"
                    )
                    global_weights = {k: _coerce_tensor(
                        v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
                    needs_rollback_flag = True
                    rollback_reason_str = f"weight_stability_check_failed_rollback_to_best_round_{BEST_ROUND_ID}"
                elif STABLE_GLOBAL_WEIGHTS is not None:
                    print(
                        f"[Cloud Server] 回退到最後一個穩定的權重（輪次: {STABLE_ROUND_ID}，穩定性檢查觸發）"
                    )
                    global_weights = {k: _coerce_tensor(
                        v).clone() for k, v in STABLE_GLOBAL_WEIGHTS.items()}
                    needs_rollback_flag = True
                    rollback_reason_str = f"weight_stability_check_failed_rollback_to_round_{STABLE_ROUND_ID}"
                elif PREVIOUS_GLOBAL_WEIGHTS is not None:
                    print(f"[Cloud Server] 沒有穩定權重可回退，回退到上一輪的權重（穩定性檢查觸發）")
                    global_weights = {k: _coerce_tensor(v).clone(
                    ) for k, v in PREVIOUS_GLOBAL_WEIGHTS.items()}
                    needs_rollback_flag = True
                    rollback_reason_str = "weight_stability_check_failed_fallback_to_previous"
                else:
                    print(f"[Cloud Server] 沒有權重可回退，使用當前權重但發出警告")

                STABILITY_CHECK_FAILURE_COUNT = 0
            else:
                print(
                    f"[Cloud Server]  觀察中：連續失敗 {STABILITY_CHECK_FAILURE_COUNT} 輪，尚未達到回退閾值 ({STABILITY_CHECK_FAILURE_THRESHOLD})，繼續使用當前權重"
                )
        else:
            if STABILITY_CHECK_FAILURE_COUNT > 0:
                print(
                    f"[Cloud Server] 穩定性檢查通過，重置失敗計數器（之前連續失敗 {STABILITY_CHECK_FAILURE_COUNT} 輪）"
                )
            STABILITY_CHECK_FAILURE_COUNT = 0
        if needs_rollback_flag:

            if BEST_GLOBAL_WEIGHTS is not None:
                print(
                    f"[Cloud Server] 回退到最佳評估權重（round={BEST_ROUND_ID}，f1={BEST_GLOBAL_F1:.4f}，原因: {rollback_reason_str}）"
                )

                global_weights = {k: _coerce_tensor(
                    v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
                needs_rollback_flag = False
                rollback_reason_str = ""
                print(f"[Cloud Server] 已回退到最佳模型權重 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})")
            elif STABLE_GLOBAL_WEIGHTS is not None:
                print(
                    f"[Cloud Server] 回退到最後一個穩定的權重（輪次: {STABLE_ROUND_ID}，原因: {rollback_reason_str}）"
                )
                global_weights = {k: _coerce_tensor(
                    v).clone() for k, v in STABLE_GLOBAL_WEIGHTS.items()}
                needs_rollback_flag = False
                rollback_reason_str = ""
            elif PREVIOUS_GLOBAL_WEIGHTS is not None:
                print(
                    f"[Cloud Server] 沒有穩定權重可回退，回退到上一輪的權重（原因: {rollback_reason_str}）"
                )
                global_weights = {k: _coerce_tensor(v).clone(
                ) for k, v in PREVIOUS_GLOBAL_WEIGHTS.items()}
                needs_rollback_flag = False
                rollback_reason_str = ""
            else:
                print(
                    f"[Cloud Server] 需要回退但沒有權重可回退（原因: {rollback_reason_str}）"
                )
                needs_rollback_flag = False

        if stability_check_passed and not needs_rollback_flag:
            STABLE_GLOBAL_WEIGHTS = {k: _coerce_tensor(
                v).clone() for k, v in global_weights.items()}
            STABLE_ROUND_ID = current_round
            print(f"[Cloud Server] 權重通過穩定性檢查，保存為穩定權重（輪次: {current_round}）")

        PREVIOUS_GLOBAL_WEIGHTS = {k: _coerce_tensor(
            v).clone() for k, v in global_weights.items()}

        try:
            _compute_and_save_global_class_weights()
        except Exception as e:
            print(f"[Cloud Server] 計算全局類別權重失敗: {e}")
        
        print(f"[Cloud Server] Enhanced FedAvg聚合完成: {len(global_weights)}層")
        
        try:
            strat = getattr(config, "AGGREGATION_STRATEGY", {}) or {}
            qg_config = strat.get("quality_gate", {})
            if qg_config.get("enabled", False):
                drop_threshold = float(qg_config.get("f1_drop_threshold", 0.30))
                quick_eval_samples = int(qg_config.get("quick_eval_samples", 1000))
                fallback_to_best = qg_config.get("fallback_to_best_agg", True)
                
                current_global_f1 = None
                try:
                    if hasattr(app, 'state') and hasattr(app.state, 'last_global_f1'):
                        current_global_f1 = float(app.state.last_global_f1)
                except:
                    pass
            
                if current_global_f1 is None or current_global_f1 <= 0:
                    try:
                        import pandas as pd
                        result_dir = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', os.getcwd()))
                        baseline_csv = os.path.join(result_dir, 'cloud_baseline.csv')
                        if os.path.exists(baseline_csv):
                            df = pd.read_csv(baseline_csv)
                            if len(df) > 0 and 'f1_score' in df.columns:
                                last_f1 = float(df['f1_score'].iloc[-1])
                                if last_f1 > 0:
                                    current_global_f1 = last_f1

                                    if hasattr(app, 'state'):
                                        app.state.last_global_f1 = last_f1
                                    print(f"[Cloud Server] Quality Gate: 從 cloud_baseline.csv 讀取上一輪 F1: {current_global_f1:.4f}")
                    except Exception as csv_exc:
                        pass
                
                if current_global_f1 is not None and current_global_f1 > 0:
                    print(f"[Cloud Server] Quality Gate: 快速評估聚合後的權重（當前 Global F1: {current_global_f1:.4f}）")
                    
                    try:
                        current_round = None
                        try:
                            if hasattr(app, 'state') and hasattr(app.state, 'current_round'):
                                current_round = app.state.current_round
                            elif hasattr(app, 'state') and hasattr(app.state, 'last_aggregation_round'):
                                current_round = app.state.last_aggregation_round
                        except Exception as round_exc:
                            print(f"[Cloud Server] Quality Gate: 獲取 current_round 失敗: {round_exc}")
                        
                        if current_round is not None:
                            print(f"[Cloud Server] Quality Gate: 使用輪次 {current_round} 進行快速評估")
                        else:
                            print(f"[Cloud Server] Quality Gate: current_round 為 None，跳過品質檢查")
                        
                        if current_round is not None:
                            eval_result = evaluate_global_model_on_csv(
                                round_id=-6000,
                                global_weights=global_weights,
                                max_samples=quick_eval_samples
                            )
                            
                            if eval_result and 'f1_score' in eval_result:
                                new_f1 = float(eval_result['f1_score'])
                                f1_drop = (current_global_f1 - new_f1) / current_global_f1 if current_global_f1 > 0 else 0.0
                                
                                print(f"[Cloud Server] Quality Gate 評估結果: 新 F1={new_f1:.4f}, 當前 F1={current_global_f1:.4f}, 下降比例={f1_drop:.2%}")
                                
                                if f1_drop > drop_threshold:
                                    print(f"[Cloud Server] Quality Gate 觸發：F1 下降 {f1_drop:.2%} > 閾值 {drop_threshold:.2%}，捨棄本輪聚合")
                                
                                    if fallback_to_best and all_weights:
                                        best_idx = performance_scores.index(max(performance_scores)) if performance_scores else 0
                                        best_agg_data = all_weights[best_idx]
                                        best_agg_id = best_agg_data.get('agg_id', 'unknown')
                                        best_agg_weights = best_agg_data['weights']
                                        
                                        print(f"[Cloud Server] 回退到最佳 Aggregator {best_agg_id} (F1={max(performance_scores):.4f}) 的權重")
                                        
                                        fusion_ratio = float(qg_config.get("fusion_ratio", 1.0))
                                        
                                        if fusion_ratio >= 1.0:
                                            global_weights = {k: _coerce_tensor(v).clone() for k, v in best_agg_weights.items()}
                                            print(f"[Cloud Server] 已使用最佳 Aggregator 權重（100% 完全替換，避免被差的 Global 權重污染）")
                                        elif global_weights and len(global_weights) > 0:
                                            fused_weights = {}
                                            for key in best_agg_weights.keys():
                                                if key in global_weights:
                                                    best_w = _coerce_tensor(best_agg_weights[key])
                                                    global_w = _coerce_tensor(global_weights[key])
                                                    if best_w.shape == global_w.shape:
                                                        fused_weights[key] = fusion_ratio * best_w + (1 - fusion_ratio) * global_w
                                                    else:
                                                        fused_weights[key] = best_w
                                                else:
                                                    fused_weights[key] = _coerce_tensor(best_agg_weights[key])
                                            global_weights = fused_weights
                                            print(f"[Cloud Server] 已融合最佳 Aggregator 權重（融合比例: {fusion_ratio:.1%}）")
                                        else:

                                            global_weights = {k: _coerce_tensor(v).clone() for k, v in best_agg_weights.items()}
                                            print(f"[Cloud Server] 已使用最佳 Aggregator 權重（完全替換）")
                                    else:
                                        print(f"[Cloud Server] 捨棄本輪聚合，維持上一輪 Global 權重")
                                        if global_weights and len(global_weights) > 0:

                                            pass
                                else:
                                    print(f"[Cloud Server] Quality Gate 通過：F1 下降 {f1_drop:.2%} < 閾值 {drop_threshold:.2%}，接受本輪聚合")
                    except Exception as eval_exc:
                        import traceback
                        print(f"[Cloud Server] Quality Gate 快速評估失敗: {eval_exc}")
                        print(f"[Cloud Server] Quality Gate 異常詳情: {traceback.format_exc()}")
                else:
                    if current_global_f1 is None:
                        print(f"[Cloud Server] Quality Gate: last_global_f1 未設置，跳過品質檢查（可能是第一輪）")
                    elif current_global_f1 <= 0:
                        print(f"[Cloud Server] Quality Gate: last_global_f1={current_global_f1} <= 0，跳過品質檢查")
        except Exception as qg_exc:
            import traceback
            print(f"[Cloud Server] Quality Gate 機制執行失敗: {qg_exc}")
            print(f"[Cloud Server] Quality Gate 異常詳情: {traceback.format_exc()}")
        
        prev_global_weights = {k: _coerce_tensor(v).clone() for k, v in global_weights.items()} if global_weights else None
        
        global_weights = _apply_server_ema(global_weights, prev_global_weights)
        
        print(f"[Cloud Server] Server EMA 平滑已應用")
        
        elite_projection_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('elite_weight_projection', {})
        min_best_f1 = float(elite_projection_cfg.get('min_best_f1', 0.3))
        if elite_projection_cfg.get('enabled', True) and BEST_GLOBAL_WEIGHTS is not None and BEST_GLOBAL_F1 > min_best_f1:
            max_distance = float(elite_projection_cfg.get('max_distance', 0.3))
            projection_method = elite_projection_cfg.get('method', 'cosine')
            projection_strength = float(elite_projection_cfg.get('strength', 0.5))
            
            try:
                if projection_method == 'cosine':
                    cosine_sim = _compute_weight_vector_cosine_similarity(BEST_GLOBAL_WEIGHTS, global_weights)
                    distance = 1.0 - cosine_sim
                    print(f"[Cloud Server] 精英權重投影檢查：餘弦相似度={cosine_sim:.4f}, 距離={distance:.4f}, 最大允許={max_distance:.4f}")
                    
                    if distance > max_distance:
                        print(f"[Cloud Server] 權重偏移過大（距離={distance:.4f} > {max_distance:.4f}），進行精英權重投影")
                        alpha = max(0.1, min(0.9, max_distance / distance * projection_strength))
                        print(f"[Cloud Server] 投影參數：alpha={alpha:.4f}（新權重保留比例）")
                        
                        for layer_name in global_weights.keys():
                            if layer_name in BEST_GLOBAL_WEIGHTS:
                                best_w = _coerce_tensor(BEST_GLOBAL_WEIGHTS[layer_name])
                                new_w = _coerce_tensor(global_weights[layer_name])
                                if isinstance(best_w, torch.Tensor) and isinstance(new_w, torch.Tensor):
                                    if best_w.shape == new_w.shape:

                                        projected_w = (1.0 - alpha) * best_w + alpha * new_w
                                        global_weights[layer_name] = projected_w
                        

                        new_cosine_sim = _compute_weight_vector_cosine_similarity(BEST_GLOBAL_WEIGHTS, global_weights)
                        new_distance = 1.0 - new_cosine_sim
                        print(f"[Cloud Server] 精英權重投影完成：新餘弦相似度={new_cosine_sim:.4f}, 新距離={new_distance:.4f}")
                    else:
                        print(f"[Cloud Server] 權重偏移在允許範圍內，無需投影")
                else:
                    total_sq_diff = 0.0
                    total_norm = 0.0
                    for layer_name in global_weights.keys():
                        if layer_name in BEST_GLOBAL_WEIGHTS:
                            best_w = _coerce_tensor(BEST_GLOBAL_WEIGHTS[layer_name])
                            new_w = _coerce_tensor(global_weights[layer_name])
                            if isinstance(best_w, torch.Tensor) and isinstance(new_w, torch.Tensor):
                                if best_w.shape == new_w.shape:
                                    diff = (new_w - best_w).norm().item() ** 2
                                    total_sq_diff += diff
                                    total_norm += best_w.norm().item() ** 2
                    
                    euclidean_distance = math.sqrt(total_sq_diff) / (math.sqrt(total_norm) + 1e-8)
                    print(f"[Cloud Server] 精英權重投影檢查：歐式距離={euclidean_distance:.4f}, 最大允許={max_distance:.4f}")
                    
                    if euclidean_distance > max_distance:
                        print(f"[Cloud Server] 權重偏移過大（距離={euclidean_distance:.4f} > {max_distance:.4f}），進行精英權重投影")
                        alpha = max(0.1, min(0.9, max_distance / euclidean_distance * projection_strength))
                        print(f"[Cloud Server] 投影參數：alpha={alpha:.4f}")
                        
                        for layer_name in global_weights.keys():
                            if layer_name in BEST_GLOBAL_WEIGHTS:
                                best_w = _coerce_tensor(BEST_GLOBAL_WEIGHTS[layer_name])
                                new_w = _coerce_tensor(global_weights[layer_name])
                                if isinstance(best_w, torch.Tensor) and isinstance(new_w, torch.Tensor):
                                    if best_w.shape == new_w.shape:
                                        projected_w = (1.0 - alpha) * best_w + alpha * new_w
                                        global_weights[layer_name] = projected_w
                        
                        new_total_sq_diff = 0.0
                        new_total_norm = 0.0
                        for layer_name in global_weights.keys():
                            if layer_name in BEST_GLOBAL_WEIGHTS:
                                best_w = _coerce_tensor(BEST_GLOBAL_WEIGHTS[layer_name])
                                new_w = _coerce_tensor(global_weights[layer_name])
                                if isinstance(best_w, torch.Tensor) and isinstance(new_w, torch.Tensor):
                                    if best_w.shape == new_w.shape:
                                        diff = (new_w - best_w).norm().item() ** 2
                                        new_total_sq_diff += diff
                                        new_total_norm += best_w.norm().item() ** 2
                        new_euclidean_distance = math.sqrt(new_total_sq_diff) / (math.sqrt(new_total_norm) + 1e-8)
                        print(f"[Cloud Server] 精英權重投影完成：新歐式距離={new_euclidean_distance:.4f}")
                    else:
                        print(f"[Cloud Server] 權重偏移在允許範圍內，無需投影")
            except Exception as e:
                print(f"[Cloud Server] 精英權重投影失敗: {e}")
                import traceback
                traceback.print_exc()
        
        norm_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('weight_norm_regularization', {})
        if norm_cfg.get('enabled', True) and global_weights and len(global_weights) > 0:
            global STABLE_NORM_HISTORY, STABLE_NORM_WINDOW, STABLE_NORM_MULTIPLIER
            
            base_hard_limit = float(norm_cfg.get('hard_limit', 200.0))
            max_norm = float(norm_cfg.get('max_global_l2_norm', 150.0))
            scaling_factor = float(norm_cfg.get('scaling_factor', 0.90))
            use_dynamic_limit = norm_cfg.get('use_dynamic_hard_limit', True)
            
            current_norm = _compute_global_l2_norm(global_weights)
            
            if use_dynamic_limit and len(STABLE_NORM_HISTORY) >= 3:
                stable_norm_mean = np.mean(STABLE_NORM_HISTORY[-STABLE_NORM_WINDOW:])
                dynamic_hard_limit = stable_norm_mean * STABLE_NORM_MULTIPLIER

                hard_limit = max(base_hard_limit, min(dynamic_hard_limit, base_hard_limit * 2.0))
                print(f"[Cloud Server] 動態 hard_limit: 穩定範數均值={stable_norm_mean:.4f}, 計算值={dynamic_hard_limit:.4f}, 最終={hard_limit:.4f} (基礎={base_hard_limit:.4f})")
            else:
                hard_limit = base_hard_limit
                if use_dynamic_limit:
                    print(f"[Cloud Server] 穩定範數歷史不足 ({len(STABLE_NORM_HISTORY)} < 3)，使用固定 hard_limit={hard_limit:.4f}")
            
            if current_norm > hard_limit:
                print(f"[Cloud Server]  perform_federated_averaging: 檢測到權重範數超過硬性上限 ({current_norm:.4f} > {hard_limit:.4f})，強制裁剪")
                global_weights = _apply_weight_norm_regularization(
                    global_weights, max_norm, scaling_factor, 
                    hard_limit=hard_limit, strict_enforcement=True
                )
                new_norm = _compute_global_l2_norm(global_weights)
                if new_norm > hard_limit:
                    scale = hard_limit / new_norm
                    _torch_local = globals().get('torch')
                    if _torch_local:
                        for layer_name in global_weights:
                            w = _coerce_tensor(global_weights[layer_name])
                            if isinstance(w, _torch_local.Tensor) and w.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                                global_weights[layer_name] = w * scale
                    new_norm = _compute_global_l2_norm(global_weights)
                print(f"[Cloud Server] perform_federated_averaging: 強制裁剪後權重範數: {new_norm:.4f} (目標≤{hard_limit:.4f})")
            elif current_norm > max_norm:
                print(f"[Cloud Server] perform_federated_averaging: 檢測到權重範數超過上限 ({current_norm:.4f} > {max_norm:.4f})，應用正則化")
                global_weights = _apply_weight_norm_regularization(
                    global_weights, max_norm, scaling_factor, 
                    hard_limit=hard_limit, strict_enforcement=True
                )
                new_norm = _compute_global_l2_norm(global_weights)
                print(f"[Cloud Server] perform_federated_averaging: 正則化後權重範數: {new_norm:.4f} (目標≤{max_norm:.4f})")
        
        if LAST_ROLLBACK_ROUND is not None and BEST_GLOBAL_WEIGHTS is not None:
            rounds_since_rollback = (current_round - LAST_ROLLBACK_ROUND) if current_round is not None and LAST_ROLLBACK_ROUND is not None else 0
            if rounds_since_rollback <= POST_ROLLBACK_ROUNDS:
                print(f"[Cloud Server] 回退後信任調整：距離回退 {rounds_since_rollback} 輪，應用信任比例 {POST_ROLLBACK_TRUST_ALPHA:.1%}")
                print(f"[Cloud Server]   - 保留 BEST_GLOBAL_WEIGHTS 比例: {POST_ROLLBACK_TRUST_ALPHA:.1%}")
                print(f"[Cloud Server]   - 保留聚合權重比例: {1.0 - POST_ROLLBACK_TRUST_ALPHA:.1%}")
                
                is_at_limits = (CURRENT_SERVER_LR_MULTIPLIER <= MIN_SERVER_LR_MULTIPLIER + 1e-6 and 
                               CURRENT_FEDPROX_MU_MULTIPLIER >= MAX_FEDPROX_MU_MULTIPLIER - 1e-6)
            
                if is_at_limits:
                    print(f"[Cloud Server] 檢測到極限狀態：學習率已達下限 ({CURRENT_SERVER_LR_MULTIPLIER:.4f})，FedProx μ 已達上限 ({CURRENT_FEDPROX_MU_MULTIPLIER:.4f})")
                    print(f"[Cloud Server] 啟用極限恢復策略（僅記錄，不實際調整）：")
                    
                    if rounds_since_rollback > 3:
                        if rounds_since_rollback % 2 == 0:
                            recovery_rate = 1.05
                            simulated_lr = min(1.0, CURRENT_SERVER_LR_MULTIPLIER * recovery_rate)
                            simulated_lr = max(MIN_SERVER_LR_MULTIPLIER, simulated_lr)
                            print(f"[Cloud Server]   - 極限恢復（log）：學習率預期緩慢恢復至 {simulated_lr:.4f}")
                        
                        if rounds_since_rollback % 2 == 0:
                            recovery_rate = 0.98
                            simulated_mu = max(1.0, CURRENT_FEDPROX_MU_MULTIPLIER * recovery_rate)
                            simulated_mu = min(MAX_FEDPROX_MU_MULTIPLIER, simulated_mu)
                            print(f"[Cloud Server]   - 極限恢復（log）：FedProx μ 預期緩慢恢復至 {simulated_mu:.4f}")
                    
                    if rounds_since_rollback <= 4:
                        dynamic_trust_alpha = 0.97
                    elif rounds_since_rollback <= 8:
                        dynamic_trust_alpha = 0.94
                    else:
                        dynamic_trust_alpha = 0.90
                    print(f"[Cloud Server]   - 極限恢復：使用更高信任比例 {dynamic_trust_alpha:.1%}（更保守）")
                else:
                    if CURRENT_SERVER_LR_MULTIPLIER < 1.0:
                        if rounds_since_rollback <= POST_ROLLBACK_ROUNDS // 2:
                            recovery_rate = 1.03
                        else:
                            recovery_rate = 1.05
                        simulated_lr = min(1.0, CURRENT_SERVER_LR_MULTIPLIER * recovery_rate)
                        simulated_lr = max(MIN_SERVER_LR_MULTIPLIER, simulated_lr)
                        print(f"[Cloud Server]   - 學習率逐步恢復（log）：乘數 ≈ {simulated_lr:.4f} (恢復期 {rounds_since_rollback}/{POST_ROLLBACK_ROUNDS}, 恢復速度: {(recovery_rate-1)*100:.0f}%)")
                    
                    if CURRENT_FEDPROX_MU_MULTIPLIER > 1.0:
                        if rounds_since_rollback <= POST_ROLLBACK_ROUNDS // 2:
                            recovery_rate = 0.97
                        else:
                            recovery_rate = 0.95
                        simulated_mu = max(1.0, CURRENT_FEDPROX_MU_MULTIPLIER * recovery_rate)
                        simulated_mu = min(MAX_FEDPROX_MU_MULTIPLIER, simulated_mu)
                        print(f"[Cloud Server]   - FedProx μ 逐步恢復（log）：乘數 ≈ {simulated_mu:.4f} (恢復期 {rounds_since_rollback}/{POST_ROLLBACK_ROUNDS}, 恢復速度: {(1-recovery_rate)*100:.0f}%)")
                    
                    base_trust_alpha = 0.80 - (rounds_since_rollback - 1) * 0.02
                    base_trust_alpha = max(0.5, base_trust_alpha)
                    
                    import random
                    noise = random.uniform(-0.05, 0.05)
                    dynamic_trust_alpha = base_trust_alpha + noise
                    dynamic_trust_alpha = max(0.5, min(0.95, dynamic_trust_alpha))
                    print(f"[Cloud Server]   - 動態信任比例: {dynamic_trust_alpha:.1%} (基礎={base_trust_alpha:.1%}, 擾動={noise:+.1%})")
                
                for layer_name in global_weights.keys():
                    if layer_name in BEST_GLOBAL_WEIGHTS:
                        best_w = _coerce_tensor(BEST_GLOBAL_WEIGHTS[layer_name])
                        agg_w = _coerce_tensor(global_weights[layer_name])
                        if isinstance(best_w, torch.Tensor) and isinstance(agg_w, torch.Tensor):
                            if best_w.shape == agg_w.shape:
                                mixed_w = dynamic_trust_alpha * best_w + (1.0 - dynamic_trust_alpha) * agg_w
                                global_weights[layer_name] = mixed_w
                
                print(f"[Cloud Server] 回退後信任調整完成：已混合 BEST_GLOBAL_WEIGHTS 和聚合權重（動態信任比例: {dynamic_trust_alpha:.1%}）")
            else:
                if rounds_since_rollback > POST_ROLLBACK_ROUNDS:
                    ROLLBACK_COUNT = 0
                    LAST_ROLLBACK_ROUND = None
                    ROLLBACK_STABLE_ROUNDS = 0
                    print(f"[Cloud Server] 回退後已恢復 {rounds_since_rollback} 輪，重置回退計數器")
        
        if global_weights:
            round_download_bytes = _compute_model_size_bytes(global_weights)
            round_download_mb = round_download_bytes / (1024 * 1024)
            print(f"[Cloud Server] 本輪通訊量（下載）: {round_download_mb:.4f} MB ({round_download_bytes:,} bytes)")
            
            import os
            try:
                if hasattr(app, 'state'):
                    app.state.round_upload_mb = round_upload_mb
                    app.state.round_download_mb = round_download_mb
                    app.state.round_total_mb = round_upload_mb + round_download_mb

                os.environ['ROUND_UPLOAD_MB'] = str(round_upload_mb)
                os.environ['ROUND_DOWNLOAD_MB'] = str(round_download_mb)
                os.environ['ROUND_TOTAL_MB'] = str(round_upload_mb + round_download_mb)
            except Exception as comm_e:
                print(f"[Cloud Server] 存儲通訊量數據失敗: {comm_e}")
        else:
            import os
            try:
                if hasattr(app, 'state'):
                    app.state.round_upload_mb = 0.0
                    app.state.round_download_mb = 0.0
                    app.state.round_total_mb = 0.0
                os.environ['ROUND_UPLOAD_MB'] = '0.0'
                os.environ['ROUND_DOWNLOAD_MB'] = '0.0'
                os.environ['ROUND_TOTAL_MB'] = '0.0'
            except Exception:
                pass
        
        return global_weights
        
    except Exception as e:
        print(f"[Cloud Server] Enhanced FedAvg聚合失敗: {e}")

        try:
            if hasattr(app, 'state'):
                app.state.round_upload_mb = 0.0
                app.state.round_download_mb = 0.0
                app.state.round_total_mb = 0.0
            import os
            os.environ['ROUND_UPLOAD_MB'] = '0.0'
            os.environ['ROUND_DOWNLOAD_MB'] = '0.0'
            os.environ['ROUND_TOTAL_MB'] = '0.0'
        except Exception:
            pass
        return {}

def _compute_and_save_global_class_weights():
    try:
        import pandas as pd
        from sklearn.utils.class_weight import compute_class_weight

        label_counts_file = os.path.join(
            config.DATA_PATH, "label_counts_per_client.csv")
        if not os.path.exists(label_counts_file):
            print(f"[Cloud Server] 找不到標籤統計文件: {label_counts_file}")
            return

        df = pd.read_csv(label_counts_file)
        num_classes = int(getattr(config, 'MODEL_CONFIG', {}).get('num_classes', 5))
        if hasattr(config, 'ALL_LABELS'):
            num_classes = len(getattr(config, 'ALL_LABELS', []))
        
        label_cols = [col for col in df.columns if col.startswith('label_')]
        global_counts = {}
        for col in label_cols:
            label_idx = int(col.split('_')[1])

            if 0 <= label_idx < num_classes:
                global_counts[label_idx] = df[col].sum()

        for cls in range(num_classes):
            if cls not in global_counts:
                global_counts[cls] = 0

        classes = sorted([cls for cls in global_counts.keys() if 0 <= cls < num_classes])
        total_samples = sum(global_counts.values())
        y = []
        for cls in classes:
            count = int(global_counts[cls])
            if count > 0:
                y.extend([cls] * count)
        
        if len(y) == 0:
            print(f"[Cloud Server] 沒有找到任何樣本，使用均勻權重")
            full_weights = np.ones(num_classes, dtype=np.float32)
        else:
            y = np.array(y)
            classes_array = np.array(classes, dtype=np.int32)
            class_weights = compute_class_weight(
                'balanced', classes=classes_array, y=y)
            full_weights = np.ones(num_classes, dtype=np.float32)
            for idx, cls in enumerate(classes):
                if 0 <= cls < num_classes:
                    full_weights[cls] = float(class_weights[idx])
        loss_config = getattr(config, 'LOSS_CONFIG', {})
        min_class_weight = float(loss_config.get('min_class_weight', 1.0))
        max_class_weight = float(loss_config.get('max_class_weight', 3.0))
        full_weights = np.clip(
            full_weights, min_class_weight, max_class_weight)
        result_dir = os.environ.get('EXPERIMENT_DIR') or config.LOG_DIR
        os.makedirs(result_dir, exist_ok=True)
        model_dir = os.path.join(result_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        weights_file = os.path.join(model_dir, "global_class_weights.npy")
        weights_dict = {
            "weights": full_weights.tolist(),
            "class_counts": {int(k): int(v) for k, v in global_counts.items() if 0 <= k < num_classes},
            "total_samples": int(total_samples),
            "num_classes": num_classes
        }
        np.save(weights_file, weights_dict, allow_pickle=True)

        print(f"[Cloud Server] 全局類別權重已計算並保存: {weights_file}")
        print(f"[Cloud Server] 類別數: {num_classes} (與配置一致)")
        print(
            f"[Cloud Server] 類別權重: {dict(zip(range(num_classes), full_weights[:num_classes]))}"
        )
        print(f"[Cloud Server] 類別分布: {dict((k, v) for k, v in global_counts.items() if 0 <= k < num_classes)}")

    except Exception as e:
        print(f"[Cloud Server] 計算全局類別權重失敗: {e}")
        import traceback
        traceback.print_exc()

RESIDUAL_BIAS_THRESHOLD = 0.1
BN_MIN_VARIANCE = 1e-3
BN_MIN_BATCHES = 32
BN_MAX_BATCHES = 2048
BN_EMA_DECAY = 0.9
BN_EMA_CACHE: Dict[str, torch.Tensor] = {}
BASELINE_MODE = bool(getattr(config, "BASELINE_MODE", False))
ENABLE_WEIGHT_AMPLIFICATION = (
    (os.environ.get('ENABLE_WEIGHT_AMPLIFICATION',
     '0').lower() not in ('0', 'false', 'no'))
    and not BASELINE_MODE
)
HIDDEN_LAYER_REPAIR_RATIO = 0.15
OUTPUT_LAYER_MEAN_THRESHOLD = -0.05
SERVER_MOMENTUM_STATE: Dict[str, torch.Tensor] = {}
PREVIOUS_GLOBAL_WEIGHTS: Optional[Dict[str, Any]] = None
STABLE_GLOBAL_WEIGHTS: Optional[Dict[str, Any]] = None
STABLE_ROUND_ID: Optional[int] = None
BEST_GLOBAL_WEIGHTS: Optional[Dict[str, Any]] = None
BEST_GLOBAL_F1: float = -1.0
PERFORMANCE_DROP_COUNT: int = 0
PERFORMANCE_DROP_THRESHOLD: int = 2
performance_warning_threshold: float = 0.20
single_drop_threshold: float = 0.40
performance_degradation_threshold: float = 0.20
BEST_ROUND_ID: Optional[int] = None
PEAK_PROTECTION_ENABLED: bool = True
PEAK_PROTECTION_THRESHOLD: float = 0.30
PEAK_PROTECTION_ROUNDS: int = 15
PEAK_PROTECTION_STRICT_MODE: bool = True
last_peak_round: Optional[int] = None
STABLE_NORM_HISTORY: list = []
STABLE_NORM_WINDOW: int = 10
STABLE_NORM_MULTIPLIER: float = 1.5
F1_DROP_OBSERVATION_COUNT: int = 0
F1_DROP_OBSERVATION_START_ROUND: Optional[int] = None
PEAK_PROTECTION_DROP_COUNT: int = 0
PEAK_PROTECTION_DROP_START_ROUND: Optional[int] = None
PEAK_PROTECTION_DROP_THRESHOLD: float = 0.20
PEAK_PROTECTION_DROP_PATIENCE: int = 3
needs_rollback_flag: bool = False
rollback_reason_str: str = ""
PREVIOUS_ROUND_ACCURACY: Optional[float] = None
KD_ALPHA_LOCKED: bool = False
LOCKED_KD_ALPHA: Optional[float] = None
LAST_KD_ALPHA_BEFORE_LOCK: Optional[float] = None
ROLLBACK_COUNT: int = 0
MAX_CONSECUTIVE_ROLLBACKS: int = 2
F1_DROP_DYNAMIC_COUNT: int = 0
POST_ROLLBACK_TRUST_ALPHA: float = 0.90
POST_ROLLBACK_ROUNDS: int = 30
HIGH_F1_PROTECTION_THRESHOLD: float = 0.95
HIGH_F1_PROTECTION_LR_REDUCTION: float = 0.5
HIGH_F1_PROTECTION_TRUST_INCREASE: float = 0.05
HIGH_F1_STABLE_ROUNDS: int = 0
HIGH_F1_MIN_STABLE_ROUNDS: int = 3
COOLING_OFF_ROLLBACK_THRESHOLD: int = 2
LAST_ROLLBACK_ROUND: Optional[int] = None
ROLLBACK_STABLE_ROUNDS: int = 0
MIN_SERVER_LR_MULTIPLIER: float = 2.0
MAX_FEDPROX_MU_MULTIPLIER: float = 2.0
TOP_N_BEST_MODELS: int = 3
BEST_MODELS_HISTORY: list = []
COSINE_SIMILARITY_THRESHOLD: float = 0.3
WEIGHT_NORM_EXPLOSION_THRESHOLD: float = 2.5
SOFT_ROLLBACK_F1_DROP_THRESHOLD: float = 0.30
HARD_ROLLBACK_F1_DROP_THRESHOLD: float = 0.07
ENABLE_ROLLBACK_MECHANISM: bool = True

if os.environ.get("ENABLE_ROLLBACK_MECHANISM", "").strip().lower() in ("1", "true", "yes"):
    ENABLE_ROLLBACK_MECHANISM = True
if "HARD_ROLLBACK_F1_DROP_THRESHOLD" in os.environ:
    try:
        HARD_ROLLBACK_F1_DROP_THRESHOLD = float(os.environ["HARD_ROLLBACK_F1_DROP_THRESHOLD"])
    except ValueError:
        pass
if "POST_ROLLBACK_ROUNDS" in os.environ:
    try:
        POST_ROLLBACK_ROUNDS = int(os.environ["POST_ROLLBACK_ROUNDS"])
    except ValueError:
        pass

LOGITS_VARIANCE_HISTORY: list = []
LOGITS_VARIANCE_DECREASE_COUNT: int = 0
MIN_LOGITS_VARIANCE_THRESHOLD: float = 0.01
STABILITY_CHECK_FAILURE_COUNT: int = 0
STABILITY_CHECK_FAILURE_THRESHOLD: int = 2
TEMPERATURE_SCALING_ENABLED: bool = True
TEMPERATURE_SCALING_T: float = 4.0
DYNAMIC_LR_PENALTY_ENABLED: bool = True
CURRENT_SERVER_LR_MULTIPLIER: float = 1.0
CURRENT_FEDPROX_MU_MULTIPLIER: float = 1.0
ACCURACY_HISTORY: list = []
ACCURACY_STAGNATION_THRESHOLD: float = 0.001
ACCURACY_STAGNATION_ROUNDS: int = 3

def _coerce_tensor(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone().float()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    return value

def _aggregate_weights_weighted_mean(weights_list, weights):
    import torch
    import numpy as np
    
    if not weights_list or len(weights_list) != len(weights):
        return {}
    
    print(f"[Cloud Server] 使用性能加權平均聚合，{len(weights_list)} 個權重")
    
    aggregated_weights = {}
    first_weights = weights_list[0]
    
    for layer_name in first_weights.keys():
        layer_weights_list = []
        for client_weights in weights_list:
            if layer_name in client_weights:
                w = _coerce_tensor(client_weights[layer_name])
                if isinstance(w, torch.Tensor):
                    layer_weights_list.append(w.cpu().numpy())
                elif isinstance(w, np.ndarray):
                    layer_weights_list.append(w)
        
        if not layer_weights_list:
            continue
        
        try:
            weighted_sum = None
            total_weight = 0.0
            
            for i, layer_w in enumerate(layer_weights_list):
                weight = float(weights[i])
                if weighted_sum is None:
                    weighted_sum = layer_w * weight
                else:
                    weighted_sum += layer_w * weight
                total_weight += weight
            
            if total_weight > 0:
                aggregated_layer = weighted_sum / total_weight
                aggregated_weights[layer_name] = torch.from_numpy(aggregated_layer).float()
        except Exception as e:
            print(f"[Cloud Server] 性能加權平均聚合層 {layer_name} 失敗: {e}，使用第一個權重")
            if layer_weights_list:
                w = layer_weights_list[0]
                if isinstance(w, torch.Tensor):
                    aggregated_weights[layer_name] = w.clone()
                elif isinstance(w, np.ndarray):
                    aggregated_weights[layer_name] = torch.from_numpy(w).float()
                else:
                    aggregated_weights[layer_name] = torch.tensor(w, dtype=torch.float32)
    
    print(f"[Cloud Server] 性能加權平均聚合完成")
    return aggregated_weights

def _aggregate_weights_median(weights_list):
    import torch
    import numpy as np
    
    if not weights_list:
        return {}
    
    print(f"[Cloud Server] 使用中位數聚合（FedMedian），{len(weights_list)} 個權重")
    
    aggregated_weights = {}
    first_weights = weights_list[0]
    
    for layer_name in first_weights.keys():

        layer_weights_list = []
        for client_weights in weights_list:
            if layer_name in client_weights:
                w = _coerce_tensor(client_weights[layer_name])
                if isinstance(w, torch.Tensor):
                    layer_weights_list.append(w.cpu().numpy())
                elif isinstance(w, np.ndarray):
                    layer_weights_list.append(w)
        
        if not layer_weights_list:
            continue
    
        try:
            layer_weights_array = np.array(layer_weights_list)
            
            if layer_weights_array.ndim == 0:
                median_weights = layer_weights_array.item()
                aggregated_weights[layer_name] = torch.tensor(median_weights, dtype=torch.float32)
            elif layer_weights_array.ndim == 1:

                median_weights = np.median(layer_weights_array)
                aggregated_weights[layer_name] = torch.tensor(median_weights, dtype=torch.float32)
            else:
                median_weights = np.median(layer_weights_array, axis=0)

                if not isinstance(median_weights, np.ndarray):
                    median_weights = np.array(median_weights)
                aggregated_weights[layer_name] = torch.from_numpy(median_weights).float()
        except Exception as e:
            print(f"[Cloud Server] 中位數聚合層 {layer_name} 失敗: {e}，使用第一個權重")

            if layer_weights_list:
                w = layer_weights_list[0]
                if isinstance(w, torch.Tensor):
                    aggregated_weights[layer_name] = w.clone()
                elif isinstance(w, np.ndarray):
                    aggregated_weights[layer_name] = torch.from_numpy(w).float()
                else:
                    aggregated_weights[layer_name] = torch.tensor(w, dtype=torch.float32)
    
    print(f"[Cloud Server] 中位數聚合完成")
    return aggregated_weights

def _aggregate_weights_trimmed_mean(weights_list, trim_ratio=0.2): 
    import torch
    import numpy as np
    
    if not weights_list:
        return {}
    
    num_clients = len(weights_list)
    trim_count = max(1, int(num_clients * trim_ratio))
    
    print(f"[Cloud Server] 使用修剪平均聚合（Trimmed Mean），{num_clients} 個權重，每端修剪 {trim_count} 個")
    
    aggregated_weights = {}
    first_weights = weights_list[0]
    
    for layer_name in first_weights.keys():

        layer_weights_list = []
        for client_weights in weights_list:
            if layer_name in client_weights:
                w = _coerce_tensor(client_weights[layer_name])
                if isinstance(w, torch.Tensor):
                    layer_weights_list.append(w.cpu().numpy())
                elif isinstance(w, np.ndarray):
                    layer_weights_list.append(w)
        
        if not layer_weights_list:
            continue
        
        layer_weights_array = np.array(layer_weights_list)
        
        if len(layer_weights_array.shape) == 1:

            sorted_weights = np.sort(layer_weights_array, axis=0)
            trimmed_weights = sorted_weights[trim_count:-trim_count] if trim_count > 0 else sorted_weights
            mean_weights = np.mean(trimmed_weights, axis=0)
        else:
            original_shape = layer_weights_array.shape[1:]
            flattened = layer_weights_array.reshape(num_clients, -1)
        
            sorted_flat = np.sort(flattened, axis=0)
            trimmed_flat = sorted_flat[trim_count:-trim_count] if trim_count > 0 else sorted_flat
            mean_flat = np.mean(trimmed_flat, axis=0)
            
            mean_weights = mean_flat.reshape(original_shape)
        
        aggregated_weights[layer_name] = torch.from_numpy(mean_weights).float()
    
    print(f"[Cloud Server] 修剪平均聚合完成")
    return aggregated_weights

def _compute_weight_vector_cosine_similarity(weights1: Dict[str, Any], weights2: Dict[str, Any]) -> float:
    if torch is None:
        return 1.0

    try:
        vec1_list = []
        vec2_list = []
        
        common_keys = set(weights1.keys()) & set(weights2.keys())
        if not common_keys:
            return 0.0
        
        for key in common_keys:
            w1 = _coerce_tensor(weights1[key])
            w2 = _coerce_tensor(weights2[key])
            
            if not isinstance(w1, torch.Tensor) or not isinstance(w2, torch.Tensor):
                continue
            if w1.dtype not in (torch.float32, torch.float64, torch.float16):
                continue
            if w2.dtype not in (torch.float32, torch.float64, torch.float16):
                continue
            
            if w1.shape != w2.shape:
                continue
            
            vec1_list.append(w1.flatten())
            vec2_list.append(w2.flatten())
        
        if not vec1_list:
            return 0.0
        
        vec1 = torch.cat(vec1_list)
        vec2 = torch.cat(vec2_list)
        
        dot_product = torch.dot(vec1, vec2).item()
        norm1 = torch.norm(vec1).item()
        norm2 = torch.norm(vec2).item()
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        return float(cosine_sim)
    except Exception as e:
        print(f"[Cloud Server] 計算餘弦相似度失敗: {e}")
        return 0.0

def _compute_model_size_bytes(weights: Dict[str, Any]) -> int:
    if torch is None:
        return 0
    
    try:
        total_params = 0
        for key, w in weights.items():
            if 'num_batches_tracked' in key:
                continue
            
            t = _coerce_tensor(w)
            if not isinstance(t, torch.Tensor):
                continue
            if t.dtype not in (torch.float32, torch.float64, torch.float16):
                continue
            
            total_params += t.numel()
        
        size_bytes = total_params * 4
        return int(size_bytes)
    except Exception as e:
        print(f"[Cloud Server] 計算模型大小失敗: {e}")
        return 0

def _analyze_aggregator_weights_with_dbi(all_weights: list, current_round: Optional[int] = None):
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score
    except Exception as e:
        print(f"[Cloud Server] DBI 權重分析跳過：缺少 sklearn 相關套件 ({e})")
        return set(), "monitor", 1.0

    try:
        security_cfg = getattr(config, "SECURITY_CONFIG", {}) or {}
        dbi_cfg = security_cfg.get("dbi_weight_anomaly", {}) or {}
        if not dbi_cfg.get("enabled", False):
            return set(), "monitor", 1.0
        pca_dim = int(dbi_cfg.get("pca_dim", 32))
        cluster_k = int(dbi_cfg.get("cluster_k", 2))
        min_cluster_ratio = float(dbi_cfg.get("min_cluster_ratio", 0.1))
        distance_threshold = float(dbi_cfg.get("distance_threshold", 1.5))
        log_top_k = int(dbi_cfg.get("log_top_k", 5))
        action = str(dbi_cfg.get("action", "monitor") or "monitor").lower()
        soft_factor = float(dbi_cfg.get("soft_factor", 0.3))

        if not all_weights or len(all_weights) < max(cluster_k, 2):
            return set(), action, soft_factor

        if torch is None:
            print("[Cloud Server] DBI 權重分析跳過：torch 不可用")
            return set(), action, soft_factor
        vectors = []
        agg_ids = []
        for agg_data in all_weights:
            agg_id = agg_data.get("agg_id", "unknown")
            weights = agg_data.get("weights", {})
            if not weights:
                continue
            flat_list = []
            for name, w in weights.items():
                t = _coerce_tensor(w)
                if not isinstance(t, torch.Tensor):
                    continue
                if t.dtype not in (torch.float32, torch.float64, torch.float16):
                    continue
                flat_list.append(t.flatten().cpu())
            if not flat_list:
                continue
            vec = torch.cat(flat_list)
            vectors.append(vec.numpy())
            agg_ids.append(agg_id)
        if len(vectors) < max(cluster_k, 2):
            return set(), action, soft_factor
        import numpy as np
        X = np.stack(vectors, axis=0)
        try:
            max_dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
            if max_dim < 1:
                print(f"[Cloud Server] DBI 權重分析：樣本數或特徵數不足，無法進行 PCA（樣本數={X.shape[0]}, 特徵數={X.shape[1]}）")
                return set(), action, soft_factor
            dim = max_dim
            pca = PCA(n_components=dim)
            X_pca = pca.fit_transform(X)
        except Exception as e:
            print(f"[Cloud Server] DBI 權重分析：PCA 失敗 ({e})，跳過本輪分析")
            return set(), action, soft_factor

        try:
            k = min(cluster_k, X_pca.shape[0])
            if k < 2:
                return set(), action, soft_factor
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            dbi_value = davies_bouldin_score(X_pca, labels)
        except Exception as e:
            print(f"[Cloud Server] DBI 權重分析：聚類或 DBI 計算失敗 ({e})")
            return set(), action, soft_factor

        unique, counts = np.unique(labels, return_counts=True)
        total = float(len(labels))
        cluster_info = []
        for cid, cnt in zip(unique, counts):
            ratio = cnt / total
            center = kmeans.cluster_centers_[cid]
            cluster_info.append((cid, cnt, ratio, center))

        main_cluster = max(cluster_info, key=lambda x: x[1])
        main_id, main_cnt, main_ratio, main_center = main_cluster

        def _dist(a, b):
            return float(np.linalg.norm(a - b))

        suspicious_clusters = []
        for cid, cnt, ratio, center in cluster_info:
            if cid == main_id:
                continue
            distance = _dist(center, main_center)
            if ratio < min_cluster_ratio and distance > distance_threshold:
                suspicious_clusters.append(
                    {
                        "cluster_id": int(cid),
                        "count": int(cnt),
                        "ratio": float(ratio),
                        "distance": float(distance),
                    }
                )

        suspicious_ids = set()
        if suspicious_clusters:
            suspicious_cluster_ids = {sc["cluster_id"] for sc in suspicious_clusters}
            for idx, lab in enumerate(labels):
                if lab in suspicious_cluster_ids:
                    suspicious_ids.add(agg_ids[idx])

        round_str = f"round={current_round}" if current_round is not None else "round=?"
        print(f"[Cloud Server] ConfShield/DBI 權重分析（{round_str}）")
        print(
            f"[Cloud Server]   - 聚合器數量: {len(labels)}, "
            f"cluster_k={k}, DBI={dbi_value:.4f}"
        )
        print(
            f"[Cloud Server]   - 主群 cluster={main_id}, "
            f"count={main_cnt}, ratio={main_ratio:.2%}"
        )
        if not suspicious_clusters:
            print("[Cloud Server]   - 未發現明顯異常小集群（monitor 模式）")
        else:
            print(
                f"[Cloud Server]   - 發現 {len(suspicious_clusters)} 個疑似異常小集群 "
                f"(min_ratio<{min_cluster_ratio:.2f}, dist>{distance_threshold:.2f})"
            )
            for sc in suspicious_clusters:
                print(
                    f"[Cloud Server]     * cluster={sc['cluster_id']}, "
                    f"count={sc['count']}, ratio={sc['ratio']:.2%}, "
                    f"dist={sc['distance']:.3f}"
                )

            if log_top_k > 0:
                for sc in suspicious_clusters:
                    cid = sc["cluster_id"]
                    members = [
                        agg_ids[i]
                        for i, lab in enumerate(labels)
                        if lab == cid
                    ]
                    members = members[:log_top_k]
                    print(
                        f"[Cloud Server]       - cluster={cid} 代表性 Aggregator: {members}"
                    )

        return suspicious_ids, action, soft_factor

    except Exception as e:
        print(f"[Cloud Server] DBI 權重分析過程發生錯誤: {e}")
        return set(), "monitor", 1.0


def _compute_global_l2_norm(weights: Dict[str, Any]) -> float:
    if not weights or len(weights) == 0:
        return 0.0
    
    try:
        _torch_local = globals().get('torch')
        if _torch_local is None:
            return 0.0
        
        total_sq = 0.0
        for layer_name, layer_weights in weights.items():

            if 'num_batches_tracked' in layer_name:
                continue
            
            w = _coerce_tensor(layer_weights)
            if isinstance(w, _torch_local.Tensor):

                if w.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                    total_sq += float(w.norm().item() ** 2)
        
        return math.sqrt(total_sq) if total_sq > 0 else 0.0
    except Exception as e:
        print(f"[Cloud Server] 計算全局 L2 範數失敗: {e}")
        return 0.0

def _apply_weight_norm_regularization(weights: Dict[str, Any], max_norm: float, scaling_factor: float = 0.95, hard_limit: Optional[float] = None, strict_enforcement: bool = False) -> Dict[str, Any]:
    if not weights or len(weights) == 0:
        return weights

    try:
        _torch_local = globals().get('torch')
        if _torch_local is None:
            return weights
        
        current_norm = _compute_global_l2_norm(weights)
        
        if hard_limit is not None and current_norm > hard_limit:
            print(f"[Cloud Server]  權重範數超過硬性上限 ({current_norm:.4f} > {hard_limit:.4f})，強制裁剪")

            scale = hard_limit / current_norm
            regularized_weights = {}
            for layer_name, layer_weights in weights.items():
                w = _coerce_tensor(layer_weights)
                if isinstance(w, _torch_local.Tensor):
                    if w.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                        regularized_weights[layer_name] = w * scale
                    else:
                        regularized_weights[layer_name] = w
                else:
                    regularized_weights[layer_name] = layer_weights
            new_norm = _compute_global_l2_norm(regularized_weights)
            print(f"[Cloud Server] 強制裁剪完成: 新範數={new_norm:.4f} (目標≤{hard_limit:.4f})")
            return regularized_weights
        
        if strict_enforcement and current_norm > max_norm * 0.9:
            print(f"[Cloud Server] 嚴格執行模式: 當前範數={current_norm:.4f} 接近上限={max_norm:.4f}，提前正則化")

            target_norm = max_norm * 0.9
            scale = target_norm / current_norm
        elif current_norm <= max_norm:
            return weights
        else:
            scale = max_norm / current_norm * scaling_factor
        
        print(f"[Cloud Server] 權重範數正則化: 當前範數={current_norm:.4f}, 上限={max_norm:.4f}, 縮放比例={scale:.4f}")
        
        regularized_weights = {}
        for layer_name, layer_weights in weights.items():
            w = _coerce_tensor(layer_weights)
            if isinstance(w, _torch_local.Tensor):

                if w.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                    regularized_weights[layer_name] = w * scale
                else:
                    regularized_weights[layer_name] = w
            else:
                regularized_weights[layer_name] = layer_weights
        
        new_norm = _compute_global_l2_norm(regularized_weights)

        if new_norm > max_norm * 1.01:
            print(f"[Cloud Server] 警告：正則化後範數仍超過上限 ({new_norm:.4f} > {max_norm:.4f})，進行二次正則化")

            scale2 = max_norm / new_norm
            for layer_name in regularized_weights:
                w = regularized_weights[layer_name]
                if isinstance(w, _torch_local.Tensor) and w.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                    regularized_weights[layer_name] = w * scale2
            new_norm = _compute_global_l2_norm(regularized_weights)
        
        print(f"[Cloud Server] 權重範數正則化完成: 新範數={new_norm:.4f} (目標≤{max_norm:.4f})")
        
        return regularized_weights
    except Exception as e:
        print(f"[Cloud Server] 權重範數正則化失敗: {e}")
        import traceback
        print(f"[Cloud Server] 詳細錯誤: {traceback.format_exc()}")
        return weights

def _check_weights_identical_numerical(prev_weights: Dict[str, Any], new_weights: Dict[str, Any], 
                                       key_layers: list[str], threshold: float = 1e-6) -> tuple[bool, int, int]:
    if not prev_weights or not new_weights:
        return False, 0, 0
    
    try:
        _torch_local = globals().get('torch')
        if _torch_local is None:
            return False, 0, 0
        
        identical_count = 0
        total_compared = 0
        
        for layer_name in key_layers:
            if layer_name not in prev_weights or layer_name not in new_weights:
                continue
            
            prev_t = _coerce_tensor(prev_weights[layer_name])
            new_t = _coerce_tensor(new_weights[layer_name])
            

            if prev_t.dtype not in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                continue
            if new_t.dtype not in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                continue
            
            if prev_t.shape != new_t.shape:
                continue
            
            l2_distance = (prev_t - new_t).norm().item()
            total_compared += 1
            
            if l2_distance < threshold:
                identical_count += 1
        
        is_identical = (total_compared > 0) and (identical_count == total_compared)
        
        return is_identical, identical_count, total_compared
    except Exception as e:
        print(f"[Cloud Server] 數值比較失敗: {e}")
        return False, 0, 0

def _check_weight_stability(current_weights: Dict[str, Any], previous_weights: Optional[Dict[str, Any]]) -> bool:
    if previous_weights is None:
        return True

    if torch is None:
        return True

    try:
        if 'output_layer.weight' not in current_weights or 'output_layer.weight' not in previous_weights:
            return True

        curr_weight = _coerce_tensor(current_weights['output_layer.weight'])
        prev_weight = _coerce_tensor(previous_weights['output_layer.weight'])

        curr_norm = curr_weight.norm().item()
        prev_norm = prev_weight.norm().item()
        if prev_norm > 0 and curr_norm < prev_norm * 0.3:
            print(
                f"[Cloud Server] 輸出層權重範數急劇下降: "
                f"{prev_norm:.4f} -> {curr_norm:.4f} "
                f"(下降 {((prev_norm - curr_norm) / prev_norm * 100):.1f}%)"
            )
            return False

        if curr_weight.shape[0] > 1:
            per_class_norms = [curr_weight[i].norm().item()
                                                   for i in range(curr_weight.shape[0])]
            norm_std = float(np.std(per_class_norms))
            norm_mean = float(np.mean(per_class_norms))
            if norm_mean > 0 and norm_std / norm_mean < 0.01:
                print(
                    f"[Cloud Server] 各類別權重範數過於接近 (std/mean={norm_std/norm_mean:.4f})，可能導致只預測單一類別")
                return False

        if 'output_layer.bias' in current_weights:
            curr_bias = _coerce_tensor(current_weights['output_layer.bias'])
            bias_values = curr_bias.cpu().numpy()
            bias_std = float(np.std(bias_values))
            bias_range = float(np.max(bias_values) - np.min(bias_values))

            if bias_range < 0.05:
                print(
                    f"[Cloud Server] 輸出層偏置範圍過小 ({bias_range:.4f})，可能導致無法區分類別"
                )
                return False


            if bias_std < 0.02:
                print(
                    f"[Cloud Server] 輸出層偏置標準差過小 ({bias_std:.4f})，可能導致只預測單一類別"
                )
                return False

        if curr_weight.shape == prev_weight.shape:
            weight_diff = (curr_weight - prev_weight).norm().item()
            prev_norm = prev_weight.norm().item()

            if prev_norm > 0 and weight_diff / prev_norm > 5.0:
                print(
                    f"[Cloud Server] 輸出層權重變化過大 ({weight_diff/prev_norm*100:.1f}%)，可能導致不穩定"
                )
                return False

        return True
    except Exception as e:
        print(f"[Cloud Server] 權重穩定性檢查失敗: {e}")

        return True

def _apply_server_momentum(
    prev_weights: Dict[str, Any],
    new_weights: Dict[str, Any],
) -> Dict[str, Any]:
    global SERVER_MOMENTUM_STATE

    cfg = getattr(config, "AGGREGATION_CONFIG", {}).get("server_momentum", {})
    if not bool(cfg.get("enabled", False)):
        return new_weights
    if not prev_weights or len(prev_weights) == 0:
        return new_weights

    _torch_local = globals().get('torch')
    if _torch_local is None:
        print(f"[Cloud Server] 警告：torch 未安裝，跳過伺服器動量更新，直接使用新權重")
        return new_weights

    momentum = float(cfg.get("momentum", 0.9))
    use_nesterov = bool(cfg.get("nesterov", False))

    updated: Dict[str, Any] = {}
    for layer_name, w_new in new_weights.items():
        w_prev = prev_weights.get(layer_name, w_new)
        w_prev_t = _coerce_tensor(w_prev)
        w_new_t = _coerce_tensor(w_new)

        if layer_name in SERVER_MOMENTUM_STATE:
            v_prev = _coerce_tensor(SERVER_MOMENTUM_STATE[layer_name])
        else:
            v_prev = _torch_local.zeros_like(w_new_t)

        delta = w_new_t - w_prev_t
        v_t = momentum * v_prev + delta
        SERVER_MOMENTUM_STATE[layer_name] = v_t

        if use_nesterov:
            w_t1 = w_prev_t + momentum * v_t + delta
        else:
            w_t1 = w_prev_t + v_t

        updated[layer_name] = w_t1

    print(
        f"[Cloud Server]  已套用伺服器動量聚合 (FedAvgM)：layers={len(updated)}, "
        f"momentum={momentum}, nesterov={use_nesterov}"
    )
    return updated

def _clamp_bn_tracker_tensor(value):
    tensor = _coerce_tensor(value)
    tensor = torch.clamp(tensor, min=float(
        BN_MIN_BATCHES), max=float(BN_MAX_BATCHES))
    return tensor

def _apply_bn_ema(layer_name: str, tensor: torch.Tensor) -> torch.Tensor:
    global BN_EMA_CACHE
    if layer_name not in BN_EMA_CACHE:
        BN_EMA_CACHE[layer_name] = tensor.clone()
        return tensor
    cached = BN_EMA_CACHE[layer_name]
    if cached.shape != tensor.shape:
        BN_EMA_CACHE[layer_name] = tensor.clone()
        return tensor
    ema_tensor = BN_EMA_DECAY * cached + (1 - BN_EMA_DECAY) * tensor
    BN_EMA_CACHE[layer_name] = ema_tensor
    return ema_tensor

def _tensor_norm(value) -> float:
    if torch is not None and isinstance(value, torch.Tensor):
        return float(value.norm().item())
    if isinstance(value, np.ndarray):
        return float(np.linalg.norm(value))
    try:
        return float(torch.tensor(value).float().norm().item())
    except Exception:
        return 0.0

def _select_best_backup_tensor(layer_name: str, sources: list):
    best_tensor = None
    best_norm = 0.0
    best_source = None
    for src in sources:
        weights = src.get('weights', {})
        if layer_name not in weights:
            continue
        candidate = weights[layer_name]
        norm_val = _tensor_norm(candidate)
        if norm_val > best_norm and math.isfinite(norm_val):
            best_norm = norm_val
            best_tensor = candidate
            best_source = src.get('agg_id')
    return best_tensor, best_norm, best_source

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_weights, aggregation_count
    
    global_weights = None
    aggregation_count = 0
    
    app.state.last_aggregation_round = None
    app.state.aggregator_weights = {}
    app.state.best_f1 = -1.0
    app.state.best_weights_path = None
    app.state.eval_drop_streak = 0
    app.state.eval_drop_patience = 3
    app.state.eval_drop_tolerance = 0.02
    
    print(f"[Cloud Server]  雲端服務器啟動完成")

    try:
        app.state.cloud_start_ts = time.time()
    except Exception:
        pass
    print(f"[Cloud Server] 初始化狀態:")
    print(f"  - 全局權重: {'已初始化' if global_weights is not None else '未初始化'}")
    print(f"  - 聚合計數: {aggregation_count}")
    print(f"  - 最後聚合輪次: {app.state.last_aggregation_round}")
    print(
        f"  - 聚合器權重緩衝區: "
        f"{len([agg_id for agg_id, weights in aggregator_weights.items() if len(weights) > 0])} 個聚合器"
    )

    if global_weights:
        print("[Cloud Server]  啟動後立即評估：已排程（global_weights 已就緒）")
        _schedule_global_test_eval(0, global_weights)
    else:
        print("[Cloud Server]  啟動後立即評估：跳過（global_weights 未初始化）")
    
    try:
        all_labels = list(getattr(config, 'ALL_LABELS', []) or [])
        num_classes = len(all_labels)

        input_dim = int(getattr(config, 'MODEL_CONFIG', {}).get(
            'input_dim', 84))
        try:
            test_path = _get_global_test_path()
            if os.path.exists(test_path):
                import pandas as _pd
                _df = _pd.read_csv(test_path, nrows=1)

                for c in ['Attack_label', 'Target Label', 'label', 'Label', 'target_label', 'Attack_type']:
                    if c in _df.columns:
                        _df = _df.drop(columns=[c])
                input_dim = max(1, _df.select_dtypes(
                    include=['number']).shape[1]) or input_dim
        except Exception:
            pass
        if num_classes > 0:
            m = _build_eval_model(input_dim=input_dim, num_classes=num_classes)

            head_ok = False
            try:
                for name, mod in m.named_modules():
                    if hasattr(mod, 'out_features'):
                        if int(getattr(mod, 'out_features')) == int(num_classes):
                            head_ok = True
                            break
            except Exception:
                pass
            print(
                f"[Cloud Server] 標籤/頭對齊檢查: "
                f"ALL_LABELS={all_labels} (num_classes={num_classes}), "
                f"input_dim={input_dim}, head_ok={head_ok}"
            )
            if not head_ok:
                msg = "cloud_startup_model_head_mismatch"
                print(f"[Cloud Server] 啟動拒絕：模型輸出維度與 ALL_LABELS 不一致")
                log_event(
                    msg,
                    f"num_classes={num_classes},labels={all_labels}",
                )
                raise RuntimeError(msg)
    except Exception:
        raise

    def monitor_and_autoheal():
        import time as _time
        while True:
            try:

                with lock:
                    aggs = dict(registered_aggregators)
                if not aggs:
                    _time.sleep(10)
                    continue

                try:
                    if hasattr(app.state, 'cloud_start_ts') and (_time.time() - app.state.cloud_start_ts) < 120:
                        _time.sleep(10)
                        continue
                except Exception:
                    pass
                rounds = {}

                for agg_id, info in aggs.items():
                    host = info.get('host', '127.0.0.1')
                    port = info.get('port')
                    if not port:
                        continue
                    base = f"http://{host}:{port}"
                    try:
                        r = requests.get(
                            f"{base}/aggregation_status", timeout=5)
                        if r.ok:
                            data = r.json()
                            rounds[agg_id] = int(data.get('current_round', 0))
                        else:
                            rounds[agg_id] = 0
                    except Exception:
                        rounds[agg_id] = 0
                if not rounds:
                    _time.sleep(10)
                    continue

                vals = list(rounds.values())
                vals_sorted = sorted(vals)
                median_round = vals_sorted[len(vals_sorted)//2]
                min_round = vals_sorted[0]
                max_round = vals_sorted[-1]
                small_count = sum(1 for v in vals if v <= 2)
                is_fresh_start = (small_count >= max(
                    1, (len(vals)+1)//2)) and (min_round <= 2) and (max_round - min_round) >= 10
                lagging = [
                    aid for aid, rd in rounds.items()
                    if (not is_fresh_start and median_round >= 5 and (rd == 0 or rd <= median_round - 5))
                ]
                leading = [
                    aid for aid, rd in rounds.items()
                    if (
                        (median_round >= 1 and rd >= median_round + 5)
                        or (is_fresh_start and rd > 5)
                    )
                ]

                for aid in leading:
                    info = aggs.get(aid)
                    if not info:
                        continue
                    host = info.get('host', '127.0.0.1')
                    port = info.get('port')
                    if not port:
                        continue
                    base = f"http://{host}:{port}"
                    try:
                        current = int(rounds.get(aid, 0))

                        step_down = max(median_round, current - 5)
                        target = step_down
                        res = requests.post(
                            f"{base}/reset_round", data={'target_round': str(target)}, timeout=5)
                        if res.ok:
                            print(
                                f"[Cloud Server] 超前聚合器回調 {aid}: {current} -> {target}"
                            )
                            log_event(
                                "autoheal_aggregator_round_reset_down",
                                f"agg={aid},from={current},to={target}",
                            )
                    except Exception as _e:
                        print(f"[Cloud Server] 自動修復聚合器 {aid} 失敗: {_e}")

                for aid in lagging:
                    info = aggs.get(aid)
                    if not info:
                        continue
                    host = info.get('host', '127.0.0.1')
                    port = info.get('port')
                    if not port:
                        continue
                    base = f"http://{host}:{port}"
                    try:
                        current = int(rounds.get(aid, 0))
                        if is_fresh_start:

                            step_up = min(current + 1, 2)
                        else:
                            step_up = min(current + 5, median_round)
                        target = step_up
                        res = requests.post(
                            f"{base}/reset_round", data={'target_round': str(target)}, timeout=5)
                        if res.ok:
                            print(
                                f"[Cloud Server] 落後聚合器對齊 {aid}: "
                                f"{current} -> {target} "
                                f"({'fresh-start' if is_fresh_start else 'median'})"
                            )
                            log_event(
                                "autoheal_aggregator_round_reset_up",
                                f"agg={aid},from={current},to={target},fresh={is_fresh_start}",
                            )
                    except Exception as _e:
                        print(f"[Cloud Server] 自動修復聚合器 {aid} 失敗: {_e}")
                _time.sleep(15)
            except Exception as e:
                print(f"[Cloud Server] 自動監控/修復循環異常: {e}")
                _time.sleep(15)

    threading.Thread(target=monitor_and_autoheal, daemon=True).start()

    try:
        _compute_and_save_global_class_weights()
    except Exception as e:
        print(f"[Cloud Server] 啟動時計算全局類別權重失敗: {e}")
    yield
    
    print(f"[Cloud Server]  雲端服務器正在關閉...")
    try:
        _evaluation_executor.shutdown(wait=True, cancel_futures=False)
        print(f"[Cloud Server] 🔚 已等待背景評估執行緒結束")
    except Exception as e:
        print(f"[Cloud Server] 結束背景評估執行緒失敗: {e}")

app = FastAPI(title="Cloud Server API", version="1.0.0", lifespan=lifespan)
_evaluation_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="eval")
should_stop_cloud_logging = False
cloud_server_id = 0
global_weights = None
aggregator_weights = defaultdict(list)
last_curve_stats = {
    'round': None,
    'effective_aggregators': 0,
    'quality_pass': 0,
    'quality_checked': 0
}
aggregation_count = 0
global_version = 0
early_stop_triggered = False

def persist_global_weights_snapshot(round_id: int, weights: Dict[str, Any]) -> None:
    try:
        experiment_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
        persist_dir = os.path.join(experiment_dir, 'persist')
        os.makedirs(persist_dir, exist_ok=True)
        snapshot_path = os.path.join(
            persist_dir, f'global_weights_round_{int(round_id):04d}.pt'
        )
        if torch is not None:
            torch.save(weights, snapshot_path)
        else:
            with open(snapshot_path, 'wb') as f:
                pickle.dump(weights, f)
        latest_path = os.path.join(persist_dir, 'global_weights_latest.pt')
        try:
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(snapshot_path, latest_path)
        except OSError:
            pass
        print(f"[Cloud Server]  已保存全局權重快照: {snapshot_path}")
    except Exception as e:
        print(f"[Cloud Server] 無法持久化全局權重快照: {e}")

class CloudServerResourceMonitor:
    def __init__(self):
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.resource_history = []
    
    def get_system_resources(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            resources = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory.available / (1024**3),
                'timestamp': time.time()
            }
            
            self.resource_history.append(resources)
            if len(self.resource_history) > 100:
                self.resource_history.pop(0)
            
            return resources
        except Exception as e:
            print(f"[Cloud Server] 資源監控失敗: {e}")
            return {'cpu_percent': 0, 'memory_percent': 0, 'memory_available_gb': 0}
    
    def is_high_load(self):
        resources = self.get_system_resources()
        return (resources['cpu_percent'] > self.cpu_threshold or 
                resources['memory_percent'] > self.memory_threshold)
    
    def get_optimization_strategy(self):
        resources = self.get_system_resources()
        
        if resources['cpu_percent'] > 90 or resources['memory_percent'] > 90:
            return 'minimal'
        elif resources['cpu_percent'] > 70 or resources['memory_percent'] > 80:
            return 'reduced'
        else:
            return 'full'

resource_monitor = CloudServerResourceMonitor()
early_stop_reason = ""

if os.environ.get('IGNORE_PERSISTED_STATE') == '1':
    print(f"[Cloud Server] 忽略持久化狀態，強制重置全局版本為0")
    global_version = 0
CLOUD_EVAL_EVERY = int(os.environ.get('CLOUD_EVAL_EVERY', '1'))

registered_aggregators = {}
aggregator_count = 0
lock = threading.Lock()

log_path = None

try:
    CLOUD_THRESHOLD = int(
        math.ceil(0.6 * max(1, getattr(config, 'NUM_AGGREGATORS', 1))))
except Exception:
    CLOUD_THRESHOLD = 1

def log_event(event, detail=""):
    server_id = cloud_server_id if cloud_server_id is not None else "unknown"
    print(f"[Cloud Server {server_id}]  {event}: {detail}")

def _compute_state_dict_hash(state_dict: Dict[str, Any], max_tensors: int = 6) -> str:
    try:
        import hashlib
        import numpy as np
        import torch

        keys = sorted(list(state_dict.keys()))
        parts: List[str] = []
        used = 0
        for k in keys:
            v = state_dict.get(k, None)
            if isinstance(v, torch.Tensor):
                b = v.detach().cpu().contiguous().numpy().tobytes()
            elif isinstance(v, np.ndarray):
                b = v.tobytes()
            else:
                continue
            parts.append(hashlib.md5(b).hexdigest()[:8])
            used += 1
            if used >= max_tensors:
                break
        if not parts:
            return "no_tensor"
        return hashlib.md5("".join(parts).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "hash_failed"

def _save_versioned_global_weights(
    weights: Dict[str, Any],
    round_id: int,
    aggregation_count: int,
    model_version: int,
) -> Optional[Dict[str, Any]]:
    try:
        import json
        import time
        import torch

        exp_dir = os.environ.get("EXPERIMENT_DIR") or config.LOG_DIR
        model_dir = os.path.join(exp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        weights_hash = _compute_state_dict_hash(weights)
        fname = f"global_weights_round{int(round_id)}_agg{int(aggregation_count)}_v{int(model_version)}.pt"
        fpath = os.path.join(model_dir, fname)
        torch.save(weights, fpath)

        meta = {
            "round_id": int(round_id),
            "aggregation_count": int(aggregation_count),
            "model_version": int(model_version),
            "weights_hash": str(weights_hash),
            "path": str(fpath),
            "mtime": float(os.path.getmtime(fpath)),
            "saved_at": float(time.time()),
        }
        with open(fpath + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return meta
    except Exception as e:
        print(f"[Cloud Server] 版本化權重保存失敗: {e}")
        return None

def _read_cloud_baseline_df(exp_dir: str):
    try:
        p = os.path.join(exp_dir, 'cloud_baseline.csv')
        if os.path.exists(p):
            import pandas as _pd
            return _pd.read_csv(p)
    except Exception:
        return None
    return None

def _check_early_stop(exp_dir: str, max_round_cap: int = 500) -> (bool, str):
    try:
        import os
        import glob
        import pandas as pd
        import numpy as _np
        
        client_f1s = []
        client_dirs = glob.glob(os.path.join(exp_dir, 'uav*'))
        for client_dir in client_dirs:
            if os.path.isdir(client_dir):
                client_id = os.path.basename(client_dir)
                curve_file = os.path.join(client_dir, f"{client_id}_curve.csv")
                if os.path.exists(curve_file):
                    try:
                        client_df = pd.read_csv(curve_file)
                        if len(client_df) > 0:
                            f1_col = None
                            for col in ['joint_f1', 'f1', 'val_f1']:
                                if col in client_df.columns:
                                    f1_col = col
                                    break
                            if f1_col:
                                client_f1s.append(client_df[f1_col].iloc[-1])
                    except Exception:
                        pass
        
        if len(client_f1s) >= 10:
            avg_client_f1 = _np.mean(client_f1s)
            max_client_f1 = _np.max(client_f1s)
            
            improving_clients = 0
            stable_clients = 0
            for client_dir in client_dirs:
                if os.path.isdir(client_dir):
                    client_id = os.path.basename(client_dir)
                    curve_file = os.path.join(client_dir, f"{client_id}_curve.csv")
                    if os.path.exists(curve_file):
                        try:
                            client_df = pd.read_csv(curve_file)
                            if len(client_df) >= 5:
                                f1_col = None
                                for col in ['joint_f1', 'f1', 'val_f1']:
                                    if col in client_df.columns:
                                        f1_col = col
                                        break
                                if f1_col:
                                    recent_f1 = client_df[f1_col].tail(5).values
                                    if recent_f1[-1] > recent_f1[0] + 0.001:
                                        improving_clients += 1
                                    elif abs(recent_f1[-1] - recent_f1[0]) < 0.005:
                                        stable_clients += 1
                        except Exception:
                            pass
            
            if avg_client_f1 > 0.92:
                df = _read_cloud_baseline_df(exp_dir)
                if df is not None and len(df) >= 30:
                    recent_f1s = df['f1_score'].tail(30).values if 'f1_score' in df.columns else []
                    if len(recent_f1s) >= 30:
                        max_recent = max(recent_f1s)
                        if improving_clients < len(client_f1s) * 0.1:
                            print(f"[Cloud Server] 平均 F1 = {avg_client_f1:.4f} > 0.92 且穩定，可提前停止")
                            return True, f"high_performance_stable(avg_f1={avg_client_f1:.4f}, improving_clients={improving_clients}/{len(client_f1s)})"
            
            if avg_client_f1 > 0.7:
                print(f"[Cloud Server] 個別客戶端平均 F1 = {avg_client_f1:.4f} > 0.7，訓練有效，不觸發早停")
                return False, ""
            
            if max_client_f1 > 0.8:
                print(f"[Cloud Server] 個別客戶端最高 F1 = {max_client_f1:.4f} > 0.8，訓練有效，不觸發早停")
                return False, ""
        
        df = _read_cloud_baseline_df(exp_dir)
        if df is None or df.empty or 'round' not in df.columns:
            return False, ""
        
        try:
            df = df.sort_values(['round', 'timestamp'])
            df = df.groupby('round', as_index=False).tail(1).sort_values('round')
        except Exception:
            df = df.sort_values('round')
        
        rounds = df['round'].tolist()
        last_round = int(rounds[-1]) if rounds else 0
        MIN_ROUNDS_FOR_EARLY_STOP = 50
        if last_round < MIN_ROUNDS_FOR_EARLY_STOP:
            return False, ""
        if last_round >= max_round_cap:
            return True, f"max_round_cap_reached({last_round} >= {max_round_cap})"
        acc = None
        if 'acc' in df.columns:
            acc = df['acc'].values
        elif 'accuracy' in df.columns:
            acc = df['accuracy'].values
        f1 = None
        if 'joint_f1' in df.columns:
            f1 = df['joint_f1'].values
        elif 'f1_score' in df.columns:
            f1 = df['f1_score'].values
        loss = df['loss'].values if 'loss' in df.columns else None
        if acc is None or f1 is None or loss is None or len(acc) < 6:
            return False, ""

        if len(acc) >= 16 and len(f1) >= 16 and len(loss) >= 8:
            prev8_acc = _np.mean(acc[-16:-8])
            curr8_acc = _np.mean(acc[-8:])
            prev8_f1 = _np.mean(f1[-16:-8])
            curr8_f1 = _np.mean(f1[-8:])
            acc_gain = curr8_acc - prev8_acc
            f1_gain = curr8_f1 - prev8_f1
            loss_not_down = _np.nanmin(loss[-8:]) >= _np.nanmin(loss[:-8])

            if (acc_gain < 0.003 and f1_gain < 0.003) and loss_not_down:

                if len(client_f1s) >= 10:

                    recent_highs = sum(1 for f1_val in client_f1s if f1_val > 0.7)
                    if recent_highs >= 5:
                        print(f"[Cloud Server] {recent_highs} 個客戶端 F1 > 0.7，不觸發早停")
                        return False, ""
                return True, f"low_improvement_and_loss_plateau(acc+={acc_gain:.4f}, f1+={f1_gain:.4f})"

        if len(acc) >= 151 and len(f1) >= 151:
            best_acc = -1.0
            best_f1 = -1.0
            no_new_high = 0
            for i in range(len(acc)):
                improved = False
                if acc[i] > best_acc + 1e-12:
                    best_acc = acc[i]
                    improved = True
                if f1[i] > best_f1 + 1e-12:
                    best_f1 = f1[i]
                    improved = True
                if improved:
                    no_new_high = 0
                else:
                    no_new_high += 1

            if no_new_high >= 150:

                if len(client_f1s) >= 10:
                    avg_client_f1 = _np.mean(client_f1s)
                    if avg_client_f1 > 0.6:
                        print(f"[Cloud Server] 個別客戶端平均 F1 = {avg_client_f1:.4f} > 0.6，不觸發早停")
                        return False, ""
                return True, "no_new_high_for_150_rounds"
    except Exception as _e:
        print(f"[Cloud Server] 早停檢查失敗: {_e}")
        import traceback
        traceback.print_exc()
    return False, ""

def _tensor_from_any(x):
    try:
        if torch is None:
            import torch as _torch_local
        else:
            _torch_local = torch
        
        if isinstance(x, np.ndarray):
            return _torch_local.from_numpy(x).float()
        elif isinstance(x, list):
            return _torch_local.tensor(x, dtype=_torch_local.float32)
        elif 'torch' in str(type(x)):
            t = x.detach()
            return t.float() if t.dtype not in (_torch_local.float32, _torch_local.float64) else t.float()
        else:
            return None
    except Exception:
        return None

def _apply_async_merge(delta_dict: Dict[str, Any], base_version: int) -> Dict[str, Any]:
    return global_weights if global_weights is not None else {}

def log_training_event_cloud(event, info: dict):
    print(f"[Cloud Server] {event}: {info}")

def save_cloud_results(aggregation_result, round_id=None):
    try:
        import csv
        import json
        global global_weights
        global STABLE_NORM_HISTORY, STABLE_NORM_WINDOW, STABLE_NORM_MULTIPLIER
        norm_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('weight_norm_regularization', {})
        base_hard_limit = float(norm_cfg.get('hard_limit', 200.0))
        hard_limit = base_hard_limit
        
        if norm_cfg.get('enabled', True) and global_weights and len(global_weights) > 0:
            max_norm = float(norm_cfg.get('max_global_l2_norm', 150.0))
            scaling_factor = float(norm_cfg.get('scaling_factor', 0.90))
            use_dynamic_limit = norm_cfg.get('use_dynamic_hard_limit', True)
            
            if use_dynamic_limit and len(STABLE_NORM_HISTORY) >= 3:
                stable_norm_mean = np.mean(STABLE_NORM_HISTORY[-STABLE_NORM_WINDOW:])
                dynamic_hard_limit = stable_norm_mean * STABLE_NORM_MULTIPLIER
                hard_limit = max(base_hard_limit, min(dynamic_hard_limit, base_hard_limit * 2.0))
            else:
                hard_limit = base_hard_limit
            
            current_norm = _compute_global_l2_norm(global_weights)
            
            if current_norm > hard_limit:
                print(f"[Cloud Server]  save_cloud_results: 檢測到權重範數超過硬性上限 ({current_norm:.4f} > {hard_limit:.4f})，強制裁剪")
                global_weights = _apply_weight_norm_regularization(
                    global_weights, max_norm, scaling_factor, 
                    hard_limit=hard_limit, strict_enforcement=True
                )
                new_norm = _compute_global_l2_norm(global_weights)
                if new_norm > hard_limit:
                    scale = hard_limit / new_norm
                    _torch_local = globals().get('torch')
                    if _torch_local:
                        for layer_name in global_weights:
                            w = _coerce_tensor(global_weights[layer_name])
                            if isinstance(w, _torch_local.Tensor) and w.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                                global_weights[layer_name] = w * scale
                    new_norm = _compute_global_l2_norm(global_weights)
                print(f"[Cloud Server] save_cloud_results: 強制裁剪後權重範數: {new_norm:.4f} (目標≤{hard_limit:.4f})")
            elif current_norm > max_norm:
                print(f"[Cloud Server] save_cloud_results: 檢測到權重範數超過上限 ({current_norm:.4f} > {max_norm:.4f})，應用正則化")
                global_weights = _apply_weight_norm_regularization(
                    global_weights, max_norm, scaling_factor, 
                    hard_limit=hard_limit, strict_enforcement=True
                )
                new_norm = _compute_global_l2_norm(global_weights)
                print(f"[Cloud Server] save_cloud_results: 正則化後權重範數: {new_norm:.4f} (目標≤{max_norm:.4f})")

        global_l2_norm = ""
        global_hash = ""
        try:
            if global_weights and len(global_weights) > 0:
                computed_norm = _compute_global_l2_norm(global_weights)
                global_l2_norm = f"{computed_norm:.6f}"
            
                hasher = hashlib.sha1()
                for layer_name in sorted(global_weights.keys()):
                    w = _coerce_tensor(global_weights[layer_name])
                    if isinstance(w, (torch.Tensor, np.ndarray)):
                        flat = w.view(-1) if hasattr(w, 'view') else w.flatten()
                        if flat.numel() > 0:
                            sample = flat[: min(2048, flat.numel())]
                            if isinstance(sample, torch.Tensor):
                                sample = sample.detach().cpu().numpy().astype(np.float32)
                            else:
                                sample = sample.astype(np.float32)
                            hasher.update(sample.tobytes())
                global_hash = hasher.hexdigest()[:8]
                
                if computed_norm > hard_limit:
                    print(f"[Cloud Server] 警告：save_cloud_results 記錄的範數 ({computed_norm:.4f}) 仍超過 hard_limit ({hard_limit:.4f})，這不應該發生！")
                
                try:
                    is_stable_round = False
                    if round_id is not None:

                        try:
                            import pandas as pd
                            result_dir = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', os.getcwd()))
                            baseline_csv = os.path.join(result_dir, 'cloud_baseline.csv')
                            if os.path.exists(baseline_csv):
                                df = pd.read_csv(baseline_csv)
                                if len(df) > 0 and 'f1_score' in df.columns:
                                    current_f1 = float(df['f1_score'].iloc[-1])

                                    if current_f1 > 0.5 and len(df) >= 3:
                                        recent_f1s = df['f1_score'].tail(3).values
                                        f1_std = np.std(recent_f1s)
                                        f1_mean = np.mean(recent_f1s)
                                        if f1_std / (f1_mean + 1e-8) < 0.1:
                                            is_stable_round = True
                        except Exception as e:
                            pass
                    
                    if is_stable_round:
                        STABLE_NORM_HISTORY.append(computed_norm)

                        if len(STABLE_NORM_HISTORY) > STABLE_NORM_WINDOW:
                            STABLE_NORM_HISTORY = STABLE_NORM_HISTORY[-STABLE_NORM_WINDOW:]
                        print(f"[Cloud Server] 記錄穩定輪次範數: {computed_norm:.4f} (歷史長度: {len(STABLE_NORM_HISTORY)})")
                except Exception as e:
                    print(f"[Cloud Server] 追蹤穩定範數失敗: {e}")
        except Exception as e:
            print(f"[Cloud Server] 無法計算 global_weights 簽名: {e}")
            import traceback
            traceback.print_exc()

        result_dir = os.environ.get('EXPERIMENT_DIR') \
            or getattr(config, 'LOG_DIR', None) \
            or config.LOG_CONFIG.get("result_dir", "result")
        os.makedirs(result_dir, exist_ok=True)
        
        server_id = cloud_server_id if cloud_server_id is not None else "unknown"
    
        curve_file = os.path.join(result_dir, "cloud_server_curve.csv")
        
        file_exists = os.path.exists(curve_file)
        
        with open(curve_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow([
                    'round',
                    'timestamp',
                    'participating_aggregators',
                    'total_data_size',
                    'aggregation_time',
                    'effective_aggregators',
                    'quality_pass_ratio',
                    'global_l2_norm',
                    'global_weight_hash',
                ])
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            participating_aggregators = len(
                aggregation_result.get('aggregator_ids', []))
            total_data_size = aggregation_result.get('total_data_size', 0)
            aggregation_time = aggregation_result.get(
                'aggregation_timestamp', time.time())

            effective_aggs = aggregation_result.get(
                'effective_aggregators', participating_aggregators)
            quality_ratio = aggregation_result.get('quality_pass_ratio', 1.0)
            
            writer.writerow([
                round_id if round_id else 'final',
                timestamp,
                participating_aggregators,
                total_data_size,
                aggregation_time,
                effective_aggs,
                f"{quality_ratio:.4f}",
                global_l2_norm,
                global_hash,
            ])
        
        print(f"[Cloud Server {server_id}] 雲端聚合曲線數據已保存到 {curve_file}")
        
        log_file = os.path.join(result_dir, "cloud_server_log.csv")
        
        log_file_exists = os.path.exists(log_file)
        
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not log_file_exists:
                writer.writerow([
                    'round', 'timestamp', 'server_id', 'participating_aggregators', 
                    'total_data_size', 'aggregator_ids', 'data_sizes'
                ])
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            aggregator_ids_str = ','.join(
                map(str, aggregation_result.get('aggregator_ids', [])))
            data_sizes_str = ','.join(
                map(str, aggregation_result.get('data_sizes', [])))
            
            writer.writerow([
                round_id if round_id else 'final',
                timestamp,
                server_id,
                participating_aggregators,
                total_data_size,
                aggregator_ids_str,
                data_sizes_str
            ])
        
        print(f"[Cloud Server {server_id}] 雲端聚合日誌數據已保存到 {log_file}")
        
        detailed_file = os.path.join(
            result_dir, "cloud_server_detailed_metrics.json")
        
        detailed_data = {
            'round_id': round_id if round_id else 'final',
            'timestamp': datetime.datetime.now().isoformat(),
            'server_id': server_id,
            'aggregation_info': {
                'participating_aggregators': participating_aggregators,
                'total_data_size': total_data_size,
                'aggregator_ids': aggregation_result.get('aggregator_ids', []),
                'data_sizes': aggregation_result.get('data_sizes', []),
                'aggregation_timestamp': aggregation_time
            },
            'model_info': {
                'has_global_weights': 'global_weights' in aggregation_result,
                'global_weights_keys': list(aggregation_result.get('global_weights', {}).keys()) if aggregation_result.get('global_weights') else []
            },
            'performance_metrics': {
                'aggregation_time': aggregation_time,
                'aggregation_count': aggregation_count
            }
        }
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"[Cloud Server {server_id}] 詳細指標已保存到 {detailed_file}")
        print(f"[Cloud Server {server_id}] 所有標準格式雲端聚合結果保存完成")
        
    except Exception as e:
        print(f"[Cloud Server {server_id}] 保存雲端聚合結果失敗: {e}")
        log_event("save_cloud_results_error", str(e))


def _get_global_test_path() -> str:
    data_path = getattr(config, 'DATA_PATH', '')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scaled_path = os.path.join(
        data_path, 'global_test_scaled.csv') if data_path else None
    raw_path = os.path.join(
        data_path, 'global_test.csv') if data_path else None

    if scaled_path and os.path.exists(scaled_path):
        print(f"[Cloud Server] 使用已標準化的測試資料: {scaled_path}")
        return scaled_path

    if raw_path and os.path.exists(raw_path):
        print(f"[Cloud Server] 使用原始測試資料（未標準化）: {raw_path}")
        return raw_path

    fallback_scaled = os.path.join(base_dir, 'data', 'global_test_scaled.csv')
    fallback_raw = os.path.join(base_dir, 'data', 'global_test.csv')

    if os.path.exists(fallback_scaled):
        print(f"[Cloud Server] 使用已標準化的測試資料（fallback）: {fallback_scaled}")
        return fallback_scaled

    if os.path.exists(fallback_raw):
        print(f"[Cloud Server] 使用原始測試資料（fallback，未標準化）: {fallback_raw}")
        return fallback_raw

    env_path = os.environ.get('GLOBAL_TEST_PATH')
    if env_path and os.path.exists(env_path):
        return env_path

    default_path = os.path.join(data_path, 'global_test.csv') if data_path else os.path.join(
        base_dir, 'data', 'global_test.csv')
    print(f"[Cloud Server] 使用預設路徑（可能不存在）: {default_path}")
    return default_path

def _get_global_test_df() -> Optional["pd.DataFrame"]:
    if pd is None:
        print("[Cloud Server] pandas 未安裝，無法載入全域測試集")
        return None
    test_path = _get_global_test_path()
    if not os.path.exists(test_path):
        print(f"[Cloud Server]  找不到全域測試集 {test_path}，跳過評測")
        return None
    with _GLOBAL_TEST_CACHE_LOCK:
        cached_path = _GLOBAL_TEST_CACHE.get("path")
        cached_df = _GLOBAL_TEST_CACHE.get("df")
        if cached_path == test_path and cached_df is not None:
            return cached_df
        try:
            df = pd.read_csv(test_path, encoding='utf-8-sig', low_memory=False)
        except Exception as e:
            print(f"[Cloud Server] 讀取全域測試集失敗: {e}")
            return None
        _GLOBAL_TEST_CACHE["path"] = test_path
        _GLOBAL_TEST_CACHE["df"] = df
        print(f"[Cloud Server] 已載入全域測試集到記憶體: {test_path}")
        return df

def _build_eval_model(input_dim: int, num_classes: int):

    import torch
    try:
        all_labels = list(getattr(config, 'ALL_LABELS', []) or [])
        expected_classes = int(
            len(all_labels)) if all_labels else int(num_classes)
    except Exception:
        expected_classes = int(num_classes)
    
    model_type = config.MODEL_CONFIG.get('type', 'dnn')

    if model_type == 'transformer':
        from models.transformer import build_transformer
        model = build_transformer(
            input_dim=input_dim,
            output_dim=expected_classes,
            d_model=config.MODEL_CONFIG.get('d_model', 128),
            num_layers=config.MODEL_CONFIG.get('num_layers', 2),
            num_heads=config.MODEL_CONFIG.get('num_heads', 4),
            d_ff=config.MODEL_CONFIG.get('d_ff', None),
            dropout=config.MODEL_CONFIG.get('dropout_rate', 0.3),
            max_seq_len=config.MODEL_CONFIG.get('max_seq_len', input_dim),
            use_positional_encoding=config.MODEL_CONFIG.get('use_positional_encoding', True)
        )
    elif model_type == 'dnn':
        from models.dnn import build_dnn
        model = build_dnn(
            input_dim=input_dim,
            output_dim=expected_classes,
            hidden_dims=config.MODEL_CONFIG.get('hidden_dims', [256, 128, 64]),
            dropout_rate=config.MODEL_CONFIG.get('dropout_rate', 0.3),
            use_batch_norm=config.MODEL_CONFIG.get('use_batch_norm', True),
            use_residual=config.MODEL_CONFIG.get('use_residual', True),
            activation=config.MODEL_CONFIG.get('activation', 'relu')
        )
    elif model_type == 'cnn':
        from models.cnn import build_cnn
        model = build_cnn(input_dim=input_dim, output_dim=expected_classes)
    else:

        from uav_client_fixed import SimpleMLP
        model = SimpleMLP(input_dim=input_dim, num_classes=expected_classes)
    
    return model

def _infer_eval_model_params_from_weights(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    try:
        keys = list(state_dict.keys())
        has_residual = any(k.startswith('residual_layers') for k in keys)
        has_input_reshape = any(k.startswith('input_reshape') for k in keys)

        in_dim = None
        out_dim = None
        hidden_dims = []
        if "layers.0.weight" in state_dict:
            w0 = state_dict["layers.0.weight"]
            if hasattr(w0, "shape") and len(w0.shape) >= 2:
                in_dim = int(w0.shape[1])
                hidden_dims.append(int(w0.shape[0]))

        idx = 1
        while f"layers.{idx}.weight" in state_dict:
            w = state_dict[f"layers.{idx}.weight"]
            if hasattr(w, "shape") and len(w.shape) >= 2:
                hidden_dims.append(int(w.shape[0]))
            idx += 1
        if "output_layer.weight" in state_dict:
            w_out = state_dict["output_layer.weight"]
            if hasattr(w_out, "shape") and len(w_out.shape) >= 2:
                out_dim = int(w_out.shape[0])

        return {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "hidden_dims": hidden_dims if hidden_dims else None,
            "has_residual": has_residual,
            "has_input_reshape": has_input_reshape,
        }
    except Exception:
        return {"in_dim": None, "out_dim": None, "hidden_dims": None, "has_residual": False, "has_input_reshape": False}


def _strip_state_dict_prefix(state_dict: Dict[str, Any], prefixes: List[str]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k
        for p in prefixes:
            if new_key.startswith(p):
                new_key = new_key[len(p):]
        cleaned[new_key] = v
    return cleaned


def _load_client_encoder_state(path: str) -> Optional[Dict[str, Any]]:
    try:
        if torch is None:
            import torch as _torch_local
        else:
            _torch_local = torch
        obj = _torch_local.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if "client_model" in obj:
                obj = obj["client_model"]
            elif "client_state_dict" in obj:
                obj = obj["client_state_dict"]
            elif "state_dict" in obj:
                obj = obj["state_dict"]
        if not isinstance(obj, dict):
            return None
        return _strip_state_dict_prefix(obj, ["client_model.", "model.", "encoder.", "client_encoder."])
    except Exception as e:

        try:
            print(f"[Cloud Server] _load_client_encoder_state 失敗: path={path}, err={e}")
        except Exception:
            pass
        return None

def _find_latest_encoder_path(exp_dir: str) -> Tuple[Optional[str], float]:
    if not exp_dir:
        return None, float("inf")
    import time
    import config_fixed as config
    candidates = []
    candidates.extend(
        [
            os.path.join(exp_dir, "client_encoder_weights.pt"),
            os.path.join(exp_dir, "client_model_weights.pt"),
            os.path.join(exp_dir, "client_model.pt"),
            os.path.join(exp_dir, "client_encoder.pt"),
        ]
    )

    try:
        if os.path.exists(exp_dir):
            for item in os.listdir(exp_dir):
                client_dir = os.path.join(exp_dir, item)
                if os.path.isdir(client_dir) and item.startswith("uav"):
                    p = os.path.join(client_dir, "client_encoder_weights.pt")
                    candidates.append(p)
    except Exception as e:
        print(f"[Cloud Server] 掃描 uav 子目錄失敗: {e}")
    best_path = None
    best_age = float("inf")
    now = time.time()

    for p in candidates:
        try:
            if os.path.exists(p):
                age_min = (now - os.path.getmtime(p)) / 60.0
                if age_min < best_age:
                    best_age = age_min
                    best_path = p
        except Exception:
            continue

    return best_path, best_age

def _load_state_dict_strict_flexible(model, state_dict: Dict[str, Any]) -> bool:

    try:
        import sys
        if 'torch' in sys.modules:
            _torch_local = sys.modules['torch']
        else:
            import torch as _torch_local
    except Exception:
        import torch as _torch_local
    
    try:
        print(f"[Cloud Server] 開始載入聚合權重...")
        print(f"  - 聚合權重層數: {len(state_dict)}")
        print(f"  - 聚合權重層名: {list(state_dict.keys())[:5]}...")
        
        model_keys = set(model.state_dict().keys())
        weight_keys = set(state_dict.keys())
        matched_keys = model_keys.intersection(weight_keys)
        missing_keys = model_keys - weight_keys
        unexpected_keys = weight_keys - model_keys
        
        print(f"  - 模型層數: {len(model_keys)}")
        print(f"  - 匹配層數: {len(matched_keys)}")
        print(f"  - 缺失層數: {len(missing_keys)}")
        print(f"  - 多餘層數: {len(unexpected_keys)}")
        
        if len(matched_keys) == 0:
            print(f"[Cloud Server] 沒有匹配的權重層，權重載入失敗")
            return False
        
        if len(missing_keys) > 0:
            print(f"[Cloud Server] 缺失權重層: {list(missing_keys)[:3]}...")
            
            bn_stats = [k for k in missing_keys if any(bn_key in k for bn_key in [
                                                       'running_mean', 'running_var', 'num_batches_tracked'])]
            if len(bn_stats) > 0:
                print(
                    f"[Cloud Server] 檢測到 {len(bn_stats)} 個 BatchNorm 統計參數缺失，嘗試從聚合權重中恢復"
                )
                
                recovered_count = 0
                for key in bn_stats[:]:

                    if key in state_dict:
                        print(f"[Cloud Server] 從聚合權重中恢復 {key}: {state_dict[key]}")
                        recovered_count += 1
                        bn_stats.remove(key)

                    elif any(bn_key in key for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']):

                        bn_layer_name = key.rsplit('.', 1)[0]
                        try:

                            model_state = model.state_dict()
                            if key in model_state:

                                if 'num_batches_tracked' in key:
                                    bn_module = None
                                    for name, module in model.named_modules():
                                        if name == bn_layer_name and (isinstance(module, _torch_local.nn.BatchNorm1d) or isinstance(module, _torch_local.nn.BatchNorm2d)):
                                            bn_module = module
                                            break
                                    if bn_module is not None:

                                        if hasattr(bn_module, 'num_batches_tracked'):
                                            state_dict[key] = bn_module.num_batches_tracked.clone()
                                            print(f"[Cloud Server] 從模型 BatchNorm 層恢復 {key}: {state_dict[key].item()}")
                                            recovered_count += 1
                                            bn_stats.remove(key)
                                        else:
                                            estimated_value = max(1, len(state_dict) // 10)
                                            state_dict[key] = _torch_local.tensor(estimated_value, dtype=_torch_local.long)
                                            print(f"[Cloud Server] 使用估算值初始化 {key}: {estimated_value}")
                                            bn_stats.remove(key)
                                    else:
                                        state_dict[key] = _torch_local.tensor(0, dtype=_torch_local.long)
                                        print(f"[Cloud Server] 使用默認值初始化 {key}: 0")
                                        bn_stats.remove(key)
                                elif 'running_mean' in key or 'running_var' in key:
                                    state_dict[key] = model_state[key].clone()
                                    if 'running_mean' in key:
                                        state_dict[key].fill_(0.0)
                                    else:
                                        state_dict[key].fill_(1.0)
                                    print(f"[Cloud Server] 使用模型當前值初始化 {key}: shape={state_dict[key].shape}")
                                    bn_stats.remove(key)
                        except Exception as e:
                            print(f"[Cloud Server] 嘗試恢復 {key} 時出錯: {e}")
                
                if len(bn_stats) > 0:
                    print(f"[Cloud Server] 仍有 {len(bn_stats)} 個 BatchNorm 參數需要初始化")
                    model_state = model.state_dict()
                for key in bn_stats:
                    if key in model_state:
                        if 'running_mean' in key or 'running_var' in key:
                            state_dict[key] = model_state[key].clone()
                            state_dict[key].fill_(
                                0.0 if 'running_mean' in key else 1.0)
                        elif 'num_batches_tracked' in key:
                            estimated_value = max(1, len(state_dict) // 10)
                            state_dict[key] = _torch_local.tensor(estimated_value, dtype=_torch_local.long)
                        print(
                                f"[Cloud Server] 初始化 {key}: {state_dict[key] if isinstance(state_dict[key], _torch_local.Tensor) and state_dict[key].numel() == 1 else state_dict[key].shape}"
                        )
        
        if len(unexpected_keys) > 0:
            print(f"[Cloud Server] 多餘權重層: {list(unexpected_keys)[:3]}...")
        
        try:
            model.load_state_dict(state_dict, strict=False)
            print(
                f"[Cloud Server] 權重載入成功 (匹配 {len(matched_keys)}/{len(model_keys)} 層)"
            )
            return True
        except Exception as e:
            print(f"[Cloud Server] 直接載入失敗: {e}")
            
            fixed = {}
            for k, v in (state_dict or {}).items():
                if isinstance(v, np.ndarray):
                    try:
                        fixed[k] = _torch_local.from_numpy(v).float()
                    except Exception:
                        continue
                elif isinstance(v, (list, tuple)):
                    try:
                        fixed[k] = _torch_local.tensor(v, dtype=_torch_local.float32)
                    except Exception:
                        continue
                else:
                    fixed[k] = v
            
            try:
                model.load_state_dict(fixed, strict=False)
                print(f"[Cloud Server] 權重載入成功 (numpy轉換後)")
                return True
            except Exception as e2:
                print(f"[Cloud Server] numpy轉換後仍失敗: {e2}")
                
                partial_fixed = {}
                model_state = model.state_dict()
                for k in matched_keys:
                    if k in fixed:

                        if k in model_state:
                            if isinstance(fixed[k], _torch_local.Tensor) and isinstance(model_state[k], _torch_local.Tensor):
                                if fixed[k].shape == model_state[k].shape:
                                    partial_fixed[k] = fixed[k]
                                else:
                                    print(f"[Cloud Server] 跳過形狀不匹配的層: {k} (權重: {fixed[k].shape}, 模型: {model_state[k].shape})")
                            else:
                                partial_fixed[k] = fixed[k]
                
                if len(partial_fixed) > 0:
                    try:
                        model.load_state_dict(partial_fixed, strict=False)
                        print(f"[Cloud Server] 權重載入成功 (部分匹配，載入 {len(partial_fixed)}/{len(matched_keys)} 層)")
                        return True
                    except Exception as e3:
                        print(f"[Cloud Server] 部分載入也失敗: {e3}")
                
                print(f"[Cloud Server] 所有載入策略都失敗，無法載入聚合權重")
                return False
                
    except Exception as e:
        print(f"[Cloud Server] 權重載入過程發生異常: {e}")
        return False

def _load_global_scaler(default_dir: str):
    try:
        data_path = getattr(config, 'DATA_PATH', '')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        preprocessor_candidates = [
            os.path.join(data_path, 'preprocessor.pkl') if data_path else None,
            os.path.join(base_dir, 'processed_data', 'preprocessor.pkl'),
            os.path.join(base_dir, 'data', 'preprocessor.pkl')
        ]
        preprocessor_candidates = [
            c for c in preprocessor_candidates if c and os.path.exists(c)]

        for preprocessor_path in preprocessor_candidates:
            try:
                print(
                    f"[Cloud Server] 嘗試從 preprocessor.pkl 載入 scaler: {preprocessor_path}"
                )
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
                    if isinstance(preprocessor, dict) and 'scaler' in preprocessor:
                        scaler = preprocessor['scaler']
                        if isinstance(scaler, StandardScaler):
                            print("[Cloud Server] 成功從 preprocessor.pkl 載入 StandardScaler")
                            return scaler
                        if hasattr(scaler, "transform"):
                            print(f"[Cloud Server] 成功從 preprocessor.pkl 載入 scaler: {type(scaler)}")
                            return scaler
                        print(f"[Cloud Server] preprocessor.pkl 中的 scaler 類型不正確: {type(scaler)}")
            except Exception as e:
                print(f"[Cloud Server] 從 preprocessor.pkl 載入失敗: {e}")
                continue


        candidates = [
            os.environ.get('SCALER_PATH'),
            os.path.join(default_dir, 'scaler.pkl'),
            os.path.join(data_path, 'scaler.pkl') if data_path else None,
            os.path.join(base_dir, 'model', 'global_scaler.pkl'),
            os.path.join(base_dir, 'data', 'scaler.pkl'),
            os.path.join(base_dir, 'scaler.pkl')
        ]

        candidates = [c for c in candidates if c]
        for p in candidates:
            if p and os.path.exists(p):
                print(f"[Cloud Server] 嘗試載入 scaler 文件: {p}")
                try:
                    with open(p, 'rb') as f:
                        scaler = pickle.load(f)

                        if isinstance(scaler, StandardScaler):
                            print(f"[Cloud Server] 成功載入 StandardScaler")
                            return scaler
                        if hasattr(scaler, "transform"):
                            print(f"[Cloud Server] 成功載入 scaler: {type(scaler)}")
                            return scaler
                        print(f"[Cloud Server] 載入的對象不是可用 scaler，類型: {type(scaler)}")
                except Exception as load_error:
                    print(f"[Cloud Server] 載入 scaler 失敗: {load_error}")

                    continue
        print(f"[Cloud Server] 未找到可用的 scaler")
    except Exception as e:
        print(f"[Cloud Server] 載入 scaler 失敗: {e}")
    return None

def _load_feature_cols(experiment_dir: str):
    try:

        data_dir = getattr(config, 'DATA_PATH', None)
        if data_dir:
            p_data = os.path.join(data_dir, 'feature_cols.json')
            if os.path.exists(p_data):
                with open(p_data, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and 'feature_cols' in obj:
                    return obj['feature_cols']
                if isinstance(obj, list):
                    return obj

        p = os.path.join(experiment_dir, 'feature_cols.json')
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            if isinstance(obj, dict) and 'feature_cols' in obj:
                return obj['feature_cols']
            if isinstance(obj, list):
                return obj
    except Exception as e:
        print(f"[Cloud Server] 載入 feature_cols.json 失敗: {e}")
    return None

def train_global_model_on_csv(round_id: int, global_weights: Dict[str, Any], epochs: int = 3) -> Optional[Dict[str, Any]]:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        test_path = _get_global_test_path()
        if not os.path.exists(test_path):
            print(f"[Cloud Server]  找不到全域測試集 {test_path}，跳過訓練")
            return None
            
        df = pd.read_csv(test_path, encoding='utf-8-sig', low_memory=False)
        if df.empty:
            print(f"[Cloud Server] 全域測試集無資料，跳過訓練")
            return None
        
        print(f"[Cloud Server]  開始雲端訓練，輪次 {round_id}")
        
        possible_label_cols = ['Attack_label',
            'Target Label', 'label', 'Label', 'target_label']
        label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
                
        if label_col is None:
            print(f"[Cloud Server] 找不到標籤欄位，跳過訓練")
            return None
            
        feature_cols = [col for col in df.columns if col != label_col]
        X = df[feature_cols].values
        y = df[label_col].values
        
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        input_dim = len(feature_cols)
        num_classes = len(label_encoder.classes_)
        model = _build_eval_model(input_dim, num_classes)
        
        if global_weights:
            try:
                import torch
                import numpy as np
                to_load = {}
                for k, v in global_weights.items():
                    if isinstance(v, np.ndarray):
                        to_load[k] = torch.from_numpy(v)
                    elif isinstance(v, torch.Tensor):
                        to_load[k] = v
                    else:
                        to_load[k] = v
                model.load_state_dict(to_load, strict=False)
                print(f"[Cloud Server] 載入全局權重成功")
            except Exception as e:
                print(f"[Cloud Server] 載入全局權重失敗: {e}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        _calib_enabled = os.environ.get("EVAL_CALIBRATE", "1").strip() != "0"
        _calib_path_env = os.environ.get("EVAL_CALIBRATION_PATH", "").strip()
        _calib_samples = os.environ.get("EVAL_CALIBRATION_SAMPLES", "2000")
        _calib_epochs = os.environ.get("EVAL_CALIBRATION_EPOCHS", "2")
        _calib_lr = os.environ.get("EVAL_CALIBRATION_LR", "0.001")
        _embed_norm = os.environ.get("EVAL_EMBED_NORM", "1")
        print(
            f"[Cloud Server] 校準狀態: enabled={_calib_enabled}, "
            f"path='{_calib_path_env or 'AUTO(val.csv)'}', "
            f"samples={_calib_samples}, epochs={_calib_epochs}, lr={_calib_lr}, "
            f"embed_norm={_embed_norm}"
        )
        if _calib_enabled:
            try:
                calib_path = _calib_path_env
                if not calib_path:
                    data_path = getattr(config, 'DATA_PATH', '')
                    prefer_scaled = bool(os.environ.get("PREFER_SCALED_DATA", str(getattr(config, "PREFER_SCALED_DATA", True))).strip() != "0")
                    if data_path:
                        scaled = os.path.join(data_path, "val_scaled.csv")
                        raw = os.path.join(data_path, "val.csv")
                        if prefer_scaled and os.path.exists(scaled):
                            calib_path = scaled
                        else:
                            calib_path = raw
                    else:
                        calib_path = ""
                if calib_path and os.path.exists(calib_path):
                    df_cal = pd.read_csv(calib_path)

                    calib_label_col = None
                    for col in ['Attack_label', 'Target Label', 'label', 'Label', 'target_label']:
                        if col in df_cal.columns:
                            calib_label_col = col
                            break
                    if calib_label_col is None:
                        print(f"[Cloud Server] 校準資料找不到標籤欄位，跳過校準")
                    else:

                        if feature_cols:
                            X_cal = df_cal[feature_cols].copy()
                        else:
                            X_cal = df_cal.drop(columns=[calib_label_col])
                        y_cal_raw = df_cal[calib_label_col]
                        if pd.api.types.is_numeric_dtype(y_cal_raw):
                            y_cal = y_cal_raw.values.astype(np.int64)
                        else:
                            y_cal = np.array([label_to_id.get(v, -1) for v in y_cal_raw.astype(str).str.strip()])
                            valid_mask = y_cal >= 0
                            X_cal = X_cal[valid_mask].copy()
                            y_cal = y_cal[valid_mask]
                        cal_df = X_cal.copy()
                        cal_df["_label"] = y_cal
                        max_cal = int(os.environ.get("EVAL_CALIBRATION_SAMPLES", "2000"))
                        if len(cal_df) > max_cal:
                            try:
                                from sklearn.model_selection import train_test_split
                                X_tmp, _, y_tmp, _ = train_test_split(
                                    cal_df.drop(columns=["_label"]), cal_df["_label"].values,
                                    train_size=max_cal, stratify=cal_df["_label"].values, random_state=42
                                )
                                X_cal, y_cal = X_tmp, y_tmp
                            except Exception:
                                cal_df = cal_df.sample(n=max_cal, random_state=42)
                                X_cal = cal_df.drop(columns=["_label"])
                                y_cal = cal_df["_label"].values
                        else:
                            X_cal = cal_df.drop(columns=["_label"])
                            y_cal = cal_df["_label"].values


                        if eval_used_encoder:
                            if eval_client_encoder is None:
                                print(f"[Cloud Server] 找不到可用的 encoder，跳過校準")
                            else:
                                eval_client_encoder.eval()
                                emb_batches = []
                                batch_size_cal = 2048
                                with torch.no_grad():
                                    for i in range(0, len(X_cal), batch_size_cal):
                                        batch_np = X_cal[i:i+batch_size_cal].values if hasattr(X_cal, 'values') else X_cal[i:i+batch_size_cal]
                                        batch_X = torch.tensor(batch_np, dtype=torch.float32).to(device)
                                        emb = eval_client_encoder.get_embedding(batch_X).cpu().numpy()
                                        emb_batches.append(emb)
                                X_cal = np.vstack(emb_batches)
                                if embed_norm_stats is not None:
                                    emb_mean, emb_std = embed_norm_stats
                                    X_cal = (X_cal - emb_mean) / emb_std
                        for p in model.parameters():
                            p.requires_grad = False
                        train_params = []
                        if hasattr(model, 'output_layer'):
                            for p in model.output_layer.parameters():
                                p.requires_grad = True
                            train_params = list(model.output_layer.parameters())
                        else:
                            last_linear = None
                            for _, module in model.named_modules():
                                if isinstance(module, _torch_local.nn.Linear):
                                    last_linear = module
                            if last_linear is not None:
                                for p in last_linear.parameters():
                                    p.requires_grad = True
                                train_params = list(last_linear.parameters())

                        if train_params:
                            calib_lr = float(os.environ.get("EVAL_CALIBRATION_LR", "0.001"))
                            calib_epochs = int(os.environ.get("EVAL_CALIBRATION_EPOCHS", "2"))
                            optimizer = _torch_local.optim.AdamW(train_params, lr=calib_lr)
                            loss_fn = _torch_local.nn.CrossEntropyLoss()
                            model.eval()
                            X_cal_np = X_cal.values if hasattr(X_cal, 'values') else X_cal
                            X_cal_tensor = _torch_local.tensor(X_cal_np, dtype=_torch_local.float32).to(device)
                            y_cal_tensor = _torch_local.tensor(y_cal, dtype=_torch_local.long).to(device)
                            batch_size_cal = 512
                            for epoch in range(calib_epochs):
                                epoch_losses = []
                                for i in range(0, len(X_cal_tensor), batch_size_cal):
                                    batch_X = X_cal_tensor[i:i+batch_size_cal]
                                    batch_y = y_cal_tensor[i:i+batch_size_cal]
                                    optimizer.zero_grad()
                                    logits = model(batch_X)
                                    loss = loss_fn(logits, batch_y)
                                    loss.backward()
                                    optimizer.step()
                                    epoch_losses.append(loss.item())
                                if epoch_losses:
                                    print(f"[Cloud Server] 校準訓練 epoch {epoch+1}/{calib_epochs}, loss={np.mean(epoch_losses):.4f}")
                        else:
                            print(f"[Cloud Server] 找不到可訓練的輸出層，跳過校準")
                else:
                    print(f"[Cloud Server] 校準資料不存在: {calib_path}")
            except Exception as e:
                print(f"[Cloud Server] 校準訓練失敗: {e}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            print(
                f"[Cloud Server] Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
            )
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
            
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_avg_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        from sklearn.metrics import f1_score
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        trained_weights = {name: param.cpu()
                                           for name, param in model.named_parameters()}
        
        result = {
            'accuracy': val_accuracy,
            'f1_score': f1,
            'loss': val_avg_loss,
            'samples': len(X_train),
            'trained_weights': trained_weights
        }
        
        print(f"[Cloud Server] 雲端訓練完成: Acc={val_accuracy:.4f}, F1={f1:.4f}")
        return result
        
    except Exception as e:
        print(f"[Cloud Server] 雲端訓練失敗: {e}")
        import traceback
        print(f"[Cloud Server] Traceback: {traceback.format_exc()}")
    return None


def _write_global_metrics(eval_res: Dict[str, Any], round_id: int) -> None:
    try:
        import csv
        exp_dir = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', 'result'))
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, "global_metrics.csv")

        attack_cfg = getattr(config, "ATTACK_CONFIG", {}) or {}
        lf_cfg = attack_cfg.get("label_flipping", {}) if attack_cfg else {}
        mp_cfg = attack_cfg.get("model_poisoning", {}) if attack_cfg else {}

        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "round": int(round_id),
            "accuracy": eval_res.get("accuracy"),
            "f1_score": eval_res.get("f1_score"),
            "joint_acc": eval_res.get("joint_acc"),
            "joint_f1": eval_res.get("joint_f1"),
            "macro_f1": eval_res.get("macro_f1"),
            "precision": eval_res.get("precision"),
            "recall": eval_res.get("recall"),
            "num_samples": eval_res.get("num_samples"),
            "attack_enabled": bool(attack_cfg.get("enabled", False)),
            "malicious_ratio": attack_cfg.get("malicious_ratio", 0.0),
            "malicious_clients": attack_cfg.get("malicious_clients", ""),
            "label_flip_enabled": bool(lf_cfg.get("enabled", False)),
            "label_flip_source": lf_cfg.get("source_label"),
            "label_flip_target": lf_cfg.get("target_label"),
            "model_poison_enabled": bool(mp_cfg.get("enabled", False)),
            "model_poison_method": mp_cfg.get("method"),
            "model_poison_sigma": mp_cfg.get("sigma"),
            "model_poison_replace_prob": mp_cfg.get("replace_prob"),
            "attack_seed": attack_cfg.get("seed", 42),
        }

        header = list(row.keys())
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[Cloud Server] 全域指標寫入失敗: {e}")


def _schedule_global_test_eval(round_id: int, weights: Dict[str, Any]) -> None:
    try:
        print(f"[Cloud Server] _schedule_global_test_eval 被調用: round_id={round_id}, weights_type={type(weights)}, weights_len={len(weights) if weights else 0}")

        enable_eval = os.environ.get("ENABLE_GLOBAL_EVAL", "1").strip()
        print(f"[Cloud Server] ENABLE_GLOBAL_EVAL={enable_eval}")
        if enable_eval == "0":
            print(f"[Cloud Server]  輪次 {round_id} 評估跳過：ENABLE_GLOBAL_EVAL=0")
            return
        if not weights:
            print(f"[Cloud Server] 輪次 {round_id} 評估跳過：weights 為空")
            return

        try:
            if not hasattr(app.state, "evaluated_rounds"):
                app.state.evaluated_rounds = set()
            eval_key = f"round_{int(round_id)}"
            if eval_key in app.state.evaluated_rounds:
                print(f"[Cloud Server]  輪次 {round_id} 評估跳過：已評估過")
                return
            app.state.evaluated_rounds.add(eval_key)
        except Exception:
            pass

        def _run_eval():
            eval_start_time = time.time()
            try:
                print(f"[Cloud Server] [進度] 開始評估輪次 {round_id}（背景執行，權重層數: {len(weights)}）", flush=True)
                max_samples = getattr(config, "GLOBAL_EVAL_MAX_SAMPLES", None)
                print(f"[Cloud Server] [進度] 評估參數: round_id={round_id}, max_samples={max_samples}", flush=True)
                
                eval_timeout = int(os.environ.get("EVAL_TIMEOUT_SECONDS", "300"))
                if eval_timeout > 0:
                    print(f"[Cloud Server] [進度] 評估超時設置: {eval_timeout} 秒 (round_id={round_id})", flush=True)
                
                eval_res = None
                eval_completed = False
                
                import threading
                eval_result_container = {'result': None, 'exception': None, 'completed': False}
                
                def _eval_worker():
                    try:
                        eval_result_container['result'] = evaluate_global_model_on_csv(round_id, weights, max_samples)
                        eval_result_container['completed'] = True
                    except Exception as e:
                        eval_result_container['exception'] = e
                        eval_result_container['completed'] = True
                
                eval_worker_thread = threading.Thread(target=_eval_worker, daemon=True)
                eval_worker_thread.start()
                eval_worker_thread.join(timeout=eval_timeout)
                
                if eval_worker_thread.is_alive():
                    print(f"[Cloud Server] 評估函數超時（{eval_timeout}秒），強制終止 (round_id={round_id})", flush=True)
                    eval_res = None
                elif eval_result_container['exception']:
                    try:
                        raise eval_result_container['exception']
                    except KeyboardInterrupt:
                        print(f"[Cloud Server] 評估被中斷 (KeyboardInterrupt, round_id={round_id})", flush=True)
                        raise
                    except RuntimeError as runtime_e:
                        error_msg = str(runtime_e)
                        print(f"[Cloud Server] 評估運行時錯誤 (RuntimeError, round_id={round_id}): {error_msg}", flush=True)
                        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                            print(f"[Cloud Server] CUDA 相關錯誤，嘗試清理緩存", flush=True)
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    print(f"[Cloud Server] 已清理 CUDA 緩存", flush=True)
                            except Exception:
                                pass
                        import traceback
                        traceback.print_exc()
                        eval_res = None
                    except Exception as eval_e:
                        print(f"[Cloud Server] 評估過程異常 (Exception, round_id={round_id}): {type(eval_e).__name__}: {eval_e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        eval_res = None
                else:
                    eval_res = eval_result_container['result']
                    eval_completed = True
                
                eval_elapsed = time.time() - eval_start_time
                print(f"[Cloud Server] [進度] 評估結果: eval_res={'not None' if eval_res is not None else 'None'}, type={type(eval_res)}, 耗時={eval_elapsed:.2f}秒 (round_id={round_id})", flush=True)
                
                if eval_res is not None:
                    try:
                        _write_global_metrics(eval_res, round_id)
                        print(f"[Cloud Server] 輪次 {round_id} 評估完成並寫入 global_metrics.csv (總耗時={eval_elapsed:.2f}秒)", flush=True)
                    except Exception as write_e:
                        print(f"[Cloud Server] 寫入 global_metrics.csv 失敗 (round_id={round_id}): {write_e}", flush=True)
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[Cloud Server] 輪次 {round_id} 評估返回 None（可能因為單類占比>90%、logits方差過低、測試數據加載失敗或CUDA錯誤）", flush=True)
            except KeyboardInterrupt:
                print(f"[Cloud Server] 評估被中斷 (KeyboardInterrupt, round_id={round_id})", flush=True)
                raise
            except Exception as e:
                eval_elapsed = time.time() - eval_start_time
                print(f"[Cloud Server] 全域測試評估失敗 (round_id={round_id}, 耗時={eval_elapsed:.2f}秒): {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=_run_eval, daemon=True)
        thread.start()
        print(f"[Cloud Server] 輪次 {round_id} 評估已提交到背景執行緒", flush=True)
    except Exception as e:
        print(f"[Cloud Server] 無法啟動全域測試評估: {e}", flush=True)
        import traceback
        traceback.print_exc()

def evaluate_global_model_on_csv(round_id: int, global_weights: Dict[str, Any], max_samples: Optional[int] = None) -> Optional[Dict[str, Any]]:
    import os
    import pandas as pd
    global CURRENT_SERVER_LR_MULTIPLIER, CURRENT_FEDPROX_MU_MULTIPLIER, needs_rollback_flag, rollback_reason_str
    global LOGITS_VARIANCE_HISTORY, LOGITS_VARIANCE_DECREASE_COUNT, F1_DROP_DYNAMIC_COUNT
    global F1_DROP_OBSERVATION_COUNT, F1_DROP_OBSERVATION_START_ROUND
    global ROLLBACK_COUNT, COOLING_OFF_ROLLBACK_THRESHOLD, MIN_SERVER_LR_MULTIPLIER, MAX_FEDPROX_MU_MULTIPLIER
    global MIN_SERVER_LR_MULTIPLIER, MAX_FEDPROX_MU_MULTIPLIER, ROLLBACK_STABLE_ROUNDS, LAST_ROLLBACK_ROUND
    global BEST_GLOBAL_F1, SOFT_ROLLBACK_F1_DROP_THRESHOLD, HARD_ROLLBACK_F1_DROP_THRESHOLD
    global HIGH_F1_PROTECTION_THRESHOLD, HIGH_F1_PROTECTION_LR_REDUCTION, HIGH_F1_PROTECTION_TRUST_INCREASE
    global HIGH_F1_STABLE_ROUNDS, HIGH_F1_MIN_STABLE_ROUNDS, POST_ROLLBACK_TRUST_ALPHA
    global ACCURACY_HISTORY, ACCURACY_STAGNATION_THRESHOLD, ACCURACY_STAGNATION_ROUNDS
    import copy
    import numpy as np

    try:
        import sys
        if 'torch' in sys.modules:
            _torch_local = sys.modules['torch']
        else:
            import torch as _torch_local
    except:
        import torch as _torch_local
    
    eval_weights = {}
    try:
        for k, v in global_weights.items():
            if isinstance(v, _torch_local.Tensor):
                eval_weights[k] = v.detach().clone().cpu()
            elif isinstance(v, np.ndarray):
                eval_weights[k] = v.copy()
            else:
                eval_weights[k] = copy.deepcopy(v)

        global_weights = eval_weights
    except Exception as e:
        print(f"[Cloud Server] 權重深度複製失敗: {e}，使用原始權重")

    try:
        in_mem_hash = _compute_state_dict_hash(global_weights)
        print(f"[Cloud Server] 🔎 Eval Weights (in-memory): round_id={round_id}, weights_hash={in_mem_hash}")
        meta = getattr(app.state, "last_saved_global_weights_meta", None)
        if isinstance(meta, dict) and meta.get("path"):
            p = meta.get("path")
            mtime = None
            try:
                if os.path.exists(p):
                    mtime = os.path.getmtime(p)
            except Exception:
                mtime = None
            print(
                f"[Cloud Server] 🔎 Last Saved Weights: path={p}, "
                f"mtime={mtime}, saved_hash={meta.get('weights_hash')}, "
                f"saved_round={meta.get('round_id')}, saved_agg={meta.get('aggregation_count')}, saved_v={meta.get('model_version')}"
            )
    except Exception as e:
        print(f"[Cloud Server] Eval/Save 版本追蹤日誌失敗: {e}")
    try:
        print(f"[Cloud Server] 開始載入測試數據 (round_id={round_id})", flush=True)
        df = _get_global_test_df()
        if df is None:
            print(f"[Cloud Server] 測試數據載入失敗 (round_id={round_id})，返回 None", flush=True)
            return None
        print(f"[Cloud Server] 測試數據載入成功 (round_id={round_id}): {len(df)} 樣本", flush=True)
        

        possible_label_cols = ['Attack_label',
            'Target Label', 'label', 'Label', 'target_label']
        label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            print(f"[Cloud Server] 找不到標籤欄位，跳過評測 (round_id={round_id})。可用欄位: {list(df.columns)[:10]}...", flush=True)
            return None
        print(f"[Cloud Server] [進度] 找到標籤欄位: {label_col} (round_id={round_id})", flush=True)
        
        if max_samples is not None and len(df) > max_samples:

            from sklearn.model_selection import train_test_split

            y_temp = df[label_col].values

            if not np.issubdtype(y_temp.dtype, np.number):
                from sklearn.preprocessing import LabelEncoder
                le_temp = LabelEncoder()
                y_encoded_temp = le_temp.fit_transform(y_temp)
            else:
                y_encoded_temp = y_temp.astype(int)
            
            try:
                unique_classes, class_counts = np.unique(y_encoded_temp, return_counts=True)
                min_class_count = min(class_counts)
                samples_per_class = max_samples // len(unique_classes)
                
                if min_class_count < samples_per_class:
                    print(f"[Cloud Server] 檢測到類別不平衡：最小類別樣本數={min_class_count}，調整採樣策略")

                    samples_per_class = max(1, min(min_class_count, samples_per_class))
                    adjusted_max_samples = samples_per_class * len(unique_classes)
                    if adjusted_max_samples < max_samples:
                        print(f"[Cloud Server] 調整採樣數量：{max_samples} → {adjusted_max_samples}（確保類別平衡）")
                        max_samples = adjusted_max_samples
                
                df_sampled, _ = train_test_split(
                    df, 
                    test_size=1 - max_samples/len(df), 
                    stratify=y_encoded_temp, 
                    random_state=42
                )
                df = df_sampled
            
                sampled_y = df[label_col].values
                if not np.issubdtype(sampled_y.dtype, np.number):
                    sampled_y_encoded = le_temp.transform(sampled_y)
                else:
                    sampled_y_encoded = sampled_y.astype(int)
                sampled_unique, sampled_counts = np.unique(sampled_y_encoded, return_counts=True)
                class_balance_ratio = min(sampled_counts) / max(sampled_counts) if max(sampled_counts) > 0 else 0.0
                
                print(f"[Cloud Server] 使用分層採樣: {len(df)} 樣本（保持類別平衡，平衡比例={class_balance_ratio:.2f}）")
            except Exception as e:

                print(f"[Cloud Server] 分層採樣失敗: {e}，使用隨機採樣")
                df = df.sample(n=max_samples, random_state=42)
        if df.empty:
            print(f"[Cloud Server] 全域測試集無資料，跳過評測")
            return None
        
        print(f"[Cloud Server] 載入測試資料: {df.shape[0]} 樣本, {df.shape[1]} 欄位")
        experiment_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
        feature_cols_hint = _load_feature_cols(experiment_dir)
        if not feature_cols_hint:
            try:
                data_dir = getattr(config, 'DATA_PATH', None)
                candidate = None
                if data_dir and os.path.isdir(data_dir):
                    for name in ["client_0.csv", "uav_0.csv", "client_1.csv"]:
                        p = os.path.join(data_dir, name)
                        if os.path.exists(p):
                            candidate = p
                            break
                if candidate:
                    tmp_df = pd.read_csv(candidate, nrows=1)
                    cols = [c for c in tmp_df.columns if str(c).lower() not in (
                        'label', 'attack_type', 'attack-type')]
                    if cols:
                        os.makedirs(experiment_dir, exist_ok=True)
                        with open(
                            os.path.join(experiment_dir, "feature_cols.json"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(cols, f, ensure_ascii=False, indent=2)
                        print(
                            f"[Cloud Server] 已自動產生 feature_cols.json（{len(cols)} 欄）"
                        )
                        feature_cols_hint = cols
            except Exception as e:
                print(f"[Cloud Server] 自動產生 feature_cols.json 失敗: {e}")
        if feature_cols_hint:
            missing = [c for c in feature_cols_hint if c not in df.columns]
            if missing:
                msg = (
                    f"feature_cols.json 缺少欄位: {missing[:10]} (共{len(missing)})"
                )
                print(f"[Cloud Server] {msg}，跳過本輪評測")
                log_event("global_eval_skipped_feature_cols_missing", msg)
                return None
        if not feature_cols_hint:
            numeric_cols = [c for c in df.columns if c not in [
                label_col, 'Attack_type'] and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                print(f"[Cloud Server] 無數值型特徵欄位，跳過評測")
                return None
            feature_cols = numeric_cols
        else:
            feature_cols = feature_cols_hint

        experiment_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
        scaler = _load_global_scaler(experiment_dir)

        try:
            if scaler is not None and hasattr(scaler, 'feature_names_in_'):
                scaler_cols = list(getattr(scaler, 'feature_names_in_'))
                miss_for_scaler = [
                    c for c in scaler_cols if c not in df.columns]
                if miss_for_scaler:
                    msg = (
                        "scaler.feature_names_in_ 缺少對應欄位: "
                        f"{miss_for_scaler[:10]} (共{len(miss_for_scaler)})"
                    )
                    print(f"[Cloud Server] {msg}，跳過本輪評測")
                    log_event("global_eval_skipped_scaler_cols_missing", msg)
                    return None

                feature_cols = scaler_cols
        except Exception as _e:
            print(f"[Cloud Server] 檢查 scaler.feature_names_in_ 失敗: {_e}")

        print(f"[Cloud Server] [進度] 開始檢查標籤類型 (round_id={round_id})", flush=True)
        y_raw = df[label_col]
        if pd.api.types.is_numeric_dtype(y_raw):

            print(f"[Cloud Server] 檢測到數值標籤，直接使用: {sorted(y_raw.unique())} (round_id={round_id})", flush=True)
            print(f"[Cloud Server] [進度] 開始處理數值標籤 (round_id={round_id})", flush=True)
            y_numeric = y_raw.values

            max_label = getattr(config, 'NUM_CLASSES', 6) - 1
            valid_mask = (y_numeric >= 0) & (y_numeric <= max_label)
            if not bool(valid_mask.any()):
                print(f"[Cloud Server] 測試集標籤超出範圍 [0,{max_label}]，跳過評測 (round_id={round_id})", flush=True)
                return None
            if not bool(valid_mask.all()):
                invalid_count = (~valid_mask).sum()
                print(f"[Cloud Server] 測試集包含 {invalid_count} 個無效標籤，已忽略 (round_id={round_id})", flush=True)
            df = df[valid_mask].copy()
            y_numeric = df[label_col].values
            y_txt = None

            print(f"[Cloud Server] [進度] 數值標籤，跳過標籤編碼器載入 (round_id={round_id})", flush=True)
            label_to_id = None
        else:
            print(f"[Cloud Server] [進度] 檢測到文字標籤，需要載入標籤編碼器 (round_id={round_id})", flush=True)
            y_txt = df[label_col].astype(str).str.strip()
            y_numeric = None

            print(f"[Cloud Server] [進度] 準備載入標籤編碼器 (round_id={round_id})", flush=True)
        try:
            data_path = getattr(config, 'DATA_PATH', '')
            experiment_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
            label_encoder_paths = [
                os.path.join(
                    data_path, 'label_encoder.pkl') if data_path else None,
                os.path.join(experiment_dir, 'label_encoder.pkl'),
                os.path.join(os.path.dirname(os.path.abspath(
                    __file__)), 'data', 'label_encoder.pkl')
            ]
            label_encoder_paths = [p for p in label_encoder_paths if p]
            print(f"[Cloud Server] [進度] 標籤編碼器候選路徑: {len(label_encoder_paths)} 個 (round_id={round_id})", flush=True)

            le_data = None
            for idx, le_path in enumerate(label_encoder_paths):
                print(f"[Cloud Server] [進度] 檢查路徑 {idx+1}/{len(label_encoder_paths)}: {le_path} (round_id={round_id})", flush=True)
                if os.path.exists(le_path):
                    print(f"[Cloud Server] [進度] 文件存在，開始讀取: {le_path} (round_id={round_id})", flush=True)
                    try:
                        import threading
                        le_data_result = {'data': None, 'error': None}

                        def load_pickle():
                            try:
                                with open(le_path, 'rb') as f:
                                    le_data_result['data'] = pickle.load(f)
                            except Exception as e:
                                le_data_result['error'] = e

                        load_thread = threading.Thread(target=load_pickle, daemon=True)
                        load_thread.start()
                        load_thread.join(timeout=10)

                        if load_thread.is_alive():
                            print(f"[Cloud Server] 載入標籤編碼器超時（10秒）: {le_path} (round_id={round_id})", flush=True)
                            continue
                        elif le_data_result['error']:
                            print(f"[Cloud Server] 載入標籤編碼器失敗: {le_data_result['error']} (round_id={round_id})", flush=True)
                            continue
                        else:
                            le_data = le_data_result['data']
                            print(f"[Cloud Server] 從 {le_path} 載入標籤編碼器 (round_id={round_id})", flush=True)
                        break
                    except Exception as e:
                        print(f"[Cloud Server] 載入標籤編碼器異常: {e} (round_id={round_id})", flush=True)
                        continue
                else:
                    print(f"[Cloud Server] [進度] 文件不存在: {le_path} (round_id={round_id})", flush=True)
            
            if le_data:
                from sklearn.preprocessing import LabelEncoder
                if isinstance(le_data, LabelEncoder):
                    if hasattr(le_data, 'classes_'):
                        label_to_id = {label: idx for idx, label in enumerate(le_data.classes_)}
                        print(f"[Cloud Server] 從 LabelEncoder.classes_ 建立映射: {len(label_to_id)} 個類別 (round_id={round_id})", flush=True)
                    else:
                        raise ValueError("LabelEncoder 物件缺少 classes_ 屬性")
                elif isinstance(le_data, dict) and 'label_mapping' in le_data:
                    label_to_id = le_data['label_mapping']
                    print(f"[Cloud Server] 使用標籤編碼器映射（字典格式）: {len(label_to_id)} 個類別 (round_id={round_id})", flush=True)
                elif isinstance(le_data, dict):
                    if 'classes' in le_data:
                        classes = le_data['classes']
                        label_to_id = {label: idx for idx, label in enumerate(classes)}
                        print(f"[Cloud Server] 從字典的 'classes' 鍵建立映射: {len(label_to_id)} 個類別 (round_id={round_id})", flush=True)
                    elif 'classes_' in le_data:
                        classes = le_data['classes_']
                        label_to_id = {label: idx for idx, label in enumerate(classes)}
                        print(f"[Cloud Server] 從字典的 'classes_' 鍵建立映射: {len(label_to_id)} 個類別 (round_id={round_id})", flush=True)
                    else:
                        raise ValueError(f"label_encoder.pkl 格式不支援: 字典缺少 'label_mapping' 或 'classes'/'classes_' 鍵")
                else:
                    raise ValueError(f"label_encoder.pkl 格式不支援: 類型為 {type(le_data)}")
            else:
                raise FileNotFoundError("找不到 label_encoder.pkl")
        except Exception as e:
            print(f"[Cloud Server] 無法載入標籤編碼器: {e}，使用 config.ALL_LABELS (round_id={round_id})", flush=True)
            all_labels = list(getattr(config, 'ALL_LABELS', []))
            if not all_labels:
                print(f"[Cloud Server] config.ALL_LABELS 為空，無法對齊標籤，跳過評測 (round_id={round_id})", flush=True)
                return None
            label_to_id = {name: i for i, name in enumerate(all_labels)}
        

        print(f"[Cloud Server] [進度] 開始處理標籤映射 (round_id={round_id})", flush=True)
        if y_txt is not None:

            known_labels = list(label_to_id.keys())
            known_mask = y_txt.isin(known_labels)
            if not bool(known_mask.any()):
                print(f"[Cloud Server] 測試集不包含任何已知標籤，跳過評測")
                return None
            if not bool(known_mask.all()):
                unknown = sorted(set(y_txt[~known_mask].unique()))
                print(f"[Cloud Server] 測試集包含未知標籤，已忽略: {unknown[:5]}{'...' if len(unknown)>5 else ''}")
                df = df[known_mask].copy()

                y_txt = df[label_col].astype(str).str.strip()
                y_numeric = np.array([label_to_id[label] for label in y_txt])
            else:

                y_numeric = np.array([label_to_id[label] for label in y_txt])

        print(f"[Cloud Server] [進度] 開始提取特徵數據 (round_id={round_id})", flush=True)
        X_df = df[feature_cols].copy()
        print(f"[Cloud Server] [進度] 特徵數據形狀: {X_df.shape} (round_id={round_id})", flush=True)

        print(f"[Cloud Server] [進度] 開始轉換為數值型別 (round_id={round_id})", flush=True)
        for c in X_df.columns:
            try:
                X_df[c] = pd.to_numeric(X_df[c], errors='coerce')
            except Exception:
                pass
        X_df = X_df.fillna(0.0)
        print(f"[Cloud Server] [進度] 數值型別轉換完成 (round_id={round_id})", flush=True)

        y = y_numeric.astype(np.int64)
        print(f"[Cloud Server] [進度] 標籤處理完成: y.shape={y.shape} (round_id={round_id})", flush=True)


        unique_labels, label_counts = np.unique(y, return_counts=True)
        label_distribution = dict(zip(unique_labels, label_counts))
        print(f"[Cloud Server] 測試數據標籤分佈: {label_distribution}")
        print(f"[Cloud Server] 測試數據總樣本數: {len(y)}, 類別數: {len(unique_labels)}")
        
        if len(unique_labels) == 0:
            print(f"[Cloud Server] 錯誤：測試數據沒有標籤")
            return None
        if len(unique_labels) == 1:
            print(f"[Cloud Server] 警告：測試數據只有一個類別 ({unique_labels[0]})，無法進行多類別評估")

        if len(unique_labels) > 1:
            max_count = max(label_counts)
            min_count = min(label_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"[Cloud Server] 類別不平衡比例: {imbalance_ratio:.2f}")
            if imbalance_ratio > 100:
                print(f"[Cloud Server] 警告：測試數據類別極度不平衡 (比例 > 100:1)")
            elif imbalance_ratio > 10:
                print(f"[Cloud Server] 警告：測試數據類別高度不平衡 (比例 > 10:1)")

        test_path = _get_global_test_path()
        is_scaled_data = 'scaled' in os.path.basename(test_path).lower()
        
        if is_scaled_data:
            print(f"[Cloud Server] 使用已標準化的測試資料，跳過標準化步驟")
            X = X_df.values.astype(np.float32)

            try:
                max_abs = float(np.nanmax(np.abs(X)))
            except Exception:
                max_abs = 0.0
            mean_abs_avg = float(np.nanmean(np.abs(X), axis=None))
            std_avg = float(np.nanmean(np.nanstd(X, axis=0)))
            scaled_like = mean_abs_avg < 0.5 and 0.5 < std_avg < 2.5

            if (not scaled_like) and (mean_abs_avg > 5 or std_avg > 5) and max_abs > 100:
                print(
                    f"[Cloud Server] 檢測到 scaled 資料範圍異常 (max|x|={max_abs:.2f})，嘗試重新套用 scaler"
                )
                scaler = _load_global_scaler(experiment_dir)
                if scaler is not None:
                    try:
                        X = scaler.transform(X_df)
                        X = np.asarray(X, dtype=np.float32)
                        print(f"[Cloud Server] 已對 scaled 測試資料重新套用 scaler: {type(scaler)}")
                    except Exception as e:
                        print(f"[Cloud Server] 重新套用 scaler 失敗: {e}")
                else:

                    X_mean = X.mean(axis=0, keepdims=True)
                    X_std = X.std(axis=0, keepdims=True) + 1e-6
                    X = (X - X_mean) / X_std
                    X = np.clip(X, -5.0, 5.0)
                    print(f"[Cloud Server] 已對 scaled 測試資料進行 Z-score + clip")
        else:
            if 'scaler' not in locals() or scaler is None:
                scaler = _load_global_scaler(experiment_dir)
        if scaler is None:

            try:
                fallback_scaler = os.path.join(os.path.dirname(__file__), 'data', 'scaler.pkl')
            except Exception:
                fallback_scaler = None

            candidates = [
                os.path.join(experiment_dir, 'scaler.pkl'),
                os.path.join(getattr(config, 'DATA_PATH', ''), 'scaler.pkl'),
                fallback_scaler
            ]
            candidates = [p for p in candidates if p and os.path.exists(p)]
            if candidates:
                try:
                    src = candidates[0]
                    dst = os.path.join(experiment_dir, 'scaler.pkl')
                    os.makedirs(experiment_dir, exist_ok=True)
                    import shutil
                    shutil.copyfile(src, dst)
                    print(f"[Cloud Server] 已自動補齊 scaler.pkl -> {dst}")
                    scaler = _load_global_scaler(experiment_dir)
                except Exception as e:
                    print(f"[Cloud Server] 複製 scaler.pkl 失敗: {e}")
        if scaler is None:
            msg = "找不到 StandardScaler (scaler.pkl)，跳過本輪評測"
            print(f"[Cloud Server] {msg}")
            log_event("global_eval_skipped_no_scaler", msg)
            return None

        try:
                n_in = int(getattr(scaler, 'n_features_in_', X_df.shape[1]))
                if n_in != len(feature_cols):
                    msg = f"scaler.n_features_in_={n_in} 與 feature_cols={len(feature_cols)} 不一致"
                    print(f"[Cloud Server] {msg}，跳過本輪評測")
                    log_event("global_eval_skipped_scaler_feature_mismatch", msg)
                    return None
        except Exception as _e:
            print(f"[Cloud Server] 無法檢查 scaler.n_features_in_: {_e}")

            print(f"[Cloud Server]  使用特徵欄位示例: {feature_cols[:5]}")
        try:
            X = scaler.transform(X_df)

            X = np.asarray(X, dtype=np.float32)
        except Exception as e:
            msg = f"scaler.transform 失敗: {e}"
            print(f"[Cloud Server] {msg}，跳過本輪評測")
            log_event("global_eval_skipped_scaler_failed", msg)
            return None

        input_dim = X.shape[1]
        all_labels = list(getattr(config, 'ALL_LABELS', []))
        num_classes = len(all_labels)
        inferred = _infer_eval_model_params_from_weights(global_weights)
        inferred_in_dim = inferred.get("in_dim")
        inferred_out_dim = inferred.get("out_dim")
        inferred_hidden = inferred.get("hidden_dims")
        has_residual = inferred.get("has_residual")
        has_input_reshape = inferred.get("has_input_reshape")
        eval_use_client_encoder = os.environ.get("EVAL_USE_CLIENT_ENCODER", "1").strip() != "0"
        
    
        eval_used_encoder = False
        eval_client_encoder = None
        embed_norm_stats = None

        if inferred_in_dim and inferred_in_dim != input_dim:
            if eval_use_client_encoder:
                client_model = None
                loaded = False
                if hasattr(app.state, "client_model_for_embedding") and app.state.client_model_for_embedding is not None:
                    try:
                        client_model = app.state.client_model_for_embedding
                        client_model.eval()
                        loaded = True
                        print(f"[Cloud Server] 使用 app.state.client_model_for_embedding 作為 encoder（最新權重）")
                    except Exception as e:
                        print(f"[Cloud Server] 使用 app.state.client_model_for_embedding 失敗: {e}")

                if not loaded:
                    import time
                    def _find_latest_encoder_path(exp_dir: str) -> Tuple[Optional[str], float]:
                        latest_path = None
                        latest_mtime = -1.0
                        if not exp_dir or not os.path.exists(exp_dir):
                            return None, -1.0
                        try:
                            root_candidates = [
                                os.path.join(exp_dir, "client_encoder_weights.pt"),
                                os.path.join(exp_dir, "client_model_weights.pt"),
                                os.path.join(exp_dir, "client_model.pt"),
                                os.path.join(exp_dir, "client_encoder.pt"),
                            ]
                            for p in root_candidates:
                                if os.path.exists(p):
                                    m = os.path.getmtime(p)
                                    if m > latest_mtime:
                                        latest_mtime = m
                                        latest_path = p
                            for item in os.listdir(exp_dir):
                                client_dir = os.path.join(exp_dir, item)
                                if os.path.isdir(client_dir) and item.startswith('uav'):
                                    p = os.path.join(client_dir, "client_encoder_weights.pt")
                                    if os.path.exists(p):
                                        m = os.path.getmtime(p)
                                        if m > latest_mtime:
                                            latest_mtime = m
                                            latest_path = p
                        except Exception as e:
                            print(f"[Cloud Server] 掃描 encoder 權重失敗: {e}")
                        age_min = (time.time() - latest_mtime) / 60.0 if latest_mtime > 0 else -1.0
                        return latest_path, age_min

                    client_weight_path = os.environ.get("CLIENT_ENCODER_WEIGHTS_PATH", "").strip()
                    exp_dir = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', 'result'))

                    if not client_weight_path:
                        latest_path, latest_age = _find_latest_encoder_path(exp_dir)
                        client_weight_path = latest_path or ""
                        if client_weight_path:
                            print(f"[Cloud Server] 選用最新 encoder 權重: {client_weight_path} (age={latest_age:.2f} 分鐘)")

                    if client_weight_path:
                        load_ok = False
                        last_err = None
                        for retry in range(3):
                            state = _load_client_encoder_state(client_weight_path)
                            if state:
                                try:
                                    client_model.load_state_dict(state, strict=False)
                                    loaded = True
                                    load_ok = True
                                    age_min = (time.time() - os.path.getmtime(client_weight_path)) / 60.0
                                    print(f"[Cloud Server] 使用 client encoder 權重: {client_weight_path} (age={age_min:.2f} 分鐘, retry={retry})")
                                    break
                                except Exception as e:
                                    last_err = e
                                    print(f"[Cloud Server] 載入 client encoder 權重失敗 (retry={retry}): {e}")
                                    time.sleep(0.5)
                            else:
                                time.sleep(0.5)
                        if not load_ok:
                            print("[Cloud Server] 找不到可用的 client encoder 權重，跳過本輪評測")
                            try:
                                exp_dir_dbg = exp_dir
                                print(f"[Cloud Server] Debug：EXPERIMENT_DIR/LOG_DIR = {exp_dir_dbg}")
                                print(f"[Cloud Server] Debug：CLIENT_ENCODER_WEIGHTS_PATH = {os.environ.get('CLIENT_ENCODER_WEIGHTS_PATH', '').strip()}")
                                cand_dbg = [
                                    os.path.join(exp_dir_dbg, "client_encoder_weights.pt"),
                                    os.path.join(exp_dir_dbg, "client_model_weights.pt"),
                                    os.path.join(exp_dir_dbg, "client_model.pt"),
                                    os.path.join(exp_dir_dbg, "client_encoder.pt"),
                                ]
                                print(
                                    "[Cloud Server] Debug：候選 encoder 路徑存在性: "
                                    + ", ".join([f"{p}={os.path.exists(p)}" for p in cand_dbg])
                                )
                                if os.path.exists(exp_dir_dbg):
                                    uav_dirs = [
                                        d for d in os.listdir(exp_dir_dbg)
                                        if d.startswith('uav') and os.path.isdir(os.path.join(exp_dir_dbg, d))
                                    ]
                                    uav_has = sum(
                                        1
                                        for d in uav_dirs
                                        if os.path.exists(os.path.join(exp_dir_dbg, d, "client_encoder_weights.pt"))
                                    )
                                    print(f"[Cloud Server] Debug：exp_dir 下 uav*/client_encoder_weights.pt 數量 = {uav_has}")
                                if last_err:
                                    print(f"[Cloud Server] Debug：最後一次載入錯誤: {last_err}")
                            except Exception as e:
                                print(f"[Cloud Server] Debug：列印 encoder 搜尋資訊失敗: {e}")
                            print("[Cloud Server] 提示：確保 client_encoder_weights.pt 文件存在且是最新的")
                            return None

                if not loaded:
                    print("[Cloud Server] 找不到可用的 client encoder 權重，跳過本輪評測")

                    try:
                        exp_dir_dbg = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', 'result'))
                        print(f"[Cloud Server] Debug：EXPERIMENT_DIR/LOG_DIR = {exp_dir_dbg}")
                        print(f"[Cloud Server] Debug：CLIENT_ENCODER_WEIGHTS_PATH = {os.environ.get('CLIENT_ENCODER_WEIGHTS_PATH', '').strip()}")

                        cand_dbg = [
                            os.path.join(exp_dir_dbg, "client_encoder_weights.pt"),
                            os.path.join(exp_dir_dbg, "client_model_weights.pt"),
                            os.path.join(exp_dir_dbg, "client_model.pt"),
                            os.path.join(exp_dir_dbg, "client_encoder.pt"),
                        ]
                        print(
                            "[Cloud Server] Debug：候選 encoder 路徑存在性: "
                            + ", ".join([f"{p}={os.path.exists(p)}" for p in cand_dbg])
                        )

                        if os.path.exists(exp_dir_dbg):
                            uav_dirs = [d for d in os.listdir(exp_dir_dbg) if d.startswith('uav') and os.path.isdir(os.path.join(exp_dir_dbg, d))]
                            uav_has = 0
                            for d in uav_dirs:
                                if os.path.exists(os.path.join(exp_dir_dbg, d, "client_encoder_weights.pt")):
                                    uav_has += 1
                            print(f"[Cloud Server] Debug：exp_dir 下 uav*/client_encoder_weights.pt 數量 = {uav_has}")
                    except Exception as e:
                        print(f"[Cloud Server] Debug：列印 encoder 搜尋資訊失敗: {e}")
                    print("[Cloud Server] 提示：確保 client_encoder_weights.pt 文件存在且是最新的")
                    return None


                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                client_model = client_model.to(device)
                emb_batches = []
                batch_size = 10000
                with torch.no_grad():
                    for i in range(0, len(X), batch_size):
                        batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
                        emb = client_model.get_embedding(batch_X).cpu().numpy()
                        emb_batches.append(emb)
                X = np.vstack(emb_batches)
                input_dim = inferred_in_dim
                eval_used_encoder = True
                eval_client_encoder = client_model
                print(f"[Cloud Server] 已使用 client encoder 轉換成 embedding: {X.shape}")
            else:
                allow_adjust = os.environ.get("ALLOW_EVAL_FEATURE_ADJUST", "1").strip() != "0"
                if not allow_adjust:
                    print("[Cloud Server] 已設定 ALLOW_EVAL_FEATURE_ADJUST=0，跳過本輪評測")
                    return None

                if input_dim < inferred_in_dim:
                    pad = np.zeros((X.shape[0], inferred_in_dim - input_dim), dtype=X.dtype)
                    X = np.hstack([X, pad])
                    input_dim = inferred_in_dim
                    print(f"[Cloud Server] 已對輸入特徵做 zero-pad: {input_dim} → {inferred_in_dim}")
                elif input_dim > inferred_in_dim:
                    X = X[:, :inferred_in_dim]
                    input_dim = inferred_in_dim
                    print(f"[Cloud Server] 已對輸入特徵做截斷: {input_dim} → {inferred_in_dim}")


        if eval_used_encoder and os.environ.get("EVAL_EMBED_NORM", "1").strip() != "0":
            try:
                emb_mean = X.mean(axis=0, keepdims=True)
                emb_std = X.std(axis=0, keepdims=True) + 1e-6
                X = (X - emb_mean) / emb_std
                embed_norm_stats = (emb_mean, emb_std)
                print(f"[Cloud Server] 已對 embedding 進行標準化 (mean/std)")
            except Exception as e:
                print(f"[Cloud Server] embedding 標準化失敗: {e}")
        print(f"[Cloud Server] 標籤/頭對齊: ALL_LABELS={all_labels} (num_classes={num_classes}), input_dim={input_dim}")
        
        if inferred_in_dim or inferred_out_dim or inferred_hidden:
                model_type = config.MODEL_CONFIG.get('type', 'dnn')
                if model_type == 'transformer':
                    from models.transformer import build_transformer
                    model = build_transformer(
                        input_dim=int(inferred_in_dim or input_dim),
                        output_dim=int(inferred_out_dim or num_classes),
                        d_model=config.MODEL_CONFIG.get('d_model', 128),
                        num_layers=config.MODEL_CONFIG.get('num_layers', 2),
                        num_heads=config.MODEL_CONFIG.get('num_heads', 4),
                        d_ff=config.MODEL_CONFIG.get('d_ff', None),
                        dropout=config.MODEL_CONFIG.get('dropout_rate', 0.3),
                        max_seq_len=config.MODEL_CONFIG.get('max_seq_len', input_dim),
                        use_positional_encoding=config.MODEL_CONFIG.get('use_positional_encoding', True)
                    )
                else:
                    from models.dnn import build_dnn
                    model = build_dnn(
                        input_dim=int(inferred_in_dim or input_dim),
                        output_dim=int(inferred_out_dim or num_classes),
                        hidden_dims=inferred_hidden or config.MODEL_CONFIG.get('hidden_dims', [256, 128, 64]),
                        dropout_rate=config.MODEL_CONFIG.get('dropout_rate', 0.3),
                        use_batch_norm=config.MODEL_CONFIG.get('use_batch_norm', True),
                        use_residual=config.MODEL_CONFIG.get('use_residual', True),
                        activation=config.MODEL_CONFIG.get('activation', 'relu')
                    )
        else:
            model = _build_eval_model(input_dim=input_dim, num_classes=num_classes)

        if inferred_out_dim and inferred_out_dim != num_classes:
            print(f"[Cloud Server] 評估輸出類別不一致: weight_out_dim={inferred_out_dim}, label_classes={num_classes}")
        

        with _torch_local.no_grad():
            initial_params = list(model.parameters())
            if initial_params:
                initial_norm = sum(p.norm().item() for p in initial_params)
                print(f"[Cloud Server] 模型初始權重總範數: {initial_norm:.4f}")
    
        if not global_weights or len(global_weights) == 0:
            print(f"[Cloud Server] global_weights 為空，無法進行評估")
            return None
        
        try:
            has_invalid = False
            for layer_name, layer_weights in global_weights.items():
                if isinstance(layer_weights, _torch_local.Tensor):
                    if _torch_local.isnan(layer_weights).any() or _torch_local.isinf(layer_weights).any():
                        print(f"[Cloud Server] 權重層 {layer_name} 包含 NaN 或 Inf，無法評估")
                        has_invalid = True
                        break
                elif isinstance(layer_weights, np.ndarray):
                    if np.isnan(layer_weights).any() or np.isinf(layer_weights).any():
                        print(f"[Cloud Server] 權重層 {layer_name} 包含 NaN 或 Inf，無法評估")
                        has_invalid = True
                        break
            if has_invalid:
                print(f"[Cloud Server] 權重包含無效值，跳過評估")
                return None
        except Exception as e:
            print(f"[Cloud Server] 權重驗證失敗: {e}，繼續評估")
        
        try:
            before_load_sum = sum(p.sum().item() for p in model.parameters())
            print(f"[Cloud Server] 載入權重前模型權重總和: {before_load_sum:.6f} (round_id={round_id})")
        except Exception:
            pass
        
        ok = _load_state_dict_strict_flexible(model, global_weights)
        if not ok:
            print(f"[Cloud Server] 無法載入聚合權重到模型，跳過評測")
            return None
        
        try:
            after_load_sum = sum(p.sum().item() for p in model.parameters())
            print(f"[Cloud Server] 載入權重後模型權重總和: {after_load_sum:.6f} (round_id={round_id})")
            if abs(before_load_sum - after_load_sum) < 1e-6:
                print(f"[Cloud Server] 警告：載入前後權重總和幾乎相同，可能權重未正確更新！")
        except Exception:
            pass
        
        model.eval()
        
        bn_layers = []
        for name, module in model.named_modules():
            if isinstance(module, _torch_local.nn.BatchNorm1d) or isinstance(module, _torch_local.nn.BatchNorm2d):
                bn_layers.append(name)
                with _torch_local.no_grad():
                    running_mean = module.running_mean
                    running_var = module.running_var
                    weight = module.weight
                    bias = module.bias
                    print(f"[Cloud Server] BatchNorm {name}: running_mean=[{running_mean.min():.4f}, {running_mean.max():.4f}], running_var=[{running_var.min():.4f}, {running_var.max():.4f}], weight=[{weight.min():.4f}, {weight.max():.4f}], bias=[{bias.min():.4f}, {bias.max():.4f}]")

                    if running_var.min() < 1e-6:
                        print(f"[Cloud Server] 警告：BatchNorm {name} 的 running_var 過小，可能導致數值不穩定")
                    if abs(running_mean.mean().item()) > 10:
                        print(f"[Cloud Server] 警告：BatchNorm {name} 的 running_mean 異常")
        
        if not bn_layers:
            print(f"[Cloud Server] 警告：模型中沒有找到 BatchNorm 層")
        

        with _torch_local.no_grad():
            sample_input = _torch_local.randn(1, input_dim)
            sample_output = model(sample_input)
            print(f"[Cloud Server] 模型測試輸出: shape={sample_output.shape}, range=[{sample_output.min():.4f}, {sample_output.max():.4f}]")
            
            total_weight_norm = sum(p.norm().item() for p in model.parameters())
            print(f"[Cloud Server] 模型權重總範數: {total_weight_norm:.4f} (round_id={round_id})")

            try:
                import hashlib
                first_param = next(model.parameters())
                if first_param is not None:
                    param_hash = hashlib.md5(first_param.detach().cpu().numpy().tobytes()).hexdigest()[:8]
                    print(f"[Cloud Server] 模型首層參數哈希: {param_hash} (round_id={round_id})")
                    
                    weight_sum = sum(p.sum().item() for p in model.parameters())
                    print(f"[Cloud Server] 模型權重總和: {weight_sum:.6f} (round_id={round_id})")
            except Exception as e:
                print(f"[Cloud Server] 參數哈希計算失敗: {e}")
            if total_weight_norm < 1e-6:
                print(f"[Cloud Server] 警告：模型權重範數極小，可能未正確載入 (round_id={round_id})")

        print(f"[Cloud Server] 開始模型推理 (round_id={round_id})", flush=True)
        
        print(f"[Cloud Server] [進度] 開始 CUDA 狀態檢查 (round_id={round_id})", flush=True)
        cuda_available = False
        print(f"[Cloud Server] CUDA 可用性檢查 (round_id={round_id}): {cuda_available} (forced CPU for eval)", flush=True)
        current_device = None
        cuda_memory_allocated = 0.0
        cuda_memory_reserved = 0.0
        
        if cuda_available:
            try:
                print(f"[Cloud Server] [進度] 開始獲取 CUDA 設備信息 (round_id={round_id})", flush=True)
                
                import threading
                import queue
                
                cuda_info_queue = queue.Queue()
                cuda_info_timeout = 5.0
                
                def _get_cuda_info():
                    try:
                        cuda_device_count = _torch_local.cuda.device_count()
                        cuda_info_queue.put(('device_count', cuda_device_count))
                        
                        if cuda_device_count > 0:
                            current_device = _torch_local.cuda.current_device()
                            cuda_info_queue.put(('current_device', current_device))
                            
                            try:
                                device_name = _torch_local.cuda.get_device_name(current_device)
                                cuda_info_queue.put(('device_name', device_name))
                            except Exception:
                                cuda_info_queue.put(('device_name', f"Device_{current_device}"))
                            
                            try:
                                mem_allocated = _torch_local.cuda.memory_allocated(current_device) / 1024**2
                                mem_reserved = _torch_local.cuda.memory_reserved(current_device) / 1024**2
                                cuda_info_queue.put(('memory', (mem_allocated, mem_reserved)))
                            except Exception:
                                cuda_info_queue.put(('memory', (0.0, 0.0)))
                        else:
                            cuda_info_queue.put(('no_device', None))
                    except Exception as e:
                        cuda_info_queue.put(('error', e))
                
                cuda_info_thread = threading.Thread(target=_get_cuda_info, daemon=True)
                cuda_info_thread.start()
                cuda_info_thread.join(timeout=cuda_info_timeout)
                
                if cuda_info_thread.is_alive():
                    print(f"[Cloud Server] CUDA 信息獲取超時（{cuda_info_timeout}秒），將使用 CPU (round_id={round_id})", flush=True)
                    cuda_available = False
                else:
                    device_count = 0
                    current_device = None
                    device_name = "Unknown"
                    cuda_memory_allocated = 0.0
                    cuda_memory_reserved = 0.0
                    
                    while not cuda_info_queue.empty():
                        try:
                            info_type, info_value = cuda_info_queue.get_nowait()
                            if info_type == 'device_count':
                                device_count = info_value
                                print(f"[Cloud Server] CUDA 設備數量: {device_count} (round_id={round_id})", flush=True)
                            elif info_type == 'current_device':
                                current_device = info_value
                                print(f"[Cloud Server] 當前 CUDA 設備索引: {current_device} (round_id={round_id})", flush=True)
                            elif info_type == 'device_name':
                                device_name = info_value
                                print(f"[Cloud Server] CUDA 設備名稱: {device_name} (round_id={round_id})", flush=True)
                            elif info_type == 'memory':
                                cuda_memory_allocated, cuda_memory_reserved = info_value
                                print(f"[Cloud Server] CUDA 內存狀態 (round_id={round_id}): allocated={cuda_memory_allocated:.2f} MB, reserved={cuda_memory_reserved:.2f} MB", flush=True)
                            elif info_type == 'no_device':
                                print(f"[Cloud Server] CUDA 設備數量為 0，將使用 CPU (round_id={round_id})", flush=True)
                                cuda_available = False
                            elif info_type == 'error':
                                print(f"[Cloud Server] CUDA 信息獲取失敗: {info_value}，將使用 CPU (round_id={round_id})", flush=True)
                                cuda_available = False
                        except queue.Empty:
                            break
                    
                    if device_count > 0 and current_device is not None:
                        print(f"[Cloud Server] [進度] CUDA 設備信息獲取完成 (round_id={round_id}): device_count={device_count}, current_device={current_device}, device_name={device_name}", flush=True)
            except RuntimeError as cuda_runtime_e:
                error_msg = str(cuda_runtime_e)
                print(f"[Cloud Server] CUDA 運行時錯誤 (round_id={round_id}): {error_msg}", flush=True)
                if "out of memory" in error_msg.lower():
                    print(f"[Cloud Server] CUDA 內存不足，將使用 CPU (round_id={round_id})", flush=True)
                    cuda_available = False
                else:
                    print(f"[Cloud Server] CUDA 運行時錯誤，將使用 CPU (round_id={round_id})", flush=True)
                    cuda_available = False
            except Exception as cuda_info_e:
                print(f"[Cloud Server] CUDA 信息獲取失敗 (round_id={round_id}): {type(cuda_info_e).__name__}: {cuda_info_e}", flush=True)
                print(f"[Cloud Server] 將回退到 CPU 模式 (round_id={round_id})", flush=True)
                cuda_available = False
                import traceback
                traceback.print_exc()
        
        print(f"[Cloud Server] [進度] CUDA 狀態檢查完成: cuda_available={cuda_available} (round_id={round_id})", flush=True)
        
        device = _torch_local.device('cuda' if cuda_available else 'cpu')
        print(f"[Cloud Server] [進度] 選擇設備: {device} (round_id={round_id})", flush=True)
        print(f"[Cloud Server] [進度] 準備移動模型到設備 (round_id={round_id})", flush=True)
        
        try:
            model_param_count = sum(p.numel() for p in model.parameters())
            model_buffer_count = sum(b.numel() for b in model.buffers())
            print(f"[Cloud Server] 模型統計 (round_id={round_id}): 參數數量={model_param_count}, 緩衝區數量={model_buffer_count}", flush=True)
            

            if cuda_available:
                try:
                    _torch_local.cuda.empty_cache()
                    print(f"[Cloud Server] 已清理 CUDA 緩存 (round_id={round_id})", flush=True)
                except Exception as cache_e:
                    print(f"[Cloud Server] CUDA 緩存清理失敗 (round_id={round_id}): {cache_e}", flush=True)
            
            model = model.to(device)
            print(f"[Cloud Server] [進度] 模型已移動到設備 {device} (round_id={round_id})", flush=True)
            
            if cuda_available:
                try:
                    after_memory_allocated = _torch_local.cuda.memory_allocated(current_device) / 1024**2
                    after_memory_reserved = _torch_local.cuda.memory_reserved(current_device) / 1024**2
                    memory_increase = after_memory_allocated - cuda_memory_allocated
                    print(f"[Cloud Server] 模型移動後 CUDA 內存 (round_id={round_id}): allocated={after_memory_allocated:.2f} MB, reserved={after_memory_reserved:.2f} MB, 增加={memory_increase:.2f} MB", flush=True)
                except Exception as mem_e:
                    print(f"[Cloud Server] CUDA 內存檢查失敗 (round_id={round_id}): {mem_e}", flush=True)
        except RuntimeError as e:
            error_msg = str(e)
            print(f"[Cloud Server] 模型移動到設備失敗 (RuntimeError, round_id={round_id}): {error_msg}", flush=True)
            if "out of memory" in error_msg.lower():
                print(f"[Cloud Server] 建議：CUDA 內存不足，嘗試清理緩存或使用 CPU", flush=True)
                try:
                    _torch_local.cuda.empty_cache()
                    print(f"[Cloud Server] 已清理 CUDA 緩存，嘗試使用 CPU", flush=True)
                    device = _torch_local.device('cpu')
                    model = model.to(device)
                    print(f"[Cloud Server] 模型已移動到 CPU (round_id={round_id})", flush=True)
                except Exception as cpu_fallback_e:
                    print(f"[Cloud Server] CPU 回退也失敗 (round_id={round_id}): {cpu_fallback_e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    return None
            else:
                import traceback
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"[Cloud Server] 模型移動到設備失敗 (Exception, round_id={round_id}): {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
        

        batch_size = 10000
        all_preds = []
        all_logits = []
        all_logits_raw = []
        
        LOGITS_CLIP_MIN = -10.0
        LOGITS_CLIP_MAX = 10.0
        
        print(f"[Cloud Server] [進度] 準備進入 torch.no_grad() 塊 (round_id={round_id})", flush=True)
        try:

            if not hasattr(model, 'parameters'):
                print(f"[Cloud Server] 模型無效：缺少 parameters 屬性 (round_id={round_id})", flush=True)
                return None
            
            with _torch_local.no_grad():

                print(f"[Cloud Server] 開始輸入數據異常檢測 (round_id={round_id}): X_shape={X.shape}, y_shape={y.shape}", flush=True)
            input_stats = {
                'min': float(X.min()),
                'max': float(X.max()),
                'mean': float(X.mean()),
                'std': float(X.std()),
                'nan_count': int(np.isnan(X).sum()),
                'inf_count': int(np.isinf(X).sum())
            }
            print(f"[Cloud Server] 輸入數據統計 (round_id={round_id}): min={input_stats['min']:.4f}, max={input_stats['max']:.4f}, mean={input_stats['mean']:.4f}, std={input_stats['std']:.4f}, nan_count={input_stats['nan_count']}, inf_count={input_stats['inf_count']}", flush=True)

            if input_stats['nan_count'] > 0:
                print(f"[Cloud Server] 警告：輸入數據包含 {input_stats['nan_count']} 個 NaN 值")

                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"[Cloud Server] 已修復 NaN 值")
            if input_stats['inf_count'] > 0:
                print(f"[Cloud Server] 警告：輸入數據包含 {input_stats['inf_count']} 個 Inf 值")

                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"[Cloud Server] 已修復 Inf 值")
            range_threshold = 100.0 if eval_used_encoder else 10.0
            if abs(input_stats['max']) > range_threshold or abs(input_stats['min']) > range_threshold:
                print(
                    f"[Cloud Server] 警告：輸入數據範圍異常 [{input_stats['min']:.2f}, {input_stats['max']:.2f}]，將進行 Z-score 標準化")
                X_mean = X.mean(axis=0, keepdims=True)
                X_std = X.std(axis=0, keepdims=True) + 1e-6
                X = (X - X_mean) / X_std
                X = np.clip(X, -5.0, 5.0)
                print(f"[Cloud Server] 已對輸入數據進行 Z-score 標準化並裁剪到 [-5.0, 5.0]")
            
            extreme_inputs = []
            extreme_inputs_with_labels = []
            total_batches = (len(X) + batch_size - 1) // batch_size
            print(f"[Cloud Server] 開始批次處理 (round_id={round_id}): 總樣本數={len(X)}, 批次大小={batch_size}, 總批次數={total_batches}", flush=True)
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                batch_num = i // batch_size + 1
                if batch_num == 1 or batch_num % 10 == 0 or batch_num == total_batches:
                    print(f"[Cloud Server] 處理批次 {batch_num}/{total_batches} (round_id={round_id}): 樣本範圍 [{i}, {min(i+batch_size, len(X))})", flush=True)
                
                batch_norms = np.linalg.norm(batch_X, axis=1)
                extreme_indices = np.where((batch_norms > 25) | (np.abs(batch_X).max(axis=1) > 35))[0]
                if len(extreme_indices) > 0:
                    for idx in extreme_indices:
                        global_idx = i + idx
                        extreme_inputs.append(global_idx)
                        extreme_inputs_with_labels.append({
                            'index': global_idx,
                            'label': int(batch_y[idx]),
                            'l2_norm': float(batch_norms[idx]),
                            'max_abs': float(np.abs(batch_X[idx]).max())
                        })
                inputs = _torch_local.tensor(batch_X, dtype=_torch_local.float32).to(device)
                
                if i == 0 and len(all_logits) == 0:
                    try:

                        if hasattr(model, 'get_embedding'):
                            hidden_features = model.get_embedding(inputs[:min(100, len(inputs))])
                            hidden_mean = hidden_features.mean().item()
                            hidden_std = hidden_features.std().item()
                            hidden_norm = hidden_features.norm().item()
                            print(f"[Cloud Server] 隱藏層特徵統計（前{min(100, len(inputs))}個樣本）: mean={hidden_mean:.4f}, std={hidden_std:.4f}, norm={hidden_norm:.4f}")
                            

                            hidden_variance = hidden_features.var(dim=0).mean().item()
                            print(f"[Cloud Server] 隱藏層特徵方差（不同樣本之間）: {hidden_variance:.4f}")
                            

                            hidden_min = hidden_features.min().item()
                            hidden_max = hidden_features.max().item()
                            print(f"[Cloud Server] 隱藏層特徵範圍: [{hidden_min:.4f}, {hidden_max:.4f}]")
                    except Exception as e:
                        print(f"[Cloud Server] 隱藏層特徵診斷失敗: {e}")
                try:
                    batch_logits = model(inputs)
                except Exception as e:
                    print(f"[Cloud Server] 模型推理失敗 (round_id={round_id}, batch={batch_num}): {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise

                batch_logits_clipped = _torch_local.clamp(batch_logits, min=LOGITS_CLIP_MIN, max=LOGITS_CLIP_MAX)
                
                if (batch_logits != batch_logits_clipped).any():
                    n_clipped = (batch_logits != batch_logits_clipped).sum().item()
                    logits_min = batch_logits.min().item()
                    logits_max = batch_logits.max().item()
                    print(f"[Cloud Server] 警告：批次 {i//batch_size + 1} 中有 {n_clipped} 個 logits 值被裁剪（原始範圍: [{logits_min:.2f}, {logits_max:.2f}]）")
                
                    batch_pred = _torch_local.argmax(batch_logits_clipped, dim=1).cpu().numpy()
                
                all_preds.append(batch_pred)
                all_logits.append(batch_logits_clipped.cpu())
                all_logits_raw.append(batch_logits.cpu())
            
            if len(extreme_inputs) > 0:
                print(f"[Cloud Server] 警告：發現 {len(extreme_inputs)} 個異常輸入樣本（索引: {extreme_inputs[:10]}{'...' if len(extreme_inputs) > 10 else ''}）")
                

                abnormal_labels = [item['label'] for item in extreme_inputs_with_labels]
                unique_abnormal_labels, abnormal_label_counts = np.unique(abnormal_labels, return_counts=True)
                abnormal_label_dist = dict(zip(unique_abnormal_labels, abnormal_label_counts))
                
                label_names = getattr(config, 'ALL_LABELS', [])
                label_name_map = {i: label_names[i] if i < len(label_names) else f"Class_{i}" for i in unique_abnormal_labels}
                
                print(f"[Cloud Server] 異常樣本標籤分佈:")
                for label_id, count in abnormal_label_dist.items():
                    label_name = label_name_map.get(label_id, f"Class_{label_id}")
                    percentage = count / len(extreme_inputs) * 100
                    print(f"  - 類別 {label_id} ({label_name}): {count} 個樣本 ({percentage:.1f}%)")
                

                avg_l2_norm = np.mean([item['l2_norm'] for item in extreme_inputs_with_labels])
                avg_max_abs = np.mean([item['max_abs'] for item in extreme_inputs_with_labels])
                max_l2_norm = np.max([item['l2_norm'] for item in extreme_inputs_with_labels])
                max_max_abs = np.max([item['max_abs'] for item in extreme_inputs_with_labels])
                print(f"[Cloud Server] 異常樣本特徵統計:")
                print(f"  - 平均 L2 範數: {avg_l2_norm:.4f} (最大: {max_l2_norm:.4f})")
                print(f"  - 平均最大絕對值: {avg_max_abs:.4f} (最大: {max_max_abs:.4f})")
                print(f"[Cloud Server] 將排除這些異常樣本，避免影響評估指標")
            
            print(f"[Cloud Server] 合併批次結果 (round_id={round_id}): 總批次數={len(all_preds)}, 總預測數={sum(len(p) for p in all_preds)}", flush=True)
            pred = np.concatenate(all_preds)
            logits = _torch_local.cat(all_logits, dim=0)
            logits_raw = _torch_local.cat(all_logits_raw, dim=0)
            print(f"[Cloud Server] 批次處理完成 (round_id={round_id}): pred_len={len(pred)}, logits_shape={logits.shape}, logits_raw_shape={logits_raw.shape}", flush=True)
            
            if len(extreme_inputs) > 0:
                mask = np.ones(len(pred), dtype=bool)
                mask[extreme_inputs] = False
            
                original_sample_count = len(pred)
                
                pred = pred[mask]
                logits = logits[mask]
                logits_raw = logits_raw[mask]
                y = y[mask]
                
                print(f"[Cloud Server] 已排除 {len(extreme_inputs)} 個異常樣本（評估樣本數: {original_sample_count} → {len(pred)}）")
            
            print(f"[Cloud Server] 開始計算預測分佈 (round_id={round_id}), pred_len={len(pred)}, y_len={len(y)}", flush=True)
            unique_pred, pred_counts = np.unique(pred, return_counts=True)
            unique_true, true_counts = np.unique(y, return_counts=True)
            pred_dist = dict(zip(unique_pred, pred_counts))
            true_dist = dict(zip(unique_true, true_counts))
            print(f"[Cloud Server] 預測分佈: {pred_dist} (round_id={round_id})", flush=True)
            print(f"[Cloud Server] 真實標籤分佈: {true_dist} (round_id={round_id})", flush=True)
            
            max_ratio = max(pred_dist.values()) / len(pred) if pred_dist else 0.0
            dominant_class = max(pred_dist, key=pred_dist.get) if pred_dist else None
            logits_var = logits_raw.var().item() if logits_raw.numel() > 0 else 0.0
            HARD_MIN_VAR = 1e-5
            WARMUP_ROUNDS = 15
            NORMAL_VAR_THRESH = 0.005

            if logits_var < HARD_MIN_VAR:
                print(f"[Cloud Server] Logits 方差極低 ({logits_var:.6f} < {HARD_MIN_VAR:.6f})，視為完全崩潰，跳過本輪結果並標記回退")
                needs_rollback_flag = True
                rollback_reason_str = f"logits_variance_{logits_var:.6f}_too_low_hard"
                return None

            if round_id is not None and round_id <= WARMUP_ROUNDS:
                if logits_var < NORMAL_VAR_THRESH:
                    print(f"[Cloud Server] WARMUP 警告：Round {round_id} logits 方差偏低 ({logits_var:.4f} < {NORMAL_VAR_THRESH:.4f})，暫不硬拒收，只做觀察")

            else:
                at_extreme_limits = (
                    ROLLBACK_COUNT >= COOLING_OFF_ROLLBACK_THRESHOLD
                    and CURRENT_SERVER_LR_MULTIPLIER <= MIN_SERVER_LR_MULTIPLIER + 1e-6
                    and CURRENT_FEDPROX_MU_MULTIPLIER >= MAX_FEDPROX_MU_MULTIPLIER - 1e-6
                )
                if logits_var < NORMAL_VAR_THRESH:
                    if at_extreme_limits:
                        print(f"[Cloud Server] Logits 方差過低，但處於極限狀態，啟用冷卻：先提高 FedProx μ 並暫緩回退")
                        CURRENT_FEDPROX_MU_MULTIPLIER = min(
                            MAX_FEDPROX_MU_MULTIPLIER,
                            CURRENT_FEDPROX_MU_MULTIPLIER * 1.10,
                        )
                        needs_rollback_flag = False
                        rollback_reason_str = ""
                    else:
                        print(f"[Cloud Server] Logits 方差過低 ({logits_var:.4f} < {NORMAL_VAR_THRESH:.4f})，視為崩潰，預設跳過本輪結果並標記回退")
                        needs_rollback_flag = True
                        rollback_reason_str = f"logits_variance_{logits_var:.4f}_too_low"
                        return None
            
            if max_ratio > 0.85:
                print(f"[Cloud Server] 嚴重退化！{max_ratio*100:.1f}% 的樣本被預測為類別 {dominant_class}")
                log_event("model_degradation_critical", f"round={round_id},dominant_class={dominant_class},ratio={max_ratio:.4f}")
                at_extreme_limits = (
                    ROLLBACK_COUNT >= COOLING_OFF_ROLLBACK_THRESHOLD
                    and CURRENT_SERVER_LR_MULTIPLIER <= MIN_SERVER_LR_MULTIPLIER + 1e-6
                    and CURRENT_FEDPROX_MU_MULTIPLIER >= MAX_FEDPROX_MU_MULTIPLIER - 1e-6
                )
                if at_extreme_limits:
                    print("[Cloud Server] 單類崩潰但處於極限狀態，啟用冷卻：提高 FedProx μ、暫緩回退，保留高信任 EMA")
                    CURRENT_FEDPROX_MU_MULTIPLIER = min(
                        MAX_FEDPROX_MU_MULTIPLIER,
                        CURRENT_FEDPROX_MU_MULTIPLIER * 1.10,
                    )
                    needs_rollback_flag = False
                    rollback_reason_str = ""
                else:
                    needs_rollback_flag = True
                    rollback_reason_str = f"prediction_distribution_{dominant_class}_{max_ratio:.2f}"


                if max_ratio > 0.90 and not at_extreme_limits:
                    print(f"[Cloud Server] 單類占比 {max_ratio*100:.1f}% > 90%，預設跳過本輪結果並標記回退")
                    print(f"[Cloud Server] 單類占比 {max_ratio*100:.1f}% > 90%，返回 None (round_id={round_id})", flush=True)
                    return None
                elif max_ratio > 0.90 and at_extreme_limits:

                    print(f"[Cloud Server] 單類占比 {max_ratio*100:.1f}% > 90%，但處於冷卻期，允許評估繼續（給模型恢復機會）")
                
                if DYNAMIC_LR_PENALTY_ENABLED and not at_extreme_limits:
                    print(f"[Cloud Server] 嚴重退化：暫不調整 SERVER_LR / FedProx μ（已關小動態懲罰，優先穩定 KD + Prototype Anchor）")
            elif max_ratio > 0.70:
                print(f"[Cloud Server] 警告：模型退化！{max_ratio*100:.1f}% 的樣本被預測為類別 {dominant_class}")
                log_event("model_degradation_detected", f"round={round_id},dominant_class={dominant_class},ratio={max_ratio:.4f}")
                
                print(f"[Cloud Server] 單一類別預測 > 70%，暫不調整 FedProx μ（保持當前 μ 乘數）")
            

    
                if DYNAMIC_LR_PENALTY_ENABLED:
                    global ACCURACY_HISTORY, ACCURACY_STAGNATION_THRESHOLD, ACCURACY_STAGNATION_ROUNDS
                    
                    should_penalize = False
                    if len(ACCURACY_HISTORY) >= ACCURACY_STAGNATION_ROUNDS:

                        recent_accs = ACCURACY_HISTORY[-ACCURACY_STAGNATION_ROUNDS:]
                        if len(recent_accs) >= 2:
                            improvements = [recent_accs[i] - recent_accs[i-1] for i in range(1, len(recent_accs))]
                            avg_improvement = sum(improvements) / len(improvements)
                            if avg_improvement < ACCURACY_STAGNATION_THRESHOLD:
                                should_penalize = True
                                print(f"[Cloud Server] 檢測到 Accuracy 停滯：最近 {ACCURACY_STAGNATION_ROUNDS} 輪平均提升 {avg_improvement:.6f} < {ACCURACY_STAGNATION_THRESHOLD:.6f}")
                            else:
                                print(f"[Cloud Server] Accuracy 仍在改善：最近 {ACCURACY_STAGNATION_ROUNDS} 輪平均提升 {avg_improvement:.6f} >= {ACCURACY_STAGNATION_THRESHOLD:.6f}，跳過學習率懲罰")
                    else:
                        should_penalize = True
                        print(f"[Cloud Server] Accuracy 歷史不足 {ACCURACY_STAGNATION_ROUNDS} 輪，使用原有邏輯")
                    
                    if should_penalize:
                        print(f"[Cloud Server] 模型退化偵測到，但已關閉基於 Accuracy 的 LR/FedProx 懲罰（保持當前乘數）")
                    else:
                        print(f"[Cloud Server] Accuracy 仍在改善，同樣不調整 LR/FedProx 乘數")
            elif max_ratio > 0.60:
                print(f"[Cloud Server] 警告：模型可能退化！{max_ratio*100:.1f}% 的樣本被預測為類別 {dominant_class}")
                log_event("model_degradation_warning", f"round={round_id},dominant_class={dominant_class},ratio={max_ratio:.4f}")
                
                if DYNAMIC_LR_PENALTY_ENABLED:
                    print(f"[Cloud Server] 可能退化：僅記錄告警，不調整 SERVER_LR / FedProx μ（保持當前乘數以降低震盪）")
            else:

                if DYNAMIC_LR_PENALTY_ENABLED:
                    print(f"[Cloud Server] 正常輪次：暫不自動恢復 SERVER_LR / FedProx μ 乘數，維持穩定設定")
            
            num_predicted_classes = len(pred_dist)
            expected_classes = len(getattr(config, 'ALL_LABELS', []))
            if expected_classes > 0 and num_predicted_classes < expected_classes * 0.5:
                print(f"[Cloud Server] 警告：模型只預測了 {num_predicted_classes}/{expected_classes} 個類別，可能無法區分類別")
                log_event("insufficient_class_prediction", f"round={round_id},predicted={num_predicted_classes},expected={expected_classes}")
            

            logits_min_raw = logits_raw.min().item()
            logits_max_raw = logits_raw.max().item()
            n_extreme_logits = ((logits_raw < LOGITS_CLIP_MIN) | (logits_raw > LOGITS_CLIP_MAX)).sum().item()
            if n_extreme_logits > 0:
                print(f"[Cloud Server] 警告：有 {n_extreme_logits} 個 logits 值超出範圍 [{LOGITS_CLIP_MIN}, {LOGITS_CLIP_MAX}]（原始範圍: [{logits_min_raw:.2f}, {logits_max_raw:.2f}]）")
            
            print(f"[Cloud Server] Logits 統計（原始）: min={logits_min_raw:.4f}, max={logits_max_raw:.4f}, mean={logits_raw.mean():.4f}, std={logits_raw.std():.4f}")
            print(f"[Cloud Server] Logits 統計（已裁剪）: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
            

            logits_np = logits.numpy()
            logits_means = [float(logits_np[:, i].mean()) for i in range(logits_np.shape[1])]
            logits_maxs = [float(logits_np[:, i].max()) for i in range(logits_np.shape[1])]
            logits_mins = [float(logits_np[:, i].min()) for i in range(logits_np.shape[1])]
            logits_stds = [float(logits_np[:, i].std()) for i in range(logits_np.shape[1])]
            print(f"[Cloud Server] 各類別 Logits 均值: {[f'{m:.4f}' for m in logits_means]}")
            print(f"[Cloud Server] 各類別 Logits 標準差: {[f'{s:.4f}' for s in logits_stds]}")
            print(f"[Cloud Server] 各類別 Logits 最大值: {[f'{m:.4f}' for m in logits_maxs]}")
            print(f"[Cloud Server] 各類別 Logits 最小值: {[f'{m:.4f}' for m in logits_mins]}")
            
            if len(logits_means) > 1:
                mean_range = max(logits_means) - min(logits_means)
                mean_std = float(np.std(logits_means))
                print(f"[Cloud Server] Logits 均值範圍: {mean_range:.6f}, 標準差: {mean_std:.6f}")
                
                if mean_range < 0.01:
                    print(f"[Cloud Server] 警告：所有類別的 Logits 均值非常接近（範圍 < 0.01），這會導致模型無法區分類別！")
                elif mean_range < 0.1:
                    print(f"[Cloud Server] 警告：所有類別的 Logits 均值較接近（範圍 < 0.1），可能導致預測不穩定")
                
                sample_logits_variance = []
                for i in range(min(100, len(logits_np))):
                    sample_logits = logits_np[i, :]
                    sample_var = float(np.var(sample_logits))
                    sample_logits_variance.append(sample_var)
                
                avg_sample_variance = float(np.mean(sample_logits_variance))
                print(f"[Cloud Server] 樣本 Logits 方差（前100個樣本平均）: {avg_sample_variance:.6f}")
                


                global LOGITS_VARIANCE_HISTORY, LOGITS_VARIANCE_DECREASE_COUNT
                

                LOGITS_VARIANCE_HISTORY.append(avg_sample_variance)
                if len(LOGITS_VARIANCE_HISTORY) > 3:
                    LOGITS_VARIANCE_HISTORY.pop(0)
                
                if len(LOGITS_VARIANCE_HISTORY) >= 2:
                    is_decreasing = True
                    for i in range(1, len(LOGITS_VARIANCE_HISTORY)):
                        if LOGITS_VARIANCE_HISTORY[i] >= LOGITS_VARIANCE_HISTORY[i-1]:
                            is_decreasing = False
                            break
                    
                    if is_decreasing:
                        LOGITS_VARIANCE_DECREASE_COUNT += 1
                        print(f"[Cloud Server] Logits 方差連續下降：歷史={[f'{v:.6f}' for v in LOGITS_VARIANCE_HISTORY]}，連續下降 {LOGITS_VARIANCE_DECREASE_COUNT} 輪")
                        

                        if LOGITS_VARIANCE_DECREASE_COUNT >= 3:

                            print(f"[Cloud Server]  Logits 方差連續 {LOGITS_VARIANCE_DECREASE_COUNT} 輪下降，模型正在變得「平庸」，原設計將增加 FedProx μ 值（現僅記錄，不實際調整）")
                            simulated_mu = min(MAX_FEDPROX_MU_MULTIPLIER, CURRENT_FEDPROX_MU_MULTIPLIER * 1.5)
                            print(f"[Cloud Server] FedProx μ 預期乘數: {CURRENT_FEDPROX_MU_MULTIPLIER:.4f} → {simulated_mu:.4f} (上限: {MAX_FEDPROX_MU_MULTIPLIER:.4f}，純 log)")
                            LOGITS_VARIANCE_DECREASE_COUNT = 0
                    else:
                        if LOGITS_VARIANCE_DECREASE_COUNT > 0:
                            print(f"[Cloud Server] Logits 方差未連續下降，重置計數器")
                            LOGITS_VARIANCE_DECREASE_COUNT = 0
                
                if avg_sample_variance < 0.01:
                    print(f"[Cloud Server] 警告：樣本 Logits 方差極小（< 0.01），所有類別的 logits 幾乎相同，這會導致模型只預測一個類別！")
                elif avg_sample_variance < 0.1:
                    print(f"[Cloud Server] 警告：樣本 Logits 方差較小（< 0.1），可能導致預測不穩定")
            
                F = _torch_local.nn.functional
            global TEMPERATURE_SCALING_ENABLED, TEMPERATURE_SCALING_T
            
            if TEMPERATURE_SCALING_ENABLED:
                scaled_logits = logits / TEMPERATURE_SCALING_T
                print(f"[Cloud Server]  溫度縮放：T={TEMPERATURE_SCALING_T:.2f}，logits 範圍: [{scaled_logits.min():.4f}, {scaled_logits.max():.4f}]")
                probs = F.softmax(scaled_logits, dim=1)
            else:
                probs = F.softmax(logits, dim=1)
            
            probs_means = [float(probs[:, i].mean()) for i in range(probs.shape[1])]
            print(f"[Cloud Server] Softmax 機率統計: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
            print(f"[Cloud Server] 各類別平均機率: {[f'{m:.4f}' for m in probs_means]}")
            
            if len(X) > 0:
                sample_input = _torch_local.tensor(X[:1000], dtype=_torch_local.float32)
                print(f"[Cloud Server] 輸入數據統計 (前1000樣本): min={sample_input.min():.4f}, max={sample_input.max():.4f}, mean={sample_input.mean():.4f}, std={sample_input.std():.4f}")
            
            class_weights = None
            use_class_weights = False
            
            if len(unique_labels) > 1:
                max_count = max(label_counts)
                min_count = min(label_counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                is_balanced = imbalance_ratio < 1.1
                
                if is_balanced:
                    print(f"[Cloud Server] 測試集類別平衡 (比例={imbalance_ratio:.2f}:1)，不使用類別權重計算 Loss")
                else:

                    try:
                        experiment_dir = os.environ.get('EXPERIMENT_DIR') or config.LOG_DIR
                        weights_file = os.path.join(experiment_dir, "model", "global_class_weights.npy")
                        if os.path.exists(weights_file):
                            weights_dict = np.load(weights_file, allow_pickle=True).item()
                            if isinstance(weights_dict, dict) and 'weights' in weights_dict:
                                class_weights = _torch_local.tensor(weights_dict['weights'], dtype=_torch_local.float32)
                                use_class_weights = True
                                print(f"[Cloud Server] 測試集不平衡 (比例={imbalance_ratio:.2f}:1)，使用全局類別權重計算 Loss: {class_weights.tolist()}")
                            elif isinstance(weights_dict, np.ndarray):
                                class_weights = _torch_local.tensor(weights_dict, dtype=_torch_local.float32)
                                use_class_weights = True
                                print(f"[Cloud Server] 測試集不平衡 (比例={imbalance_ratio:.2f}:1)，使用全局類別權重計算 Loss: {class_weights.tolist()}")
                    except Exception as e:
                        print(f"[Cloud Server] 載入類別權重失敗: {e}，使用標準 CrossEntropyLoss")
            
            if use_class_weights and class_weights is not None:
                criterion = _torch_local.nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = _torch_local.nn.CrossEntropyLoss()
            y_tensor = _torch_local.tensor(y, dtype=_torch_local.long)
            loss = float(criterion(logits, y_tensor).item())
        except Exception as e:
            print(f"[Cloud Server] 模型推理失敗 (round_id={round_id}): {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

        print(f"[Cloud Server] [進度] 開始計算評估指標 (round_id={round_id})", flush=True)
        acc = float(accuracy_score(y, pred))
        try:
            f1 = float(f1_score(y, pred, average='macro'))
        except Exception as f1_e:
            print(f"[Cloud Server] [進度] F1 計算失敗: {f1_e}，使用默認值 0.0 (round_id={round_id})", flush=True)
            f1 = 0.0
        print(f"[Cloud Server] [進度] 評估指標計算完成: acc={acc:.4f}, f1={f1:.4f} (round_id={round_id})", flush=True)

        joint_acc = acc
        joint_f1 = f1
        
        correct_predictions = int((pred == y).sum())
        total_samples = len(y)
        calculated_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
        print(f"[Cloud Server] Accuracy 計算詳情 (round_id={round_id}):")
        print(f"  正確預測數: {correct_predictions}/{total_samples}")
        print(f"  計算的 accuracy: {calculated_acc:.10f}")
        print(f"  sklearn accuracy_score: {acc:.10f}")
        print(f"  差異: {abs(calculated_acc - acc):.10e}")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, pred, labels=list(range(len(set(y)))))
        print(f"[Cloud Server] 混淆矩陣摘要 (round_id={round_id}):")
        print(f"  矩陣形狀: {cm.shape}")
        print(f"  對角線元素（正確預測）: {cm.diagonal().tolist()}")
        print(f"  對角線總和: {cm.diagonal().sum()}")
        print(f"  非對角線元素總和: {(cm.sum() - cm.diagonal().sum())}")
        
        try:
            import hashlib
            weight_hash_parts = []
            param_count = 0
            for name, param in model.named_parameters():
                if param_count < 3:
                    weight_hash_parts.append(hashlib.md5(param.detach().cpu().numpy().tobytes()).hexdigest()[:8])
                    param_count += 1
            combined_hash = hashlib.md5(''.join(weight_hash_parts).encode()).hexdigest()[:16]
            print(f"[Cloud Server] 權重哈希值 (round_id={round_id}): {combined_hash}")
        except Exception as e:
            print(f"[Cloud Server] 權重哈希計算失敗: {e}")
        
        try:
            from sklearn.metrics import f1_score as f1_score_func
            per_class_f1 = f1_score_func(y, pred, average=None, zero_division=0)
            all_labels = list(getattr(config, 'ALL_LABELS', []))
            
            if len(all_labels) == len(per_class_f1):

                f1_per_class_dict = {i: float(f1_val) for i, f1_val in enumerate(per_class_f1)}
                app.state.last_f1_per_class = f1_per_class_dict
                
                for i, (class_name, class_f1) in enumerate(zip(all_labels, per_class_f1)):
                    if class_f1 < 0.01:
                        print(f"[Cloud Server] 警告：類別 {class_name} (類別 {i}) 的 F1 分數過低 ({class_f1:.4f})，模型無法正確識別此類別")
                        log_event("low_class_f1_warning", f"round={round_id},class={i},class_name={class_name},f1={class_f1:.4f}")

                f1_summary = {name: f"{f1_val:.4f}" for name, f1_val in zip(all_labels, per_class_f1)}
                print(f"[Cloud Server] 各類別 F1 分數: {f1_summary}")
        except Exception as e:
            print(f"[Cloud Server] 各類別 F1 分數計算失敗: {e}")
        
        try:
            import pandas as pd
            experiment_dir_for_trend = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
            baseline_csv = os.path.join(experiment_dir_for_trend, 'cloud_baseline.csv')
            if os.path.exists(baseline_csv):
                try:
                    df_history = pd.read_csv(baseline_csv)
                    if len(df_history) > 0:
                        recent_rounds = df_history.tail(5)
                        if len(recent_rounds) > 1:
                            acc_trend = recent_rounds['accuracy'].values
                            f1_trend = recent_rounds['f1_score'].values
                            acc_improving = acc > acc_trend[-1] if len(acc_trend) > 0 else False
                            f1_improving = f1 > f1_trend[-1] if len(f1_trend) > 0 else False
                            acc_avg_recent = acc_trend[-3:].mean() if len(acc_trend) >= 3 else acc_trend.mean()
                            acc_avg_early = acc_trend[:3].mean() if len(acc_trend) >= 3 else acc_trend[0]
                        
                            ACCURACY_HISTORY.append(acc)
                            if len(ACCURACY_HISTORY) > 10:
                                ACCURACY_HISTORY = ACCURACY_HISTORY[-10:]
                            
                            print(f"[Cloud Server] 性能趨勢分析:")
                            print(f"  當前: 準確率={acc:.4f}, F1={f1:.4f}")
                            print(f"  近期平均: 準確率={acc_avg_recent:.4f}")
                            print(f"  早期平均: 準確率={acc_avg_early:.4f}")
                            if acc_avg_recent > acc_avg_early:
                                improvement = (acc_avg_recent - acc_avg_early) / acc_avg_early * 100
                                print(f"  準確率改善: {improvement:.2f}%")
                            else:
                                decline = (acc_avg_early - acc_avg_recent) / acc_avg_early * 100
                                print(f"  準確率下降: {decline:.2f}%")
                except Exception as e:
                    print(f"[Cloud Server] 性能趨勢分析失敗: {e}")
        except Exception:
            pass
        

        total_params = sum(p.numel() for p in model.parameters())
        non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
        print(f"[Cloud Server] 模型參數: 總數={total_params}, 非零={non_zero_params}, 零參數比例={(1-non_zero_params/total_params)*100:.2f}%")
        cls_report = classification_report(y, pred, output_dict=True, zero_division=0)
        per_class_f1 = {f"f1_class_{int(k)}": float(v['f1-score']) for k, v in cls_report.items() if k.isdigit()}
        support = {f"support_class_{int(k)}": int(v['support']) for k, v in cls_report.items() if k.isdigit()}
        label_names = list(getattr(config, 'ALL_LABELS', []))
        label_name_fields = {f"label_name_{i}": str(name) for i, name in enumerate(label_names)}

        try:
            num_classes = len(label_names) if label_names else int(max(max(y), max(pred))) + 1
        except Exception:
            num_classes = int(max(pred)) + 1 if len(pred) > 0 else 0
        pred_support = {}
        pred_ratio = {}
        max_pred_ratio = None
        max_pred_class = None
        if num_classes > 0 and len(pred) > 0:
            try:
                pred = np.asarray(pred).astype(int)
                pred_counts = np.bincount(pred, minlength=num_classes)
                total_pred = float(len(pred))
                if total_pred > 0:
                    for i in range(num_classes):
                        cnt = int(pred_counts[i])
                        ratio = float(cnt / total_pred)
                        pred_support[f"pred_support_class_{i}"] = cnt
                        pred_ratio[f"pred_ratio_class_{i}"] = ratio
                    max_idx = int(np.argmax(pred_counts))
                    max_pred_class = max_idx
                    max_pred_ratio = float(pred_counts[max_idx] / total_pred)
            except Exception as e:
                print(f"[Cloud Server] 預測類別分佈計算失敗: {e}")
                pred_support = {}
                pred_ratio = {}
                max_pred_ratio = None
                max_pred_class = None

        try:
            cm = confusion_matrix(y, pred, labels=list(range(len(label_names))))
            pairs = []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i != j and cm[i, j] > 0:
                        pairs.append(((i, j), int(cm[i, j])))
            pairs.sort(key=lambda x: x[1], reverse=True)
            top5 = pairs[:5]
            confusion_summary = {f"cm_top{i+1}": f"{label_names[a]}→{label_names[b]}:{cnt}" for i, ((a, b), cnt) in enumerate(top5)}
        except Exception:
            confusion_summary = {}
        result = {
            'round': int(round_id),
            'samples': int(len(y)),
            'accuracy': acc,
            'f1_score': f1,
            'joint_acc': joint_acc,
            'joint_f1': joint_f1,
            'loss': loss,
            'debug_forced_baseline': False,
            **per_class_f1,
            **support,
            **pred_support,
            **pred_ratio,
            'max_pred_ratio': max_pred_ratio if max_pred_ratio is not None else 0.0,
            'max_pred_class': max_pred_class if max_pred_class is not None else -1,
            **label_name_fields,
            **confusion_summary
        }
        
        try:
            per_class_f1_array = []
            num_classes = len(label_names)
            for i in range(num_classes):
                f1_key = f"f1_class_{i}"
                class_f1 = per_class_f1.get(f1_key, 0.0)
                per_class_f1_array.append(class_f1)
            
            result_dir = os.environ.get('EXPERIMENT_DIR') or config.LOG_DIR
            model_dir = os.path.join(result_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            weights_file = os.path.join(model_dir, "global_class_weights.npy")
            
            if os.path.exists(weights_file):
                weights_dict = np.load(weights_file, allow_pickle=True).item()
                current_weights = np.array(weights_dict.get('weights', np.ones(num_classes, dtype=np.float32)))
            else:

                print(f"[Cloud Server] 全局類別權重文件不存在，創建默認權重")
                loss_config = getattr(config, 'LOSS_CONFIG', {})
                min_class_weight = float(loss_config.get('min_class_weight', 1.5))
                max_class_weight = float(loss_config.get('max_class_weight', 30.0))

                initial_weight = (min_class_weight + max_class_weight) / 2.0
                current_weights = np.ones(num_classes, dtype=np.float32) * initial_weight
                weights_dict = {
                    'weights': current_weights.tolist(),
                    'num_classes': num_classes,
                    'last_updated_round': 0
                }
            
            print(f"[Cloud Server] 已跳過自動類別權重更新（暫時固定 global_class_weights 以降低震盪）")
        except Exception as e:
            print(f"[Cloud Server] 自適應權重更新失敗: {e}")
            import traceback
            traceback.print_exc()

        try:
            if hasattr(app, "state"):
                last_f1_same = getattr(app.state, "last_eval_f1", None)
                repeat_count = int(getattr(app.state, "repeat_f1_count", 0))
                if last_f1_same is not None and abs(f1 - last_f1_same) < 1e-6:
                    repeat_count += 1
                else:
                    repeat_count = 0
                app.state.last_eval_f1 = float(f1)
                app.state.repeat_f1_count = repeat_count

                if repeat_count >= 2:
                    boost = 1.5
                    old_mul = CURRENT_SERVER_LR_MULTIPLIER
                    simulated_new = min(2.0, CURRENT_SERVER_LR_MULTIPLIER * boost)
                    print(f"[Cloud Server] ⚡ 檢測到連續僵死 F1（{repeat_count+1} 輪相同），原設計將暫時提高 SERVER_LR 乘數 {old_mul:.3f} -> {simulated_new:.3f} 以打破停滯（現已改為純 log，不實際調整）")

                    app.state.repeat_f1_count = 0
        except Exception as unlock_exc:
            print(f"[Cloud Server] 解鎖機制處理異常: {unlock_exc}")

        try:
            import torch
            import asyncio
            result_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
            best_path = getattr(app.state, 'best_weights_path', None)
            if not best_path:
                best_path = os.path.join(result_dir, 'best_global_weights.pt')
                app.state.best_weights_path = best_path
            best_f1 = float(getattr(app.state, 'best_f1', -1.0))
            tolerance = float(getattr(app.state, 'eval_drop_tolerance', 0.03))
            patience = int(getattr(app.state, 'eval_drop_patience', 3))
            app.state.last_global_f1 = float(f1)
        
            
            if BEST_GLOBAL_F1 > 0:
                f1_drop_ratio = (BEST_GLOBAL_F1 - f1) / BEST_GLOBAL_F1 if BEST_GLOBAL_F1 > 0 else 0.0
                
                if SOFT_ROLLBACK_F1_DROP_THRESHOLD <= f1_drop_ratio < HARD_ROLLBACK_F1_DROP_THRESHOLD:
                    print(f"[Cloud Server]  軟回退觸發：F1下降 {f1_drop_ratio*100:.1f}% ({BEST_GLOBAL_F1:.4f} → {f1:.4f})")
                    print(f"[Cloud Server]   - 不回退權重，但（現已改為純 log）僅記錄原本預期的學習率調整")

                    simulated_lr_mult = max(MIN_SERVER_LR_MULTIPLIER, CURRENT_SERVER_LR_MULTIPLIER * 0.7)
                    print(f"[Cloud Server]   - 預期學習率乘數: {CURRENT_SERVER_LR_MULTIPLIER:.4f} → {simulated_lr_mult:.4f}（僅 log，不實際變更）")

                    if f1_drop_ratio >= 0.20:
                        global KD_ALPHA_LOCKED, LOCKED_KD_ALPHA, LAST_KD_ALPHA_BEFORE_LOCK
                        if not KD_ALPHA_LOCKED:
                            KD_ALPHA_LOCKED = True

                            if LAST_KD_ALPHA_BEFORE_LOCK is not None:
                                LOCKED_KD_ALPHA = LAST_KD_ALPHA_BEFORE_LOCK
                            print(f"[Cloud Server]   -  KD Alpha 已鎖定（F1下降 {f1_drop_ratio*100:.1f}% > 20%）")
                            if LOCKED_KD_ALPHA is not None:
                                print(f"[Cloud Server]   - 鎖定值: {LOCKED_KD_ALPHA:.4f}")
            
                elif f1_drop_ratio >= HARD_ROLLBACK_F1_DROP_THRESHOLD:
                    print(f"[Cloud Server]  硬回退觸發：F1下降 {f1_drop_ratio*100:.1f}% ({BEST_GLOBAL_F1:.4f} → {f1:.4f})")
                    if ENABLE_ROLLBACK_MECHANISM:
                        print(f"[Cloud Server]   - 回退到最佳模型並進入凍結模式")
                        needs_rollback_flag = True
                        rollback_reason_str = f"hard_rollback_f1_drop_{f1_drop_ratio:.2f}"
                    else:
                        print(f"[Cloud Server]   - 回退機制已禁用，繼續訓練（可能導致性能下降）")
            
            if f1 >= HIGH_F1_PROTECTION_THRESHOLD:
                HIGH_F1_STABLE_ROUNDS += 1
                if HIGH_F1_STABLE_ROUNDS >= HIGH_F1_MIN_STABLE_ROUNDS:

                    old_lr_mult = CURRENT_SERVER_LR_MULTIPLIER
                    simulated_lr_mult = max(MIN_SERVER_LR_MULTIPLIER, CURRENT_SERVER_LR_MULTIPLIER * HIGH_F1_PROTECTION_LR_REDUCTION)
                    print(f"[Cloud Server] 高F1保護機制啟用：F1={f1:.4f} (已穩定{HIGH_F1_STABLE_ROUNDS}輪，僅記錄預期調整)")
                    print(f"[Cloud Server]   - 預期學習率乘數：{old_lr_mult:.4f} → {simulated_lr_mult:.4f} (降低{(1-HIGH_F1_PROTECTION_LR_REDUCTION)*100:.0f}%，純 log)")

                    old_trust = POST_ROLLBACK_TRUST_ALPHA
                    simulated_trust = min(0.95, POST_ROLLBACK_TRUST_ALPHA + HIGH_F1_PROTECTION_TRUST_INCREASE)
                    if simulated_trust > old_trust:
                        print(f"[Cloud Server]   - 預期信任比例：{old_trust:.2f} → {simulated_trust:.2f}（純 log，不實際變更）")
            else:
                if HIGH_F1_STABLE_ROUNDS > 0:
                    HIGH_F1_STABLE_ROUNDS = 0
                    print(f"[Cloud Server] 高F1保護機制關閉：F1={f1:.4f} < {HIGH_F1_PROTECTION_THRESHOLD:.2f}")
            
            if f1 < 0.3 and CURRENT_SERVER_LR_MULTIPLIER < 1.0:
                recovery_rate = 1.05
                old_lr_mult = CURRENT_SERVER_LR_MULTIPLIER
                simulated_lr_mult = min(1.0, CURRENT_SERVER_LR_MULTIPLIER * recovery_rate)
                if simulated_lr_mult > old_lr_mult:
                    print(f"[Cloud Server]  動態學習率調整（純 log）：低F1={f1:.4f}，預期學習率逐步恢復：{old_lr_mult:.4f} → {simulated_lr_mult:.4f}")
            
            app.state.last_f1 = f1

            try:
                per_class_f1_dict = {}
                num_classes = len(label_names)
                for i in range(num_classes):
                    f1_key = f"f1_class_{i}"
                    class_f1 = per_class_f1.get(f1_key, 0.0)
                    per_class_f1_dict[i] = class_f1
                app.state.last_per_class_f1 = per_class_f1_dict
            except Exception as e:
                print(f"[Cloud Server] 保存各類別 F1 分數失敗: {e}")
                app.state.last_per_class_f1 = None
            
            if f1 > best_f1 + 1e-6:
                app.state.best_f1 = f1
                app.state.eval_drop_streak = 0
                try:
                    torch.save(global_weights, best_path)
                    print(f"[Cloud Server]  已更新最佳權重到 {best_path} (F1={f1:.4f})")
                except Exception as e:
                    print(f"[Cloud Server] 儲存最佳權重失敗: {e}")
            else:
                if (best_f1 - f1) > tolerance:
                    app.state.eval_drop_streak = int(getattr(app.state, 'eval_drop_streak', 0)) + 1
                else:
                    app.state.eval_drop_streak = 0

            if int(getattr(app.state, 'eval_drop_streak', 0)) >= patience:
                rollback_path = getattr(app.state, 'best_weights_path', None)
                if rollback_path and os.path.exists(rollback_path):
                    try:
                        rollback_sd = torch.load(rollback_path, map_location='cpu')
                        if isinstance(rollback_sd, dict):
                            globals()['global_weights'] = rollback_sd
                            globals()['global_version'] = globals().get('global_version', 0) + 1
                            print(f"[Cloud Server] 🔁 觸發回退至最佳權重 (F1={best_f1:.4f}, path={rollback_path}), 更新版本={globals().get('global_version')}")

                            try:
                                broadcast_round = int(round_id) if round_id is not None else -1
                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                if loop.is_running():
                                    loop.create_task(_immediate_broadcast_global_weights(broadcast_round))
                                else:
                                    loop.run_until_complete(_immediate_broadcast_global_weights(broadcast_round))
                            except Exception as e:
                                print(f"[Cloud Server] 回退後廣播失敗: {e}")
                    except Exception as e:
                        print(f"[Cloud Server] 回退讀取失敗: {e}")
                app.state.eval_drop_streak = 0
        except Exception as e:
            print(f"[Cloud Server] 評估連降回退邏輯失敗: {e}")

        try:
            result_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
            os.makedirs(result_dir, exist_ok=True)
            out_path = os.path.join(result_dir, 'cloud_baseline.csv')
            import csv
            import fcntl
            import json
            from typing import Dict as _Dict, Any as _Any
            
            file_exists = os.path.exists(out_path)
            file_size = os.path.getsize(out_path) if file_exists else 0
            need_header = not file_exists or file_size == 0
            
            existing_rounds = set()
            if file_exists and file_size > 0:
                try:
                    with open(out_path, 'r', encoding='utf-8') as check_f:
                        reader = csv.DictReader(check_f)
                        for row in reader:
                            if 'round' in row:
                                try:
                                    existing_rounds.add(int(row['round']))
                                except (ValueError, TypeError):
                                    pass
                except Exception as check_e:
                    print(f"[Cloud Server] 檢查現有輪次失敗: {check_e}")
            
            if 'f1_score' in result:
                current_f1 = float(result['f1_score'])

                historical_max_f1 = float(os.environ.get('HISTORICAL_MAX_F1', str(BEST_GLOBAL_F1 if BEST_GLOBAL_F1 > 0 else current_f1)))
                if current_f1 > historical_max_f1:
                    historical_max_f1 = current_f1
                    os.environ['HISTORICAL_MAX_F1'] = str(historical_max_f1)
                
                if historical_max_f1 > HIGH_F1_PROTECTION_THRESHOLD:

                    os.environ['HIGH_F1_LR_DECAY'] = '1'
                    os.environ['CURRENT_GLOBAL_F1'] = str(historical_max_f1)
                    print(f"[Cloud Server]  歷史最高 F1={historical_max_f1:.4f} > {HIGH_F1_PROTECTION_THRESHOLD:.4f}，已啟用學習率衰減（當前 F1={current_f1:.4f}）")
                else:

                    if 'HIGH_F1_LR_DECAY' in os.environ:
                        del os.environ['HIGH_F1_LR_DECAY']
                        print(f"[Cloud Server]   歷史最高 F1={historical_max_f1:.4f} <= {HIGH_F1_PROTECTION_THRESHOLD:.4f}，已恢復正常學習率")
            
            if round_id < 0:
                print(f"[Cloud Server]  品質檢查評估 (round_id={round_id})，跳過 CSV 寫入")
                return result
            
            if round_id in existing_rounds:
                print(f"[Cloud Server]  輪次 {round_id} 已存在於 cloud_baseline.csv，跳過重複寫入")

                return result
            else:
                base_fields = ['round','samples','accuracy','f1_score','loss','timestamp']
                extra_fields = sorted([k for k in result.keys() if k not in ['round','samples','accuracy','f1_score','loss']])
                
                file_mode = 'r+' if os.path.exists(out_path) and os.path.getsize(out_path) > 0 else 'a'
                with open(out_path, file_mode, newline='', encoding='utf-8') as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        try:

                            current_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                            if current_size == 0:
                                need_header = True
                            
                            if current_size > 0:
                                f.seek(0)
                                reader = csv.DictReader(f)
                                for row in reader:
                                    if 'round' in row:
                                        try:
                                            if int(row['round']) == round_id:
                                                print(f"[Cloud Server]  輪次 {round_id} 在獲取鎖後發現已存在，跳過寫入")
                                                return result
                                        except (ValueError, TypeError):
                                            pass
                                f.seek(0, 2)
                            
                            try:
                                if hasattr(app, 'state'):
                                    round_upload_mb = float(getattr(app.state, 'round_upload_mb', 0.0))
                                    round_download_mb = float(getattr(app.state, 'round_download_mb', 0.0))
                                    round_total_mb = float(getattr(app.state, 'round_total_mb', 0.0))
                                else:
                                    round_upload_mb = float(os.environ.get('ROUND_UPLOAD_MB', '0.0'))
                                    round_download_mb = float(os.environ.get('ROUND_DOWNLOAD_MB', '0.0'))
                                    round_total_mb = float(os.environ.get('ROUND_TOTAL_MB', '0.0'))
                            except Exception as comm_read_e:
                                print(f"[Cloud Server] 讀取通訊量數據失敗: {comm_read_e}，使用預設值 0.0")
                                round_upload_mb = 0.0
                                round_download_mb = 0.0
                                round_total_mb = 0.0
                            
                            cumulative_total_mb = round_total_mb
                            if current_size > 0:
                                try:
                                    f.seek(0)
                                    reader = csv.DictReader(f)
                                    prev_cumulative = 0.0
                                    for prev_row in reader:
                                        if 'cumulative_comm_mb' in prev_row:
                                            try:
                                                prev_cumulative = float(prev_row['cumulative_comm_mb'])
                                            except (ValueError, TypeError):
                                                pass
                                    cumulative_total_mb = prev_cumulative + round_total_mb
                                    f.seek(0, 2)
                                except Exception as cum_e:
                                    print(f"[Cloud Server] 計算累積通訊量失敗: {cum_e}")
                            
                            writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields + ['upload_mb', 'download_mb', 'round_comm_mb', 'cumulative_comm_mb'])
                            if need_header:
                                writer.writeheader()
                            row = dict(result)
                            row['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            row['upload_mb'] = round_upload_mb
                            row['download_mb'] = round_download_mb
                            row['round_comm_mb'] = round_total_mb
                            row['cumulative_comm_mb'] = cumulative_total_mb
                            writer.writerow(row)
                            f.flush()
                            print(f"[Cloud Server] 已寫入 cloud_baseline.csv: round={round_id}, f1={row.get('f1_score', 'N/A')}, 通訊量={round_total_mb:.4f} MB (累積={cumulative_total_mb:.4f} MB)")

                            try:
                                dyn_cfg = getattr(config, "DYNAMIC_CLASS_WEIGHTING", {}) or {}
                                if dyn_cfg.get("enabled", False):
                                    filename = dyn_cfg.get("filename", "dynamic_class_weights.json")
                                    eps = float(dyn_cfg.get("eps", 1e-3))
                                    normalize_mean1 = bool(dyn_cfg.get("normalize_mean1", True))
                                    beta = float(dyn_cfg.get("ema_beta", 0.85))
                                    min_w = float(dyn_cfg.get("min_weight", 0.5))
                                    max_w = float(dyn_cfg.get("max_weight", 5.0))
                                    extra_top_k = int(dyn_cfg.get("extra_boost_top_k", 0) or 0)
                                    extra_factor = float(dyn_cfg.get("extra_boost_factor", 1.15))
                                    extra_min_f1 = float(dyn_cfg.get("extra_boost_min_f1", 0.0) or 0.0)
                                    use_sliding_window = bool(dyn_cfg.get("use_sliding_window", True))
                                    sliding_window_size = int(dyn_cfg.get("sliding_window_size", 3) or 3)
                                    update_frequency = int(dyn_cfg.get("update_frequency", 1) or 1)
                                    dyn_path = os.path.join(result_dir, filename)
                                    should_update = True
                                    if update_frequency > 1:
                                        prev_round = None
                                        if os.path.exists(dyn_path):
                                            try:
                                                with open(dyn_path, "r", encoding="utf-8") as rf:
                                                    prev_obj = json.load(rf) or {}
                                                prev_round = prev_obj.get("round")
                                            except Exception:
                                                pass
                                        if prev_round is not None:
                                            rounds_since_update = int(round_id) - int(prev_round)
                                            if rounds_since_update < update_frequency:
                                                should_update = False
                                                print(f"[Cloud Server]  動態權重更新頻率控制：距離上次更新 {rounds_since_update} 輪 < {update_frequency}，跳過本次更新")

                                    if not should_update:
                                        pass
                                    else:
                                        f1_keys = ["f1_class_0", "f1_class_1", "f1_class_2", "f1_class_3", "f1_class_4"]
                                        if all(k in row for k in f1_keys):
                                            f1_vals = []
                                            if use_sliding_window and sliding_window_size > 1:
                                                try:
                                                    import pandas as pd
                                                    baseline_df = pd.read_csv(out_path)
                                                    if len(baseline_df) > 0:
                                                        recent_rounds = baseline_df.tail(sliding_window_size - 1)
                                                        avg_f1_per_class = []
                                                        for k in f1_keys:
                                                            if k in recent_rounds.columns:
                                                                avg_f1 = float(recent_rounds[k].mean())
                                                                avg_f1_per_class.append(avg_f1)
                                                            else:
                                                                avg_f1_per_class.append(0.0)

                                                        current_f1_per_class = []
                                                        for k in f1_keys:
                                                            try:
                                                                current_f1_per_class.append(float(row.get(k, 0.0)))
                                                            except Exception:
                                                                current_f1_per_class.append(0.0)

                                                        window_weight = 1.0 / sliding_window_size
                                                        current_weight = 1.0 - (sliding_window_size - 1) * window_weight
                                                        for i in range(len(f1_keys)):
                                                            smoothed_f1 = current_weight * current_f1_per_class[i] + (1.0 - current_weight) * avg_f1_per_class[i]
                                                            f1_vals.append(smoothed_f1)
                                                        print(f"[Cloud Server] 使用滑動窗口平均 F1（窗口大小={sliding_window_size}）：當前={[f'{x:.4f}' for x in current_f1_per_class]}，平均={[f'{x:.4f}' for x in f1_vals]}")
                                                    else:
                                                        for k in f1_keys:
                                                            try:
                                                                f1_vals.append(float(row.get(k, 0.0)))
                                                            except Exception:
                                                                f1_vals.append(0.0)
                                                except Exception as window_e:

                                                    print(f"[Cloud Server] 滑動窗口計算失敗: {window_e}，回退到單輪 F1")
                                                    for k in f1_keys:
                                                        try:
                                                            f1_vals.append(float(row.get(k, 0.0)))
                                                        except Exception:
                                                            f1_vals.append(0.0)
                                            else:
                                                for k in f1_keys:
                                                    try:
                                                        f1_vals.append(float(row.get(k, 0.0)))
                                                    except Exception:
                                                        f1_vals.append(0.0)

                                            raw = []
                                            for v in f1_vals:
                                                vv = max(v, eps)
                                                raw.append(1.0 / vv)

                                            if normalize_mean1:
                                                m = (sum(raw) / max(1, len(raw)))
                                                if m > 0:
                                                    raw = [x / m for x in raw]

                                            raw = [min(max(x, min_w), max_w) for x in raw]

                                            if extra_top_k > 0 and extra_factor > 1.0:
                                                idx_f1_pairs = list(enumerate(f1_vals))
                                                idx_f1_pairs.sort(key=lambda p: (p[1] if p[1] is not None else 0.0))
                                                boosted = 0
                                                for idx, f1_v in idx_f1_pairs:
                                                    if boosted >= extra_top_k:
                                                        break

                                                    if f1_v is None:
                                                        continue
                                                    if f1_v >= extra_min_f1:
                                                        continue

                                                    before = raw[idx]
                                                    after = before * extra_factor
                                                    after = min(max(after, min_w), max_w)
                                                    raw[idx] = after
                                                    boosted += 1
                                                    print(f"[Cloud Server]  額外 boost 類別 {idx}：F1={f1_v:.4f}，權重 {before:.4f} → {after:.4f}")

                                            prev = None
                                            if os.path.exists(dyn_path):
                                                try:
                                                    with open(dyn_path, "r", encoding="utf-8") as rf:
                                                        prev_obj = json.load(rf) or {}
                                                    prev = prev_obj.get("class_weight")
                                                except Exception:
                                                    prev = None
                                            if isinstance(prev, dict):
                                                smoothed = []
                                                for i, cur in enumerate(raw):

                                                    pv = prev.get(str(i), prev.get(i, None))
                                                    try:
                                                        pv = float(pv)
                                                    except Exception:
                                                        pv = None
                                                    if pv is None:
                                                        smoothed.append(cur)
                                                    else:
                                                        smoothed.append(beta * pv + (1.0 - beta) * cur)
                                                raw = smoothed
                                                print(f"[Cloud Server] EMA 平滑（beta={beta}）：權重已平滑")

                                            payload: _Dict[str, _Any] = {
                                                "round": int(round_id),
                                                "source": "cloud_f1",
                                                "f1": {str(i): float(f1_vals[i]) for i in range(len(f1_vals))},
                                                "class_weight": {str(i): float(raw[i]) for i in range(len(raw))},
                                                "timestamp": row.get("timestamp"),
                                            }
                                            with open(dyn_path, "w", encoding="utf-8") as wf:
                                                json.dump(payload, wf, ensure_ascii=False, indent=2)
                                            print(f"[Cloud Server]  已寫入動態 class weight: {dyn_path} -> {payload['class_weight']}")
                            except Exception as _dyn_e:
                                print(f"[Cloud Server] 生成 dynamic_class_weights.json 失敗: {_dyn_e}")
                        except Exception as write_exc:
                            print(f"[Cloud Server] 寫入 cloud_baseline.csv 時發生錯誤 (round={round_id}): {write_exc}")
                            import traceback
                            traceback.print_exc()
                    except (IOError, OSError) as lock_e:

                        print(f"[Cloud Server] 文件鎖不可用: {lock_e}，使用無鎖寫入")

                        current_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                        if current_size == 0:
                            need_header = True
                        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
                        if need_header:
                            writer.writeheader()
                        row = dict(result)
                        row['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow(row)
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            try:
                fairness_path = os.path.join(result_dir, 'fairness_summary.csv')
                f_exists = os.path.exists(fairness_path)

                per_client_f1 = []
                for name in os.listdir(result_dir):
                    subdir = os.path.join(result_dir, name)
                    if os.path.isdir(subdir) and name.startswith('uav'):
                        split_eval = os.path.join(subdir, f"{name}_split_eval.csv")
                        curve = os.path.join(subdir, f"{name}_curve.csv")
                        f1_val = None
                        try:
                            if os.path.exists(split_eval):
                                import pandas as _pd
                                _df = _pd.read_csv(split_eval)
                                if 'f1' in _df.columns:
                                    f1_val = float(_df.tail(1)['f1'].values[0])
                        except Exception:
                            f1_val = None
                        if f1_val is None:
                            try:
                                if os.path.exists(curve):
                                    import pandas as _pd
                                    _df = _pd.read_csv(curve)
                                    if 'f1' in _df.columns:
                                        f1_val = float(_df.tail(1)['f1'].values[0])
                            except Exception:
                                f1_val = None
                        if f1_val is not None:
                            per_client_f1.append(f1_val)
                if per_client_f1:
                    per_client_f1.sort()
                    q10_idx = max(0, int(len(per_client_f1) * 0.10) - 1)
                    q10 = per_client_f1[q10_idx]
                    worst = per_client_f1[0]
                    with open(fairness_path, 'a', newline='', encoding='utf-8') as ff:
                        w = csv.writer(ff)
                        if not f_exists:
                            w.writerow(['round','q10_macro_f1','min_macro_f1','num_clients_used'])
                        w.writerow([int(round_id), q10, worst, len(per_client_f1)])
            except Exception as _e:
                print(f"[Cloud Server] 公平性統計失敗: {_e}")

            f1_value = f1 if 'f1' in locals() else 0.0
            acc_value = acc if 'acc' in locals() else 0.0
            loss_value = loss if 'loss' in locals() else float('inf')
            y_len = len(y) if 'y' in locals() else 0
            print(f"[Cloud Server] [進度] 全域基準評測完成: acc={acc_value:.4f}, f1={f1_value:.4f}, loss={loss_value:.4f}, samples={y_len} (round_id={round_id})", flush=True)
        except Exception as e:
            print(f"[Cloud Server] 寫入 cloud_baseline.csv 失敗: {e}")
        return result
    except Exception as e:
        print(f"[Cloud Server] 全域基準評測失敗: {e}")
        return None

@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "cloud_server_id": cloud_server_id,
        "aggregation_count": aggregation_count,
        "aggregator_count": aggregator_count,
        "registered_aggregators": list(registered_aggregators.keys())
    })

@app.post("/register_aggregator")
async def register_aggregator(
    aggregator_id: int = Form(...),
    status: str = Form("ready"),
    port: int = Form(...)
):
    try:
        global aggregator_count
        
        with lock:
            registered_aggregators[aggregator_id] = {
                'status': status,
                'host': '127.0.0.1',
                'port': port,
                'register_time': time.time(),
                'last_heartbeat': time.time()
            }
            aggregator_count = len(registered_aggregators)
        
        print(f"[Cloud Server]  聚合器 {aggregator_id} 註冊成功")
        print(f"  - 狀態: {status}")
        print(f"  - 端口: {port}")
        print(f"  - 總聚合器數量: {aggregator_count}")
        
        log_event("aggregator_registered", f"aggregator_id={aggregator_id},status={status},port={port}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"聚合器 {aggregator_id} 註冊成功",
            "aggregator_id": aggregator_id,
            "total_aggregators": aggregator_count
        })
        
    except Exception as e:
        error_msg = f"聚合器註冊失敗: {str(e)}"
        print(f"[Cloud Server] {error_msg}")
        return JSONResponse(content={"status": "error", "message": error_msg}, status_code=500)
    
    try:
        if not hasattr(app.state, "last_evaluated_round"):
            app.state.last_evaluated_round = -1
        if round_id != app.state.last_evaluated_round:
            print(f"[Cloud Server] Round 變更檢測: {app.state.last_evaluated_round} → {round_id}，將強制觸發全域評估")
            app.state.last_evaluated_round = round_id

    except Exception as e:
        print(f"[Cloud Server] Round 變更檢測失敗: {e}")
    
    try:
        if early_stop_triggered:
            return JSONResponse(content={"status":"stopped","message":f"early_stop: {early_stop_reason}"}, status_code=200)
        
        if torch is None:
            raise HTTPException(status_code=500, detail="PyTorch not available")
        
        data_bytes = await aggregated_data.read()
        aggregated_data_dict = pickle.loads(data_bytes)
    
        aggregated_embedding = aggregated_data_dict.get('embedding')
        confidence = aggregated_data_dict.get('confidence', 0.5)
        participating_clients = aggregated_data_dict.get('participating_clients', [])
        aggregator_id = aggregated_data_dict.get('aggregator_id', 0)
        aggregated_client_predictions = aggregated_data_dict.get('client_predictions')
        
        if aggregated_embedding is None:
            print(f"[Cloud Server] 錯誤: 聚合內嵌表示為空")
            raise HTTPException(status_code=400, detail="Empty aggregated embedding")
        

        if not isinstance(aggregated_embedding, torch.Tensor):
            if isinstance(aggregated_embedding, (list, np.ndarray)):
                aggregated_embedding = torch.tensor(aggregated_embedding, dtype=torch.float32)
            else:
                raise ValueError(f"不支持的內嵌表示類型: {type(aggregated_embedding)}")
        
        global_prototypes = None
        print(f"  - 輪次: {round_id}")
        print(f"  - 聚合器: {aggregator_id}")
        print(f"  - 參與客戶端: {len(participating_clients)} 個")
        print(f"  - 平均置信度: {confidence:.4f}")
        print(f"  - 內嵌表示形狀: {aggregated_embedding.shape}")


        server_training_cfg = getattr(config, "SERVER_TRAINING_CONFIG", {}) or {}
        server_training_enabled = bool(server_training_cfg.get("enabled", True))

        if server_training_enabled:


            updated_weights = await train_server_model_with_embedding(
                round_id,
                aggregated_embedding,
                confidence,
                participating_clients,
                aggregated_client_predictions,
                global_prototypes,
            )

            try:
                if hasattr(app.state, "evaluated_rounds"):
                    eval_key = f"round_{int(round_id)}"
                    if eval_key in app.state.evaluated_rounds:
                        print(f"[Cloud Server] Round {round_id} 已評估過，但檢測到 round 變更，強制重新評估")
                        app.state.evaluated_rounds.discard(eval_key)
            except Exception:
                pass


            _schedule_global_test_eval(round_id, updated_weights or {})
        else:

            print(f"[Cloud Server]  SERVER_TRAINING_CONFIG.enabled=False，跳過雲端 KD / Prototype Anchor，僅做 FedAvg+FedProx 聚合與評估 (round={round_id})")

            try:
                current_weights = global_weights or {}
            except NameError:
                current_weights = {}

            try:
                if hasattr(app.state, "evaluated_rounds"):
                    eval_key = f"round_{int(round_id)}"
                    if eval_key in app.state.evaluated_rounds:
                        print(f"[Cloud Server] Round {round_id} 已評估過，但檢測到 round 變更，強制重新評估")
                        app.state.evaluated_rounds.discard(eval_key)
            except Exception:
                pass
            _schedule_global_test_eval(round_id, current_weights)
        
        return JSONResponse(content={
            "status": "received",
            "message": "異質性聚合結果已接收並用於伺服器訓練",
            "round_id": round_id
        })
        
    except Exception as e:
        error_msg = f"接收異質性聚合結果失敗: {str(e)}"
        print(f"[Cloud Server] {error_msg}")
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": error_msg}, status_code=500)

async def train_server_model_with_embedding(
    round_id: int,
    aggregated_embedding: torch.Tensor,
    confidence: float,
    participating_clients: List[int],
    aggregated_client_predictions: Optional[List[float]] = None,
    global_prototypes: Optional[Dict[int, torch.Tensor]] = None
):
    global global_weights
    updated_weights = global_weights.copy() if global_weights else {}
    return updated_weights

def compute_knowledge_distillation_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    temperature: float = 4.0,
    loss_type: str = 'mse',
    use_temperature: bool = True
) -> torch.Tensor:
    student_mean = torch.mean(student_output, dim=-1, keepdim=True)
    student_std = torch.std(student_output, dim=-1, keepdim=True) + 1e-7
    student_output = (student_output - student_mean) / student_std
    
    teacher_mean = torch.mean(teacher_output, dim=-1, keepdim=True)
    teacher_std = torch.std(teacher_output, dim=-1, keepdim=True) + 1e-7
    teacher_output = (teacher_output - teacher_mean) / teacher_std

    try:
        import config as config_module
        model_config = getattr(config_module, 'MODEL_CONFIG', {})
        kd_config = model_config.get('knowledge_distillation', {})
        logit_bias = kd_config.get('logit_bias', {})
        
        if logit_bias:
            for class_idx, bias_value in logit_bias.items():
                class_idx = int(class_idx)
                if class_idx < student_output.size(-1):
                    student_output[:, class_idx] += bias_value
                    teacher_output[:, class_idx] += bias_value

                    if not hasattr(compute_knowledge_distillation_loss, '_bias_logged'):
                        print(f"[Cloud Server]  破局配置 4.0：類別 {class_idx} Logit 偏置 +{bias_value:.2f}（強制起跳）")
                        compute_knowledge_distillation_loss._bias_logged = True
    except Exception as e:
        pass
    
    LOGITS_CLIP_MIN = -5.0
    LOGITS_CLIP_MAX = 5.0
    student_output = torch.clamp(student_output, min=LOGITS_CLIP_MIN, max=LOGITS_CLIP_MAX)
    teacher_output = torch.clamp(teacher_output, min=LOGITS_CLIP_MIN, max=LOGITS_CLIP_MAX)
    
    if use_temperature and temperature > 1.0:

        student_soft = torch.softmax(student_output / temperature, dim=-1)
        teacher_soft = torch.softmax(teacher_output / temperature, dim=-1)
    else:
        student_soft = torch.softmax(student_output, dim=-1)
        teacher_soft = torch.softmax(teacher_output, dim=-1)
    
    if loss_type == 'kl_div':

        import torch.nn.functional as F

        loss = F.kl_div(
            F.log_softmax(student_output / temperature, dim=-1),
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
    else:

        loss = torch.nn.functional.mse_loss(student_soft, teacher_soft)
    
    return loss

async def _train_with_labeled_data(
    server_model: torch.nn.Module,
    server_dataset_path: str,
    aggregated_embedding: torch.Tensor,
    kd_enabled: bool,
    kd_temperature: float,
    kd_alpha: float,
    kd_loss_type: str,
    use_temperature: bool,
    server_epochs: int,
    device: torch.device,
    round_id: int,
    server_lr: float = 2e-3
):
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    
    try:
        if pd is None:
            print(f"[Cloud Server] pandas 未安裝，無法讀取數據集")
            return
        
        df = pd.read_csv(server_dataset_path, encoding='utf-8-sig', low_memory=False)
        if df.empty:
            print(f"[Cloud Server] 伺服器數據集為空")
            return
        
        possible_label_cols = ['Attack_label', 'Target Label', 'label', 'Label', 'target_label']
        label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"[Cloud Server] 找不到標籤欄位，改用知識蒸餾模式")
            await _train_with_knowledge_distillation_only(
                server_model, aggregated_embedding, 0.5,
                kd_temperature, kd_loss_type, use_temperature,
                server_epochs, device, round_id
            )
            return
        
        feature_cols = [col for col in df.columns if col != label_col]
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        weight_decay = float(os.environ.get("SERVER_WEIGHT_DECAY", getattr(config, "WEIGHT_DECAY", 1e-4)))
        optimizer = optim.Adam(server_model.parameters(), lr=server_lr, weight_decay=weight_decay, amsgrad=True)
        print(f"[Cloud Server] 伺服器優化器配置: lr={server_lr}, weight_decay={weight_decay}, amsgrad=True")
        ce_loss_fn = nn.CrossEntropyLoss()
        
        teacher_output_cache = None
        
        print(f"[Cloud Server]  開始訓練（有標註數據 + 知識蒸餾）")
        server_model.train()
        
        for epoch in range(server_epochs):
            total_loss = 0.0
            total_ce_loss = 0.0
            total_kd_loss = 0.0
            num_batches = 0
            

            if app.state.client_model_for_embedding is None or app.state.input_dim != X_train.shape[1]:
                app.state.input_dim = X_train.shape[1]
                client_model_params = {
                    'input_dim': app.state.input_dim,
                    'embedding_dim': embedding_dim,
                    'output_dim': output_dim,
                    'hidden_dims': heterogeneous_fl_config.get('client_hidden_dims', [512, 256, 128]),
                    'dropout_rate': config.MODEL_CONFIG.get('dropout_rate', 0.3),
                    'use_batch_norm': config.MODEL_CONFIG.get('use_batch_norm', True),
                    'use_residual': config.MODEL_CONFIG.get('use_residual', True),
                    'activation': config.MODEL_CONFIG.get('activation', 'relu')
                }
                app.state.client_model_for_embedding = ClientModel(**client_model_params).to(device)
                app.state.client_model_for_embedding.eval()
                print(f"[Cloud Server]  初始化客戶端模型（用於生成內嵌表示）: input_dim={app.state.input_dim}")
            
            client_model = app.state.client_model_for_embedding
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():

                    teacher_embedding, _ = client_model(features)

                    if features.size(0) == 1:
                        teacher_embedding = aggregated_embedding.expand(1, -1)
                    else:
                        pass
                
                student_output = server_model(teacher_embedding)
                
                ce_loss = ce_loss_fn(student_output, labels)
                
                kd_loss = torch.tensor(0.0, device=device)
                if kd_enabled:
                    with torch.no_grad():
                        teacher_output = server_model(aggregated_embedding.expand(features.size(0), -1))
                    
                    kd_loss = compute_knowledge_distillation_loss(
                        student_output,
                        teacher_output,
                        kd_temperature,
                        kd_loss_type,
                        use_temperature
                    )
                

                if kd_enabled:
                    total_loss_batch = (1 - kd_alpha) * ce_loss + kd_alpha * kd_loss
                else:
                    total_loss_batch = ce_loss
                

                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                total_ce_loss += ce_loss.item()
                if kd_enabled:
                    total_kd_loss += kd_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            avg_ce_loss = total_ce_loss / max(num_batches, 1)
            avg_kd_loss = total_kd_loss / max(num_batches, 1) if kd_enabled else 0.0
            
            print(f"[Cloud Server] Epoch {epoch+1}/{server_epochs}: "
                  f"Loss={avg_loss:.4f}, CE={avg_ce_loss:.4f}, KD={avg_kd_loss:.4f}")
        
    except Exception as e:
        print(f"[Cloud Server] 有標註數據訓練失敗: {e}")
        import traceback
        traceback.print_exc()

async def _train_with_knowledge_distillation_only(
    server_model: torch.nn.Module,
    aggregated_embedding: torch.Tensor,
    confidence: float,
    kd_temperature: float,
    kd_loss_type: str,
    use_temperature: bool,
    server_epochs: int,
    device: torch.device,
    round_id: int,
    server_lr: float = 2e-3,
    kd_config: Optional[Dict[str, Any]] = None,
    aggregated_client_predictions: Optional[List[float]] = None,
    client_diversity_weights: Optional[Dict[int, float]] = None,
):
    import torch.optim as optim
    import config_fixed as config
    from collections import deque
    
    if kd_config is None:
        kd_config = config.MODEL_CONFIG.get('knowledge_distillation', {})
    
    kd_enabled = kd_config.get('enabled', True)
    disable_kd_env = os.environ.get("DISABLE_KD", "0").strip() == "1"
    if disable_kd_env:
        kd_enabled = False
        print(f"[Cloud Server] DISABLE_KD=1：KD 訓練已禁用，僅執行 Prototype Anchor Step（如果啟用）")
    
    try:
        min_server_lr = float(os.environ.get("SERVER_MIN_LR", "0.0001"))
        effective_server_lr = max(float(server_lr), min_server_lr)
        weight_decay = float(os.environ.get("SERVER_WEIGHT_DECAY", getattr(config, "WEIGHT_DECAY", 1e-4)))
        optimizer = optim.Adam(server_model.parameters(), lr=effective_server_lr, weight_decay=weight_decay, amsgrad=True)
        print(f"[Cloud Server] 伺服器優化器配置: lr={effective_server_lr}, weight_decay={weight_decay}, amsgrad=True (min_lr={min_server_lr})")
        
        print(f"[Cloud Server]  開始知識蒸餾訓練（無標註數據）")
        

        aggregated_embedding = aggregated_embedding.to(device)
        if aggregated_embedding.dim() == 1:
            aggregated_embedding = aggregated_embedding.unsqueeze(0)

        replay_max = int(os.environ.get("SERVER_EMBED_REPLAY_MAX", "512"))
        train_batch_size = int(os.environ.get("SERVER_EMBED_TRAIN_BATCH", "64"))
        steps_per_round = int(os.environ.get("SERVER_EMBED_TRAIN_STEPS", "10"))
        if not hasattr(app.state, "server_embedding_replay") or app.state.server_embedding_replay is None:
            app.state.server_embedding_replay = deque(maxlen=replay_max)
            print(f"[Cloud Server] 初始化 embedding replay buffer: maxlen={replay_max}")

        with torch.no_grad():
            for i in range(aggregated_embedding.size(0)):
                app.state.server_embedding_replay.append(aggregated_embedding[i].detach().clone())

        replay_size = len(app.state.server_embedding_replay)
        if replay_size < 4:
            aggregated_embedding_batch = aggregated_embedding.repeat(4, 1)
        else:
            aggregated_embedding_batch = aggregated_embedding
        if steps_per_round < 1:
            steps_per_round = 1

        effective_batch = max(4, min(train_batch_size, replay_size))
        effective_steps = max(1, steps_per_round)
        print(f"[Cloud Server]  Replay buffer 狀態: size={replay_size}, batch={effective_batch}, steps={effective_steps}")
        
        use_embedding_as_teacher = True
        use_cross_layer_teacher = False
        teacher_output_batch = None

        if use_embedding_as_teacher:
            print(f"[Cloud Server]  破局配置 3.7：使用 embedding 生成教師信號（而非直接使用客戶端預測），避免學習偏見模式")
            use_cross_layer_teacher = False
        elif aggregated_client_predictions is not None:

            try:
                teacher_output_batch = torch.tensor(aggregated_client_predictions, dtype=torch.float32, device=device)
                if teacher_output_batch.dim() == 1:
                    teacher_output_batch = teacher_output_batch.unsqueeze(0)
                if teacher_output_batch.size(0) == 1:
                    teacher_output_batch = teacher_output_batch.repeat(2, 1)

                max_prob = torch.max(torch.softmax(teacher_output_batch, dim=-1), dim=-1)[0].mean().item()
                if max_prob > 0.3:
                    print(f"[Cloud Server] 破局配置 3.7：客戶端預測偏見（最大機率={max_prob:.4f} > 0.3），降低教師權重")
                    teacher_output_batch = teacher_output_batch * 0.5
                print(f"[Cloud Server]  破局配置 3.2：使用跨層教師機制（客戶端預測作為教師，已過濾偏見）: {teacher_output_batch.shape}")
                use_cross_layer_teacher = True
            except Exception as e:
                print(f"[Cloud Server] 使用客戶端預測作為教師模型失敗: {e}，回退到自我蒸餾")
                use_cross_layer_teacher = False
        
        def _freeze_bn(m: torch.nn.Module):
            for mod in m.modules():
                if isinstance(mod, torch.nn.BatchNorm1d):
                    mod.eval()

        server_model.train()
        _freeze_bn(server_model)
        
        for epoch in range(server_epochs):
            for step in range(effective_steps):
                if replay_size >= effective_batch:
                    idx = torch.randint(low=0, high=replay_size, size=(effective_batch,))
                    batch = torch.stack([app.state.server_embedding_replay[int(i)] for i in idx], dim=0).to(device)
                else:
                    batch = aggregated_embedding.repeat(effective_batch, 1)
                training_base_embedding = batch

                if teacher_output_batch is None:
                    server_model.eval()
                    with torch.no_grad():
                        teacher_output_batch = server_model(training_base_embedding).detach().clone()
                    server_model.train()
                    _freeze_bn(server_model)

                disable_mixup_env = os.environ.get("DISABLE_MIXUP", "0").strip() == "1"
                use_mixup = not disable_mixup_env and global_prototypes is not None and len(global_prototypes) > 0
                if disable_mixup_env:
                    print(f"[Cloud Server] DISABLE_MIXUP=1：Manifold Mixup 已禁用，僅使用原始 embedding")
                mixup_prob = 0.95
                
                target_class_candidates = []
                if use_mixup:

                    for k in range(5):
                        if k in global_prototypes and global_prototypes[k] is not None:
                            target_class_candidates.append(k)
                    if not target_class_candidates:
                        target_class_candidates = list(global_prototypes.keys())
                
                target_class = None
                if use_mixup and target_class_candidates:
                    rand_val = torch.rand(1).item()
                    should_mixup = rand_val < mixup_prob
                    if should_mixup:
                        candidate_f1_scores = {}
                        if hasattr(app.state, 'last_f1_per_class') and app.state.last_f1_per_class:
                            for candidate_class in target_class_candidates:
                                candidate_f1_scores[candidate_class] = app.state.last_f1_per_class.get(candidate_class, 0.0)
                        else:
                            for candidate_class in target_class_candidates:
                                candidate_f1_scores[candidate_class] = 0.0
                        min_f1 = min(candidate_f1_scores.values())
                        low_f1_candidates = [c for c, f1 in candidate_f1_scores.items() if f1 == min_f1]
                        target_class = int(np.random.choice(low_f1_candidates))
                        if epoch == 0 and step == 0 and len(low_f1_candidates) > 1:
                            print(f"[Cloud Server]  破局配置 4.2：選擇 F1 最低的類別進行 Mixup（F1={min_f1:.4f}，候選：{low_f1_candidates}，選擇：{target_class}）")
                
                if target_class is not None and target_class in global_prototypes and global_prototypes[target_class] is not None:
                    target_proto = global_prototypes[target_class].to(device)
                    if target_proto.dim() == 1:
                        target_proto = target_proto.unsqueeze(0)
                    
                    lam = np.random.uniform(0.4, 0.6)
                    
                    if target_proto.size(0) == 1:
                        target_proto = target_proto.repeat(training_base_embedding.size(0), 1)
                    elif target_proto.size(0) != training_base_embedding.size(0):
                        target_proto = target_proto[:training_base_embedding.size(0)]
                    
                    mixed_embedding = lam * training_base_embedding + (1 - lam) * target_proto
                    training_embedding = mixed_embedding
                    mixup_enabled = True
                else:
                    training_embedding = training_base_embedding
                    mixup_enabled = False
                
                noise_scale = 0.01
                perturbed_embedding = training_embedding + torch.randn_like(training_embedding) * noise_scale
                
                student_output = server_model(perturbed_embedding)
                
                if use_cross_layer_teacher and teacher_output_batch is not None:
                    with torch.no_grad():
                        prediction_diff = torch.mean(torch.abs(torch.softmax(student_output, dim=-1) - torch.softmax(teacher_output_batch, dim=-1)))
                        semantic_gate_threshold = 0.3
                        if prediction_diff > semantic_gate_threshold:
                            print(f"[Cloud Server]  破局配置 3.2：語義一致性檢查失敗（差異={prediction_diff:.4f} > {semantic_gate_threshold}），原設計將增加 FedProx μ（現僅記錄）")
                            global CURRENT_FEDPROX_MU_MULTIPLIER, MAX_FEDPROX_MU_MULTIPLIER
                            old_mu = CURRENT_FEDPROX_MU_MULTIPLIER
                            simulated_mu = min(MAX_FEDPROX_MU_MULTIPLIER, CURRENT_FEDPROX_MU_MULTIPLIER * 1.2)
                            if simulated_mu > old_mu:
                                print(f"[Cloud Server]  FedProx μ 預期增加: {old_mu:.4f} → {simulated_mu:.4f}（純 log，不實際變更）")
                
                kd_loss_weight = 1.0
                diversity_weight_scale = 1.0
                proto_loss_scale = 1.0

                if client_diversity_weights and len(client_diversity_weights) > 0:
                    avg_diversity = sum(client_diversity_weights.values()) / len(client_diversity_weights)
                    kd_loss_weight = np.sqrt(avg_diversity) if avg_diversity > 0 else 0.5
                    kd_loss_weight = max(0.3, min(1.0, kd_loss_weight))
                    if epoch == 0:
                        print(f"[Cloud Server]  破局配置 3.7：加權知識蒸餾（平均多樣性={avg_diversity:.4f}，KD損失權重={kd_loss_weight:.4f}）")
                
                if mixup_enabled and 'lam' in locals() and 'target_class' in locals():
                    original_label = torch.softmax(teacher_output_batch, dim=-1)
                    target_class_tensor = torch.zeros_like(original_label)
                    target_class_tensor[:, target_class] = 1.0
                    
                    target_class_f1 = 0.0
                    if hasattr(app.state, 'last_f1_per_class') and app.state.last_f1_per_class:
                        target_class_f1 = app.state.last_f1_per_class.get(target_class, 0.0)
                    
                    f1_threshold = 0.1
                    use_hard_label = target_class_f1 < f1_threshold
                    
                    if use_hard_label:
                        mixed_label = target_class_tensor
                        lam_adjusted = 0.0
                        mixup_weight_multiplier = max(1.5, min(2.0, 2.0 * (1.0 - target_class_f1 / f1_threshold)))
                    else:
                        lam_adjusted = max(0.2, min(0.7, 0.5 + 0.2 * (target_class_f1 - f1_threshold) / (0.5 - f1_threshold)))
                        lam_adjusted = max(0.2, min(lam_adjusted, lam))
                        mixed_label = lam_adjusted * original_label + (1 - lam_adjusted) * target_class_tensor
                        mixup_weight_multiplier = max(1.0, min(2.0, 2.0 * (1.0 - target_class_f1)))
                    
                    mixed_logits = torch.log(mixed_label + 1e-7)
                    if kd_enabled:
                        kd_loss = compute_knowledge_distillation_loss(
                            student_output,
                            mixed_logits,
                            kd_temperature,
                            kd_loss_type,
                            use_temperature
                        ) * kd_loss_weight * mixup_weight_multiplier
                    else:
                        kd_loss = torch.tensor(0.0, device=device)
                    if epoch == 0:
                        label_type = "硬標籤" if use_hard_label else "軟標籤"
                        lam_info = f"{lam:.3f}→{lam_adjusted:.3f}(F1={target_class_f1:.3f})" if lam_adjusted != lam else f"{lam:.3f}"
            else:
                if kd_enabled:
                    kd_loss = compute_knowledge_distillation_loss(
                        student_output,
                        teacher_output_batch,
                        kd_temperature,
                        kd_loss_type,
                        use_temperature
                    ) * kd_loss_weight
                else:
                    kd_loss = torch.tensor(0.0, device=device)
                
                diversity_penalty_enabled = kd_config.get('diversity_penalty', False)
                diversity_weight = kd_config.get('diversity_weight', 0.1) * diversity_weight_scale
                entropy_loss = torch.tensor(0.0, device=device)
                
                if diversity_penalty_enabled:
                    student_probs = torch.softmax(student_output, dim=-1)
                    entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-7), dim=-1)
                    avg_entropy = torch.mean(entropy)
                    max_entropy = torch.log(torch.tensor(student_output.size(-1), dtype=torch.float32, device=device))
                    entropy_loss = 1.0 - (avg_entropy / max_entropy)
            
                    max_prob = torch.max(student_probs, dim=-1)[0]
                    avg_max_prob = torch.mean(max_prob)
                    avg_max_prob_float = float(avg_max_prob.item())
                    
                    class_probs = torch.mean(student_probs, dim=0)
                    max_class_prob = torch.max(class_probs).item()
                    max_class_idx = torch.argmax(class_probs).item()
                    
                    mode_collapse_threshold = 0.5
                    if max_class_prob > mode_collapse_threshold:
                        print(f"[Cloud Server]  模式崩塌檢測：類別 {max_class_idx} 預測比例={max_class_prob:.4f} > {mode_collapse_threshold}，增強多樣性懲罰")
                        diversity_weight = diversity_weight * 2.0
                        if max_class_idx == 1 and not mixup_enabled and use_mixup and 1 in target_class_candidates:
                            print(f"[Cloud Server]  緊急修復：類別 1 模式崩塌，強制觸發 Mixup")
            
                    if avg_max_prob_float > 0.3:
                        print(f"[Cloud Server]  破局配置 3.4：檢測到預測分佈過於集中（平均最大機率={avg_max_prob_float:.4f} > 0.3），應用多樣性懲罰（熵={avg_entropy.item():.4f}/{max_entropy.item():.4f}）")

                disable_prototype = os.environ.get("DISABLE_PROTOTYPE", "0").strip() == "1"
                disable_mixup_env = os.environ.get("DISABLE_MIXUP", "0").strip() == "1"
                prototype_alignment_loss = torch.tensor(0.0, device=device)
                prototype_alignment_weight = float(os.environ.get("PROTOTYPE_ALIGNMENT_WEIGHT", "0.5"))
                
                prototype_target_class = None
                if not disable_prototype and global_prototypes and len(global_prototypes) > 0:
                    if mixup_enabled and target_class is not None:
                        prototype_target_class = target_class
                    elif disable_mixup_env:
                        candidate_f1_scores = {}
                        if hasattr(app.state, 'last_f1_per_class') and app.state.last_f1_per_class:
                            for k in global_prototypes.keys():
                                candidate_f1_scores[k] = app.state.last_f1_per_class.get(k, 0.0)
                        else:
                            for k in global_prototypes.keys():
                                candidate_f1_scores[k] = 0.0
                        if candidate_f1_scores:
                            min_f1 = min(candidate_f1_scores.values())
                            low_f1_candidates = [c for c, f1 in candidate_f1_scores.items() if f1 == min_f1]
                            prototype_target_class = int(np.random.choice(low_f1_candidates))
                            if epoch == 0 and step == 0:
                                print(f"[Cloud Server] DISABLE_MIXUP=1：選擇 F1 最低的類別進行 Prototype Alignment（F1={min_f1:.4f}，候選：{low_f1_candidates}，選擇：{prototype_target_class}）")
                
                if not disable_prototype and prototype_target_class is not None and prototype_target_class in global_prototypes:
                    target_prototype = global_prototypes[prototype_target_class].to(device)
                    if target_prototype.dim() == 1:
                        target_prototype = target_prototype.unsqueeze(0)
                
                    if target_prototype.size(0) == 1:
                        target_prototype = target_prototype.repeat(training_embedding.size(0), 1)
                    elif target_prototype.size(0) != training_embedding.size(0):
                        target_prototype = target_prototype[:training_embedding.size(0)]
                    
                    prototype_alignment_loss = torch.mean((training_embedding - target_prototype) ** 2)

                    if epoch == 0:
                        print(f"[Cloud Server]  方案 1：原型對齊損失（目標類別={prototype_target_class}，損失={prototype_alignment_loss.item():.4f}，權重={prototype_alignment_weight:.2f}）")
                
                if kd_enabled:
                    kd_loss_value = kd_loss.item()

                    loss_threshold = 15.0
                    if kd_loss_value > loss_threshold:
                        scale_factor = loss_threshold / kd_loss_value
                        kd_loss = kd_loss * scale_factor
                        print(f"[Cloud Server] 警告：KD Loss 過大 ({kd_loss_value:.4f} > {loss_threshold:.4f})，已線性縮放至 {kd_loss.item():.4f}（縮放因子={scale_factor:.4f}，保持梯度）")
                    else:
                        print(f"[Cloud Server] KD Loss 正常: {kd_loss_value:.4f}")
                    
                    if kd_loss_value < 1e-6:
                        print(f"[Cloud Server] 警告：KD Loss 過小 ({kd_loss_value:.6f})，可能表示模型已收斂或需要調整參數")

                        if teacher_output_batch is not None:
                            with torch.no_grad():
                                output_diff = torch.mean(torch.abs(student_output - teacher_output_batch))
                                print(f"[Cloud Server] 輸出差異: {output_diff.item():.6f}")
                else:
                    print(f"[Cloud Server] KD 已禁用，跳過 KD Loss 檢查")
                
                if kd_enabled:
                    total_loss = kd_loss + diversity_weight * entropy_loss + prototype_alignment_loss * (prototype_alignment_weight * proto_loss_scale)
                else:
                    total_loss = diversity_weight * entropy_loss + prototype_alignment_loss * (prototype_alignment_weight * proto_loss_scale)
                if diversity_penalty_enabled or prototype_alignment_loss.item() > 0:
                    entropy_loss_value = entropy_loss.item()
                    proto_loss_value = prototype_alignment_loss.item()
                    proto_loss_info = f", Prototype Loss={proto_loss_value:.4f} (weight={prototype_alignment_weight:.2f})" if proto_loss_value > 0 else ""
                    total_loss_value = total_loss.item()
                    kd_loss_info = f"KD Loss={kd_loss.item():.4f}, " if kd_enabled else ""
                    print(f"[Cloud Server] 破局配置 3.2：總損失: {kd_loss_info}Entropy Loss={entropy_loss_value:.4f} (weight={diversity_weight:.2f}){proto_loss_info}, Total={total_loss_value:.4f}")
                
                optimizer.zero_grad()
                total_loss.backward()
                
                total_grad_norm = 0.0
                for param in server_model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                max_grad_norm = 500.0
                if total_grad_norm > max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_grad_norm)
                    print(f"[Cloud Server] 梯度範數過大 ({total_grad_norm:.4f} > {max_grad_norm:.4f})，已裁剪")
                    total_grad_norm = max_grad_norm
                
                if total_grad_norm < 1e-6:
                    print(f"[Cloud Server] 警告：梯度範數過小 ({total_grad_norm:.6f})，可能無法有效更新模型")
                
                optimizer.step()
                
                kd_loss_info = f"KD Loss={kd_loss.item():.4f}, " if kd_enabled else ""
                print(f"[Cloud Server] Epoch {epoch+1}/{server_epochs}, Step {step+1}/{effective_steps}: {kd_loss_info}Grad Norm={total_grad_norm:.4f}")

        disable_prototype = os.environ.get("DISABLE_PROTOTYPE", "0").strip() == "1"
        if not disable_prototype and global_prototypes and len(global_prototypes) > 0:
            try:
                anchor_protos = []
                anchor_labels = []
                num_classes = student_output.size(-1)
                for cls in range(num_classes):
                    if cls in global_prototypes and global_prototypes[cls] is not None:
                        p = global_prototypes[cls].to(device)
                        if p.dim() == 1:
                            p = p.unsqueeze(0)
                        anchor_protos.append(p)
                        anchor_labels.append(cls)
                if anchor_protos and anchor_labels:
                    anchor_batch = torch.cat(anchor_protos, dim=0)
                    anchor_targets = torch.tensor(anchor_labels, dtype=torch.long, device=device)
                    anchor_repeat = 3
                    anchor_batch = anchor_batch.repeat(anchor_repeat, 1)
                    anchor_targets = anchor_targets.repeat(anchor_repeat)
                    server_model.train()

                    for mod in server_model.modules():
                        if isinstance(mod, torch.nn.BatchNorm1d):
                            mod.eval()
                    anchor_logits = server_model(anchor_batch)
                    class_weight_tensor = None
                    try:
                        if hasattr(app.state, "last_f1_per_class") and app.state.last_f1_per_class:
                            f1_per_class_dict = app.state.last_f1_per_class
                            num_classes = anchor_logits.size(-1)
                            import numpy as _np
                            base_cls_weights = _np.ones(num_classes, dtype=_np.float32)

                            gamma = 1.0
                            for cls_idx in range(num_classes):
                                f1_val = float(f1_per_class_dict.get(cls_idx, 0.0))
                                if f1_val >= 0.3:
                                    base_cls_weights[cls_idx] = 1.0 + gamma * f1_val
                            class_weight_tensor = torch.tensor(base_cls_weights, dtype=torch.float32, device=device)
                    except Exception:
                        class_weight_tensor = None

                    ce_anchor = torch.nn.functional.cross_entropy(
                        anchor_logits,
                        anchor_targets,
                        weight=class_weight_tensor,
                    )

                    anchor_weight = 2.0
                    anchor_loss = ce_anchor * anchor_weight
                    optimizer.zero_grad()
                    anchor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_grad_norm)
                    optimizer.step()

            except Exception as e:
                print(f"[Cloud Server] Prototype Anchor Step 失敗: {e}")
        
    except Exception as e:
        print(f"[Cloud Server] 知識蒸餾訓練失敗: {e}")
        import traceback
        traceback.print_exc()

@app.post("/upload_aggregated_weights")
async def upload_aggregated_weights(
    aggregator_id: int = Form(...),
    round_id: int = Form(...),
    model_version: int = Form(...),
    participating_clients: str = Form(...),
    weights: UploadFile = File(...)
):
    global global_weights, aggregation_count, last_curve_stats
    global early_stop_triggered, early_stop_reason, aggregator_weights
    
    print(f"[Cloud Server] 收到聚合器 {aggregator_id} 的上傳請求 (輪次: {round_id}, 模型版本: {model_version})")
    request_start_time = time.time()
    
    try:
        if early_stop_triggered:
            print(f"[Cloud Server]  早停已觸發，拒絕聚合器 {aggregator_id} 的上傳")
            return JSONResponse(content={"status":"stopped","message":f"early_stop: {early_stop_reason}"}, status_code=200)
        
        print(f"[Cloud Server] 開始讀取上傳文件...")
        weights_bytes = await weights.read()
        print(f"[Cloud Server] 文件讀取完成，大小: {len(weights_bytes) / (1024 * 1024):.2f} MB")
    
        print(f"[Cloud Server] 開始反序列化數據...")
        upload_data = pickle.loads(weights_bytes)
        print(f"[Cloud Server] 數據反序列化完成")
        
        try:
            participating_clients_list = json.loads(participating_clients)
        except json.JSONDecodeError:
            participating_clients_list = []
        
        print(f"[Cloud Server] 聚合器 {aggregator_id} 上傳詳情:")
        print(f"  - 輪次: {round_id}")
        print(f"  - 模型版本: {model_version}")
        print(f"  - 參與客戶端: {len(participating_clients_list)} 個")
        print(f"  - 請求處理耗時: {time.time() - request_start_time:.2f}s")
        
        expected_round = 1
        if hasattr(app.state, "last_aggregation_round") and app.state.last_aggregation_round:
            try:
                expected_round = int(app.state.last_aggregation_round) + 1
            except Exception:
                expected_round = 1
        
        agg_cfg = getattr(config, 'AGGREGATION_CONFIG', {}) or {}
        late_lag = agg_cfg.get('late_upload_max_round_lag', 1)
        future_tol = agg_cfg.get('future_round_tolerance', 2)
        
        catch_up_threshold = 5
        if round_id > expected_round + catch_up_threshold:
            last_agg_round = int(app.state.last_aggregation_round) if hasattr(app.state, "last_aggregation_round") and app.state.last_aggregation_round else 0
            if round_id > last_agg_round:
                print(f"[Cloud Server] 檢測到協調器追趕場景：上傳輪次 {round_id} 遠超期望輪次 {expected_round}，動態放寬限制（last_agg_round={last_agg_round}）")
                allowed_min = max(1, last_agg_round + 1)
                allowed_max = round_id + future_tol
            else:
                allowed_min = max(1, expected_round - late_lag)
                allowed_max = expected_round + future_tol
        else:
            allowed_min = max(1, expected_round - late_lag)
            allowed_max = expected_round + future_tol
        
        if round_id < allowed_min or round_id > allowed_max:
            print(f"[Cloud Server]  略過聚合器 {aggregator_id} 上傳：輪次 {round_id} 不在允許範圍 [{allowed_min}, {allowed_max}] (期望輪次={expected_round})")
            return JSONResponse(content={"status": "out_of_sync", "message": f"round {round_id} not expected (allowed {allowed_min}-{allowed_max})"}, status_code=200)
        
        aggregated_weights = upload_data.get('aggregated_weights', {})
        aggregation_stats = upload_data.get('aggregation_stats', {})
        
        if not aggregated_weights:
            print(f"[Cloud Server] 錯誤: 聚合權重為空")
            raise HTTPException(status_code=400, detail="Empty aggregated weights")
        
        try:
            all_weights_flat = np.concatenate([v.flatten() for v in aggregated_weights.values()])
        except Exception as e:
            print(f"[Cloud Server] 錯誤: 聚合權重數據無效: {e}")
            raise HTTPException(status_code=400, detail="Invalid aggregated weights data")
        
        if all_weights_flat.size == 0:
            print(f"[Cloud Server] 警告: 聚合權重數據為空")
            raise HTTPException(status_code=400, detail="Empty aggregated weights data")
        
        weight_mean = all_weights_flat.mean()
        weight_std = all_weights_flat.std()
        
        print(f"[Cloud Server] 聚合權重統計:")
        print(f"  - 均值: {weight_mean:.6f}")
        print(f"  - 標準差: {weight_std:.6f}")
        print(f"  - 權重層數: {len(aggregated_weights)}")
        
        mean_abs_max = 5.0
        std_max = 10.0
        if np.isnan(weight_mean) or np.isinf(weight_mean) or np.isnan(weight_std) or np.isinf(weight_std):
            print(f"[Cloud Server] 錯誤: 聚合權重包含 NaN/Inf")
            raise HTTPException(status_code=400, detail="Invalid aggregated weights: NaN/Inf detected")
        if abs(weight_mean) > mean_abs_max or weight_std > std_max:
            print(f"[Cloud Server] 拒收聚合權重：均值/標準差異常 (mean={weight_mean:.3f}, std={weight_std:.3f})")
            raise HTTPException(status_code=400, detail=f"Invalid aggregated weights: mean/std abnormal (>{mean_abs_max},{std_max})")
        
        import asyncio as py_asyncio
        global_performance_score = aggregation_stats.get('performance_score')
        global_accuracy = aggregation_stats.get('accuracy')
        global_f1_score = aggregation_stats.get('f1_score')
        aggregation_stats['performance_source'] = aggregation_stats.get('performance_source', 'aggregator_reported')

        try:
            eval_timeout = 10.0
            max_samples = 100
            loop = py_asyncio.get_running_loop()
            eval_future = loop.run_in_executor(
                None,
                evaluate_global_model_on_csv,
                -(round_id * 1000 + aggregator_id),
                aggregated_weights,
                max_samples
            )
            eval_result = await py_asyncio.wait_for(eval_future, timeout=eval_timeout)
            if eval_result:
                ga = eval_result.get('accuracy', global_accuracy)
                gf = eval_result.get('f1_score', global_f1_score)
                if ga is not None and gf is not None:
                    global_accuracy = ga
                    global_f1_score = gf
                    global_performance_score = (ga + gf) / 2
                    aggregation_stats['performance_source'] = 'global_eval'

                    max_pred_ratio = eval_result.get('max_pred_ratio')
                    max_pred_class = eval_result.get('max_pred_class')
                    if max_pred_ratio is not None:
                        aggregation_stats['max_pred_ratio'] = float(max_pred_ratio)
                    if max_pred_class is not None:
                        try:
                            aggregation_stats['max_pred_class'] = int(max_pred_class)
                        except Exception:
                            aggregation_stats['max_pred_class'] = -1
                    print(
                        f"[Cloud Server] 上傳評估成功 (max_samples={max_samples}, timeout={eval_timeout}s): "
                        f"acc={ga:.4f}, f1={gf:.4f}, perf={global_performance_score:.4f}, "
                        f"max_pred_ratio={aggregation_stats.get('max_pred_ratio', 'N/A')}"
                    )
        except py_asyncio.TimeoutError:
            print(f"[Cloud Server] ⏰ 上傳評估超時（{eval_timeout}s），保持聚合器上報分數")
        except Exception as e:
            print(f"[Cloud Server] 上傳評估失敗，保持聚合器上報分數: {e}")
        
        with lock:
            aggregator_weights[aggregator_id].append({
                'aggregated_weights': aggregated_weights,
                'round_id': round_id,
                'model_version': model_version,
                'participating_clients': participating_clients_list,
                'aggregation_stats': aggregation_stats,
                'data_size': upload_data.get('data_size', 1000),
                'timestamp': time.time(),
                'weight_stats': {
                    'mean': weight_mean,
                    'std': weight_std,
                    'num_layers': len(aggregated_weights)
                }
            })

            if len(aggregator_weights[aggregator_id]) > 10:
                aggregator_weights[aggregator_id] = aggregator_weights[aggregator_id][-10:]

            prune_before = time.time() - 600
            aggregator_weights[aggregator_id] = [w for w in aggregator_weights[aggregator_id] if w.get('timestamp', 0) >= prune_before]
            
            app.state.aggregator_weights = {agg_id: weights for agg_id, weights in aggregator_weights.items() if len(weights) > 0}
        
        current_round = round_id
        
        print(f"[Cloud Server] 輪次檢查調試:")
        print(f"  - 當前輪次: {current_round}")
        print(f"  - app.state存在: {hasattr(app, 'state')}")
        print(f"  - last_aggregation_round存在: {hasattr(app.state, 'last_aggregation_round') if hasattr(app, 'state') else False}")
        print(f"  - 上次聚合輪次: {getattr(app.state, 'last_aggregation_round', 'None') if hasattr(app, 'state') else 'None'}")
        
        cloud_threshold = CLOUD_THRESHOLD
        
        try:
            total_aggs = max(1, len(registered_aggregators))
        except Exception:
            total_aggs = 1
        
        cfg = getattr(config, 'AGGREGATION_CONFIG', {}) or {}
        cfg_quorum = cfg.get('aggregator_quorum') or cfg.get('min_aggregators_for_global')
        
        available_aggs = len([agg_id for agg_id, agg_info in registered_aggregators.items() if agg_info.get('status') == 'healthy'])
        if available_aggs == 0:
            available_aggs = total_aggs
        
        if cfg_quorum is None:
            quorum_required = max(1, int(math.ceil(0.6 * available_aggs)))
        else:
            quorum_required = max(1, int(cfg_quorum))
        
        quorum_required = min(quorum_required, available_aggs)
        
        if (hasattr(app.state, 'last_aggregation_round') and 
            app.state.last_aggregation_round is not None):
            if current_round <= app.state.last_aggregation_round:
                print(f"[Cloud Server]  輪次 {current_round} 已聚合過（上次聚合輪次: {app.state.last_aggregation_round}），跳過重複聚合")
                return JSONResponse(content={"status":"already_aggregated","message":f"round {current_round} already aggregated"})

            elif current_round > app.state.last_aggregation_round + 1:
                skipped_rounds = current_round - app.state.last_aggregation_round - 1
                print(f"[Cloud Server] 警告：輪次不連續！跳過了 {skipped_rounds} 個輪次（上次: {app.state.last_aggregation_round}, 當前: {current_round}）")
                print(f"[Cloud Server] 允許聚合輪次 {current_round}（即使跳過了 {skipped_rounds} 個輪次）")
                try:
                    log_training_event_cloud("round_gap_detected", {
                        'round_id': current_round,
                        'last_round': app.state.last_aggregation_round,
                        'skipped_rounds': skipped_rounds,
                        'detail': f"Gap detected: last={app.state.last_aggregation_round}, current={current_round}"
                    })
                except Exception:
                    pass
        
        current_aggregator_count = len([agg_id for agg_id, weights in aggregator_weights.items() if len(weights) > 0])
        if current_aggregator_count >= quorum_required:
            print(f"[Cloud Server]  輪次 {current_round} 已有足夠聚合器權重 ({current_aggregator_count}/{quorum_required})，開始全局聚合")

        elif current_aggregator_count < quorum_required:
            print(f"[Cloud Server]  輪次 {current_round} 等待更多聚合器權重 ({current_aggregator_count}/{quorum_required})")

        if not hasattr(app.state, 'aggregation_start_time'):
            app.state.aggregation_start_time = {}
        
        current_time = time.time()
        if current_round not in app.state.aggregation_start_time:
            app.state.aggregation_start_time[current_round] = current_time
        
        max_wait_time = cfg.get('max_wait_time', 180)
        if current_time - app.state.aggregation_start_time[current_round] > max_wait_time:
            print(f"[Cloud Server] ⏰ 輪次 {current_round} 聚合超時({max_wait_time}s)，強制進行聚合")
            quorum_required = 1
        
        print(f"[Cloud Server] 動態quorum調整: 配置={cfg_quorum}, 總數={total_aggs}, 可用={available_aggs}, 要求={quorum_required}")
        round_tolerance = 1
        current_round_reports = sum(1 for wlist in aggregator_weights.values() 
                                     if any(abs(w.get('round_id', 0) - current_round) <= round_tolerance for w in wlist))

        if current_round_reports >= quorum_required:
            try:
                dbi_weights_list = []
                for agg_id, weights_list in aggregator_weights.items():
                    round_candidates = [w for w in weights_list if abs(w.get('round_id', 0) - current_round) <= round_tolerance]
                    if round_candidates:
                        latest_weight = round_candidates[-1]
                        agg_weights = latest_weight.get('aggregated_weights', {})
                        if agg_weights:
                            dbi_weights_list.append({
                                'agg_id': agg_id,
                                'weights': agg_weights,
                                'performance_score': latest_weight.get('aggregation_stats', {}).get('performance_score', 0.5),
                                'data_size': latest_weight.get('data_size', 1000)
                            })
                
                if len(dbi_weights_list) >= 2:
                    dbi_suspicious_ids, dbi_action, dbi_soft_factor = _analyze_aggregator_weights_with_dbi(
                        dbi_weights_list, current_round=current_round
                    )
                    dbi_suspicious_ids = set(dbi_suspicious_ids or [])
                
                    if dbi_action == "hard" and dbi_suspicious_ids:
                        print(
                            f"[Cloud Server] ConfShield/DBI 硬剔除：標記 {len(dbi_suspicious_ids)} 個可疑聚合器 "
                            f"({dbi_suspicious_ids})，將從聚合中排除"
                        )

                        if not hasattr(app.state, 'dbi_excluded_aggregators'):
                            app.state.dbi_excluded_aggregators = {}
                        app.state.dbi_excluded_aggregators[current_round] = dbi_suspicious_ids
                    elif dbi_action == "soft" and dbi_suspicious_ids:
                        print(
                            f"[Cloud Server] ConfShield/DBI 軟降權：標記 {len(dbi_suspicious_ids)} 個可疑聚合器 "
                            f"({dbi_suspicious_ids})，將在聚合時應用 soft_factor={dbi_soft_factor:.3f}"
                        )

                        if not hasattr(app.state, 'dbi_soft_weights'):
                            app.state.dbi_soft_weights = {}
                        app.state.dbi_soft_weights[current_round] = {
                            'suspicious_ids': dbi_suspicious_ids,
                            'soft_factor': dbi_soft_factor
                        }
                    else:

                        print(f"[Cloud Server] ConfShield/DBI 監測模式：未發現需要處理的可疑聚合器")
                else:
                    print(f"[Cloud Server] ConfShield/DBI 跳過：聚合器數量不足 ({len(dbi_weights_list)} < 2)")
            except Exception as e:
                print(f"[Cloud Server] DBI 檢測過程發生錯誤: {e}")
                import traceback
                traceback.print_exc()
        
            q_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('quality_quorum', {})
            quality_enabled = bool(q_cfg.get('enabled', False))
            min_delta = float(q_cfg.get('min_delta', 0.0))
            min_pass = int(q_cfg.get('min_pass', quorum_required))
            timeout_seconds = int(q_cfg.get('timeout_seconds', 120))

            def quick_eval_delta_f1(agg_id: int) -> tuple[Optional[float], str]:
                try:
                    wlist = aggregator_weights.get(agg_id, [])
                    cands = [w for w in wlist if w.get('round_id') == current_round]
                    if not cands:
                        return (None, f"agg={agg_id}: 找不到當前輪次 {current_round} 的權重")
                    candidate = cands[-1]
                    cand_weights = candidate.get('aggregated_weights')
                    if cand_weights is None or len(cand_weights) == 0:
                        return (None, f"agg={agg_id}: 權重為空")
                    
                    base_eval = {'macro_f1': 0.0, 'f1_score': 0.0}
                    base_macro_f1 = 0.0
                    try:
                        base_round_id = -(current_round - 1)

                        max_samples = getattr(config, "GLOBAL_EVAL_MAX_SAMPLES", None)
                        base_eval = evaluate_global_model_on_csv(base_round_id, global_weights, max_samples)
                        if base_eval is None:
                            base_eval = {'macro_f1': 0.0, 'f1_score': 0.0}
                    except Exception as e:
                        print(f"[Cloud Server] 評估基準權重失敗 (round={current_round-1}): {e}")
                        base_eval = {'macro_f1': 0.0, 'f1_score': 0.0}
                    
                    base_macro_f1 = float(base_eval.get('f1_score', 0.0)) if 'f1_score' in base_eval else float(base_eval.get('macro_f1', 0.0))
                    
                    try:
                        quality_check_round_id = -current_round
                        max_samples = getattr(config, "GLOBAL_EVAL_MAX_SAMPLES", None)
                        tmp_eval = evaluate_global_model_on_csv(quality_check_round_id, cand_weights, max_samples)
                        if tmp_eval is None:
                            return (None, f"agg={agg_id}: 評估返回 None（可能缺少測試集或標籤編碼器）")
                    except Exception as e:
                        error_msg = f"agg={agg_id}: 評估異常 - {type(e).__name__}: {str(e)[:100]}"
                        print(f"[Cloud Server] {error_msg}")
                        return (None, error_msg)
                    
                    cand_macro_f1 = float(tmp_eval.get('f1_score', 0.0)) if 'f1_score' in tmp_eval else float(tmp_eval.get('macro_f1', 0.0))
                    if not isinstance(cand_macro_f1, (int, float)) or (isinstance(cand_macro_f1, float) and (cand_macro_f1 != cand_macro_f1 or cand_macro_f1 == float('inf'))):
                        return (-1e9, f"agg={agg_id}: 評估結果異常 (cand_macro_f1={cand_macro_f1})")
                    
                    delta = cand_macro_f1 - base_macro_f1
                    return (delta, "")
                except Exception as e:
                    error_msg = f"agg={agg_id}: 未預期異常 - {type(e).__name__}: {str(e)[:100]}"
                    print(f"[Cloud Server] {error_msg}")
                    return (None, error_msg)

            quality_pass_ids = []
            quality_errors = {}
            if quality_enabled:
                for aid in list(aggregator_weights.keys()):
                    delta, error_msg = quick_eval_delta_f1(aid)
                    if delta is None:
                        if error_msg:
                            quality_errors[aid] = error_msg
                            print(f"[Cloud Server]  品質檢查: agg={aid}, 跳過（原因: {error_msg[:60]})")
                        continue
                    if error_msg:
                        quality_errors[aid] = error_msg
                    print(f"[Cloud Server]  品質檢查: agg={aid}, ΔmacroF1={delta:.6f}" + (f" (錯誤: {error_msg[:50]})" if error_msg else ""))
                    if delta >= min_delta:
                        quality_pass_ids.append(aid)
                    else:
                        if aid not in quality_errors:
                            quality_errors[aid] = f"ΔmacroF1={delta:.6f} < min_delta {min_delta}"
            
                if len(quality_pass_ids) == 0 and current_round_reports > 0:
                    error_summary = "; ".join([f"agg{k}: {v[:50]}" for k, v in list(quality_errors.items())[:3]])
                    print(f"[Cloud Server] 品質門檻警告：所有聚合器品質檢查失敗，降級為使用所有聚合器進行聚合")
                    print(f"[Cloud Server] 錯誤摘要: {error_summary}")
                    log_event("quality_gate_degraded", f"round={current_round},reports={current_round_reports},errors={len(quality_errors)}")

                    quality_pass_ids = list(aggregator_weights.keys())
                    print(f"[Cloud Server] 降級處理：使用所有 {len(quality_pass_ids)} 個聚合器進行聚合")

                    if len(quality_pass_ids) < min_pass:
                        print(f"[Cloud Server] 降級後聚合器數量 ({len(quality_pass_ids)}) 仍不足 min_pass ({min_pass})，降低 min_pass 要求")
                        min_pass = len(quality_pass_ids)

            
                if len(quality_pass_ids) < min_pass:
                    try:
                        if getattr(app.state, 'quality_wait_round', None) != current_round:
                            app.state.quality_wait_round = current_round
                            app.state.quality_wait_started_at = time.time()
                    except Exception:
                        pass
                    elapsed = 0
                    try:
                        elapsed = time.time() - getattr(app.state, 'quality_wait_started_at', time.time())
                    except Exception:
                        pass
                    if elapsed < timeout_seconds:
                        print(f"[Cloud Server]  等待品質 quorum: 通過={len(quality_pass_ids)}/{min_pass}, elapsed={elapsed:.1f}s < {timeout_seconds}s")
                        log_event("waiting_for_quality", f"round={current_round},pass={len(quality_pass_ids)},need={min_pass},elapsed={elapsed:.1f}")
                        return JSONResponse(content={"status":"waiting","message":"waiting for quality quorum"})
                    else:

                        if len(quality_pass_ids) > 0:
                            print(f"[Cloud Server] 品質 quorum 超時 {timeout_seconds}s，啟用 fallback（使用 {len(quality_pass_ids)} 個通過的聚合器）")
                            log_event("quality_timeout_fallback", f"round={current_round},pass={len(quality_pass_ids)},need={min_pass}")
                        else:

                            print(f"[Cloud Server]  品質 quorum 超時且無通過聚合器，拒絕聚合")
                            log_event("quality_timeout_rejected", f"round={current_round},pass=0,need={min_pass}")
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "status": "rejected",
                                    "message": f"品質 quorum 超時且無通過聚合器（輪次 {current_round}）",
                                    "round": current_round,
                                    "quality_pass": 0,
                                    "total_reports": current_round_reports
                                }
                            )

            pass_count = len(quality_pass_ids) if quality_enabled else current_round_reports
            last_curve_stats = {
                'round': current_round,
                'effective_aggregators': current_round_reports,
                'quality_pass': pass_count,
                'quality_checked': current_round_reports
            }

            print(f"[Cloud Server]  quorum 達成，開始全局聚合 (輪次: {current_round}, 回報: {current_round_reports}/{total_aggs}, 閾值: {quorum_required}{', 品質達標='+str(len(quality_pass_ids)) if quality_enabled else ''})")
            

            original_aggregator_weights = aggregator_weights.copy() if quality_enabled and quality_pass_ids else None
            if quality_enabled and quality_pass_ids:

                filtered_weights = {aid: aggregator_weights[aid] for aid in quality_pass_ids if aid in aggregator_weights}
                if filtered_weights:
                    aggregator_weights = filtered_weights
                    print(f"[Cloud Server]  品質過濾：使用 {len(quality_pass_ids)} 個通過品質檢查的聚合器進行聚合")
                else:
                    print(f"[Cloud Server] 警告：品質過濾後無有效聚合器，使用原始聚合器")
            

            try:
                dbi_excluded = set()
                if hasattr(app.state, 'dbi_excluded_aggregators') and current_round in app.state.dbi_excluded_aggregators:
                    dbi_excluded = app.state.dbi_excluded_aggregators[current_round]
                    if dbi_excluded:

                        before_count = len(aggregator_weights)
                        aggregator_weights = {aid: w for aid, w in aggregator_weights.items() if aid not in dbi_excluded}
                        after_count = len(aggregator_weights)
                        if before_count > after_count:
                            print(
                                f"[Cloud Server] ConfShield/DBI 硬剔除應用：排除 {before_count - after_count} 個可疑聚合器 "
                                f"({dbi_excluded & set(original_aggregator_weights.keys() if original_aggregator_weights else {})})"
                            )
                
                dbi_soft_info = None
                if hasattr(app.state, 'dbi_soft_weights') and current_round in app.state.dbi_soft_weights:
                    dbi_soft_info = app.state.dbi_soft_weights[current_round]
                    if dbi_soft_info:
                        print(
                            f"[Cloud Server] ConfShield/DBI 軟降權將在聚合時應用："
                            f"{len(dbi_soft_info['suspicious_ids'])} 個可疑聚合器將乘以 {dbi_soft_info['soft_factor']:.3f}"
                        )
            except Exception as e:
                print(f"[Cloud Server] DBI 應用過程發生錯誤: {e}")
                import traceback
                traceback.print_exc()
            
            try:
                dual_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('dual_weighting', {})
                if bool(dual_cfg.get('enabled', False)):
                    try:
                        beta_data = float(dual_cfg.get('beta_data', 1.0))
                        beta_pred = float(dual_cfg.get('beta_pred', 1.0))
                        alpha_clip = float(dual_cfg.get('alpha_clip', 0.5))
                        dual_meta = collect_dual_meta_for_round(current_round)
                        dual_weights = compute_dual_weights(dual_meta, beta_data, beta_pred, alpha_clip)

                        prev_global_weights = global_weights if global_weights is not None else {}
                        global_weights = perform_global_aggregation(dual_weights)

                    except Exception:
                        global_weights = perform_global_aggregation()
                else:
                    prev_global_weights = global_weights if global_weights is not None else {}
                    new_global_weights = perform_federated_averaging()

                    weight_update_rejected = False

                    if not new_global_weights or len(new_global_weights) == 0:
                        print(f"[Cloud Server] 警告：perform_federated_averaging 返回空權重，使用上一輪權重 (round={current_round})")
                        if prev_global_weights and len(prev_global_weights) > 0:
                            global_weights = {k: _coerce_tensor(v).clone() for k, v in prev_global_weights.items()}
                        else:
                            global_weights = {}
                    else:
                        if prev_global_weights and len(prev_global_weights) > 0:
                            try:
                                update_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('weight_update_condition', {})
                                use_numerical = update_cfg.get('use_numerical_comparison', True)
                                l2_threshold = update_cfg.get('l2_distance_threshold', 1e-4)
                                key_layers = update_cfg.get('key_layers', ['output_layer.weight', 'layers.0.weight', 'input_reshape.weight'])
                                
                                if not key_layers:
                                    key_layers = []
                                    for key in ['output_layer.weight', 'layers.0.weight', 'input_reshape.weight']:
                                        if key in prev_global_weights and key in new_global_weights:
                                            key_layers.append(key)

                                    if not key_layers:
                                        all_keys = list(prev_global_weights.keys())
                                        key_layers = [k for k in all_keys[:3] if k in new_global_weights]
                                
                                if use_numerical:

                                    is_identical, identical_count, total_compared = _check_weights_identical_numerical(
                                        prev_global_weights, new_global_weights, key_layers, l2_threshold
                                    )
                                    
                                    if total_compared > 0:
                                        identical_ratio = identical_count / total_compared
                                        if is_identical:
                                            print(f"[Cloud Server] 警告：新聚合權重與上一輪相同 (round={current_round}, {identical_count}/{total_compared} 層相同, L2閾值={l2_threshold:.2e})")
                                        elif identical_ratio >= 0.8:
                                            print(f"[Cloud Server] 警告：新聚合權重與上一輪高度相似 (round={current_round}, {identical_count}/{total_compared} 層相同, 比例={identical_ratio:.2%})")
                                        else:
                                            print(f"[Cloud Server] 權重已更新 (round={current_round}, {identical_count}/{total_compared} 層相同, 比例={identical_ratio:.2%})")
                                else:
                                    import hashlib
                                    _torch_local = globals().get('torch')
                                    if _torch_local is None:
                                        print(f"[Cloud Server] 權重哈希比較跳過: torch 未安裝")
                                    else:
                                        identical_layers = 0
                                        total_layers_compared = 0
                                        layer_hashes = {}
                                        
                                        for layer_name in key_layers:
                                            prev_tensor = _coerce_tensor(prev_global_weights[layer_name])
                                            new_tensor = _coerce_tensor(new_global_weights[layer_name])
                                            
                                            if prev_tensor.dtype not in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                                                continue
                                            if new_tensor.dtype not in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                                                continue
                                            
                                            if prev_tensor.shape != new_tensor.shape:
                                                continue
                                            
                                            prev_hash = hashlib.md5(prev_tensor.cpu().numpy().tobytes()).hexdigest()[:8]
                                            new_hash = hashlib.md5(new_tensor.cpu().numpy().tobytes()).hexdigest()[:8]
                                            layer_hashes[layer_name] = (prev_hash, new_hash)
                                            total_layers_compared += 1
                                            
                                            if prev_hash == new_hash:
                                                identical_layers += 1
                                        
                                        if total_layers_compared > 0:
                                            identical_ratio = identical_layers / total_layers_compared
                                            if identical_ratio >= 0.8:
                                                print(f"[Cloud Server] 警告：新聚合權重與上一輪高度相似 (round={current_round}, {identical_layers}/{total_layers_compared} 層相同, 比例={identical_ratio:.2%})")
                                                for layer_name, (prev_h, new_h) in layer_hashes.items():
                                                    if prev_h == new_h:
                                                        print(f"  - {layer_name}: hash={prev_h} (相同)")
                                                    else:
                                                        print(f"  - {layer_name}: prev={prev_h}, new={new_h} (不同)")
                                            else:
                                                print(f"[Cloud Server] 權重已更新 (round={current_round}, {identical_layers}/{total_layers_compared} 層相同, 比例={identical_ratio:.2%})")
                            except Exception as hash_e:
                                print(f"[Cloud Server] 權重比較失敗: {hash_e}")
                                import traceback
                                print(f"[Cloud Server] 詳細錯誤: {traceback.format_exc()}")

                        updated_weights = _apply_server_momentum(prev_global_weights, new_global_weights)
                    
                        weight_update_rejected = False
                        
                        if prev_global_weights and len(prev_global_weights) > 0 and updated_weights and len(updated_weights) > 0:
                            try:

                                update_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('weight_update_condition', {})
                                use_numerical = update_cfg.get('use_numerical_comparison', True)
                                l2_threshold = update_cfg.get('l2_distance_threshold', 1e-4)
                                key_layers = update_cfg.get('key_layers', ['output_layer.weight', 'layers.0.weight', 'input_reshape.weight'])
                                
                                _torch_local = globals().get('torch')
                                if _torch_local is None:
                                    print(f"[Cloud Server] 警告：torch 未安裝，跳過權重更新驗證，使用動量更新結果")
                                    global_weights = updated_weights
                                else:
                                    if use_numerical:
                                        is_identical_updated, updated_identical_count, updated_total_compared = _check_weights_identical_numerical(
                                            prev_global_weights, updated_weights, key_layers, l2_threshold
                                        )
                                        
                                        if is_identical_updated and updated_total_compared > 0:
                                            print(f"[Cloud Server] 警告：動量更新後權重未變化，檢查新聚合權重是否與上一輪不同")
                                            is_identical_new, new_vs_prev_count, new_total_compared = _check_weights_identical_numerical(
                                                prev_global_weights, new_global_weights, key_layers, l2_threshold
                                            )
                                            
                                            if is_identical_new and new_total_compared > 0:
                                                print(f"[Cloud Server] 錯誤：新聚合權重與上一輪完全相同，拒絕更新 (round={current_round}, L2閾值={l2_threshold:.2e})")
                                                global_weights = {k: _coerce_tensor(v).clone() for k, v in prev_global_weights.items()}
                                                weight_update_rejected = True
                                                print(f"[Cloud Server] 保持上一輪權重，跳過本輪聚合")
                                            else:
                                                print(f"[Cloud Server] 新聚合權重與上一輪不同 ({new_total_compared - new_vs_prev_count}/{new_total_compared} 個關鍵層不同)，使用新聚合權重")
                                                global_weights = {k: _coerce_tensor(v).clone() for k, v in new_global_weights.items()}
                                        else:
                                            global_weights = updated_weights
                                            if updated_total_compared > 0:
                                                print(f"[Cloud Server] 權重已更新: {updated_total_compared - updated_identical_count}/{updated_total_compared} 個關鍵層有變化")
                                    else:
                                        import hashlib
                                        updated_count = 0
                                        for key in key_layers:
                                            if key in prev_global_weights and key in updated_weights:
                                                prev_t = _coerce_tensor(prev_global_weights[key])
                                                new_t = _coerce_tensor(updated_weights[key])
                                                if prev_t.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16) and new_t.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                                                    if prev_t.shape == new_t.shape:
                                                        prev_hash = hashlib.md5(prev_t.cpu().numpy().tobytes()).hexdigest()[:8]
                                                        new_hash = hashlib.md5(new_t.cpu().numpy().tobytes()).hexdigest()[:8]
                                                        if prev_hash != new_hash:
                                                            updated_count += 1
                                        
                                        if updated_count == 0 and len(key_layers) > 0:
                                            print(f"[Cloud Server] 警告：動量更新後權重未變化，檢查新聚合權重是否與上一輪不同")
                                            new_vs_prev_count = 0
                                            for key in key_layers:
                                                if key in prev_global_weights and key in new_global_weights:
                                                    prev_t = _coerce_tensor(prev_global_weights[key])
                                                    new_t = _coerce_tensor(new_global_weights[key])
                                                    if prev_t.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16) and new_t.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                                                        if prev_t.shape == new_t.shape:
                                                            prev_hash = hashlib.md5(prev_t.cpu().numpy().tobytes()).hexdigest()[:8]
                                                            new_hash = hashlib.md5(new_t.cpu().numpy().tobytes()).hexdigest()[:8]
                                                            if prev_hash != new_hash:
                                                                new_vs_prev_count += 1
                                            
                                            if new_vs_prev_count == 0:
                                                print(f"[Cloud Server] 錯誤：新聚合權重與上一輪完全相同，拒絕更新 (round={current_round})")
                                                global_weights = {k: _coerce_tensor(v).clone() for k, v in prev_global_weights.items()}
                                                weight_update_rejected = True
                                                print(f"[Cloud Server] 保持上一輪權重，跳過本輪聚合")
                                            else:
                                                print(f"[Cloud Server] 新聚合權重與上一輪不同 ({new_vs_prev_count}/{len(key_layers)} 個關鍵層不同)，使用新聚合權重")
                                                global_weights = {k: _coerce_tensor(v).clone() for k, v in new_global_weights.items()}
                                        else:
                                            global_weights = updated_weights
                                            print(f"[Cloud Server] 權重已更新: {updated_count}/{len(key_layers)} 個關鍵層有變化")
                            except Exception as hash_e:
                                print(f"[Cloud Server] 權重更新驗證失敗: {hash_e}，使用動量更新結果")
                                import traceback
                                print(f"[Cloud Server] 詳細錯誤: {traceback.format_exc()}")
                                global_weights = updated_weights
                        else:
                            global_weights = {k: _coerce_tensor(v).clone() for k, v in new_global_weights.items()}
                            print(f"[Cloud Server] 首次聚合，直接使用新權重")
                
                if original_aggregator_weights is not None:
                    aggregator_weights = original_aggregator_weights
                
                if global_weights is None or len(global_weights) == 0:
                    print(f"[Cloud Server] 警告：全局聚合後 global_weights 為空，嘗試使用 perform_global_aggregation")
                    try:
                        global_weights = perform_global_aggregation()
                    except Exception as e:
                        print(f"[Cloud Server] perform_global_aggregation 也失敗: {e}")
                        global_weights = {}
                
                if global_weights:
                    try:
                        import torch
                        weight_valid = True
                        for layer_name, layer_weights in list(global_weights.items())[:3]:
                            if isinstance(layer_weights, torch.Tensor):
                                if torch.isnan(layer_weights).any() or torch.isinf(layer_weights).any():
                                    print(f"[Cloud Server] 警告：聚合後權重層 {layer_name} 包含 NaN 或 Inf")
                                    weight_valid = False
                            elif isinstance(layer_weights, np.ndarray):
                                if np.isnan(layer_weights).any() or np.isinf(layer_weights).any():
                                    print(f"[Cloud Server] 警告：聚合後權重層 {layer_name} 包含 NaN 或 Inf")
                                    weight_valid = False
                        if not weight_valid:
                            print(f"[Cloud Server] 警告：聚合後的權重包含無效值，但繼續使用")
                    except Exception as e:
                        print(f"[Cloud Server] 權重驗證失敗: {e}")
                
                if weight_update_rejected:
                    print(f"[Cloud Server]  權重更新被拒絕，但仍需廣播並更新 ACK (round={current_round})")

                    try:
                        import asyncio
                        try:
                            loop = asyncio.get_running_loop()
                            asyncio.create_task(_immediate_broadcast_global_weights(current_round))
                            print(f"[Cloud Server]  已觸發立即廣播全局權重（輪次: {current_round}，即使權重未更新）")
                        except RuntimeError:
                            import threading
                            def run_broadcast():
                                try:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    loop.run_until_complete(_immediate_broadcast_global_weights(current_round))
                                    loop.close()
                                except Exception as e:
                                    print(f"[Cloud Server] 線程中立即廣播失敗: {e}")
                            thread = threading.Thread(target=run_broadcast, daemon=True)
                            thread.start()
                            print(f"[Cloud Server]  已在線程中觸發立即廣播全局權重（輪次: {current_round}）")
                    except Exception as e:
                        print(f"[Cloud Server] 廣播失敗: {e}")
                    
                    try:
                        for agg_id, weights_list in aggregator_weights.items():
                            before_count = len(weights_list)
                            aggregator_weights[agg_id] = [w for w in weights_list if w.get('round_id') != current_round]
                            after_count = len(aggregator_weights[agg_id])
                            if before_count > after_count:
                                print(f"[Cloud Server] 🧹 已清理聚合器 {agg_id} 的輪次 {current_round} 權重緩衝區 ({before_count} -> {after_count})")
                    except Exception as e:
                        print(f"[Cloud Server] 清理聚合器權重緩衝區失敗: {e}")
                    
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "rejected",
                            "message": f"權重與上一輪完全相同，拒絕更新 (round={current_round})，但已廣播",
                            "round": current_round,
                            "reason": "weights_identical"
                        }
                    )
                
                aggregation_count += 1
                
                app.state.last_aggregation_round = current_round
                
                if global_weights and len(global_weights) > 0:
                    norm_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('weight_norm_regularization', {})
                    if norm_cfg.get('enabled', True):
                        max_norm = float(norm_cfg.get('max_global_l2_norm', 150.0))
                        hard_limit = float(norm_cfg.get('hard_limit', 200.0))
                        scaling_factor = float(norm_cfg.get('scaling_factor', 0.90))
                        warn_threshold = float(norm_cfg.get('warn_threshold', 120.0))
                        strict_enforcement = bool(norm_cfg.get('strict_enforcement', True))
                        
                        global last_peak_round
                        is_peak_protection_active = False
                        if PEAK_PROTECTION_ENABLED and last_peak_round is not None:
                            rounds_since_peak = current_round - last_peak_round
                            if rounds_since_peak <= PEAK_PROTECTION_ROUNDS:
                                is_peak_protection_active = True
                                max_norm = max_norm * 0.7
                                hard_limit = hard_limit * 0.7
                                warn_threshold = warn_threshold * 0.7
                                scaling_factor = scaling_factor * 0.90
                                print(f"[Cloud Server] 峰值保護期間：使用更嚴格的正則化（max_norm={max_norm:.2f}, hard_limit={hard_limit:.2f}, warn_threshold={warn_threshold:.2f}, scaling_factor={scaling_factor:.3f}）")
                        
                        current_norm = _compute_global_l2_norm(global_weights)
                        
                        if current_norm > hard_limit:
                            print(f"[Cloud Server]  權重範數超過硬性上限 ({current_norm:.4f} > {hard_limit:.4f})，強制裁剪{'（峰值保護模式）' if is_peak_protection_active else ''}")
                            global_weights = _apply_weight_norm_regularization(global_weights, max_norm, scaling_factor, hard_limit=hard_limit, strict_enforcement=strict_enforcement)
                            new_norm = _compute_global_l2_norm(global_weights)
                            if new_norm > hard_limit:
                                print(f"[Cloud Server] 警告：正則化後仍超過硬性上限 ({new_norm:.4f} > {hard_limit:.4f})，進行二次強制裁剪")

                                scale = hard_limit / new_norm
                                _torch_local = globals().get('torch')
                                if _torch_local:
                                    for layer_name in global_weights:
                                        w = _coerce_tensor(global_weights[layer_name])
                                        if isinstance(w, _torch_local.Tensor) and w.dtype in (_torch_local.float32, _torch_local.float64, _torch_local.float16):
                                            global_weights[layer_name] = w * scale
                                new_norm = _compute_global_l2_norm(global_weights)
                            print(f"[Cloud Server] 強制裁剪後權重範數: {new_norm:.4f} (目標≤{hard_limit:.4f})")

                        elif current_norm > warn_threshold:
                            print(f"[Cloud Server] 警告：全局權重 L2 範數過大 ({current_norm:.4f} > {warn_threshold:.4f}){'（峰值保護模式）' if is_peak_protection_active else ''}")
                        
                        if current_norm > max_norm:
                            print(f"[Cloud Server] 權重範數超過上限 ({current_norm:.4f} > {max_norm:.4f})，應用正則化{'（峰值保護模式）' if is_peak_protection_active else ''}")
                            global_weights = _apply_weight_norm_regularization(global_weights, max_norm, scaling_factor, hard_limit=hard_limit, strict_enforcement=strict_enforcement)
                            new_norm = _compute_global_l2_norm(global_weights)
                            print(f"[Cloud Server] 正則化後權重範數: {new_norm:.4f} (目標≤{max_norm:.4f})")
                        elif strict_enforcement and current_norm > warn_threshold:
                            print(f"[Cloud Server] 嚴格執行模式：權重範數接近警告閾值 ({current_norm:.4f} > {warn_threshold:.4f})，提前正則化{'（峰值保護模式）' if is_peak_protection_active else ''}")

                            global_weights = _apply_weight_norm_regularization(global_weights, warn_threshold, 0.98, hard_limit=hard_limit, strict_enforcement=False)
                            new_norm = _compute_global_l2_norm(global_weights)
                            print(f"[Cloud Server] 提前正則化後權重範數: {new_norm:.4f}")
                        else:
                            print(f"[Cloud Server] 全局權重 L2 範數正常 ({current_norm:.4f} ≤ {warn_threshold:.4f}){'（峰值保護模式）' if is_peak_protection_active else ''}")
                    
                    print(f"[Cloud Server] 全局權重已更新: {len(global_weights)} 層，輪次 {current_round}")
                else:
                    print(f"[Cloud Server] 警告：全局權重更新後為空，可能影響後續評估和廣播")

                if global_weights:
                    persist_global_weights_snapshot(current_round, global_weights)
                
                try:
                    for agg_id, weights_list in aggregator_weights.items():

                        before_count = len(weights_list)
                        aggregator_weights[agg_id] = [w for w in weights_list if w.get('round_id') != current_round]
                        after_count = len(aggregator_weights[agg_id])
                        if before_count > after_count:
                            print(f"[Cloud Server] 🧹 已清理聚合器 {agg_id} 的輪次 {current_round} 權重緩衝區 ({before_count} -> {after_count})")
                except Exception as e:
                    print(f"[Cloud Server] 清理聚合器權重緩衝區失敗: {e}")
                
                print(f"[Cloud Server] 全局聚合完成 (第{aggregation_count}次, 輪次: {current_round})")
                log_event("global_aggregation_completed", f"count={aggregation_count},round={current_round},quorum={quorum_required}")

                try:
                    per_agg_data_sizes = []
                    for agg_id, weights_list in aggregator_weights.items():

                        round_candidates = [w for w in weights_list if w.get('round_id') == current_round]
                        if round_candidates:
                            size = int(round_candidates[-1].get('data_size', 0))
                        elif weights_list:

                            size = int(weights_list[-1].get('data_size', 0))
                        else:
                            size = 0
                        per_agg_data_sizes.append(max(0, size))
                    total_data_size_real = sum(per_agg_data_sizes)
                except Exception as e:
                    print(f"[Cloud Server] 計算真實 total_data_size 失敗，退回使用 len(weights_list): {e}")
                    per_agg_data_sizes = [len(wl) for wl in aggregator_weights.values()]
                    total_data_size_real = sum(per_agg_data_sizes)

                log_training_event_cloud(
                    'global_aggregated',
                    {
                        'round_id': current_round,
                        'participating_aggregators': len(aggregator_weights),
                        'total_data_size': total_data_size_real,
                        'aggregation_count': aggregation_count
                    }
                )
                
                try:
                    if aiohttp is not None:

                        import asyncio
                        try:
                            loop = asyncio.get_running_loop()

                            asyncio.create_task(_immediate_broadcast_global_weights(current_round))
                            print(f"[Cloud Server]  已觸發立即廣播全局權重（輪次: {current_round}）")
                        except RuntimeError:
                            import threading
                            def run_broadcast():
                                try:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    loop.run_until_complete(_immediate_broadcast_global_weights(current_round))
                                    loop.close()
                                except Exception as e:
                                    print(f"[Cloud Server] 線程中立即廣播失敗: {e}")
                            thread = threading.Thread(target=run_broadcast, daemon=True)
                            thread.start()
                            print(f"[Cloud Server]  已在線程中觸發立即廣播全局權重（輪次: {current_round}）")
                except Exception as e:
                    print(f"[Cloud Server] 立即廣播失敗: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    active_aggs = len([k for k, v in aggregator_weights.items() if v])
                    per_agg_data_sizes = []
                    for agg_id, weights_list in aggregator_weights.items():
                        round_candidates = [w for w in weights_list if w.get('round_id') == current_round]
                        if round_candidates:
                            size = int(round_candidates[-1].get('data_size', 0))
                        elif weights_list:
                            size = int(weights_list[-1].get('data_size', 0))
                        else:
                            size = 0
                        per_agg_data_sizes.append(max(0, size))
                    total_data_size_real = sum(per_agg_data_sizes)
                    cloud_aggregation_result = {
                        'global_weights': global_weights,
                        'aggregator_ids': list(aggregator_weights.keys()),
                        'data_sizes': per_agg_data_sizes,
                        'total_data_size': total_data_size_real,
                        'aggregation_count': aggregation_count,
                        'round_id': current_round,
                        'timestamp': time.time(),
                        'effective_aggregators': last_curve_stats.get('effective_aggregators', active_aggs),
                        'quality_pass_ratio': (
                            last_curve_stats.get('quality_pass', active_aggs) /
                            max(1, last_curve_stats.get('quality_checked', active_aggs))
                        )
                    }
                    
                    save_cloud_results(cloud_aggregation_result, current_round)
                    
                except Exception as e:
                    print(f"[Cloud Server] 保存聚合結果失敗: {e}")
                
                eval_key = f"round_{current_round}"
                if not hasattr(app.state, 'evaluated_rounds'):
                    app.state.evaluated_rounds = set()
                
                if eval_key not in app.state.evaluated_rounds:
                    if global_weights is None or len(global_weights) == 0:
                        print(f"[Cloud Server] 輪次 {current_round} 評估跳過：global_weights 為空")
                        app.state.evaluated_rounds.add(eval_key)
                    else:
                        import copy
                        import torch
                        weights_copy = {}
                        try:
                            for k, v in global_weights.items():
                                if isinstance(v, torch.Tensor):
                                    weights_copy[k] = v.detach().clone().cpu()
                                elif isinstance(v, np.ndarray):
                                    weights_copy[k] = v.copy()
                                elif isinstance(v, (list, tuple)):
                                    weights_copy[k] = copy.deepcopy(v)
                                else:
                                    weights_copy[k] = copy.deepcopy(v)

                        except Exception as e:
                            print(f"[Cloud Server] 權重複製失敗: {e}，無法進行評估")
                            app.state.evaluated_rounds.add(eval_key)
                            return JSONResponse(content={"status":"error","message":f"權重複製失敗: {e}"})
                        
                    def run_evaluation(round_id: int, weights: dict):
                        import sys
                        import traceback
                        global BEST_GLOBAL_F1, BEST_GLOBAL_WEIGHTS, BEST_ROUND_ID, PERFORMANCE_DROP_COUNT
                        global F1_DROP_OBSERVATION_COUNT, F1_DROP_OBSERVATION_START_ROUND
                        global global_weights, needs_rollback_flag, rollback_reason_str
                        global last_peak_round
                        try:
                            print(f"[Cloud Server] 開始評估輪次 {round_id}（背景執行）", flush=True)
                            if not weights or len(weights) == 0:
                                print(f"[Cloud Server] 評估失敗：傳入的權重為空 (round={round_id})")
                                return
                            max_samples = getattr(config, "GLOBAL_EVAL_MAX_SAMPLES", None)
                            eval_res = evaluate_global_model_on_csv(round_id, weights, max_samples)
                            if eval_res is not None:
                                print(f"[Cloud Server] 輪次 {round_id} 評估完成: 準確率={eval_res.get('accuracy', 0):.4f}, F1={eval_res.get('f1_score', 0):.4f}")
                                log_event("global_baseline_evaluated", json.dumps(eval_res))
                                _write_global_metrics(eval_res, round_id)
                                try:
                                    f1_value = float(eval_res.get('f1_score', eval_res.get('macro_f1', 0.0)))
                                    acc_value = float(eval_res.get('accuracy', 0.0))
                                    elite_projection_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('elite_weight_projection', {})
                                    f1_drop_threshold = float(elite_projection_cfg.get('f1_drop_threshold', 0.3))
                                    enable_f1_based_protection = elite_projection_cfg.get('enable_f1_based_protection', True)
                                    
                                    if f1_value > BEST_GLOBAL_F1:
                                        BEST_GLOBAL_F1 = f1_value
                                        BEST_ROUND_ID = round_id
                                        BEST_GLOBAL_WEIGHTS = {k: _coerce_tensor(v).clone() for k, v in weights.items()}
                                        
                                        global BEST_MODELS_HISTORY, TOP_N_BEST_MODELS
                                        BEST_MODELS_HISTORY.append((round_id, f1_value, {k: _coerce_tensor(v).clone() for k, v in weights.items()}))
                                        BEST_MODELS_HISTORY.sort(key=lambda x: x[1], reverse=True)
                                        BEST_MODELS_HISTORY = BEST_MODELS_HISTORY[:TOP_N_BEST_MODELS]
                                        print(f"[Cloud Server] 🏅 更新最佳模型快照 round={round_id}, f1={f1_value:.4f}, acc={acc_value:.4f}")
                                        print(f"[Cloud Server]  多樣性緩衝: 保留 {len(BEST_MODELS_HISTORY)} 個最佳模型 (Top-{TOP_N_BEST_MODELS})")
                                        PERFORMANCE_DROP_COUNT = 0
                                        
                                        global ROLLBACK_STABLE_ROUNDS
                                        if LAST_ROLLBACK_ROUND is not None:
                                            ROLLBACK_STABLE_ROUNDS += 1
                                            print(f"[Cloud Server] F1 提升，穩定輪次計數器: {ROLLBACK_STABLE_ROUNDS}")
                                        
                                        if F1_DROP_OBSERVATION_COUNT > 0:
                                            print(f"[Cloud Server] F1 提升，重置觀察期計數器")
                                            F1_DROP_OBSERVATION_COUNT = 0
                                            F1_DROP_OBSERVATION_START_ROUND = None
                                        
                                        if PEAK_PROTECTION_ENABLED and f1_value >= PEAK_PROTECTION_THRESHOLD:
                                            last_peak_round = round_id
                                            print(f"[Cloud Server] 啟用峰值保護（F1={f1_value:.4f} >= {PEAK_PROTECTION_THRESHOLD:.2f}，保護期 {PEAK_PROTECTION_ROUNDS} 輪）")
                                        
                                        try:
                                            exp_dir = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', 'result'))
                                            os.makedirs(exp_dir, exist_ok=True)
                                            best_weights_path = os.path.join(exp_dir, "best_global_weights.pt")
                                            best_meta_path = os.path.join(exp_dir, "best_global_meta.json")
                                            if torch is not None:
                                                torch.save(BEST_GLOBAL_WEIGHTS, best_weights_path)
                                                print(f"[Cloud Server]  已立即保存最佳權重到 {best_weights_path} (round={round_id}, f1={f1_value:.4f})")
                                            meta_obj = {
                                                "round": BEST_ROUND_ID,
                                                "best_f1": BEST_GLOBAL_F1,
                                                "saved_at": datetime.datetime.now().isoformat()
                                            }
                                            with open(best_meta_path, "w", encoding="utf-8") as mf:
                                                json.dump(meta_obj, mf, ensure_ascii=False, indent=2)
                                            print(f"[Cloud Server]  已立即保存最佳模型元數據到 {best_meta_path} (round={round_id}, f1={f1_value:.4f})")
                                        except Exception as persist_exc:
                                            print(f"[Cloud Server] 立即保存最佳模型快照失敗: {persist_exc}")
                                    elif BEST_GLOBAL_F1 > 0:
                                        performance_drop = (BEST_GLOBAL_F1 - f1_value) / BEST_GLOBAL_F1 if BEST_GLOBAL_F1 > 0 else 0.0
                                        
                                        global F1_DROP_DYNAMIC_COUNT
                                        
                                        if enable_f1_based_protection:
                                            if performance_drop > 0.50:
                                                print(f"[Cloud Server]  極端下降檢測：F1 下降 {performance_drop:.2%} > 50%，立即回退")
                                                print(f"[Cloud Server] 立即回退到最佳模型 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})")
                                                if BEST_GLOBAL_WEIGHTS is not None:
                                                    global_weights = {k: _coerce_tensor(v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
                                                    print(f"[Cloud Server] 已立即回退到最佳模型權重 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})")
                                                    needs_rollback_flag = True
                                                    rollback_reason_str = f"extreme_drop_{performance_drop*100:.1f}%_immediate_rollback"

                                                F1_DROP_OBSERVATION_COUNT = 0
                                                F1_DROP_OBSERVATION_START_ROUND = None
                                                F1_DROP_DYNAMIC_COUNT = 0
                                            elif performance_drop < 0.15:
                                                if F1_DROP_OBSERVATION_COUNT > 0 or F1_DROP_DYNAMIC_COUNT > 0:
                                                    print(f"[Cloud Server] F1 小幅下降 {performance_drop:.2%} < 15%，視為正常震盪（Exploration），重置計數器")
                                                    F1_DROP_OBSERVATION_COUNT = 0
                                                    F1_DROP_OBSERVATION_START_ROUND = None
                                                    F1_DROP_DYNAMIC_COUNT = 0
                                            elif performance_drop > f1_drop_threshold:
                                                F1_DROP_DYNAMIC_COUNT += 1
                                                
                                                dynamic_threshold = max(1, int(3.0 - (performance_drop - 0.15) / 0.10 * 2))
                                                
                                                if F1_DROP_OBSERVATION_START_ROUND is None:
                                                    F1_DROP_OBSERVATION_START_ROUND = round_id
                                                    print(f"[Cloud Server] 檢測到 F1 下降 {performance_drop:.2%}（15% < drop <= 50%），開始動態計數（觸發閾值: {dynamic_threshold} 輪）")
                                                else:
                                                    rounds_in_count = round_id - F1_DROP_OBSERVATION_START_ROUND + 1
                                                    print(f"[Cloud Server] F1 下降持續中（動態計數 {F1_DROP_DYNAMIC_COUNT}/{dynamic_threshold} 輪，已持續 {rounds_in_count} 輪，下降 {performance_drop:.2%}）")
                                                
                                                if F1_DROP_DYNAMIC_COUNT >= dynamic_threshold:
                                                    print(f"[Cloud Server]  動態計數器觸發：F1 持續下降 {F1_DROP_DYNAMIC_COUNT} 輪（閾值: {dynamic_threshold}），觸發回退")
                                                    print(f"[Cloud Server] 回退到最佳模型 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})")
                                                    if BEST_GLOBAL_WEIGHTS is not None:
                                                        global_weights = {k: _coerce_tensor(v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
                                                        print(f"[Cloud Server] 已回退到最佳模型權重 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})")
                                                        needs_rollback_flag = True
                                                        rollback_reason_str = f"dynamic_drop_{performance_drop*100:.1f}%_{F1_DROP_DYNAMIC_COUNT}_rounds"

                                                    F1_DROP_OBSERVATION_COUNT = 0
                                                    F1_DROP_OBSERVATION_START_ROUND = None
                                                    F1_DROP_DYNAMIC_COUNT = 0
                                            else:
                                                if F1_DROP_OBSERVATION_COUNT > 0 or F1_DROP_DYNAMIC_COUNT > 0:
                                                    print(f"[Cloud Server] F1 恢復正常（下降 {performance_drop:.2%} < 閾值 {f1_drop_threshold:.2%}），重置計數器")
                                                    F1_DROP_OBSERVATION_COUNT = 0
                                                    F1_DROP_OBSERVATION_START_ROUND = None
                                                    F1_DROP_DYNAMIC_COUNT = 0
                                        
                                        if not (enable_f1_based_protection and performance_drop > f1_drop_threshold):
                                            if performance_drop > 0.15:
                                                PERFORMANCE_DROP_COUNT += 1
                                                print(f"[Cloud Server] 性能下降檢測: BEST_F1={BEST_GLOBAL_F1:.4f}, 當前F1={f1_value:.4f}, 下降={performance_drop*100:.1f}%, 連續下降次數={PERFORMANCE_DROP_COUNT}")
                                                
                                                if PERFORMANCE_DROP_COUNT >= 2:

                                                    needs_rollback_flag = True
                                                    rollback_reason_str = f"performance_degradation_{performance_drop*100:.1f}%_for_{PERFORMANCE_DROP_COUNT}_rounds"
                                                    print(f"[Cloud Server] 觸發回退：性能連續下降 {PERFORMANCE_DROP_COUNT} 輪，下降 {performance_drop*100:.1f}%")
                                            else:
                                                PERFORMANCE_DROP_COUNT = 0
                                        else:
                                            PERFORMANCE_DROP_COUNT = 0
                                        
                                        if BEST_GLOBAL_WEIGHTS is None:
                                            print(f"[Cloud Server] 警告：BEST_GLOBAL_WEIGHTS 為 None，嘗試從當前權重恢復")
                                            try:
                                                BEST_GLOBAL_WEIGHTS = {
                                                    k: _coerce_tensor(v).clone()
                                                    for k, v in weights.items()
                                                }
                                                print(f"[Cloud Server] 已從當前權重恢復 BEST_GLOBAL_WEIGHTS (round={round_id})")
                                            except Exception as e:
                                                print(f"[Cloud Server] 無法恢復 BEST_GLOBAL_WEIGHTS: {e}")
                                                print(f"[Cloud Server] 跳過性能下降檢測（BEST_GLOBAL_WEIGHTS 為 None，無法恢復）")
                                        else:
                                            if enable_f1_based_protection and performance_drop > f1_drop_threshold:

                                                print(f"[Cloud Server] 觀察期已啟用（F1 下降 {performance_drop:.2%}），跳過單次大幅下降檢測")
                                            else:
                                                performance_drop_recheck = (BEST_GLOBAL_F1 - f1_value) / BEST_GLOBAL_F1
                                                
                                                is_peak_protection_active = False
                                                effective_warning_threshold = performance_warning_threshold
                                                effective_single_drop_threshold = single_drop_threshold
                                                effective_degradation_threshold = performance_degradation_threshold
                                                
                                                if PEAK_PROTECTION_ENABLED and last_peak_round is not None:
                                                    rounds_since_peak = round_id - last_peak_round
                                                    if rounds_since_peak <= PEAK_PROTECTION_ROUNDS:
                                                        is_peak_protection_active = True
                                                        effective_warning_threshold = performance_warning_threshold * 0.5
                                                        effective_single_drop_threshold = single_drop_threshold * 0.5
                                                        effective_degradation_threshold = performance_degradation_threshold * 0.5
                                                        print(f"[Cloud Server] 峰值保護活躍中（距離峰值{rounds_since_peak}輪）：使用更嚴格閾值（警告={effective_warning_threshold*100:.1f}%, 單次下降={effective_single_drop_threshold*100:.1f}%, 退化={effective_degradation_threshold*100:.1f}%）")
                                                        
                                                        global PEAK_PROTECTION_DROP_COUNT, PEAK_PROTECTION_DROP_START_ROUND
                                                        if performance_drop_recheck > PEAK_PROTECTION_DROP_THRESHOLD:

                                                            if PEAK_PROTECTION_DROP_START_ROUND is None:

                                                                PEAK_PROTECTION_DROP_START_ROUND = round_id
                                                                PEAK_PROTECTION_DROP_COUNT = 1
                                                                print(f"[Cloud Server] 峰值保護：檢測到 F1 下降 {performance_drop_recheck*100:.1f}% > {PEAK_PROTECTION_DROP_THRESHOLD*100:.1f}%，開始追蹤連續下降（第 1 輪）")
                                                            else:
                                                                PEAK_PROTECTION_DROP_COUNT += 1
                                                                rounds_in_drop = round_id - PEAK_PROTECTION_DROP_START_ROUND + 1
                                                                print(f"[Cloud Server] 峰值保護：F1 下降持續中（連續 {PEAK_PROTECTION_DROP_COUNT}/{PEAK_PROTECTION_DROP_PATIENCE} 輪，已持續 {rounds_in_drop} 輪，下降 {performance_drop_recheck*100:.1f}%）")
                                                                

                                                                if PEAK_PROTECTION_DROP_COUNT >= PEAK_PROTECTION_DROP_PATIENCE:
                                                                    print(f"[Cloud Server]  峰值保護：連續 {PEAK_PROTECTION_DROP_COUNT} 輪 F1 下降 > {PEAK_PROTECTION_DROP_THRESHOLD*100:.1f}%，立即回退到峰值模型")
                                                                    print(f"[Cloud Server] 立即回退到最佳模型 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})")
                                                                    if BEST_GLOBAL_WEIGHTS is not None:

                                                                        global_weights = {k: _coerce_tensor(v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
                                                                        print(f"[Cloud Server] 已立即回退到最佳模型權重 (round={BEST_ROUND_ID})")
                                                                        needs_rollback_flag = True
                                                                        rollback_reason_str = f"peak_protection_consecutive_drop_{PEAK_PROTECTION_DROP_COUNT}_rounds_{performance_drop_recheck*100:.1f}%"

                                                                    PEAK_PROTECTION_DROP_COUNT = 0
                                                                    PEAK_PROTECTION_DROP_START_ROUND = None
                                                                    PERFORMANCE_DROP_COUNT = 0
                                                        else:
                                                            if PEAK_PROTECTION_DROP_COUNT > 0:
                                                                print(f"[Cloud Server] 峰值保護：F1 恢復正常（下降 {performance_drop_recheck*100:.1f}% < {PEAK_PROTECTION_DROP_THRESHOLD*100:.1f}%），重置連續下降計數器")
                                                                PEAK_PROTECTION_DROP_COUNT = 0
                                                                PEAK_PROTECTION_DROP_START_ROUND = None
                                                
                                                print(f"[Cloud Server] 性能下降檢查: BEST_F1={BEST_GLOBAL_F1:.4f}, 當前F1={f1_value:.4f}, 下降={performance_drop_recheck*100:.1f}%")
                                                
                                                if performance_drop_recheck > effective_warning_threshold:

                                                    PERFORMANCE_DROP_COUNT += 1
                                                    print(f"[Cloud Server] 性能下降檢測: F1從{BEST_GLOBAL_F1:.4f}降至{f1_value:.4f} (下降{performance_drop_recheck*100:.1f}%)")
                                                    
                                                    if performance_drop_recheck > effective_single_drop_threshold:
                                                        print(f"[Cloud Server]  單次大幅下降檢測: 下降{performance_drop_recheck*100:.1f}% > {effective_single_drop_threshold*100:.1f}%，立即回退{'（峰值保護模式）' if is_peak_protection_active else ''}")
                                                        print(f"[Cloud Server] 立即回退到最佳模型 (round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f})")

                                                        if BEST_GLOBAL_WEIGHTS is not None:
                                                            global_weights = {k: _coerce_tensor(v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
                                                            print(f"[Cloud Server] 已立即回退到最佳模型權重 (round={BEST_ROUND_ID})")
                                                        needs_rollback_flag = True
                                                        rollback_reason_str = f"single_large_drop_{performance_drop_recheck*100:.1f}%_rollback_to_best_round_{BEST_ROUND_ID}"
                                                        PERFORMANCE_DROP_COUNT = 0
                                                        print(f"[Cloud Server] 注意：已回退，當前輪次使用最佳模型權重，下一輪聚合也將使用最佳權重")
                                                        


                                                        print(f"[Cloud Server] 保留 Round {round_id} 的評估記錄（F1={f1_value:.4f}），即使性能下降也不刪除，以便分析訓練狀態")

                                                        try:
                                                            exp_dir = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', 'result'))
                                                            csv_path = os.path.join(exp_dir, 'cloud_baseline.csv')
                                                            if os.path.exists(csv_path) and BEST_GLOBAL_WEIGHTS is not None:
                                                                import pandas as pd
                                                                df = pd.read_csv(csv_path)

                                                                if BEST_ROUND_ID not in df['round'].values:
                                                                    max_samples = getattr(config, "GLOBAL_EVAL_MAX_SAMPLES", None)
                                                                    best_eval_res = evaluate_global_model_on_csv(BEST_ROUND_ID, BEST_GLOBAL_WEIGHTS, max_samples)
                                                                    if best_eval_res:
                                                                        print(f"[Cloud Server] 已重新評估最佳模型 (Round {BEST_ROUND_ID}) 並更新 CSV")
                                                        except Exception as cleanup_exc:
                                                            print(f"[Cloud Server] 更新最佳模型記錄異常: {cleanup_exc}")

                                                    elif performance_drop_recheck > effective_degradation_threshold and PERFORMANCE_DROP_COUNT >= PERFORMANCE_DROP_THRESHOLD:
                                                        print(f"[Cloud Server] 連續下降計數: {PERFORMANCE_DROP_COUNT}/{PERFORMANCE_DROP_THRESHOLD}")
                                                        print(f"[Cloud Server] 立即回退（連續{PERFORMANCE_DROP_COUNT}輪下降，回退到最佳模型 round={BEST_ROUND_ID}, f1={BEST_GLOBAL_F1:.4f}）")

                                                        if BEST_GLOBAL_WEIGHTS is not None:
                                                            global_weights = {k: _coerce_tensor(v).clone() for k, v in BEST_GLOBAL_WEIGHTS.items()}
                                                            print(f"[Cloud Server] 已立即回退到最佳模型權重 (round={BEST_ROUND_ID})")
                                                        needs_rollback_flag = True
                                                        rollback_reason_str = f"performance_degradation_{performance_drop_recheck*100:.1f}%_consecutive_{PERFORMANCE_DROP_COUNT}_rollback_to_best_round_{BEST_ROUND_ID}"
                                                        PERFORMANCE_DROP_COUNT = 0
                                                        print(f"[Cloud Server] 注意：已回退，當前輪次使用最佳模型權重，下一輪聚合也將使用最佳權重")
                                                    
                                                        print(f"[Cloud Server] 保留 Round {round_id} 的評估記錄（F1={f1_value:.4f}），即使性能下降也不刪除，以便分析訓練狀態")

                                                        try:
                                                            exp_dir = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', 'result'))
                                                            csv_path = os.path.join(exp_dir, 'cloud_baseline.csv')
                                                            if os.path.exists(csv_path) and BEST_GLOBAL_WEIGHTS is not None:
                                                                import pandas as pd
                                                                df = pd.read_csv(csv_path)

                                                                if BEST_ROUND_ID not in df['round'].values:
                                                                    max_samples = getattr(config, "GLOBAL_EVAL_MAX_SAMPLES", None)
                                                                    best_eval_res = evaluate_global_model_on_csv(BEST_ROUND_ID, BEST_GLOBAL_WEIGHTS, max_samples)
                                                                    if best_eval_res:
                                                                        print(f"[Cloud Server] 已重新評估最佳模型 (Round {BEST_ROUND_ID}) 並更新 CSV")
                                                        except Exception as cleanup_exc:
                                                            print(f"[Cloud Server] 更新最佳模型記錄異常: {cleanup_exc}")
                                                    else:
                                                        print(f"[Cloud Server] 連續下降計數: {PERFORMANCE_DROP_COUNT}/{PERFORMANCE_DROP_THRESHOLD}")
                                                        print(f"[Cloud Server] 建議：繼續觀察，給模型學習機會（已連續下降{PERFORMANCE_DROP_COUNT}輪）")
                                                else:
                                                    if PERFORMANCE_DROP_COUNT > 0:
                                                        print(f"[Cloud Server] 性能下降幅度減小，重置下降計數器")
                                                    PERFORMANCE_DROP_COUNT = 0
                                            
                                    try:
                                        exp_dir = os.environ.get('EXPERIMENT_DIR', getattr(config, 'LOG_DIR', 'result'))
                                        os.makedirs(exp_dir, exist_ok=True)
                                        best_weights_path = os.path.join(exp_dir, "best_global_weights.pt")
                                        best_meta_path = os.path.join(exp_dir, "best_global_meta.json")
                                        if torch is not None:
                                            torch.save(BEST_GLOBAL_WEIGHTS, best_weights_path)
                                            print(f"[Cloud Server]  已保存最佳權重到 {best_weights_path}")
                                        meta_obj = {
                                            "round": BEST_ROUND_ID,
                                            "best_f1": BEST_GLOBAL_F1,
                                            "saved_at": datetime.datetime.now().isoformat()
                                        }
                                        with open(best_meta_path, "w", encoding="utf-8") as mf:
                                            json.dump(meta_obj, mf, ensure_ascii=False, indent=2)
                                        print(f"[Cloud Server]  已保存最佳模型描述到 {best_meta_path}")
                                    except Exception as persist_exc:
                                        print(f"[Cloud Server] 保存最佳模型快照失敗: {persist_exc}")
                                except Exception as best_exc:
                                    print(f"[Cloud Server] 更新最佳模型快照失敗: {best_exc}")
                            else:
                                print(f"[Cloud Server] 輪次 {round_id} 評估失敗，返回None")
                                print(f"[Cloud Server] 可能原因：找不到 global_test.csv 或評估過程中出錯")

                                try:
                                    log_event("global_baseline_evaluation_failed", f"round={round_id},reason=returned_none")
                                except:
                                    pass
                        except Exception as e:
                            import traceback
                            error_trace = traceback.format_exc()
                            print(f"[Cloud Server] 輪次 {round_id} 評估異常: {e}", flush=True)
                            print(f"[Cloud Server] 評估異常詳情:\n{error_trace}", flush=True)

                            try:
                                log_event("global_baseline_evaluation_exception", f"round={round_id},error={str(e)}")
                            except:
                                pass
                            traceback.print_exc()
                        except BaseException as e:

                            import traceback
                            error_trace = traceback.format_exc()
                            print(f"[Cloud Server] 輪次 {round_id} 評估發生嚴重異常: {e}", flush=True)
                            print(f"[Cloud Server] 評估異常詳情:\n{error_trace}", flush=True)
                            traceback.print_exc()
                    
                    _evaluation_executor.submit(run_evaluation, current_round, weights_copy)
                    print(f"[Cloud Server] 輪次 {current_round} 評估已提交到背景執行緒，不會阻塞健康檢查 (權重層數: {len(weights_copy)})")
                    
                    app.state.evaluated_rounds.add(eval_key)
                else:
                    print(f"[Cloud Server]  輪次 {current_round} 已評估過，跳過重複評估")
                
                try:
                    experiment_config = getattr(config, 'EXPERIMENT_CONFIG', {})
                    early_stopping_enabled = experiment_config.get('early_stopping', True)
                    
                    stop = False
                    reason = ""
                    
                    if not early_stopping_enabled:
                        print(f"[Cloud Server]  早停機制已禁用（根據 EXPERIMENT_CONFIG）")
                    else:
                        exp_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
                        stop, reason = _check_early_stop(exp_dir, max_round_cap=int(getattr(config, 'CONVERGENCE_CONFIG', {}).get('max_rounds', 100)))
                    
                    if stop:
                        early_stop_triggered = True
                        early_stop_reason = reason

                        try:
                            exp_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
                            marker = os.path.join(exp_dir, 'EARLY_STOPPED.txt')
                            with open(marker, 'w', encoding='utf-8') as mf:
                                mf.write(f"early_stop: {reason}\nround: {current_round}\n")
                        except Exception:
                            pass
                        log_event("early_stop_triggered", reason)
                except Exception as e:
                    print(f"[Cloud Server] 早停檢查異常: {e}")

                for agg_id in list(aggregator_weights.keys()):
                    aggregator_weights[agg_id] = [w for w in aggregator_weights[agg_id] if w.get('round_id') != current_round]

            
            except Exception as e:
                print(f"[Cloud Server] 全局聚合失敗: {e}")
                log_event("global_aggregation_failed", str(e))
                raise HTTPException(status_code=500, detail=f"Global aggregation failed: {str(e)}")
        else:
            print(f"[Cloud Server]  等待更多聚合器 (輪次: {current_round}, 回報: {current_round_reports}/{total_aggs}, quorum: {quorum_required})")
            log_event("waiting_for_aggregators", f"round={current_round},current={current_round_reports},threshold={quorum_required}")
        
        response_time = time.time() - request_start_time
        print(f"[Cloud Server] 聚合器 {aggregator_id} 上傳處理完成 (總耗時: {response_time:.2f}s)")
        
        return JSONResponse(content={
            "status": "success",
            "message": "聚合權重上傳成功",
            "aggregator_id": aggregator_id,
            "round_id": round_id,
            "aggregation_count": aggregation_count
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = f"聚合權重上傳失敗: {str(e)}"
        print(f"[Cloud Server] {error_msg}")
        print(f"[Cloud Server] 異常詳情:\n{error_trace}")
        log_event("aggregated_weights_upload_failed", f"agg={aggregator_id},round={round_id},error={str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload_aggregated_delta")
async def upload_aggregated_delta(
    aggregator_id: int = Form(...),
    base_version: int = Form(...),
    round_id: int = Form(...),
    model_version: int = Form(...),
    delta: UploadFile = File(...)
):
    global global_weights, global_version

    try:
        try:
            delta_bytes = await delta.read()
            delta_payload = pickle.loads(delta_bytes)
        except Exception as e:
            msg = f"無法解析 delta 資料: {e}"
            print(f"[Cloud Server] {msg}")
            return JSONResponse(content={"status": "error", "message": msg}, status_code=400)

        if not isinstance(delta_payload, dict):
            msg = "delta 資料格式錯誤（預期為 dict）"
            print(f"[Cloud Server] {msg}")
            return JSONResponse(content={"status": "error", "message": msg}, status_code=400)

        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float()
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            if isinstance(x, (list, tuple)):
                try:
                    return torch.tensor(x, dtype=torch.float32)
                except Exception:
                    return None
            return None

        delta_tensors: Dict[str, torch.Tensor] = {}
        for k, v in delta_payload.items():
            t = _to_tensor(v)
            if t is not None:
                delta_tensors[k] = t

        if global_weights is None or not isinstance(global_weights, dict) or not global_weights:
            global_weights = {k: v.clone() for k, v in delta_tensors.items()}
            global_version = int(base_version) + 1
            print(f"[Cloud Server] 首次接收 delta，初始化 global_weights（版本={global_version}）")
            return {"status": "ok", "new_version": int(global_version)}

        if int(base_version) != int(global_version):
            msg = f"版本不一致，base_version={base_version} current={global_version}"

            debug_version_conflict = os.environ.get("DEBUG_VERSION_CONFLICT", "0").strip() == "1"
            if debug_version_conflict:
                print(f"[Cloud Server] [調試] {msg}")

            if not hasattr(upload_aggregated_delta, '_version_conflict_count'):
                upload_aggregated_delta._version_conflict_count = 0
                upload_aggregated_delta._version_success_count = 0
            upload_aggregated_delta._version_conflict_count += 1
            if upload_aggregated_delta._version_conflict_count % 20 == 0:
                total = upload_aggregated_delta._version_conflict_count + upload_aggregated_delta._version_success_count
                conflict_rate = upload_aggregated_delta._version_conflict_count / total * 100 if total > 0 else 0
                print(f"[Cloud Server] 版本衝突統計: 衝突={upload_aggregated_delta._version_conflict_count}, 成功={upload_aggregated_delta._version_success_count}, 衝突率={conflict_rate:.1f}% (這是並發環境下的正常現象)")
            return JSONResponse(content={"status": "conflict", "message": msg}, status_code=409)

        merged: Dict[str, Any] = {}
        for k, base_val in (global_weights or {}).items():
            base_t = _to_tensor(base_val)
            d_t = delta_tensors.get(k)
            if base_t is not None and d_t is not None and base_t.shape == d_t.shape:
                merged[k] = (base_t + d_t).clone()
            elif d_t is not None:

                print(f"[Cloud Server] 權重鍵 {k} 形狀不符或缺失，使用 delta 覆蓋")
                merged[k] = d_t.clone()
            else:
                if base_t is not None:
                    merged[k] = base_t.clone()
                else:
                    merged[k] = base_val

        for k, d_t in delta_tensors.items():
            if k not in merged:
                merged[k] = d_t.clone()

        global_weights = merged
        global_version = int(global_version) + 1
        
        if not hasattr(upload_aggregated_delta, '_version_success_count'):
            upload_aggregated_delta._version_success_count = 0
        upload_aggregated_delta._version_success_count += 1
        
        delta_size_bytes = _compute_model_size_bytes(delta_tensors)
        delta_size_mb = delta_size_bytes / (1024 * 1024)
        
        if not hasattr(app.state, 'round_upload_bytes'):
            app.state.round_upload_bytes = {}
        if round_id not in app.state.round_upload_bytes:
            app.state.round_upload_bytes[round_id] = 0
        app.state.round_upload_bytes[round_id] += delta_size_bytes
    
        global_size_bytes = _compute_model_size_bytes(global_weights)
        global_size_mb = global_size_bytes / (1024 * 1024)
    
        round_upload_mb = app.state.round_upload_bytes[round_id] / (1024 * 1024)
        round_download_mb = global_size_mb
        round_total_mb = round_upload_mb + round_download_mb
        
        app.state.round_upload_mb = round_upload_mb
        app.state.round_download_mb = round_download_mb
        app.state.round_total_mb = round_total_mb
        
        print(f"[Cloud Server] 已合併來自聚合器 {aggregator_id} 的 delta（round={round_id}, new_version={global_version}）")
        print(f"[Cloud Server] 本輪通訊量（delta 上傳）: {delta_size_mb:.4f} MB, 累積上傳: {round_upload_mb:.4f} MB, 下載: {round_download_mb:.4f} MB")
        log_event("delta_merged", f"agg={aggregator_id},round={round_id},version={global_version}")
        
        with lock:
            if aggregator_id not in aggregator_weights:
                aggregator_weights[aggregator_id] = []
            
            weight_mean = 0.0
            weight_std = 0.0
            try:
                if global_weights:
                    all_weights_flat = torch.cat([_coerce_tensor(w).flatten() for w in global_weights.values() if _coerce_tensor(w) is not None])
                    weight_mean = float(all_weights_flat.mean().item())
                    weight_std = float(all_weights_flat.std().item())
            except Exception:
                pass
            
            aggregator_weights[aggregator_id].append({
                'aggregated_weights': {k: v.clone() for k, v in global_weights.items()},
                'round_id': round_id,
                'model_version': model_version,
                'participating_clients': [],
                'aggregation_stats': {
                    'performance_score': 0.5,
                    'accuracy': None,
                    'f1_score': None
                },
                'data_size': 1000,
                'timestamp': time.time(),
                'weight_stats': {
                    'mean': weight_mean,
                    'std': weight_std,
                    'num_layers': len(global_weights)
                }
            })
            
            if len(aggregator_weights[aggregator_id]) > 10:
                aggregator_weights[aggregator_id] = aggregator_weights[aggregator_id][-10:]
            
            prune_before = time.time() - 600
            aggregator_weights[aggregator_id] = [w for w in aggregator_weights[aggregator_id] if w.get('timestamp', 0) >= prune_before]
        
        try:
            try:
                total_aggs = max(1, len(registered_aggregators))
            except Exception:
                total_aggs = 1
            
            cfg = getattr(config, 'AGGREGATION_CONFIG', {}) or {}
            cfg_quorum = cfg.get('aggregator_quorum') or cfg.get('min_aggregators_for_global')
            
            available_aggs = len([agg_id for agg_id, agg_info in registered_aggregators.items() if agg_info.get('status') == 'healthy'])
            if available_aggs == 0:
                available_aggs = total_aggs
            
            if cfg_quorum is None:
                quorum_required = max(1, int(math.ceil(0.6 * available_aggs)))
            else:
                quorum_required = max(1, int(cfg_quorum))
            quorum_required = min(quorum_required, available_aggs)
            
            round_tolerance = 1
            current_round_reports = sum(1 for wlist in aggregator_weights.values() 
                                       if any(abs(w.get('round_id', 0) - round_id) <= round_tolerance for w in wlist))
            
            if current_round_reports >= quorum_required:
                dbi_weights_list = []
                for agg_id, weights_list in aggregator_weights.items():
                    round_candidates = [w for w in weights_list if abs(w.get('round_id', 0) - round_id) <= round_tolerance]
                    if round_candidates:
                        latest_weight = round_candidates[-1]
                        agg_weights = latest_weight.get('aggregated_weights', {})
                        if agg_weights:
                            dbi_weights_list.append({
                                'agg_id': agg_id,
                                'weights': agg_weights,
                                'performance_score': latest_weight.get('aggregation_stats', {}).get('performance_score', 0.5),
                                'data_size': latest_weight.get('data_size', 1000)
                            })
                
                if len(dbi_weights_list) >= 2:
                    dbi_suspicious_ids, dbi_action, dbi_soft_factor = _analyze_aggregator_weights_with_dbi(
                        dbi_weights_list, current_round=round_id
                    )
                    dbi_suspicious_ids = set(dbi_suspicious_ids or [])
                    

                    if dbi_action == "hard" and dbi_suspicious_ids:
                        print(
                            f"[Cloud Server] ConfShield/DBI 硬剔除：標記 {len(dbi_suspicious_ids)} 個可疑聚合器 "
                            f"({dbi_suspicious_ids})，將從後續處理中排除"
                        )
                        if not hasattr(app.state, 'dbi_excluded_aggregators'):
                            app.state.dbi_excluded_aggregators = {}
                        app.state.dbi_excluded_aggregators[round_id] = dbi_suspicious_ids
                    elif dbi_action == "soft" and dbi_suspicious_ids:
                        print(
                            f"[Cloud Server] ConfShield/DBI 軟降權：標記 {len(dbi_suspicious_ids)} 個可疑聚合器 "
                            f"({dbi_suspicious_ids})，將在聚合時應用 soft_factor={dbi_soft_factor:.3f}"
                        )
                        if not hasattr(app.state, 'dbi_soft_weights'):
                            app.state.dbi_soft_weights = {}
                        app.state.dbi_soft_weights[round_id] = {
                            'suspicious_ids': dbi_suspicious_ids,
                            'soft_factor': dbi_soft_factor
                        }
                    else:
                        print(f"[Cloud Server] ConfShield/DBI 監測模式：未發現需要處理的可疑聚合器")
                else:
                    print(f"[Cloud Server] ConfShield/DBI 跳過：聚合器數量不足 ({len(dbi_weights_list)} < 2)")
        except Exception as e:
            print(f"[Cloud Server] DBI 檢測過程發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            if not hasattr(app.state, "last_evaluated_round"):
                app.state.last_evaluated_round = -1
            
            last_round = app.state.last_evaluated_round
            if round_id != last_round:
                print(f"[Cloud Server] Round 變更檢測: {last_round} → {round_id}，將強制觸發全域評估", flush=True)
                
                if hasattr(app.state, "evaluated_rounds"):
                    eval_key = f"round_{int(round_id)}"
                    if eval_key in app.state.evaluated_rounds:
                        print(f"[Cloud Server] Round {round_id} 已評估過，但檢測到 round 變更，強制重新評估", flush=True)
                        app.state.evaluated_rounds.discard(eval_key)
                
                if global_weights and len(global_weights) > 0:
                    print(f"[Cloud Server] 準備觸發全域評估: round_id={round_id}, weights_keys={list(global_weights.keys())[:5]}..., weights_len={len(global_weights)}", flush=True)
                    _schedule_global_test_eval(round_id, global_weights)
                else:
                    print(f"[Cloud Server] 無法觸發全域評估: global_weights 為空或無效 (round_id={round_id}, global_weights={global_weights is not None}, len={len(global_weights) if global_weights else 0})", flush=True)
                
                app.state.last_evaluated_round = round_id
            else:
                if last_round == -1 and global_weights and len(global_weights) > 0:
                    print(f"[Cloud Server] 首次聚合檢測，觸發全域評估: round_id={round_id}", flush=True)
                    _schedule_global_test_eval(round_id, global_weights)
                app.state.last_evaluated_round = round_id
        except Exception as e:
            print(f"[Cloud Server] Round 變更評估觸發失敗: {e}", flush=True)
            import traceback
            traceback.print_exc()

        return {"status": "ok", "new_version": int(global_version)}

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"聚合 delta 上傳失敗: {e}"
        print(f"[Cloud Server] {error_msg}")
        print(f"[Cloud Server] 異常詳情:\n{error_trace}")
        log_event("delta_upload_failed", f"agg={aggregator_id},round={round_id},error={e}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload_weights")
async def upload_weights(
    aggregator_id: int = Form(...),
    data_size: int = Form(...),
    weights: UploadFile = File(...)
):
    global global_weights, aggregation_count
    
    try:
        upload_data = pickle.loads(weights.file.read())
        
        if not isinstance(upload_data, dict):
            print(f"[Cloud Server] 錯誤: 上傳數據不是字典格式")
            raise HTTPException(status_code=400, detail="Invalid data format")
        
        client_weights = upload_data.get('client_weights', {})
        server_model_state = upload_data.get('server_model_state', None)
        client_id = upload_data.get('client_id', 0)
        
        if not client_weights:
            print(f"[Cloud Server] 警告: 客戶端權重為空")
            raise HTTPException(status_code=400, detail="Empty client weights")
        
        try:
            all_weights_flat = np.concatenate([v.flatten() for v in client_weights.values()])
        except Exception as e:
            print(f"[Cloud Server] 錯誤: 權重數據無效: {e}")
            raise HTTPException(status_code=400, detail="Invalid weights data")
        
        if all_weights_flat.size == 0:
            print(f"[Cloud Server] 警告: 權重數據為空")
            raise HTTPException(status_code=400, detail="Empty weights data")
        
        weight_mean = all_weights_flat.mean()
        weight_std = all_weights_flat.std()
        
        print(f"[Cloud Server] 收到聚合器 {aggregator_id} 的客戶端 {client_id} 權重")
        print(f"  權重統計: mean={weight_mean:.6f}, std={weight_std:.6f}")
        print(f"  數據量: {data_size}")
        
        if abs(weight_mean) > 10.0:
            print(f"[Cloud Server] 警告: 權重均值過高 ({weight_mean:.6f})")
        if weight_std > 5.0:
            print(f"[Cloud Server] 警告: 權重標準差過大 ({weight_std:.6f})")
        if np.isnan(weight_mean) or np.isinf(weight_mean):
            print(f"[Cloud Server] 錯誤: 權重包含 NaN 或 Inf")
            raise HTTPException(status_code=400, detail="Invalid weights: NaN or Inf detected")
        
        with lock:
            aggregator_weights[aggregator_id].append({
                'client_weights': client_weights,
                'server_model_state': server_model_state,
                'data_size': data_size,
                'timestamp': time.time(),
                'weight_stats': {
                    'mean': weight_mean,
                    'std': weight_std
                }
            })
        
        if len(aggregator_weights) >= CLOUD_THRESHOLD:
            print(f"[Cloud Server]  開始全局聚合 (聚合器數量: {len(aggregator_weights)})")
            
            try:
                global_weights = perform_global_aggregation()
                aggregation_count += 1
                
                print(f"[Cloud Server] 全局聚合完成 (第{aggregation_count}次)")
                log_event("global_aggregation_completed", f"count={aggregation_count}")
                
                try:
                    cloud_aggregation_result = {
                        'global_weights': global_weights,
                        'aggregator_ids': list(aggregator_weights.keys()),
                        'data_sizes': [len(weights_list) for weights_list in aggregator_weights.values()],
                        'total_data_size': sum(len(weights_list) for weights_list in aggregator_weights.values()),
                        'aggregation_timestamp': time.time()
                    }
                    save_cloud_results(cloud_aggregation_result, aggregation_count)
                except Exception as e:
                    print(f"[Cloud Server] 保存雲端聚合結果失敗: {e}")
                
            except Exception as e:
                print(f"[Cloud Server] 全局聚合失敗: {e}")
                log_event("global_aggregation_failed", str(e))
                raise HTTPException(status_code=500, detail=f"Global aggregation failed: {e}")
        
        log_event("weights_uploaded", f"aggregator_id={aggregator_id},data_size={data_size}")
        
        return {
            "status": "weights_received",
            "aggregator_id": aggregator_id,
            "aggregation_count": aggregation_count,
            "global_weights_available": global_weights is not None
        }
        
    except Exception as e:
        error_msg = f"處理權重上傳時發生錯誤: {str(e)}"
        print(f"[Cloud Server] {error_msg}")
        log_event("upload_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload_split_features")
async def upload_split_features(
    client_id: int = Form(...),
    data_size: int = Form(...),
    batch_size: int = Form(...),
    features: UploadFile = File(...),
    label: UploadFile = File(...)
):
    try:
        split_cfg = getattr(config, 'SPLIT_LEARNING_CONFIG', {}) or {}
        if not bool(split_cfg.get('enabled', False)):
            return JSONResponse(content={"status": "disabled", "message": "split learning disabled"}, status_code=400)
        print(f"[Cloud Server] 接收客戶端 {client_id} 的Split Learning特徵")
        
        features_bytes = await features.read()
        if len(features_bytes) == 0:
            raise HTTPException(status_code=400, detail="特徵數據為空")
        
        labels_bytes = await label.read()
        if len(labels_bytes) == 0:
            raise HTTPException(status_code=400, detail="標籤數據為空")
        
        import pickle
        features_array = pickle.loads(features_bytes)
        print(f"[Cloud Server] 特徵形狀: {features_array.shape}")
        
        if not hasattr(app.state, 'server_backend'):
            from models.split_learning import ServerBackEnd
            import torch
            
            app.state.server_backend = ServerBackEnd(
                split_dim=config.MODEL_CONFIG["split_dim"],
                num_classes=config.MODEL_CONFIG["output_dim"],
                use_attn=config.MODEL_CONFIG["use_attention"],
                use_residual=config.MODEL_CONFIG["use_residual"]
            )

            smoothing = 0.05
            app.state.server_criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
            app.state.server_optimizer = torch.optim.Adam(
                app.state.server_backend.parameters(), 
                lr=config.SERVER_LR
            )
            print(f"[Cloud Server] ServerBackEnd模型初始化成功")
        
        import torch
        features_tensor = torch.tensor(features_array, dtype=torch.float32, requires_grad=True)
        
        labels_np = pickle.loads(labels_bytes)
        actual_batch_size = labels_np.size
        
        labels_np = labels_np.astype(np.int64)
        
        if abs(actual_batch_size - int(batch_size)) > 50:
            print(f"[Cloud Server] 標籤數量與聲明批次大小差異較大: {actual_batch_size} vs {batch_size}")
            print(f"[Cloud Server]  使用實際標籤數量: {actual_batch_size}")
        
        if features_tensor.shape[0] != actual_batch_size:
            print(f"[Cloud Server] 特徵與標籤數量不一致: {features_tensor.shape[0]} vs {actual_batch_size}")

            min_size = min(features_tensor.shape[0], actual_batch_size)
            print(f"[Cloud Server]  調整到較小數量: {min_size}")

            features_tensor = features_tensor[:min_size]

            features_tensor.requires_grad_(True)
            labels_np = labels_np[:min_size]
            actual_batch_size = min_size
        
        print(f"[Cloud Server] 標籤統計: min={labels_np.min()}, max={labels_np.max()}, unique={np.unique(labels_np)}")
        
        valid_labels = (labels_np >= 0) & (labels_np <= 3)
        if not np.all(valid_labels):
            print(f"[Cloud Server] 發現無效標籤值，進行修正")
            invalid_count = np.sum(~valid_labels)
            print(f"[Cloud Server]  無效標籤數量: {invalid_count}")
            
            labels_np = labels_np.copy()
            labels_np[~valid_labels] = 0
            print(f"[Cloud Server]  已將無效標籤替換為0")
        
        labels_tensor = torch.tensor(labels_np, dtype=torch.long)
        
        app.state.server_backend.train()
        logits = app.state.server_backend(features_tensor)
        loss = app.state.server_criterion(logits, labels_tensor)
        app.state.server_optimizer.zero_grad()
        loss.backward()
        
        gradients = features_tensor.grad.clone().detach().numpy()
        
        print(f"[Cloud Server] 計算完成 - Loss: {loss.item():.4f}")
        print(f"[Cloud Server] 梯度形狀: {gradients.shape}")
    
        log_event("split_features_processed", f"client_id={client_id},loss={loss.item():.4f}")
        
        return {
            "status": "success",
            "message": "特徵處理成功",
            "loss": float(loss.item()),
            "batch_size": int(batch_size)
        }
        
    except Exception as e:
        error_msg = f"處理Split Learning特徵時發生錯誤: {str(e)}"
        print(f"[Cloud Server] {error_msg}")
        log_event("split_features_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/get_global_weights")
async def get_global_weights():
    global global_weights
    if global_weights is None:
        print(f"[Cloud Server] 全局權重尚未初始化，進行臨時初始化為空權重")
        log_event("global_weights_requested", "lazy_initialize")
        global_weights = {}
    
    try:
        weights_bytes = pickle.dumps(global_weights)
        
        print(f"[Cloud Server] 返回全局權重")
        log_event("global_weights_requested", "success")
        
        return Response(
            content=weights_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=global_weights.pkl"}
        )
        
    except Exception as e:
        error_msg = f"序列化全局權重時發生錯誤: {str(e)}"
        print(f"[Cloud Server] {error_msg}")
        log_event("get_weights_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/get_global_weights_with_version")
async def get_global_weights_with_version():
    global global_weights, global_version
    try:
        payload = {
            'version': int(global_version),
            'weights': global_weights if global_weights is not None else {}
        }
        content = pickle.dumps(payload)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=global_with_version.pkl"}
        )
    except Exception as e:
        error_msg = f"序列化全局權重與版本時發生錯誤: {str(e)}"
        print(f"[Cloud Server] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/global_status")
async def global_status():
    try:
        status = {
            "cloud_server_id": cloud_server_id,
            "aggregation_count": aggregation_count,
            "has_global_weights": global_weights is not None,
            "aggregator_count": len(aggregator_weights),
            "cloud_threshold": CLOUD_THRESHOLD,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        aggregator_details = {}
        for agg_id, weights_list in aggregator_weights.items():
            aggregator_details[agg_id] = {
                "weight_count": len(weights_list),
                "last_upload": weights_list[-1]['timestamp'] if weights_list else None
            }
        status["aggregator_details"] = aggregator_details
        
        print(f"[Cloud Server] 狀態查詢: {status}")
        log_event("status_requested", "success")
        
        return status
        
    except Exception as e:
        error_msg = f"獲取全局狀態時發生錯誤: {str(e)}"
        print(f"[Cloud Server] {error_msg}")
        log_event("status_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def _apply_server_ema(new_weights: dict, prev_weights: dict = None) -> dict:
    global global_weights
    try:
        ema_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('server_ema', {}) or {}
        if not ema_cfg or not bool(ema_cfg.get('enabled', False)):
            return new_weights
        decay = float(ema_cfg.get('decay', 0.985))
        import torch
        
        if prev_weights is None:
            prev_weights = global_weights
        
        if prev_weights is None or len(prev_weights) == 0:
            return new_weights
        
        smoothed = {}
        for k, v in new_weights.items():
            prev = prev_weights.get(k, None)
            try:
                if prev is None:
                    smoothed[k] = v
                else:
                    pt = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
                    gt = prev if isinstance(prev, torch.Tensor) else torch.tensor(prev, dtype=torch.float32)
                    smoothed[k] = (decay * gt + (1.0 - decay) * pt).float()
            except Exception:
                smoothed[k] = v
        return smoothed
    except Exception:
        return new_weights

def perform_global_aggregation():
    global global_weights
    print(f"[Cloud Server] 開始執行全局聚合...")
    
    all_weights = []
    all_data_sizes = []
    all_server_states = []
    all_performance_scores = []
    
    for aggregator_id, weights_list in aggregator_weights.items():
        for weight_info in weights_list:
            if 'aggregated_weights' in weight_info:
                all_weights.append(weight_info['aggregated_weights'])
            elif 'client_weights' in weight_info:

                all_weights.append(weight_info['client_weights'])
            else:
                print(f"[Cloud Server] 未知的權重格式: {list(weight_info.keys())}")
                continue
                
            data_size = weight_info.get('data_size', 1000)
            all_data_sizes.append(data_size)
            
            performance_score = None
            aggregation_stats = weight_info.get('aggregation_stats', {})
            if aggregation_stats:
                if 'accuracy' in aggregation_stats and 'f1_score' in aggregation_stats:
                    performance_score = (aggregation_stats['accuracy'] + aggregation_stats['f1_score']) / 2
                elif 'accuracy' in aggregation_stats:
                    performance_score = aggregation_stats['accuracy']
                elif 'f1_score' in aggregation_stats:
                    performance_score = aggregation_stats['f1_score']
                elif 'performance_score' in aggregation_stats:
                    performance_score = aggregation_stats['performance_score']
            
            if performance_score is None:
                performance_score = 0.5
                print(f"[Cloud Server] 聚合器 {aggregator_id} 未提供性能分數，使用默認值 0.5")
            
            all_performance_scores.append(performance_score)
            
            if 'server_model_state' in weight_info and weight_info['server_model_state'] is not None:
                all_server_states.append(weight_info['server_model_state'])
    
    if not all_weights:
        raise ValueError("沒有權重數據進行聚合")
    
    print(f"[Cloud Server] 聚合 {len(all_weights)} 個權重，總數據量: {sum(all_data_sizes)}")
    print(f"[Cloud Server] 性能分數: {[f'{p:.4f}' for p in all_performance_scores]}")
    
    if global_weights:
        prev_weight_stats = {}
        for layer_name, layer_weight in global_weights.items():
            if isinstance(layer_weight, (torch.Tensor, np.ndarray)):
                weight_array = layer_weight.cpu().numpy() if isinstance(layer_weight, torch.Tensor) else layer_weight
                prev_weight_stats[layer_name] = {
                    'mean': float(np.mean(weight_array)),
                    'std': float(np.std(weight_array)),
                    'norm': float(np.linalg.norm(weight_array))
                }
        print(f"[Cloud Server] 聚合前全局權重統計: {len(prev_weight_stats)} 層")
        if prev_weight_stats:
            sample_layer = list(prev_weight_stats.keys())[0]
            print(f"[Cloud Server]   範例層 ({sample_layer}): mean={prev_weight_stats[sample_layer]['mean']:.6f}, std={prev_weight_stats[sample_layer]['std']:.6f}, norm={prev_weight_stats[sample_layer]['norm']:.6f}")
    
    aggregated_client_weights = aggregate_client_weights(all_weights, all_data_sizes, all_performance_scores)
    
    if aggregated_client_weights:
        agg_weight_stats = {}
        for layer_name, layer_weight in aggregated_client_weights.items():
            if isinstance(layer_weight, (torch.Tensor, np.ndarray)):
                weight_array = layer_weight.cpu().numpy() if isinstance(layer_weight, torch.Tensor) else layer_weight
                agg_weight_stats[layer_name] = {
                    'mean': float(np.mean(weight_array)),
                    'std': float(np.std(weight_array)),
                    'norm': float(np.linalg.norm(weight_array))
                }
        print(f"[Cloud Server] 聚合後權重統計: {len(agg_weight_stats)} 層")
        if agg_weight_stats:
            sample_layer = list(agg_weight_stats.keys())[0]
            print(f"[Cloud Server]   範例層 ({sample_layer}): mean={agg_weight_stats[sample_layer]['mean']:.6f}, std={agg_weight_stats[sample_layer]['std']:.6f}, norm={agg_weight_stats[sample_layer]['norm']:.6f}")
        
        if global_weights and prev_weight_stats:
            weight_changes = {}
            for layer_name in agg_weight_stats.keys():
                if layer_name in prev_weight_stats:
                    prev_norm = prev_weight_stats[layer_name]['norm']
                    curr_norm = agg_weight_stats[layer_name]['norm']
                    norm_change = abs(curr_norm - prev_norm)
                    relative_change = norm_change / (prev_norm + 1e-8)
                    weight_changes[layer_name] = {
                        'abs_change': norm_change,
                        'relative_change': relative_change
                    }
            if weight_changes:
                avg_abs_change = np.mean([w['abs_change'] for w in weight_changes.values()])
                avg_rel_change = np.mean([w['relative_change'] for w in weight_changes.values()])
                print(f"[Cloud Server] 權重變化統計: 平均絕對變化={avg_abs_change:.6f}, 平均相對變化={avg_rel_change:.6f}")
                if avg_abs_change < 1e-6:
                    print(f"[Cloud Server] 警告：權重變化極小，可能未有效聚合")
    
    aggregated_server_state = None
    if all_server_states:
        aggregated_server_state = aggregate_server_states(all_server_states)
    
    prev_global_weights = global_weights.copy() if global_weights else None
    aggregated_smoothed = _apply_server_ema(aggregated_client_weights, prev_global_weights)
    
    if aggregated_smoothed:
        ema_weight_stats = {}
        for layer_name, layer_weight in aggregated_smoothed.items():
            if isinstance(layer_weight, (torch.Tensor, np.ndarray)):
                weight_array = layer_weight.cpu().numpy() if isinstance(layer_weight, torch.Tensor) else layer_weight
                ema_weight_stats[layer_name] = {
                    'mean': float(np.mean(weight_array)),
                    'std': float(np.std(weight_array)),
                    'norm': float(np.linalg.norm(weight_array))
                }
        print(f"[Cloud Server] EMA 平滑後權重統計: {len(ema_weight_stats)} 層")
        if ema_weight_stats:
            sample_layer = list(ema_weight_stats.keys())[0]
            print(f"[Cloud Server]   範例層 ({sample_layer}): mean={ema_weight_stats[sample_layer]['mean']:.6f}, std={ema_weight_stats[sample_layer]['std']:.6f}, norm={ema_weight_stats[sample_layer]['norm']:.6f}")
    
    norm_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('weight_norm_regularization', {})
    if norm_cfg.get('enabled', True) and aggregated_smoothed and len(aggregated_smoothed) > 0:
        hard_limit = float(norm_cfg.get('hard_limit', 200.0))
        max_norm = float(norm_cfg.get('max_global_l2_norm', 150.0))
        scaling_factor = float(norm_cfg.get('scaling_factor', 0.90))
        
        current_norm = _compute_global_l2_norm(aggregated_smoothed)
        
        if current_norm > hard_limit:
            print(f"[Cloud Server]  perform_global_aggregation: 檢測到權重範數超過硬性上限 ({current_norm:.4f} > {hard_limit:.4f})，強制裁剪")
            aggregated_smoothed = _apply_weight_norm_regularization(
                aggregated_smoothed, max_norm, scaling_factor, 
                hard_limit=hard_limit, strict_enforcement=True
            )
            new_norm = _compute_global_l2_norm(aggregated_smoothed)
            if new_norm > hard_limit:
                scale = hard_limit / new_norm
                for layer_name in aggregated_smoothed:
                    w = _coerce_tensor(aggregated_smoothed[layer_name])
                    if isinstance(w, torch.Tensor) and w.dtype in (torch.float32, torch.float64, torch.float16):
                        aggregated_smoothed[layer_name] = w * scale
                new_norm = _compute_global_l2_norm(aggregated_smoothed)
            print(f"[Cloud Server] perform_global_aggregation: 強制裁剪後權重範數: {new_norm:.4f} (目標≤{hard_limit:.4f})")
        elif current_norm > max_norm:
            print(f"[Cloud Server] perform_global_aggregation: 檢測到權重範數超過上限 ({current_norm:.4f} > {max_norm:.4f})，應用正則化")
            aggregated_smoothed = _apply_weight_norm_regularization(
                aggregated_smoothed, max_norm, scaling_factor, 
                hard_limit=hard_limit, strict_enforcement=True
            )
            new_norm = _compute_global_l2_norm(aggregated_smoothed)
            print(f"[Cloud Server] perform_global_aggregation: 正則化後權重範數: {new_norm:.4f} (目標≤{max_norm:.4f})")
    
    global_weights = aggregated_smoothed
    
    print(f"[Cloud Server] 全局聚合完成")
    
    return global_weights

def aggregate_client_weights(weights_list, data_sizes, performance_scores=None):
    import torch
    
    if not weights_list:
        return {}
    
    agg_method_cfg = getattr(config, 'AGGREGATION_CONFIG', {}).get('aggregation_method', {})
    agg_method_type = agg_method_cfg.get('type', 'weighted')
    trim_ratio = float(agg_method_cfg.get('trim_ratio', 0.2))
    
    print(f"[Cloud Server] 聚合方法配置: type={agg_method_type}, trim_ratio={trim_ratio}")
    
    if agg_method_type == 'median':
        print(f"[Cloud Server] 使用中位數聚合（FedMedian）")
        return _aggregate_weights_median(weights_list)
    elif agg_method_type == 'trimmed':
        print(f"[Cloud Server] 使用修剪平均聚合（Trimmed Mean），trim_ratio={trim_ratio}")
        return _aggregate_weights_trimmed_mean(weights_list, trim_ratio)
    
    total_data_size = sum(data_sizes)

    strategy_config = getattr(config, 'AGGREGATION_STRATEGY', {})
    strategy_type = strategy_config.get('type', 'weighted')
    data_weight_ratio = strategy_config.get('data_weight', 0.7)
    performance_weight_ratio = strategy_config.get('performance_weight', 0.3)
    min_performance = strategy_config.get('min_performance', 0.1)
    
    if total_data_size == 0:
        print(f"[Cloud Server] 總數據量為0，使用平均聚合")

        weight_factors = [1.0 / len(data_sizes)] * len(data_sizes)
    else:
        if strategy_type == 'weighted' or performance_scores is None:
            weight_factors = [size / total_data_size for size in data_sizes]
        elif strategy_type == 'performance' and performance_scores:
            max_perf = max(performance_scores)
            min_perf = min(performance_scores)
            if max_perf > min_perf:
                normalized_perfs = [(p - min_perf) / (max_perf - min_perf) for p in performance_scores]
            else:
                normalized_perfs = [1.0] * len(performance_scores)
            normalized_perfs = [max(min_performance, p) for p in normalized_perfs]
            total_perf = sum(normalized_perfs)
            weight_factors = [p / total_perf if total_perf > 0 else 1.0 / len(performance_scores) 
                              for p in normalized_perfs]
        else:
            data_weights = [size / total_data_size for size in data_sizes]

            if performance_scores:
                max_perf = max(performance_scores)
                min_perf = min(performance_scores)
                if max_perf > min_perf:
                    perf_weights = [(max(min_performance, (p - min_perf) / (max_perf - min_perf))) 
                                  for p in performance_scores]
                else:
                    perf_weights = [1.0] * len(performance_scores)
                total_perf = sum(perf_weights)
                perf_weights = [p / total_perf if total_perf > 0 else 1.0 / len(performance_scores) 
                               for p in perf_weights]
            else:
                perf_weights = [1.0 / len(data_sizes)] * len(data_sizes)

            weight_factors = [data_weight_ratio * dw + performance_weight_ratio * pw 
                           for dw, pw in zip(data_weights, perf_weights)]

            total_weight = sum(weight_factors)
            weight_factors = [w / total_weight if total_weight > 0 else 1.0 / len(weight_factors) 
                            for w in weight_factors]
    
    print(f"[Cloud Server] 聚合策略: {strategy_type}")
    print(f"[Cloud Server] 權重因子: {[f'{w:.3f}' for w in weight_factors]}")
    print(f"[Cloud Server] 數據大小: {data_sizes}")
    print(f"[Cloud Server] 總數據大小: {total_data_size}")
    if performance_scores:
        print(f"[Cloud Server] 性能分數: {[f'{p:.3f}' for p in performance_scores]}")

    aggregated_weights = {}
    first_weights = weights_list[0]
    
    layer_stats_before = {}
    for layer_name in first_weights.keys():
        if isinstance(first_weights[layer_name], torch.Tensor):
            layer_stats_before[layer_name] = {
                'norm': first_weights[layer_name].norm().item(),
                'mean': first_weights[layer_name].mean().item(),
                'std': first_weights[layer_name].std().item()
            }
    
    for layer_name in first_weights.keys():
        first_layer = first_weights[layer_name]
    
        if isinstance(first_layer, torch.Tensor):
            if first_layer.dtype != torch.float32 and first_layer.dtype != torch.float64:
                print(f"[Cloud Server] 權重層 {layer_name} 類型為 {first_layer.dtype}，轉換為 float32")
                first_layer = first_layer.float()
            
            aggregated_layer = torch.zeros_like(first_layer, dtype=torch.float32)
        elif isinstance(first_layer, np.ndarray):

            first_layer_t = torch.from_numpy(first_layer).float()
            aggregated_layer = torch.zeros_like(first_layer_t, dtype=torch.float32)
        else:
            aggregated_layer = 0.0
        
        client_layer_norms = []
        client_layer_means = []
        client_layer_stds = []
        weighted_contributions = []
        for i, client_weights in enumerate(weights_list):
            weight_factor = weight_factors[i]
            client_layer = client_weights[layer_name]
            
            if isinstance(client_layer, torch.Tensor):
                if client_layer.dtype != torch.float32 and client_layer.dtype != torch.float64:
                    print(f"[Cloud Server] 客戶端 {i} 權重層 {layer_name} 類型為 {client_layer.dtype}，轉換為 float32")
                    client_layer = client_layer.float()
                
                client_layer_norms.append(client_layer.norm().item())
                client_layer_means.append(client_layer.mean().item())
                client_layer_stds.append(client_layer.std().item())
                
                weight_tensor = torch.tensor(weight_factor, dtype=torch.float32, device=client_layer.device)
                aggregated_layer += weight_tensor * client_layer
            elif isinstance(client_layer, np.ndarray):
                client_layer_t = torch.from_numpy(client_layer).float()
                client_layer_norms.append(client_layer_t.norm().item())
                client_layer_means.append(client_layer_t.mean().item())
                client_layer_stds.append(client_layer_t.std().item())
                weight_tensor = torch.tensor(weight_factor, dtype=torch.float32)
                aggregated_layer += weight_tensor * client_layer_t
            elif isinstance(client_layer, list):
                try:
                    client_layer_t = torch.tensor(client_layer, dtype=torch.float32)
                    client_layer_norms.append(client_layer_t.norm().item())
                    weight_tensor = torch.tensor(weight_factor, dtype=torch.float32)
                    aggregated_layer += weight_tensor * client_layer_t
                except Exception:
                    pass
            else:
                try:
                    aggregated_layer += weight_factor * float(client_layer)
                except Exception:
                    try:
                        client_layer_t = torch.tensor(client_layer, dtype=torch.float32)
                        weight_tensor = torch.tensor(weight_factor, dtype=torch.float32)
                        aggregated_layer += weight_tensor * client_layer_t
                    except Exception:
                        continue
        
        if isinstance(aggregated_layer, torch.Tensor):
            layer_norm_after = aggregated_layer.norm().item()
            layer_mean_after = aggregated_layer.mean().item()
            layer_std_after = aggregated_layer.std().item()
            
            if layer_name in layer_stats_before:
                layer_norm_before = layer_stats_before[layer_name]['norm']
                if layer_norm_before > 0:
                    scale_ratio = layer_norm_after / layer_norm_before
                    if scale_ratio < 0.1 or scale_ratio > 10:
                        print(f"[Cloud Server] 警告：層 {layer_name} 權重範數變化異常 (前={layer_norm_before:.4f}, 後={layer_norm_after:.4f}, 比例={scale_ratio:.4f})")
            
            if abs(layer_mean_after) > 10 or layer_std_after > 10:
                print(f"[Cloud Server] 警告：層 {layer_name} 權重值異常 (mean={layer_mean_after:.4f}, std={layer_std_after:.4f})")
            
            if client_layer_norms:
                min_norm = min(client_layer_norms)
                max_norm = max(client_layer_norms)
                if max_norm > 0 and (max_norm / min_norm) > 100:
                    print(f"[Cloud Server] 警告：層 {layer_name} 客戶端權重範數差異過大 (min={min_norm:.4f}, max={max_norm:.4f}, 比例={max_norm/min_norm:.2f})")
        
        aggregated_weights[layer_name] = aggregated_layer
    
    if aggregated_weights:
        print(f"[Cloud Server] 權重聚合完成統計:")
        print(f"  - 總層數: {len(aggregated_weights)}")
        if client_layer_norms:
            print(f"  - 客戶端權重範數範圍: [{min(client_layer_norms):.4f}, {max(client_layer_norms):.4f}]")
            print(f"  - 客戶端權重範數平均: {np.mean(client_layer_norms):.4f}")
        if client_layer_means:
            print(f"  - 客戶端權重均值範圍: [{min(client_layer_means):.4f}, {max(client_layer_means):.4f}]")
        if client_layer_stds:
            print(f"  - 客戶端權重標準差範圍: [{min(client_layer_stds):.4f}, {max(client_layer_stds):.4f}]")
        
        sample_layer = list(aggregated_weights.keys())[0]
        if isinstance(aggregated_weights[sample_layer], torch.Tensor):
            sample_norm = aggregated_weights[sample_layer].norm().item()
            sample_mean = aggregated_weights[sample_layer].mean().item()
            sample_std = aggregated_weights[sample_layer].std().item()
            print(f"  - 範例層 ({sample_layer}): norm={sample_norm:.4f}, mean={sample_mean:.4f}, std={sample_std:.4f}")
    
    return aggregated_weights

def aggregate_server_states(server_states):
    import torch
    
    if not server_states:
        return None
    
    aggregated_state = {}
    first_state = server_states[0]
    
    for key in first_state.keys():
        first_param = first_state[key]
        
        if isinstance(first_param, torch.Tensor):
            if 'num_batches_tracked' in key:
                aggregated_param = torch.zeros_like(first_param, dtype=torch.float32)
            else:
                if first_param.dtype != torch.float32 and first_param.dtype != torch.float64:
                    print(f"[Cloud Server] 服務器狀態 {key} 類型為 {first_param.dtype}，轉換為 float32")
                    first_param = first_param.float()
                
                aggregated_param = torch.zeros_like(first_param, dtype=torch.float32)
        else:
            aggregated_param = 0.0
        
        for state in server_states:
            param = state[key]
            
            if isinstance(param, torch.Tensor):
                if 'num_batches_tracked' in key:
                    param = param.float()
                    aggregated_param += param
                else:
                    if param.dtype != torch.float32 and param.dtype != torch.float64:
                        param = param.float()
                    
                    aggregated_param += param
            else:
                aggregated_param += float(param)
        
        if isinstance(aggregated_param, torch.Tensor):
            aggregated_param /= len(server_states)
            if 'num_batches_tracked' in key:
                aggregated_param = _clamp_bn_tracker_tensor(aggregated_param)
                aggregated_param = aggregated_param.long()
        else:
            aggregated_param /= len(server_states)
        
        aggregated_state[key] = aggregated_param
    
    return aggregated_state

def log_cloud_status():
    global should_stop_cloud_logging
    while not should_stop_cloud_logging:
        time.sleep(60)
        try:
            print(f"[Cloud Server] 狀態報告 - 聚合次數: {aggregation_count}, 聚合器數量: {aggregator_count}")
            log_event("status_report", f"aggregation_count={aggregation_count},aggregator_count={aggregator_count}")
            
        except Exception as e:
            print(f"[Cloud Server] 狀態報告錯誤: {e}")
    
    print(f"[Cloud Server] 狀態報告線程已停止")

async def _immediate_broadcast_global_weights(round_id: int):
    global global_weights, global_version, registered_aggregators
    
    if global_weights is None or len(global_weights) == 0:
        print(f"[Cloud Server] 全局權重為空，跳過立即廣播")
        return

    _schedule_global_test_eval(round_id, global_weights)
    
    print(f"[Cloud Server]  開始立即廣播全局權重 (輪次: {round_id}, 版本: {global_version})")
    
    broadcast_success = 0
    broadcast_failed = 0
    
    for agg_id, agg_info in registered_aggregators.items():
        try:
            agg_url = f"http://{agg_info['host']}:{agg_info['port']}"
            
            broadcast_data = {
                'global_weights': global_weights,
                'server_weights': global_weights,
                'global_version': global_version,
                'timestamp': time.time(),
                'broadcast_type': 'immediate',
                'round_id': round_id
            }
            
            weights_bytes = pickle.dumps(broadcast_data)
            
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('weights', weights_bytes, filename='global_weights.pkl')
                data.add_field('global_version', str(global_version))
                data.add_field('broadcast_type', 'immediate')
                data.add_field('round_id', str(round_id))
                
                async with session.post(f"{agg_url}/receive_global_weights", data=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        broadcast_success += 1
                        print(f"[Cloud Server] 立即廣播成功給聚合器 {agg_id}")
                    else:
                        broadcast_failed += 1
                        print(f"[Cloud Server] 立即廣播給聚合器 {agg_id} 失敗: HTTP {resp.status}")
                        
        except asyncio.TimeoutError:
            broadcast_failed += 1
            print(f"[Cloud Server] 立即廣播給聚合器 {agg_id} 超時")
        except Exception as e:
            broadcast_failed += 1
            print(f"[Cloud Server] 立即廣播給聚合器 {agg_id} 異常: {e}")
    
    print(f"[Cloud Server] 立即廣播完成: 成功={broadcast_success}, 失敗={broadcast_failed} (輪次: {round_id})")

async def broadcast_global_weights():
    import asyncio
    global should_stop_cloud_logging
    
    if aiohttp is None:
        print(f"[Cloud Server] 廣播功能不可用：aiohttp 未安裝")
        return
    
    broadcast_interval = 300
    
    while not should_stop_cloud_logging:
        try:
            await asyncio.sleep(broadcast_interval)
            
            if global_weights is None or len(global_weights) == 0:
                print(f"[Cloud Server] 全局權重為空，跳過廣播")
                continue
            
            current_round = getattr(app.state, 'current_round', None) or getattr(app.state, 'last_aggregation_round', None) or 0

            _schedule_global_test_eval(current_round, global_weights)
            
            print(f"[Cloud Server] 開始定期廣播全局權重 (版本: {global_version}, 輪次: {current_round})")
        
            broadcast_success = 0
            broadcast_failed = 0
            
            for agg_id, agg_info in registered_aggregators.items():
                try:
                    agg_url = f"http://{agg_info['host']}:{agg_info['port']}"
                    
                    broadcast_data = {
                        'global_weights': global_weights,
                        'server_weights': global_weights,
                        'global_version': global_version,
                        'timestamp': time.time(),
                        'broadcast_type': 'periodic',
                        'round_id': current_round
                    }
                    
                    weights_bytes = pickle.dumps(broadcast_data)
                    
                    async with aiohttp.ClientSession() as session:
                        data = aiohttp.FormData()
                        data.add_field('weights', weights_bytes, filename='global_weights.pkl')
                        data.add_field('global_version', str(global_version))
                        data.add_field('broadcast_type', 'periodic')
                        data.add_field('round_id', str(current_round))
                        
                        async with session.post(f"{agg_url}/receive_global_weights", data=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            if resp.status == 200:
                                broadcast_success += 1
                                print(f"[Cloud Server] 成功廣播給聚合器 {agg_id}")
                            else:
                                broadcast_failed += 1
                                print(f"[Cloud Server] 廣播給聚合器 {agg_id} 失敗: HTTP {resp.status}")
                                
                except asyncio.TimeoutError:
                    broadcast_failed += 1
                    print(f"[Cloud Server] 廣播給聚合器 {agg_id} 超時")
                except Exception as e:
                    broadcast_failed += 1
                    print(f"[Cloud Server] 廣播給聚合器 {agg_id} 異常: {e}")
            
            log_event("global_weights_broadcast", f"success={broadcast_success},failed={broadcast_failed},version={global_version}")
            print(f"[Cloud Server] 廣播完成: 成功={broadcast_success}, 失敗={broadcast_failed}")
            
        except asyncio.CancelledError:
            print(f"[Cloud Server] 廣播線程被取消")
            break
        except Exception as e:
            print(f"[Cloud Server] 廣播全局權重異常: {e}")
            import traceback
            traceback.print_exc()
            log_event("broadcast_error", str(e))

            await asyncio.sleep(60)
    
    print(f"[Cloud Server] 廣播線程已停止")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Cloud Server for Federated Learning')
    parser.add_argument('--port', type=int, default=None, help='Port to run the server on')
    parser.add_argument('--host', type=str, default=None, help='Host to bind the server to')
    args = parser.parse_args()
    
    server_port = args.port if args.port is not None else config.NETWORK_CONFIG['cloud_server']['port']
    server_host = args.host if args.host is not None else config.NETWORK_CONFIG['cloud_server']['host']
    
    experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
    if experiment_dir:
        result_dir = experiment_dir
        print(f"[Cloud Server] 使用環境變量實驗目錄: {result_dir}")
    else:
        import datetime
        now = datetime.datetime.now().strftime("tokyo_drone_fixed_%Y%m%d_%H%M%S")
        result_dir = os.path.join(config.LOG_DIR, now)
        os.makedirs(result_dir, exist_ok=True)
        print(f"[Cloud Server] 創建新實驗目錄: {result_dir}")
    
    os.makedirs(result_dir, exist_ok=True)
    
    os.environ['EXPERIMENT_DIR'] = result_dir
    os.environ['LOG_DIR'] = result_dir
    
    print(f"[Cloud Server]  啟動中...")
    print(f"  服務地址: http://{server_host}:{server_port}")
    print(f"  職責: 全局權重聚合")
    print(f"  使用實驗目錄: {result_dir}")
    print(f"  日誌格式: {config.LOG_CONFIG.get('result_log_format', 'csv')}")
    
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((server_host, server_port))
        sock.close()
        if result == 0:
            print(f"[Cloud Server] 警告：端口 {server_port} 已被占用，可能導致啟動失敗")
            print(f"[Cloud Server] 建議：檢查是否有其他 Cloud Server 實例正在運行")
    except Exception as e:
        print(f"[Cloud Server] 端口檢查失敗: {e}")
    
    marker_file = os.path.join(result_dir, "cloud_server_using.txt")
    try:
        with open(marker_file, "w", encoding="utf-8") as f:
            f.write(f"雲端服務器使用此實驗目錄\n")
            f.write(f"使用時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"雲端服務器ID: {cloud_server_id}\n")
            f.write(f"目錄路徑: {result_dir}\n")
            f.write(f"服務地址: http://{server_host}:{server_port}\n")
    except Exception as e:
        print(f"[Cloud Server] 無法創建標記文件: {e}")
    
    threading.Thread(target=log_cloud_status, daemon=True).start()
    
    def run_broadcast():
        try:
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:

                task = loop.create_task(broadcast_global_weights())

                loop.run_forever()
            except Exception as e:
                print(f"[Cloud Server] 廣播線程異常: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Cloud Server] 廣播線程啟動失敗: {e}")
            import traceback
            traceback.print_exc()
    
    broadcast_thread = threading.Thread(target=run_broadcast, daemon=True)
    broadcast_thread.start()
    print(f"[Cloud Server] 啟動定期廣播線程（每5分鐘廣播一次全局權重）")

    print(f"[Cloud Server] 啟動完成!")
    print(f"[Cloud Server] 📁 實驗目錄: {result_dir}")
    print(f"[Cloud Server] 其他組件將自動搜尋並使用此目錄")
    
    try:
        uvicorn.run(
            app, 
            host=server_host, 
            port=server_port,
            access_log=True,
            log_level="info"
        )
    except OSError as e:
        if "Address already in use" in str(e) or "address is already in use" in str(e).lower():
            print(f"[Cloud Server] 端口 {server_port} 已被占用，無法啟動")
            print(f"[Cloud Server] 解決方案：")
            print(f"  1. 檢查是否有其他 Cloud Server 實例正在運行")
            print(f"  2. 使用不同的端口：python cloud_server_fixed.py --port <其他端口>")
            print(f"  3. 終止占用端口的進程：lsof -ti:{server_port} | xargs kill -9")
        else:
            print(f"[Cloud Server] 啟動失敗: {e}")
        raise
    except Exception as e:
        print(f"[Cloud Server] 啟動異常: {e}")
        import traceback
        traceback.print_exc()
        raise 