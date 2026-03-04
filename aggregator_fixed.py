                      
                       
"""
聚合器 - 修復版本
專注於解決權重分發和輪次管理問題
"""

import asyncio
import json
import pickle
import time
import traceback
import argparse
import sys
import os
from typing import Dict, List, Any, Optional
import datetime
import math

                 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
import uvicorn
import aiohttp

                                            
import importlib
CONFIG_MODULE = os.environ.get("CONFIG_MODULE", "config_fixed")
try:
    config = importlib.import_module(CONFIG_MODULE)
    print(f"[Aggregator]  使用配置模組: {CONFIG_MODULE}")
except Exception as e:
    print(f"[Aggregator] 載入配置模組 {CONFIG_MODULE} 失敗，回退到 config_fixed: {e}")
    import config_fixed as config

                   
from models.dnn import NetworkAttackDNN

                
try:
    from models.regional_aggregation import RegionalAggregator
    REGIONAL_AGGREGATION_AVAILABLE = True
    print("[Aggregator]  區域性聚合模組已載入")
except ImportError as e:
    REGIONAL_AGGREGATION_AVAILABLE = False
    print(f"[Aggregator] 區域性聚合模組載入失敗: {e}，將使用標準 FedAvg")

      
                               
app = FastAPI(title="Federated Learning Aggregator", version="2.0")

@app.on_event("startup")
async def startup_event():
    try:
                       
        asyncio.create_task(periodic_round_sync_check())
        
                                       
        async def delayed_register():
            await asyncio.sleep(3)               
            await register_with_cloud_server()
        
        asyncio.create_task(delayed_register())
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] 啟動註冊任務失敗: {e}")

       
aggregator_id = 0
round_count = 0
last_completed_round = 0
round_clients = []
client_weights_buffer = {}
client_data_sizes = {}
                    
client_predictions_buffer = {}
round_start_time = None
client_timeout_status = {}
global_weights = None
server_global_weights = None
model_version = 1
server_model_version = -1
is_training_phase = False
aggregator_port = 8000

         
def initialize_global_weights():
    global global_weights
    if global_weights is None:
        print(f"[Aggregator {aggregator_id}] 初始化全局權重...")
                       
        input_dim = int(config.MODEL_CONFIG.get('input_dim', 22))
        num_classes = int(config.MODEL_CONFIG.get('output_dim', config.MODEL_CONFIG.get('num_classes', 5)))
                  
        try:
            if input_dim <= 0:
                data_dir = getattr(config, 'DATA_PATH', '')
                sample_csv = None
                if os.path.isdir(data_dir):
                    for fn in os.listdir(data_dir):
                        if fn.endswith('.csv'):
                            sample_csv = os.path.join(data_dir, fn)
                            break
                if sample_csv:
                    import pandas as pd
                    df_head = pd.read_csv(sample_csv, nrows=1)
                    label_col = getattr(config, 'LABEL_COL', 'label')
                    if label_col in df_head.columns:
                        input_dim = max(1, len(df_head.columns) - 1)
                    else:
                        input_dim = max(1, len(df_head.columns))
        except Exception:
            pass
        
                                           
        model_type = config.MODEL_CONFIG.get('type', 'dnn')
        dropout_rate = float(config.MODEL_CONFIG.get('dropout_rate', 0.3))
        
        if model_type == 'transformer':
                                    
            from models.transformer import build_transformer
            model = build_transformer(
                input_dim=input_dim,
                output_dim=num_classes,
                d_model=config.MODEL_CONFIG.get('d_model', 128),
                num_layers=config.MODEL_CONFIG.get('num_layers', 2),
                num_heads=config.MODEL_CONFIG.get('num_heads', 4),
                d_ff=config.MODEL_CONFIG.get('d_ff', None),
                dropout=dropout_rate,
                max_seq_len=config.MODEL_CONFIG.get('max_seq_len', input_dim),
                use_positional_encoding=config.MODEL_CONFIG.get('use_positional_encoding', True)
            )
            print(f"[Aggregator {aggregator_id}]  使用 Transformer 模型初始化全局權重")
        elif model_type == 'dnn':
                         
            from models.dnn import build_dnn
            model = build_dnn(
                input_dim=input_dim,
                output_dim=num_classes,
                hidden_dims=config.MODEL_CONFIG.get('hidden_dims', [256, 128, 64]),
                dropout_rate=dropout_rate,
                use_batch_norm=config.MODEL_CONFIG.get('use_batch_norm', True),
                use_residual=config.MODEL_CONFIG.get('use_residual', True),
                activation=config.MODEL_CONFIG.get('activation', 'relu')
            )
            print(f"[Aggregator {aggregator_id}] 使用 DNN 模型初始化全局權重")
        elif model_type == 'cnn':
                       
            from models.cnn import build_cnn
            model = build_cnn(input_dim=input_dim, output_dim=num_classes)
            print(f"[Aggregator {aggregator_id}] 使用 CNN 模型初始化全局權重")
        else:
                     
            from models.dnn import build_dnn
            model = build_dnn(
                input_dim=input_dim,
                output_dim=num_classes,
                hidden_dims=config.MODEL_CONFIG.get('hidden_dims', [256, 128, 64]),
                dropout_rate=dropout_rate
            )
            print(f"[Aggregator {aggregator_id}] 未知模型類型 {model_type}，回退到 DNN")
        
        global_weights = model.state_dict()
        print(f"[Aggregator {aggregator_id}] 全局權重初始化完成，權重層數: {len(global_weights)}")
    return global_weights

def reset_global_weights():
    global global_weights
    print(f"[Aggregator {aggregator_id}] 強制重置全局權重...")
    global_weights = None
    return initialize_global_weights()

    
federated_config = config.FEDERATED_CONFIG
AGGREGATION_TIMEOUT = int(getattr(config, 'AGGREGATION_CONFIG', {}).get('max_wait_time', 120))
PARTIAL_AGGREGATION_ENABLED = bool(getattr(config, 'AGGREGATION_CONFIG', {}).get('partial_aggregation_enabled', True))
MIN_PARTIAL_RATIO = float(getattr(config, 'AGGREGATION_CONFIG', {}).get('min_partial_ratio', 0.3))

                             
AGG_CFG = getattr(config, 'AGGREGATION_CONFIG', {})
STALE_POLICY = str(AGG_CFG.get('stale_policy', 'allow'))                                
MAX_STALENESS = int(AGG_CFG.get('max_staleness', 1))
STALENESS_DECAY_LAMBDA = float(AGG_CFG.get('staleness_decay_lambda', 0.7))
DATASIZE_ALPHA_MAX_MULTIPLIER = float(AGG_CFG.get('alpha_max_multiplier', 3.0))                            

ROBUST_CFG = AGG_CFG.get('robust', {}) or {}
ROBUST_METHOD = str(ROBUST_CFG.get('method', 'none'))                                      
ROBUST_TRIM_RATIO = float(ROBUST_CFG.get('trim_ratio', 0.2))

CLIP_CFG = AGG_CFG.get('clip', {}) or {}
CLIP_NORM = float(CLIP_CFG.get('norm', 0.0))                

                               
NORM_GUARD_CFG = AGG_CFG.get('norm_guard', {}) or {}
NORM_GUARD_ENABLED = bool(NORM_GUARD_CFG.get('enabled', True))
NORM_GUARD_K = float(NORM_GUARD_CFG.get('k', 3.5))
NORM_GUARD_PENALTY = float(NORM_GUARD_CFG.get('penalty_factor', 0.15))

TRUST_CFG = AGG_CFG.get('trust', {}) or {}
TRUST_ENABLED = bool(TRUST_CFG.get('enabled', True))
TRUST_DECAY = float(TRUST_CFG.get('decay', 0.9))
TRUST_GAIN = float(TRUST_CFG.get('gain', 0.01))

                
CLOUD_SERVER_CONFIG = {
    "enabled": True,
    "url": config.NETWORK_CONFIG['cloud_server']['url'],
    "upload_after_aggregation": True,
    "timeout": 30
}

          
simplified_client_selector = None
global_performance = 0.5

        
server_model = None
optimizer = None
criterion = None

      
total_aggregations_done = 0

     
buffer_lock = asyncio.Lock()
aggregation_lock = asyncio.Lock()

        
simplified_aggregation_strategy = None

       
should_stop_logging = False

def log_event(event, detail=""):
    print(f"[Aggregator {aggregator_id}]  {event}: {detail}")

def save_aggregation_stats():
    stats = {
        "aggregator_id": aggregator_id,
        "round_count": round_count,
        "current_clients": len(round_clients),
        "buffer_size": len(client_weights_buffer),
        "model_version": model_version,
        "is_training_phase": is_training_phase
    }
    print(f"[Aggregator {aggregator_id}] 統計: {stats}")

async def register_with_cloud_server():
    max_retries = 5
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            if not CLOUD_SERVER_CONFIG.get("enabled", False):
                print(f"[Aggregator {aggregator_id}]  Cloud Server 連接已禁用")
                return
            
            cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
            if not cloud_url:
                print(f"[Aggregator {aggregator_id}] Cloud Server URL 未配置")
                return
            
            print(f"[Aggregator {aggregator_id}]  嘗試連接 Cloud Server (嘗試 {attempt + 1}/{max_retries})")
            
            data = aiohttp.FormData()
            data.add_field('aggregator_id', str(aggregator_id))
            data.add_field('status', 'ready')
            data.add_field('port', str(aggregator_port))
            
            timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{cloud_url}/register_aggregator", data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"[Aggregator {aggregator_id}]  已向Cloud Server註冊成功")
                        print(f"  - 響應: {result}")
                        log_event("cloud_register_success", f"status={resp.status}, attempt={attempt + 1}")
                        return             
                    else:
                        text = await resp.text()
                        print(f"[Aggregator {aggregator_id}] Cloud 註冊失敗: HTTP {resp.status}")
                        print(f"  - 錯誤詳情: {text}")
                        log_event("cloud_register_failed", f"status={resp.status}, attempt={attempt + 1}, error={text}")
                        
        except asyncio.TimeoutError:
            print(f"[Aggregator {aggregator_id}] ⏰ Cloud 註冊超時 (嘗試 {attempt + 1}/{max_retries})")
            log_event("cloud_register_timeout", f"attempt={attempt + 1}")
        except aiohttp.ClientConnectorError as e:
            print(f"[Aggregator {aggregator_id}] 🔌 Cloud 連接失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
            log_event("cloud_register_connection_error", f"attempt={attempt + 1}, error={str(e)}")
        except Exception as e:
            print(f"[Aggregator {aggregator_id}] Cloud 註冊異常 (嘗試 {attempt + 1}/{max_retries}): {e}")
            log_event("cloud_register_exception", f"attempt={attempt + 1}, error={str(e)}")
        
                      
        delay = base_delay * (2 ** attempt)
        print(f"[Aggregator {aggregator_id}]  等待 {delay:.1f} 秒後重試...")
        await asyncio.sleep(delay)
    
             
    print(f"[Aggregator {aggregator_id}] Cloud 註冊失敗，已嘗試 {max_retries} 次")
    log_event("cloud_register_final_failure", f"max_retries={max_retries}")
    
                                        
    print(f"[Aggregator {aggregator_id}] 允許聚合器在無Cloud Server模式下繼續運行")
    log_event("cloud_server_disabled_fallback", "聚合器將在本地模式下運行")
    
               
    try:
        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
        print(f"[Aggregator {aggregator_id}] 診斷信息:")
        print(f"  - Cloud URL: {cloud_url}")
        print(f"  - 聚合器端口: {aggregator_port}")
        print(f"  - 超時設置: {CLOUD_SERVER_CONFIG.get('timeout', 30)}s")
        print(f"  - 啟用狀態: {CLOUD_SERVER_CONFIG.get('enabled', False)}")
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] 診斷信息收集失敗: {e}")

async def upload_aggregated_weights_to_cloud(round_id_value: int, weights_state: Dict[str, Any], participating_clients: List[int]):
    try:
        if not CLOUD_SERVER_CONFIG.get("enabled", False) or not CLOUD_SERVER_CONFIG.get("upload_after_aggregation", False):
            return
        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
        if not cloud_url:
            return
                                   
        aggregated_weights_payload = {}
        for key, value in (weights_state or {}).items():
            try:
                if isinstance(value, torch.Tensor):
                    if value.device.type != 'cpu':
                        value_cpu = value.detach().cpu()
                    else:
                        value_cpu = value.detach()
                    aggregated_weights_payload[key] = value_cpu.numpy()
                elif isinstance(value, np.ndarray):
                    aggregated_weights_payload[key] = value
                else:
                                      
                    aggregated_weights_payload[key] = np.array(value, dtype=np.float32)
            except Exception:
                             
                continue

        upload_dict = {
            'aggregated_weights': aggregated_weights_payload,
            'aggregation_stats': {
                'num_layers': len(aggregated_weights_payload),
                'timestamp': time.time(),
                'participating_clients': participating_clients,
            }
        }
        weights_bytes = pickle.dumps(upload_dict)
        data = aiohttp.FormData()
        data.add_field('aggregator_id', str(aggregator_id))
        data.add_field('round_id', str(round_id_value))
        data.add_field('model_version', str(model_version))
        data.add_field('participating_clients', json.dumps(participating_clients))
        data.add_field('weights', weights_bytes, filename='aggregated_weights.pkl')
        timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{cloud_url}/upload_aggregated_weights", data=data) as resp:
                if resp.status == 200:
                    print(f"[Aggregator {aggregator_id}]  已上傳聚合權重到Cloud Server (round={round_id_value})")
                    log_event("cloud_upload_success", f"round={round_id_value}")
                else:
                    text = await resp.text()
                    print(f"[Aggregator {aggregator_id}] 上傳聚合權重到Cloud失敗: {resp.status} {text}")
                    log_event("cloud_upload_failed", f"round={round_id_value}, status={resp.status}")
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] 上傳聚合權重異常: {e}")
        log_event("cloud_upload_exception", f"round={round_id_value}, error={e}")

async def upload_delta_to_cloud(base_version: int, delta_state: Dict[str, Any], round_id_value: int):
    try:
        if not CLOUD_SERVER_CONFIG.get("enabled", False):
            return
        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
        if not cloud_url:
            return
        
                       
        def _compute_delta(local_weights, base_weights_dict):
            def _to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.detach(    ).cpu().float()
                elif isinstance(x, np.ndarray):
                    return torch.from_numpy(x).float()
                elif isinstance(x, list):
                    try:
                        return torch.tensor(x, dtype=torch.float32)
                    except Exception:
                        return None
                return None
            
            delta_result = {}
            if isinstance(base_weights_dict, dict) and base_weights_dict:
                for k, v in local_weights.items():
                    g = _to_tensor(v)
                    b = _to_tensor(base_weights_dict.get(k))
                    if g is not None and b is not None and g.shape == b.shape:
                        delta_result[k] = (g - b)
                    else:
                        if g is not None:
                            delta_result[k] = g
            else:
                                        
                for k, v in local_weights.items():
                    t = _to_tensor(v)
                    if t is not None:
                        delta_result[k] = t
            return delta_result
        
                        
        def _serialize_delta(delta_dict):
            delta_payload = {}
            for k, v in (delta_dict or {}).items():
                if isinstance(v, torch.Tensor):
                    delta_payload[k] = v.detach().cpu().numpy()
                elif isinstance(v, np.ndarray):
                    delta_payload[k] = v
                else:
                    try:
                        delta_payload[k] = np.array(v, dtype=np.float32)
                    except Exception:
                        continue
            return pickle.dumps(delta_payload)
        
        timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
        max_retries = int(getattr(config, 'AGGREGATION_CONFIG', {}).get('cas_max_retries', 4))
        base_backoff = float(getattr(config, 'AGGREGATION_CONFIG', {}).get('cas_backoff_seconds', 0.5))
        import random
        
        current_base_version = base_version
        current_delta = delta_state
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            attempt = 0
            while attempt < max_retries:
                             
                delta_bytes = _serialize_delta(current_delta)
                data = aiohttp.FormData()
                data.add_field('aggregator_id', str(aggregator_id))
                data.add_field('base_version', str(current_base_version))
                data.add_field('round_id', str(round_id_value))
                data.add_field('model_version', str(model_version))
                data.add_field('delta', delta_bytes, filename='delta.pkl')
                
                async with session.post(f"{cloud_url}/upload_aggregated_delta", data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"[Aggregator {aggregator_id}]  已上傳 delta，new_version={result.get('new_version')}")
                        log_event("cloud_delta_upload_success", f"base_version={current_base_version}")
                        return
                    elif resp.status == 409:
                                                        
                        print(f"[Aggregator {aggregator_id}] CAS 衝突 (base_version={current_base_version})，重新獲取版本並重算 delta ({attempt+1}/{max_retries})")
                        try:
                                      
                            async with session.get(f"{cloud_url}/get_global_weights_with_version") as get_resp:
                                if get_resp.status == 200:
                                    payload = pickle.loads(await get_resp.read())
                                    new_base_version = int(payload.get('version', 0))
                                    new_base_weights = payload.get('weights', {})
                                    print(f"[Aggregator {aggregator_id}]  重新獲取版本: {current_base_version} → {new_base_version}")
                                                
                                    current_base_version = new_base_version
                                    current_delta = _compute_delta(global_weights, new_base_weights)
                                    delay = base_backoff * (2 ** attempt) + random.uniform(-0.1, 0.1)
                                    await asyncio.sleep(max(0.0, delay))
                                    attempt += 1
                                    continue
                                else:
                                    print(f"[Aggregator {aggregator_id}]  重新獲取版本失敗: HTTP {get_resp.status}")
                                    attempt += 1
                                    await asyncio.sleep(base_backoff)
                                    continue
                        except Exception as e:
                            print(f"[Aggregator {aggregator_id}]  重新獲取版本異常: {e}")
                            attempt += 1
                            await asyncio.sleep(base_backoff)
                            continue
                    else:
                        text = await resp.text()
                        print(f"[Aggregator {aggregator_id}] 上傳 delta 失敗: {resp.status} {text}")
                        log_event("cloud_delta_upload_failed", f"status={resp.status}")
                        return
            
                      
            print(f"[Aggregator {aggregator_id}] 上傳 delta 達到最大重試次數 ({max_retries})，放棄")
            log_event("cloud_delta_upload_max_retries", f"max_retries={max_retries}")
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] 上傳 delta 異常: {e}")
        log_event("cloud_delta_upload_exception", str(e))

async def fetch_global_weights_from_cloud_with_retry(max_retries: int = 15, delay_seconds: float = 2.0) -> bool:
    try:
        if not CLOUD_SERVER_CONFIG.get("enabled", False):
            return False
        cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
        if not cloud_url:
            return False

        timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                        if resp.status == 200:
                            weights_bytes = await resp.read()
                            payload = pickle.loads(weights_bytes)
                            server_version = int(payload.get('version', 0))
                            server_weights = payload.get('weights', {})

                                              
                            updated_weights = {}
                            for k, v in (server_weights or {}).items():
                                if isinstance(v, torch.Tensor):
                                    updated_weights[k] = v.detach().cpu()
                                elif isinstance(v, np.ndarray):
                                    updated_weights[k] = torch.from_numpy(v).float()
                                elif isinstance(v, list):
                                    try:
                                        updated_weights[k] = torch.tensor(v, dtype=torch.float32)
                                    except Exception:
                                        continue
                                else:
                                    try:
                                        updated_weights[k] = torch.tensor(v, dtype=torch.float32)
                                    except Exception:
                                        continue

                            if updated_weights:
                                global global_weights, model_version
                                global_weights = updated_weights
                                model_version = int(server_version)
                                print(f"[Aggregator {aggregator_id}]  已同步Cloud全局權重（cloud_version={server_version}）")
                                log_event("cloud_sync_success", f"cloud_version={server_version}")
                                return True
                        elif resp.status == 404:
                                            
                            print(f"[Aggregator {aggregator_id}] Cloud 全局權重尚未可用（404），重試 {attempt+1}/{max_retries}")
                        else:
                            text = await resp.text()
                            print(f"[Aggregator {aggregator_id}] 獲取Cloud全局權重失敗: {resp.status} {text}")
            except Exception as inner_e:
                print(f"[Aggregator {aggregator_id}] 連接Cloud獲取權重失敗（嘗試 {attempt+1}/{max_retries}）: {inner_e}")

            await asyncio.sleep(delay_seconds)

        log_event("cloud_sync_timeout", f"retries={max_retries}")
        return False
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] 同步Cloud全局權重異常: {e}")
        log_event("cloud_sync_exception", str(e))
        return False

async def periodic_round_sync_check():
    """🔧 新增：定期檢查並同步聚合器輪次"""
    while True:
        try:
            await asyncio.sleep(60)            
            await check_and_sync_rounds()
        except Exception as e:
            print(f"[Aggregator {aggregator_id}]  輪次同步檢查異常: {e}")
            await asyncio.sleep(30)           

async def check_and_sync_rounds():
    """🔧 新增：檢查並同步所有聚合器輪次"""
    try:
                              
        num_aggregators = getattr(config, 'NUM_AGGREGATORS', 4)
        aggregator_urls = [
            f"http://127.0.0.1:{8000 + i}" for i in range(num_aggregators)
        ]
        
        current_rounds = []
        
        for i, url in enumerate(aggregator_urls):
            try:
                async with aiohttp.ClientSession() as session:
                                               
                    timeout = aiohttp.ClientTimeout(total=5)                   
                    async with session.get(f"{url}/current_round", timeout=timeout) as response:
                        if response.status == 200:
                            data = await response.json()
                            current_round = data.get('current_round', 0)
                            current_rounds.append(current_round)
                        else:
                                           
                            error_text = await response.text()
                            print(f"[Aggregator {aggregator_id}]  檢查聚合器 {i} 輪次失敗: HTTP {response.status} - {error_text[:100]}")
                            current_rounds.append(0)
            except asyncio.TimeoutError:
                                  
                print(f"[Aggregator {aggregator_id}]  檢查聚合器 {i} 輪次超時（可能未啟動或網絡延遲）")
                current_rounds.append(0)
            except Exception as e:
                                        
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"[Aggregator {aggregator_id}]  檢查聚合器 {i} 輪次失敗: {error_msg}")
                current_rounds.append(0)
        
                                        
        if aggregator_id >= len(current_rounds):
            print(f"[Aggregator {aggregator_id}]  聚合器ID {aggregator_id} 超出範圍 (0-{len(current_rounds)-1})，跳過同步檢查")
            return
        
                
        if len(current_rounds) == 0:
            print(f"[Aggregator {aggregator_id}]  無法獲取任何聚合器輪次，跳過同步檢查")
            return
            
        max_round = max(current_rounds)
        current_round = current_rounds[aggregator_id]
        
                          
        if max_round > current_round + 1:             
            print(f"[Aggregator {aggregator_id}]  檢測到輪次落後: 當前{current_round}輪，最高{max_round}輪，開始同步...")
            
            try:
                async with aiohttp.ClientSession() as session:
                    data = aiohttp.FormData()
                    data.add_field('target_round', str(max_round))
                    
                    async with session.post(f"{aggregator_urls[aggregator_id]}/reset_round", data=data, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"[Aggregator {aggregator_id}] 輪次同步成功: {current_round} -> {max_round}")
                            log_event("round_sync", f"自動同步: {current_round} -> {max_round}")
                        else:
                            print(f"[Aggregator {aggregator_id}] 輪次同步失敗: HTTP {response.status}")
            except Exception as e:
                print(f"[Aggregator {aggregator_id}] 輪次同步異常: {e}")
        
                
        round_status = ", ".join([f"聚合器{i}={current_rounds[i]}輪" for i in range(len(current_rounds))])
        print(f"[Aggregator {aggregator_id}] 輪次狀態: {round_status}, 最高={max_round}輪")
        
    except Exception as e:
        print(f"[Aggregator {aggregator_id}] 輪次同步檢查失敗: {e}")
        import traceback
        traceback.print_exc()

def log_training_event_aggregator(event, info: dict):
    """記錄訓練事件（簡化版）"""
    print(f"[Aggregator {aggregator_id}] {event}: {info}")

def get_global_performance_metric():
    """獲取全局性能指標（簡化版：返回固定值）"""
    return 0.5

def select_clients_for_round(round_id, total_clients=None):
    """🔧 改進：實現分層非同步 FL 的客戶端選擇策略"""
    assigned_clients = get_assigned_clients()
    if not assigned_clients:
        return []
    
                             
    import random
    
                  
    random.seed(round_id + aggregator_id * 1000)
    
                        
                   
    participation_cfg = config.FEDERATED_CONFIG.get('participation_strategy', {})
    if round_id <= participation_cfg.get('early_rounds', {}).get('threshold', 20):
        participation_ratio = participation_cfg.get('early_rounds', {}).get('ratio', 0.8)
    elif round_id <= participation_cfg.get('mid_rounds', {}).get('threshold', 100):
        participation_ratio = participation_cfg.get('mid_rounds', {}).get('ratio', 0.7)
    elif round_id <= participation_cfg.get('late_rounds', {}).get('threshold', 200):
        participation_ratio = participation_cfg.get('late_rounds', {}).get('ratio', 0.6)
    else:
        participation_ratio = participation_cfg.get('final_rounds', {}).get('ratio', 0.5)
    
                 
    if total_clients is None:
        total_clients = len(assigned_clients)
    num_to_select = max(2, int(total_clients * participation_ratio))
    
          
    selected_clients = random.sample(assigned_clients, min(num_to_select, len(assigned_clients)))
    
                               
    if not hasattr(app.state, 'not_selected_streak'):
        app.state.not_selected_streak = {}
    
                                                           
    for cid in assigned_clients:
        if cid not in app.state.not_selected_streak:
            app.state.not_selected_streak[cid] = 0
    
                 
    for cid in assigned_clients:
        if cid in selected_clients:
            app.state.not_selected_streak[cid] = 0
        else:
            app.state.not_selected_streak[cid] = app.state.not_selected_streak.get(cid, 0) + 1
        
                      
    forced_clients = [cid for cid in assigned_clients if app.state.not_selected_streak.get(cid, 0) >= 3]
    for cid in forced_clients:
        if cid not in selected_clients:
            selected_clients.append(cid)
            print(f"[Aggregator {aggregator_id}] 強制加入連續3輪未被選中的客戶端: {cid}")
    
           
    selected_clients = sorted(list(set(selected_clients)))
    
    print(f"[Aggregator {aggregator_id}] 分層非同步 FL 客戶端選擇: {selected_clients} (參與率: {len(selected_clients)}/{len(assigned_clients)} = {len(selected_clients)/len(assigned_clients):.1%})")
    
    return selected_clients

def calculate_dynamic_participation_ratio(round_id, assigned_clients):
    """計算動態參與率"""
    dynamic_config = config.FEDERATED_CONFIG.get('client_selection', {}).get('dynamic_participation', {})
    base_ratio = float(config.FEDERATED_CONFIG.get('client_selection', {}).get('base_participation_ratio', 0.8))
    min_ratio = float(config.FEDERATED_CONFIG.get('client_selection', {}).get('min_participation_ratio', 0.4))
    max_ratio = float(config.FEDERATED_CONFIG.get('client_selection', {}).get('max_participation_ratio', 1.0))
    
    if not dynamic_config.get('enabled', False):
        return base_ratio
    
             
    if not hasattr(app.state, 'performance_history'):
        app.state.performance_history = {}
    
    if not hasattr(app.state, 'participation_ratios'):
        app.state.participation_ratios = {base_ratio}
    
              
    avg_improvement = calculate_average_performance_improvement(assigned_clients, round_id)
    
                 
    improvement_factor = dynamic_config.get('improvement_factor', 0.1)
    penalty_factor = dynamic_config.get('penalty_factor', 0.05)
    min_adjustment = dynamic_config.get('min_adjustment', 0.05)
    max_adjustment = dynamic_config.get('max_adjustment', 0.2)
    smoothing_factor = dynamic_config.get('smoothing_factor', 0.8)
    
            
    if avg_improvement > 0.01:        
        adjustment = min(max_adjustment, improvement_factor * avg_improvement)
        new_ratio = min(max_ratio, base_ratio + adjustment)
    elif avg_improvement < -0.01:        
        adjustment = min(max_adjustment, penalty_factor * abs(avg_improvement))
        new_ratio = max(min_ratio, base_ratio - adjustment)
    else:        
        new_ratio = base_ratio
    
          
    if app.state.participation_ratios:
        last_ratio = max(app.state.participation_ratios)
        new_ratio = smoothing_factor * last_ratio + (1 - smoothing_factor) * new_ratio
    
              
    new_ratio = max(min_ratio, min(max_ratio, new_ratio))
    
             
    app.state.participation_ratios.add(new_ratio)
    
    print(f"[Aggregator {aggregator_id}] 動態參與率調整: {base_ratio:.3f} -> {new_ratio:.3f} (改善={avg_improvement:.4f})")
    
    return new_ratio

def calculate_client_performance_scores(assigned_clients, round_id):
    return {cid: 0.5 for cid in assigned_clients}

def calculate_average_performance_improvement(assigned_clients, round_id):
    return 0.0

def record_client_performance(client_id, results_data, round_id):
    """記錄客戶端性能（簡化版：僅輸出日誌）"""
    if isinstance(results_data, dict):
        f1 = results_data.get('f1_score', results_data.get('accuracy', 0.0))
        print(f"[Aggregator {aggregator_id}] 客戶端 {client_id} 輪次 {round_id} 性能: {f1:.4f}")


def perform_standard_fedavg(client_weights_list, client_data_sizes_list):
    """執行標準FedAvg聚合"""
    if not client_weights_list:
        return {}
    
    print(f"[Aggregator] 開始標準FedAvg聚合，{len(client_weights_list)}個客戶端")
    
            
    total_data_size = max(1, sum(max(0, s) for s in client_data_sizes_list))
    
                  
    client_weights = [max(0, size) / total_data_size for size in client_data_sizes_list]

                                                   
                                                      

                                  
    try:
        num_clients = max(1, len(client_weights))
        alpha_max = DATASIZE_ALPHA_MAX_MULTIPLIER * (1.0 / num_clients)
        client_weights = [min(w, alpha_max) for w in client_weights]
        s = sum(client_weights)
        if s > 0:
            client_weights = [w / s for w in client_weights]
    except Exception:
        pass
    
    print(f"[Aggregator] 客戶端數據大小: {client_data_sizes_list}")
    print(f"[Aggregator] 客戶端權重: {[f'{w:.3f}' for w in client_weights]}")
    print(f"[Aggregator] 總數據量: {total_data_size}")
    
             
    aggregated_weights = {}
    first_weights = client_weights_list[0]
    
                                                
    for layer_name in first_weights.keys():
                                                        
        if ('running_mean' in layer_name or 
            'running_var' in layer_name or 
            'num_batches_tracked' in layer_name):
            if global_weights is not None and layer_name in global_weights and isinstance(global_weights[layer_name], torch.Tensor):
                aggregated_weights[layer_name] = global_weights[layer_name].clone().float()
            else:
                aggregated_weights[layer_name] = (
                    first_weights[layer_name].clone().float()
                    if isinstance(first_weights[layer_name], torch.Tensor)
                    else first_weights[layer_name]
                )
            continue
        first_layer = first_weights[layer_name]
        
                              
        if isinstance(first_layer, torch.Tensor):
            if first_layer.dtype not in (torch.float32, torch.float64):
                first_layer = first_layer.float()
            aggregated_layer = torch.zeros_like(first_layer, dtype=torch.float64)
        else:
            aggregated_layer = 0.0
        
                           
        for i, client_weight_dict in enumerate(client_weights_list):
            weight = client_weights[i]
            client_layer = client_weight_dict[layer_name]
            
            if isinstance(client_layer, torch.Tensor):
                if client_layer.dtype not in (torch.float32, torch.float64):
                    client_layer = client_layer.float()
                                      
                if CLIP_NORM and CLIP_NORM > 0:
                    layer_norm = torch.norm(client_layer.reshape(-1).to(dtype=torch.float64), p=2)
                    if torch.isfinite(layer_norm) and layer_norm > CLIP_NORM:
                        scale = CLIP_NORM / (layer_norm + 1e-12)
                        client_layer = (client_layer.to(dtype=torch.float64) * scale).to(dtype=client_layer.dtype)
                wt = float(weight)
                                 
                aggregated_layer += client_layer.to(dtype=torch.float64) * wt
            elif isinstance(client_layer, list):
                           
                print(f"[Aggregator]  檢測到列表類型權重層 {layer_name}，轉換為張量")
                try:
                    client_layer = torch.tensor(client_layer, dtype=torch.float64)
                    aggregated_layer += client_layer * float(weight)
                except Exception as e:
                    print(f"[Aggregator] 列表轉張量失敗: {e}")
                    continue
            else:
                try:
                    aggregated_layer += float(weight) * float(client_layer)
                except Exception as e:
                    print(f"[Aggregator] 權重轉換失敗: {e}")
                    continue
        
                                                          
                       
        if isinstance(aggregated_layer, torch.Tensor):
            aggregated_weights[layer_name] = aggregated_layer.to(dtype=torch.float32)
        else:
            aggregated_weights[layer_name] = aggregated_layer
    
            
    aggregated_weights = apply_smoothing_to_weights(aggregated_weights)
    
    print(f"[Aggregator] 標準FedAvg聚合完成")
    print(f"[Aggregator] 聚合權重層數: {len(aggregated_weights)}")
    
    return aggregated_weights

def apply_smoothing_to_weights(new_weights):
    """應用平滑機制到權重，減少震盪"""
    global global_weights
    
                        
    if (globals().get('round_count', 0) or 0) < 2:
        return new_weights
    
    if global_weights is None or len(global_weights) == 0:
                       
        return new_weights
    
            
    smoothing_factor = config.FEDERATED_CONFIG.get('aggregation', {}).get('smoothing_factor', 0.5)
    
    print(f"[Aggregator] 應用平滑機制，平滑因子: {smoothing_factor}")
    
    smoothed_weights = {}
    
    def _to_cpu_f32(x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.device.type != 'cpu':
                x = x.cpu()
            return x.float()
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, list):
            try:
                return torch.tensor(x, dtype=torch.float32)
            except Exception:
                return x
        return x

    def _to_float_scalar(x: Any) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.item() if x.numel() == 1 else x.float().mean().item())
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    for layer_name in new_weights.keys():
        if layer_name in global_weights:
            old_val = _to_cpu_f32(global_weights[layer_name])
            new_val = _to_cpu_f32(new_weights[layer_name])
            if isinstance(old_val, torch.Tensor) and isinstance(new_val, torch.Tensor):
                smoothed_weights[layer_name] = smoothing_factor * old_val + (1 - smoothing_factor) * new_val
            else:
                try:
                    o = _to_float_scalar(old_val)
                    n = _to_float_scalar(new_val)
                    smoothed_weights[layer_name] = smoothing_factor * o + (1 - smoothing_factor) * n
                except Exception:
                    smoothed_weights[layer_name] = new_weights[layer_name]
        else:
                     
            smoothed_weights[layer_name] = new_weights[layer_name]
    
    print(f"[Aggregator]  權重平滑完成")
    return smoothed_weights

def check_aggregation_conditions():
    """檢查聚合條件"""
    current_clients = len(client_weights_buffer)
                        
    try:
        expected_clients = len(round_clients)              
    except Exception:
        expected_clients = 5                
    configured_min = int(config.AGGREGATION_CONFIG.get("min_clients_for_aggregation", 2))
                         
    min_clients = max(2, configured_min)           
    min_clients = min(min_clients, expected_clients)
    max_wait_time = config.AGGREGATION_CONFIG.get("max_wait_time", 60)          
    force_aggregation = config.AGGREGATION_CONFIG.get("force_aggregation_after_timeout", True)
    
    print(f"[Aggregator] 聚合條件檢查:")
    print(f"[Aggregator]   - 當前客戶端: {current_clients}")
    print(f"[Aggregator]   - 期望客戶端: {expected_clients}")
    print(f"[Aggregator]   - 最小要求: {min_clients}")
    print(f"[Aggregator]   - 客戶端列表: {list(client_weights_buffer.keys())}")
    
                    
    participation_ratio = current_clients / expected_clients if expected_clients > 0 else 0
    min_participation_ratio = 0.0                
    
                   
    if current_clients >= min_clients and participation_ratio >= min_participation_ratio:
        print(f"[Aggregator]  達到聚合條件: {current_clients} 個客戶端 (參與率: {participation_ratio:.1%})")
        return True
    
                   
    if round_start_time is not None:
        elapsed_time = time.time() - round_start_time
        quick_time = max_wait_time * 0.6                     
        
        if elapsed_time > quick_time and current_clients >= min_clients:
            print(f"[Aggregator]  快速聚合（等待時間60%）: {current_clients} 個客戶端 (等待 {elapsed_time:.1f}s)")
            return True
        
            
    if round_start_time is not None:
        elapsed_time = time.time() - round_start_time
        if elapsed_time > max_wait_time:
                             
            threshold = max(1, int(config.AGGREGATION_CONFIG.get("min_clients_for_aggregation", 1)))
            if force_aggregation and current_clients >= threshold:
                print(f"[Aggregator] 超時強制聚合: {current_clients} 個客戶端 (等待 {elapsed_time:.1f}s)")
                return True
            else:
                print(f"[Aggregator] 超時但客戶端數量不足，跳過聚合: {current_clients} < {threshold}")
                return False
    
    print(f"[Aggregator]  等待更多客戶端: {current_clients}/{min_clients} (參與率: {participation_ratio:.1%})")
    return False

def get_assigned_clients():
    """獲取分配給此聚合器的客戶端列表"""
    assigned_clients = []
    total_clients = int(getattr(config, 'NUM_CLIENTS', 10))
    for client_id in range(total_clients):
        if client_id % config.NUM_AGGREGATORS == aggregator_id:
            assigned_clients.append(client_id)
    
    print(f"[Aggregator {aggregator_id}] 分配客戶端: {assigned_clients}")
    return assigned_clients

def convert_numpy_values(obj):
    """轉換numpy值為原生Python類型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_values(item) for item in obj]
    else:
        return obj

@app.get("/health")
async def health_check():
    """健康檢查"""
    try:
        health_status = {
            "status": "healthy",
            "aggregator_id": aggregator_id,
            "round_count": round_count,
            "buffer_size": len(client_weights_buffer),
            "global_weights_available": global_weights is not None,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_version": model_version,
            "training_phase": is_training_phase
        }
        
        return convert_numpy_values(health_status)
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

@app.post("/start_federated_round")
async def start_federated_round(round_id: int = Form(...), force: bool = Form(False)):
    """開始新一輪聯邦學習"""
    global round_count, round_clients, client_weights_buffer, round_start_time, client_timeout_status, is_training_phase
    
    print(f"[Aggregator] 收到開始第{round_id}輪聯邦學習請求 (強制模式: {force})")
    print(f"[Aggregator] 當前狀態: 輪次={round_count}, 選中客戶端={round_clients}")
    
          
    log_event("start_federated_round", f"收到輪次 {round_id} 啟動請求 (強制: {force})")
    
                       
    min_clients_required = config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', 3)
    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')

                
    if round_id < round_count and not force:
        print(f"[Aggregator]  收到過期輪次啟動請求: {round_id} < {round_count}")
        log_event("round_start_stale_request", f"請求輪次 {round_id} 小於當前 {round_count}")
        return {
            "round_id": round_count,
            "selected_clients": round_clients,
            "min_clients_required": min_clients_required,
            "aggregation_timeout": AGGREGATION_TIMEOUT,
            "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
            "selection_strategy": "simplified",
            "training_flow": training_flow,
            "status": "stale_round_request"
        }

                      
    if round_id == round_count and round_start_time is not None and not force:
                              
        current_time = time.time()
        if current_time - round_start_time > 600:        
            print(f"[Aggregator] 檢測到輪次卡住超過10分鐘，允許重啟第{round_id}輪")
            log_event("round_stuck_detected", f"輪次 {round_id} 卡住超過10分鐘，強制重啟")
                           
            round_start_time = None
            is_training_phase = False
                        
            client_weights_buffer = {}
            client_timeout_status = {}
        else:
            print(f"[Aggregator]  第{round_id}輪已經啟動，跳過重複啟動")
            log_event("round_already_started", f"輪次 {round_id} 已啟動")
            return {
                "round_id": round_id,
                "selected_clients": round_clients,
                "min_clients_required": min_clients_required,
                "aggregation_timeout": AGGREGATION_TIMEOUT,
                "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                "selection_strategy": "simplified",
                "training_flow": training_flow,
                "status": "already_started"
            }

                   
    if force:
                     
        print(f"[Aggregator] 強制模式：設置輪次 {round_id}")
        round_count = round_id
        round_start_time = time.time()
        is_training_phase = True
                                                  
        client_weights_buffer = {}
        client_timeout_status = {}
        
                 
        round_clients = select_clients_for_round(round_id)
        print(f"[Aggregator] 強制模式：選中客戶端 {round_clients}")
        
        log_event("round_force_started", f"強制啟動輪次 {round_id}, 選中客戶端 {round_clients}")
        return {
            "round_id": round_id,
            "selected_clients": round_clients,
            "min_clients_required": min_clients_required,
            "aggregation_timeout": AGGREGATION_TIMEOUT,
            "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
            "selection_strategy": "simplified",
            "training_flow": training_flow,
            "status": "force_started"
        }
    
                         
    if round_id > round_count:
        if round_id > round_count + 1:                
            print(f"[Aggregator]  輪次跳躍過大: 請求={round_id}, 當前={round_count}")
            log_event("round_advance_blocked", f"large_skip: current={round_count}, request={round_id}")
            return {
                "round_id": round_count,
                "selected_clients": round_clients,
                "min_clients_required": min_clients_required,
                "aggregation_timeout": AGGREGATION_TIMEOUT,
                "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                "selection_strategy": "simplified",
                "training_flow": training_flow,
                "status": "large_skip_rejected"
            }
        
                        
        if len(client_weights_buffer) > 20:                
            print(f"[Aggregator]  緩衝區過多: {len(client_weights_buffer)} 個權重未聚合")
            log_event("round_advance_blocked", f"buffer_overflow={len(client_weights_buffer)}")
            return {
                "round_id": round_count,
                "selected_clients": round_clients,
                "min_clients_required": min_clients_required,
                "aggregation_timeout": AGGREGATION_TIMEOUT,
                "partial_aggreation_enabled": PARTIAL_AGGREGATION_ENABLED,
                "selection_strategy": "simplified",
                "training_flow": training_flow,
                "status": "buffer_overflow"
            }
        
        print(f"[Aggregator] 輪次更新: {round_count} -> {round_id}")
        round_count = round_id
    
              
    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
    
    if training_flow == 'train_then_select':
                                
        print(f"[Aggregator]  使用新流程：先訓練後選擇")
        assigned_clients = []
        for client_id in range(10):
            if client_id % config.NUM_AGGREGATORS == aggregator_id:
                assigned_clients.append(client_id)
        round_clients = assigned_clients
        is_training_phase = True
        print(f"[Aggregator]  新流程：分配給此聚合器的客戶端可以參與訓練: {round_clients}")
    else:
                        
        print(f"[Aggregator]  使用原有流程：先選擇後訓練")
        round_clients = select_clients_for_round(round_id)
        is_training_phase = False
    
                 
    client_weights_buffer.clear()
    client_data_sizes.clear()
    
                  
    round_start_time = time.time()
    client_timeout_status.clear()
                   
    if not hasattr(app.state, 'aggregation_done'):
        app.state.aggregation_done = False
        app.state.aggregation_done_round = -1
    else:
        app.state.aggregation_done = False
        app.state.aggregation_done_round = round_count
    
    print(f"[Aggregator]  開始第{round_id}輪聯邦學習")
    print(f"[Aggregator]  客戶端選擇策略: {training_flow}")
    print(f"[Aggregator]  聚合超時設置: {AGGREGATION_TIMEOUT}s")
    print(f"[Aggregator]  選中客戶端列表: {round_clients} (共 {len(round_clients)} 台)")
    
    log_training_event_aggregator(
        'round_started',
        {
            'round_id': round_id,
            'selected_len': len(round_clients),
            'buffer_size': len(client_weights_buffer),
            'model_version': model_version,
            'detail': str(round_clients)
        }
    )
    
    return {
        "round_id": round_id,
        "selected_clients": round_clients,
        "min_clients_required": min_clients_required,
        "aggregation_timeout": AGGREGATION_TIMEOUT,
        "partial_aggregation_enabled": PARTIAL_AGGREGATION_ENABLED,
        "selection_strategy": "simplified",
        "training_flow": training_flow
    }

@app.post("/upload_federated_weights")
async def upload_federated_weights(
    client_id: int = Form(...),
    data_size: int = Form(...),
    round_id: int = Form(...),
    commit_id: str = Form(""),
                         
    client_predictions: Optional[UploadFile] = File(None),
    weights: Optional[UploadFile] = File(None)
):
    """接收客戶端上傳的聯邦學習權重"""
    global global_weights                     
    global client_weights_buffer, client_data_sizes, client_predictions_buffer, round_count, round_clients, global_performance, is_training_phase, global_weights, model_version
    
    print(f"[Aggregator] 收到客戶端 {client_id} 權重上傳請求")
    print(f"[Aggregator] 當前狀態: 輪次={round_count}, 客戶端輪次={round_id}")
    print(f"[Aggregator] 當前緩衝區: {len(client_weights_buffer)} 個客戶端")
    print(f"[Aggregator] 選中客戶端: {round_clients}")
    
                
    if STALE_POLICY == 'strict':
        if round_id != round_count:
            print(f"[Aggregator] 嚴格同步：拒收非當前輪次的權重 (client_round={round_id}, agg_round={round_count})")
            return {
                "status": "rejected",
                "message": "嚴格同步策略：僅接受當前輪次",
                "expected_round": round_count
            }
    elif STALE_POLICY == 'decay':
        if round_id < round_count - MAX_STALENESS:
            print(f"[Aggregator] 衰減策略：超過允許的落後輪次 MAX_STALENESS={MAX_STALENESS}")
            return {
                "status": "rejected",
                "message": "超過允許落後輪次",
                "expected_round": round_count
            }
        elif round_id < round_count:
            print(f"[Aggregator]  衰減策略：接受落後 {round_count - round_id} 輪，將在權重中施加衰減")
                                           
            staleness = round_count - round_id
        else:
            staleness = 0
    else:
                                       
        if round_id < round_count - 50:
            print(f"[Aggregator]  客戶端輪次 {round_id} 嚴重落後於聚合器輪次 {round_count}，拒絕處理")
            log_event("weight_upload_rejected", f"客戶端輪次嚴重落後: {round_id} < {round_count - 10}")
            return {
                "status": "rejected",
                "message": "輪次嚴重不匹配",
                "expected_round": round_count
            }
        elif round_id < round_count:
            print(f"[Aggregator] 客戶端輪次 {round_id} 落後{round_count - round_id}輪，但允許參與當前輪次 {round_count}")
            round_id = round_count
    
          
    if round_id != round_count:
        print(f"[Aggregator] 輪次不匹配: 客戶端輪次={round_id}, 聚合器輪次={round_count}")
        
                                   
        if round_count == 0:
            print(f"[Aggregator] 聚合器初始狀態，調整到客戶端輪次 {round_id}")
            round_count = round_id
            training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
            if training_flow == 'train_then_select':
                round_clients = get_assigned_clients()
                is_training_phase = True
            else:
                round_clients = select_clients_for_round(round_id)
                is_training_phase = False
            print(f"[Aggregator] 開始第{round_id}輪")
        else:
                                   
            if round_id > round_count:
                print(f"[Aggregator] 客戶端輪次超前，調整到客戶端輪次 {round_id}")
                
                if len(client_weights_buffer) == 0:
                    print(f"[Aggregator] 緩衝區為空，安全調整輪次")
                    round_count = round_id
                    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
                    if training_flow == 'train_then_select':
                        round_clients = get_assigned_clients()
                        is_training_phase = True
                    else:
                        round_clients = select_clients_for_round(round_id)
                        is_training_phase = False
                    print(f"[Aggregator] 開始第{round_id}輪")
                else:
                    print(f"[Aggregator]  緩衝區有 {len(client_weights_buffer)} 個客戶端權重，保持當前輪次等待聚合")
                    round_id = round_count
            elif round_id < round_count:
                print(f"[Aggregator] 客戶端輪次落後，保持當前輪次 {round_count}")
            else:
                print(f"[Aggregator] 輪次匹配，保持當前輪次 {round_count}")
    
                   
    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
    
    if training_flow == 'train_then_select':
        if client_id not in round_clients:
            print(f"[Aggregator]  客戶端 {client_id} 不在分配列表中，但仍在訓練階段接受")
    else:
        if client_id not in round_clients:
            print(f"[Aggregator]  客戶端 {client_id} 不在選中列表中")
            print(f"[Aggregator]  仍然接受客戶端 {client_id} 的權重以保持系統穩定性")
    
    try:
                               
        client_prediction = None
        if client_predictions is not None:
            try:
                predictions_bytes = await client_predictions.read()
                client_prediction = pickle.loads(predictions_bytes)
                print(f"[Aggregator]  收到客戶端 {client_id} 預測概率分布: {client_prediction}")
            except Exception as pred_error:
                print(f"[Aggregator]  解析客戶端預測概率分布失敗: {pred_error}")
                client_prediction = None
        
                
        if weights is None:
            return {
                "status": "rejected",
                "message": "weights field is required for standard federated learning",
                "current_round": round_count
            }
        
        weights_data = pickle.loads(await weights.read())
                                     
        def _to_cpu_f32_layer(x):
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.device.type != 'cpu':
                    x = x.cpu()
                return x.float()
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            elif isinstance(x, list):
                try:
                    return torch.tensor(x, dtype=torch.float32)
                except Exception:
                    return x
            return x
        if isinstance(weights_data, dict):
            normalized_weights = {}
            for k, v in weights_data.items():
                                           
                if k.startswith('_teacher_model'):
                    continue
                normalized_weights[k] = _to_cpu_f32_layer(v)
            weights_data = normalized_weights
        
                 
        async with buffer_lock:
                                                
            if commit_id:
                if not hasattr(app.state, 'idempotent_commits'):
                    app.state.idempotent_commits = set()
                idem_key = (int(round_id), int(client_id), str(commit_id))
                if idem_key in app.state.idempotent_commits:
                    print(f"[Aggregator]  冪等：忽略重複提交 (round={round_id}, client={client_id}, commit={commit_id})")
                    return {
                        "status": "received",
                        "message": "duplicate_commit_ignored",
                        "buffer_size": len(client_weights_buffer),
                        "current_round": round_count
                    }
                app.state.idempotent_commits.add(idem_key)

                                 
            if global_weights is None:
                                                        
                try:
                    initialize_global_weights()
                    if global_weights is not None:
                        print(f"[Aggregator] 全局權重為空，已根據配置初始化（模型類型: {config.MODEL_CONFIG.get('type', 'dnn')}）")
                    else:
                                             
                        print(f"[Aggregator] 全局權重為空，使用客戶端 {client_id} 的權重作為初始模板")
                        filtered_weights = {
                            k: (v.detach().cpu().float().clone() if isinstance(v, torch.Tensor) else v)
                            for k, v in (weights_data.items() if isinstance(weights_data, dict) else [])
                            if not k.startswith('_teacher_model')
                        }
                        global_weights = filtered_weights
                except Exception as e:
                                         
                    print(f"[Aggregator]  全局權重初始化失敗: {e}，使用客戶端 {client_id} 的權重作為初始模板")
                    filtered_weights = {
                        k: (v.detach().cpu().float().clone() if isinstance(v, torch.Tensor) else v)
                        for k, v in (weights_data.items() if isinstance(weights_data, dict) else [])
                        if not k.startswith('_teacher_model')
                    }
                    global_weights = filtered_weights
            if isinstance(weights_data, dict) and isinstance(global_weights, dict):
                gw_keys = set(global_weights.keys())
                wk_keys = set(weights_data.keys())
                if gw_keys != wk_keys:
                    missing = list(gw_keys - wk_keys)
                    extra = list(wk_keys - gw_keys)
                    
                                                              
                                                       
                    filtered_missing = [k for k in missing if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']) and not k.startswith('_teacher_model')]
                    filtered_extra = [k for k in extra if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']) and not k.startswith('_teacher_model')]
                    
                                                          
                                                            
                    global_is_old_arch = (
                        any('layers.' in k or 'residual_layers.' in k or 'batch_norms.' in k for k in filtered_missing) and
                        any('transformer_blocks.' in k or 'input_projection.' in k or 'classifier.' in k for k in filtered_extra)
                    )
                                                            
                    client_is_old_arch = (
                        any('transformer_blocks.' in k or 'input_projection.' in k or 'classifier.' in k for k in filtered_missing) and
                        any('layers.' in k or 'residual_layers.' in k or 'batch_norms.' in k for k in filtered_extra)
                    )
                    
                    if global_is_old_arch:
                                                   
                        model_type = config.MODEL_CONFIG.get('type', 'dnn')
                        print(f"[Aggregator]  檢測到架構變更（全局權重是舊架構，客戶端權重是新架構），更新全局權重架構")
                        print(f"  - 缺失的鍵（舊架構）: {len(filtered_missing)} 個")
                        print(f"  - 新增的鍵（新架構）: {len(filtered_extra)} 個")
                                                                   
                        reset_global_weights()
                                            
                        gw_keys = set(global_weights.keys())
                                            
                        missing = list(gw_keys - wk_keys)
                        extra = list(wk_keys - gw_keys)
                        filtered_missing = [k for k in missing if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']) and not k.startswith('_teacher_model')]
                        filtered_extra = [k for k in extra if not any(bn_key in k for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']) and not k.startswith('_teacher_model')]
                        print(f"[Aggregator]  全局權重已重新初始化為 {model_type} 架構，權重層數: {len(global_weights)}")
                        print(f"[Aggregator] 重新檢查後 - missing: {len(filtered_missing)}, extra: {len(filtered_extra)}")
                    elif client_is_old_arch:
                                                    
                        print(f"[Aggregator]  檢測到客戶端 {client_id} 使用舊架構（DNN），全局權重已更新為新架構（Transformer）")
                        print(f"  - 全局權重缺少的鍵（新架構）: {len(filtered_missing)} 個")
                        print(f"  - 客戶端權重多餘的鍵（舊架構）: {len(filtered_extra)} 個")
                        print(f"[Aggregator] 拒絕客戶端 {client_id} 的權重，請客戶端更新為 Transformer 架構後重試")
                        raise ValueError(f"客戶端 {client_id} 使用舊架構（DNN），全局權重已更新為新架構（Transformer）。請客戶端更新後重試。")
                    
                    if filtered_missing or filtered_extra:
                        missing_display = filtered_missing[:5]
                        extra_display = filtered_extra[:5]
                        raise ValueError(f"權重鍵不一致 missing={missing_display} extra={extra_display}")
                    else:
                        print(f"[Aggregator]  檢測到 BatchNorm 參數或 teacher 模型權重差異，自動忽略: missing={len(missing)} extra={len(extra)}")
                                                   
                        common_keys = gw_keys & wk_keys
                        weights_data = {k: weights_data[k] for k in common_keys if k in weights_data and not k.startswith('_teacher_model')}
                        global_weights = {k: global_weights[k] for k in common_keys if k in global_weights and not k.startswith('_teacher_model')}
                for k, v in weights_data.items():
                    if isinstance(v, torch.Tensor):
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            raise ValueError(f"權重含 NaN/Inf: {k}")
                        if k in global_weights and isinstance(global_weights[k], torch.Tensor):
                            if tuple(v.shape) != tuple(global_weights[k].shape):
                                                    
                                                       
                                                                   
                                print(
                                    f"[Aggregator]  權重形狀不符，將以客戶端權重覆蓋全局模板: "
                                    f"{k} {tuple(global_weights[k].shape)} -> {tuple(v.shape)}"
                                )
                                global_weights[k] = v.detach().cpu().float().clone()

                                   
        if client_id in client_weights_buffer:
            print(f"[Aggregator]  客戶端 {client_id} 權重已存在，覆蓋舊權重")
        
              
        client_weights_buffer[client_id] = weights_data
        client_data_sizes[client_id] = max(0, int(data_size))
        if client_prediction is not None:
            client_predictions_buffer[client_id] = client_prediction
                               
        if STALE_POLICY == 'decay':
            if not hasattr(app.state, 'client_staleness'):
                app.state.client_staleness = {}
            app.state.client_staleness[client_id] = int(locals().get('staleness', 0))
        
        print(f"[Aggregator]  客戶端 {client_id} 權重已接收")
        print(f"[Aggregator] 緩衝區狀態: {len(client_weights_buffer)}/{len(round_clients)} 客戶端")

                
        log_training_event_aggregator(
            'weights_received',
            {
                'round_id': round_count,
                'selected_len': len(round_clients),
                'buffer_size': len(client_weights_buffer),
                'received_client': client_id,
                'model_version': model_version
            }
        )
        
                                 
        do_aggregate = False
        try:
                               
            required_ratio = 0.7              
            required_min = 3               
            current_clients = len(client_weights_buffer)
            expected_clients = max(1, len(round_clients))
            participation = current_clients / expected_clients
            
                     
            if current_clients >= max(required_min, int(expected_clients * required_ratio)):
                do_aggregate = True
                print(f"[Aggregator]  達到聚合門檻: {current_clients}/{expected_clients} ({participation:.1%})")
                           
            elif round_start_time is not None and (time.time() - round_start_time) > AGGREGATION_TIMEOUT:
                min_clients = max(4, int(config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', 4)))
                if current_clients >= min_clients:
                    do_aggregate = True
                    print(f"[Aggregator]  超時強制聚合: {current_clients} 個客戶端")
                else:
                    print(f"[Aggregator]  超時但客戶端不足: {current_clients} < {min_clients}")
            else:
                print(f"[Aggregator]  等待更多客戶端: {current_clients}/{expected_clients} ({participation:.1%})")
        except Exception:
            do_aggregate = check_aggregation_conditions()

        if do_aggregate:
                                  
            if aggregation_lock.locked():
                print(f"[Aggregator]  聚合進行中，暫不重入")
                return {
                    "status": "received",
                    "message": "aggregation_in_progress",
                    "buffer_size": len(client_weights_buffer),
                    "current_round": round_count
                }
            async with aggregation_lock:
                                      
                try:
                    last_ids = getattr(app.state, 'last_aggregated_client_ids', set())
                    last_round = getattr(app.state, 'last_aggregated_round', -1)
                    current_ids = set(client_weights_buffer.keys())
                    if last_round == round_count and current_ids == last_ids:
                                                    
                        allow_after = 60
                        elapsed = 0
                        if round_start_time is not None:
                            try:
                                elapsed = time.time() - round_start_time
                            except Exception:
                                elapsed = 0
                        if elapsed < allow_after:
                            print(f"[Aggregator]  聚合條件達成但與上次聚合客戶相同，等待更多客戶... ({elapsed:.0f}s/<{allow_after}s)")
                            return {
                                "status": "received",
                                "message": "已聚合相同客戶集合，等待新客戶加入",
                                "buffer_size": len(client_weights_buffer),
                                "current_round": round_count
                            }
                        else:
                            print(f"[Aggregator]  相同客戶集合等待超過 {allow_after}s，放行聚合以推進輪次")
                except Exception:
                    pass

            print(f"[Aggregator]  滿足聚合條件，開始聚合...")
            
                    
            client_ids_in_buffer = list(client_weights_buffer.keys())
            client_weights_list = [client_weights_buffer[cid] for cid in client_ids_in_buffer]
            client_data_sizes_list = [client_data_sizes[cid] for cid in client_ids_in_buffer]

                                                                
            if NORM_GUARD_ENABLED or TRUST_ENABLED:
                               
                l2_list = []
                for w in client_weights_list:
                    total = 0.0
                    for v in w.values():
                        if isinstance(v, torch.Tensor):
                            total += float(torch.norm(v.reshape(-1).to(dtype=torch.float64), p=2))
                        elif isinstance(v, np.ndarray):
                            total += float(np.linalg.norm(v.reshape(-1)))
                        elif isinstance(v, list):
                            try:
                                total += float(np.linalg.norm(np.array(v, dtype=np.float32).reshape(-1)))
                            except Exception:
                                continue
                    l2_list.append(total)
                if l2_list:
                    median_l2 = float(np.median(l2_list))
                else:
                    median_l2 = 0.0

                           
                if NORM_GUARD_ENABLED:
                             
                    if not hasattr(app.state, 'client_adaptive_weights'):
                        app.state.client_adaptive_weights = {}
                    if not hasattr(app.state, 'client_performance_history'):
                        app.state.client_performance_history = {}
                    
                    new_sizes = []
                    for idx, cid in enumerate(client_ids_in_buffer):
                        size = float(client_data_sizes_list[idx])
                        adj = 1.0
                        
                                     
                        if median_l2 > 0 and l2_list[idx] > NORM_GUARD_K * median_l2:
                            adj *= NORM_GUARD_PENALTY
                                          
                            app.state.client_adaptive_weights[cid] = app.state.client_adaptive_weights.get(cid, 1.0) * 0.95
                        
                                    
                        adaptive_weight = app.state.client_adaptive_weights.get(cid, 1.0)
                        
                                              
                        if median_l2 > 0:
                            norm_ratio = l2_list[idx] / median_l2
                            
                                                        
                            if norm_ratio < 0.65:                    
                                adaptive_weight *= 1.15
                            elif norm_ratio < 0.9:         
                                adaptive_weight *= 1.05
                            elif norm_ratio > 1.9:                  
                                adaptive_weight *= 0.85
                            elif norm_ratio > 1.35:        
                                adaptive_weight *= 0.95
                            
                                       
                            if hasattr(app.state, 'client_performance_history'):
                                if cid in app.state.client_performance_history:
                                    perf_history = app.state.client_performance_history[cid]
                                    if len(perf_history) >= 3:
                                        recent_avg = sum(perf_history[-3:]) / 3
                                        if recent_avg > 0.25:        
                                            adaptive_weight *= 1.05
                                        elif recent_avg < 0.20:        
                                            adaptive_weight *= 0.95
                        
                                   
                        adaptive_weight = max(0.3, min(3.0, adaptive_weight))
                        app.state.client_adaptive_weights[cid] = adaptive_weight
                        
                                   
                        if not hasattr(app.state, 'client_performance_history'):
                            app.state.client_performance_history = {}
                        if cid not in app.state.client_performance_history:
                            app.state.client_performance_history[cid] = []
                        
                                               
                        if median_l2 > 0:
                            estimated_performance = max(0.1, min(0.5, 1.0 - norm_ratio * 0.3))
                            app.state.client_performance_history[cid].append(estimated_performance)
                                        
                            if len(app.state.client_performance_history[cid]) > 10:
                                app.state.client_performance_history[cid] = app.state.client_performance_history[cid][-10:]
                        
                                    
                        adj *= adaptive_weight
                        
                        new_sizes.append(size * adj)
                    
                    client_data_sizes_list = new_sizes
                    
                               
                    adaptive_weights = [app.state.client_adaptive_weights.get(cid, 1.0) for cid in client_ids_in_buffer]
                    avg_weight = sum(adaptive_weights) / len(adaptive_weights)
                    print(f"[Aggregator] 自適應權重統計: 平均={avg_weight:.3f}, 範圍=[{min(adaptive_weights):.3f}, {max(adaptive_weights):.3f}]")
                
                                 
                    if not hasattr(app.state, 'performance_history'):
                        app.state.performance_history = []
                    
                                       
                    if median_l2 > 0:
                        avg_norm_ratio = sum(l2_list) / len(l2_list) / median_l2
                        estimated_global_performance = max(0.1, min(0.5, 1.0 - avg_norm_ratio * 0.3))
                        app.state.performance_history.append(estimated_global_performance)
                        
                                    
                        if len(app.state.performance_history) > 20:
                            app.state.performance_history = app.state.performance_history[-20:]
                        
                                  
                        if len(app.state.performance_history) >= 5:
                            recent_5 = app.state.performance_history[-5:]
                            if all(recent_5[i] <= recent_5[i-1] for i in range(1, len(recent_5))):
                                print(f"[Aggregator]  檢測到性能下降趨勢，可能過擬合")
                                           
                                for cid in client_ids_in_buffer:
                                    current_weight = app.state.client_adaptive_weights.get(cid, 1.0)
                                    app.state.client_adaptive_weights[cid] = max(0.3, current_weight * 0.9)
                                                               
            if STALE_POLICY == 'decay':
                staleness_list = [int(getattr(app.state, 'client_staleness', {}).get(cid, 0)) for cid in client_ids_in_buffer]
                decay_weights = [math.exp(-STALENESS_DECAY_LAMBDA * s) for s in staleness_list]
                                                 
                client_data_sizes_list = [max(0.0, ds) * float(dw) for ds, dw in zip(client_data_sizes_list, decay_weights)]
            
                                                
            client_weight_stats = {}                                                             
            try:
                                             
                for idx, client_id in enumerate(client_ids_in_buffer):
                    if idx < len(client_weights_list):
                        weights = client_weights_list[idx]
                        if not isinstance(weights, dict):
                            continue
                        all_weights_flat = []
                        for key, value in weights.items():
                            if isinstance(value, torch.Tensor):
                                all_weights_flat.extend(value.cpu().numpy().flatten().tolist())
                            elif isinstance(value, (list, np.ndarray)):
                                if isinstance(value, np.ndarray):
                                    all_weights_flat.extend(value.flatten().tolist())
                                else:
                                    all_weights_flat.extend(value)
                        
                        if all_weights_flat:
                            client_weight_stats[client_id] = {
                                'norm': float(np.linalg.norm(all_weights_flat)),
                                'mean': float(np.mean(all_weights_flat)),
                                'std': float(np.std(all_weights_flat)),
                                'min': float(np.min(all_weights_flat)),
                                'max': float(np.max(all_weights_flat))
                            }
            except Exception as e:
                print(f"[Aggregator]  計算客戶端權重統計失敗: {e}")
            
                  
            try:
                log_event("aggregation_started", f"開始聚合 {len(client_weights_list)} 個客戶端權重")
                
                                  
                regional_config = getattr(config, 'REGIONAL_AGGREGATION_CONFIG', {}) or {}
                use_regional = regional_config.get('enabled', False) and REGIONAL_AGGREGATION_AVAILABLE
                
                if use_regional:
                    print(f"[Aggregator]  使用區域性聚合策略（ConfShield + Regional Alignment）")
                    try:
                                   
                        confshield_cfg = regional_config.get('confshield', {})
                        alignment_cfg = regional_config.get('regional_alignment', {})
                        
                        regional_aggregator = RegionalAggregator(
                            confshield_config=confshield_cfg,
                            alignment_config=alignment_cfg
                        )
                        
                                 
                        sizes_int: List[int] = [int(x) for x in client_data_sizes_list]
                        aggregated_weights, agg_info = regional_aggregator.aggregate(
                            client_weights_list,
                            sizes_int,
                            global_weights if global_weights else {}
                        )
                        
                                
                        print(f"[Aggregator] 區域性聚合信息:")
                        print(f"  - 原始客戶端數: {agg_info['original_count']}")
                        print(f"  - 過濾後客戶端數: {agg_info['filtered_count']}")
                        print(f"  - 特徵對齊: {'已應用' if agg_info['alignment_applied'] else '未應用'}")
                        
                        if not aggregated_weights and regional_config.get('fallback_to_fedavg', True):
                            print(f"[Aggregator]  區域性聚合失敗，回退到標準 FedAvg")
                            aggregated_weights = perform_standard_fedavg(client_weights_list, client_data_sizes_list)
                    except Exception as reg_e:
                        print(f"[Aggregator]  區域性聚合異常: {reg_e}，回退到標準 FedAvg")
                        if regional_config.get('fallback_to_fedavg', True):
                            aggregated_weights = perform_standard_fedavg(client_weights_list, client_data_sizes_list)
                        else:
                            raise
                else:
                                 
                    aggregated_weights = perform_standard_fedavg(client_weights_list, client_data_sizes_list)
                
                print(f"[Aggregator]  聚合完成")
                
            except Exception as e:
                print(f"[Aggregator] 聚合失敗: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "status": "error",
                    "message": f"聚合失敗: {str(e)}"
                }
            
            if aggregated_weights:
                                          
                weight_stats = {}
                try:
                                                 
                                 
                    all_weights_flat = []
                    for key, value in aggregated_weights.items():
                        if isinstance(value, torch.Tensor):
                            all_weights_flat.extend(value.cpu().numpy().flatten().tolist())
                        elif isinstance(value, (list, np.ndarray)):
                            if isinstance(value, np.ndarray):
                                all_weights_flat.extend(value.flatten().tolist())
                            else:
                                all_weights_flat.extend(value)
                    
                    if all_weights_flat:
                        weight_stats = {
                            'mean': float(np.mean(all_weights_flat)),
                            'std': float(np.std(all_weights_flat)),
                            'min': float(np.min(all_weights_flat)),
                            'max': float(np.max(all_weights_flat)),
                            'norm': float(np.linalg.norm(all_weights_flat))
                        }
                except Exception as e:
                    print(f"[Aggregator]  計算聚合權重統計失敗: {e}")
                    weight_stats = {}
                                   
                try:
                    import csv, datetime, os
                    result_dir = os.environ.get('EXPERIMENT_DIR', config.LOG_DIR)
                    os.makedirs(result_dir, exist_ok=True)
                    csv_path = os.path.join(result_dir, f"aggregator_{aggregator_id}_participation.csv")
                    file_exists = os.path.exists(csv_path)
                    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                        fieldnames = [
                            'timestamp','round_id','model_version','num_selected','num_buffered',
                            'participation_ratio','client_ids','data_sizes','adaptive_weights',
                            'weight_mean','weight_std','weight_norm',               
                            'client_weight_norms','client_weight_means','client_weight_stds'                            
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        adaptive_weights_row = []
                        try:
                            adaptive_weights_row = [app.state.client_adaptive_weights.get(cid, 1.0) for cid in client_ids_in_buffer]
                        except Exception:
                            adaptive_weights_row = []
                        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        participation_ratio = 0.0
                        try:
                            denom_clients = max(1, len(round_clients))
                            participation_ratio = len(client_weights_buffer) / denom_clients
                        except Exception:
                            participation_ratio = 0.0
                        
                        
                                                      
                        client_weight_norms_str = ''
                        client_weight_means_str = ''
                        client_weight_stds_str = ''
                        try:
                                                               
                            norms_list = [f"{client_weight_stats.get(cid, {}).get('norm', 0.0):.6f}" for cid in client_ids_in_buffer]
                            means_list = [f"{client_weight_stats.get(cid, {}).get('mean', 0.0):.6f}" for cid in client_ids_in_buffer]
                            stds_list = [f"{client_weight_stats.get(cid, {}).get('std', 0.0):.6f}" for cid in client_ids_in_buffer]
                            client_weight_norms_str = ','.join(norms_list)
                            client_weight_means_str = ','.join(means_list)
                            client_weight_stds_str = ','.join(stds_list)
                        except Exception as e:
                            print(f"[Aggregator]  收集客戶端權重統計失敗: {e}")
                        
                        writer.writerow({
                            'timestamp': ts,
                            'round_id': round_count,
                            'model_version': model_version,
                            'num_selected': len(round_clients),
                            'num_buffered': len(client_weights_buffer),
                            'participation_ratio': participation_ratio,
                            'client_ids': ','.join(map(str, client_ids_in_buffer)),
                            'data_sizes': ','.join(map(str, [client_data_sizes.get(cid, 0) for cid in client_ids_in_buffer])),
                            'adaptive_weights': ','.join([f"{w:.3f}" for w in adaptive_weights_row]) if adaptive_weights_row else '',
                            'weight_mean': weight_stats.get('mean', 0.0),               
                            'weight_std': weight_stats.get('std', 0.0),               
                            'weight_norm': weight_stats.get('norm', 0.0),               
                            'client_weight_norms': client_weight_norms_str,                
                            'client_weight_means': client_weight_means_str,                
                            'client_weight_stds': client_weight_stds_str                 
                        })
                except Exception as _e:
                    print(f"[Aggregator]  寫入參與統計失敗: {_e}")
                          
                if global_performance is None:
                    global_performance = 0.5
                else:
                    denom_clients = max(1, len(round_clients))
                    participation_ratio = len(client_weights_buffer) / denom_clients
                    global_performance = 0.9 * global_performance + 0.1 * participation_ratio
                
                print(f"[Aggregator] 更新全局性能指標: {global_performance:.3f}")
                
                                    
                global_weights = aggregated_weights
                
                                       
                model_version += 1
                print(f"[Aggregator]  更新 model_version: {model_version}")
                
                                 
                                   
                denom_clients2 = max(1, len(round_clients))
                participation_ratio = len(client_weights_list) / denom_clients2
                             
                will_advance = participation_ratio >= 0.6              
                
                if will_advance:
                                                         
                    print(f"[Aggregator]  聚合完成，保持在第 {round_count} 輪")
                    print(f"[Aggregator] 本輪聚合統計: {len(client_weights_list)} 個客戶端參與 (參與率: {participation_ratio:.1%})")
                else:
                    print(f"[Aggregator]  參與率不足，不推進輪次: {len(client_weights_list)}/{denom_clients2} ({participation_ratio:.1%})")
                print(f"[Aggregator] 本輪聚合統計: {len(client_weights_list)} 個客戶端參與")
                
                                       
                if will_advance:
                    app.state.aggregated_in_round = round_count - 1
                    app.state.aggregation_done = True
                    app.state.aggregation_done_round = round_count - 1
                else:
                    app.state.aggregated_in_round = round_count
                    app.state.aggregation_done = True
                    app.state.aggregation_done_round = round_count
                app.state.last_aggregated_client_ids = set(client_weights_buffer.keys())
                app.state.last_aggregated_round = round_count if not will_advance else (round_count - 1)
                
                             
                participating_clients = list(client_weights_buffer.keys())
                                              
                if will_advance:
                    client_weights_buffer.clear()
                    client_data_sizes.clear()
                
                print(f"[Aggregator]  聚合完成，清空緩衝區")
                log_event("aggregation_completed", f"聚合完成，參與客戶端: {participating_clients}")
                                                                
                try:
                    global last_completed_round
                    last_completed_round = round_count
                except Exception:
                    pass
                           
                try:
                    globals()['total_aggregations_done'] = int(globals().get('total_aggregations_done', 0)) + 1
                except Exception:
                    globals()['total_aggregations_done'] = 1
                save_aggregation_stats()

                                      
                try:
                                              
                    cloud_url = CLOUD_SERVER_CONFIG.get("url", "").rstrip('/')
                    base_version = 0
                    base_weights = None
                    if cloud_url:
                        try:
                            timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
                            async with aiohttp.ClientSession(timeout=timeout) as session:
                                async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                                    if resp.status == 200:
                                        payload = pickle.loads(await resp.read())
                                        base_version = int(payload.get('version', 0))
                                        base_weights = payload.get('weights', {})
                        except Exception as e:
                            print(f"[Aggregator {aggregator_id}]  取得 cloud base 版本失敗: {e}")
                                             
                    def _to_tensor(x):
                        if isinstance(x, torch.Tensor):
                            return x.detach().cpu().float()
                        elif isinstance(x, np.ndarray):
                            return torch.from_numpy(x).float()
                        elif isinstance(x, list):
                            try:
                                return torch.tensor(x, dtype=torch.float32)
                            except Exception:
                                return None
                        return None
                    delta = {}
                    if isinstance(base_weights, dict) and base_weights:
                        for k, v in global_weights.items():
                            g = _to_tensor(v)
                            b = _to_tensor(base_weights.get(k))
                            if g is not None and b is not None and g.shape == b.shape:
                                delta[k] = (g - b)
                            else:
                                                           
                                if g is not None:
                                    delta[k] = g
                    else:
                                                
                        for k, v in global_weights.items():
                            t = _to_tensor(v)
                            if t is not None:
                                delta[k] = t

                                                                 
                    try:
                        await upload_delta_to_cloud(base_version, delta, round_count - 1)
                    except Exception as e:
                        print(f"[Aggregator {aggregator_id}]  上傳 delta 初次失敗: {e}")
                            
                        try:
                            timeout = aiohttp.ClientTimeout(total=CLOUD_SERVER_CONFIG.get('timeout', 30))
                            async with aiohttp.ClientSession(timeout=timeout) as session:
                                async with session.get(f"{cloud_url}/get_global_weights_with_version") as resp:
                                    if resp.status == 200:
                                        payload = pickle.loads(await resp.read())
                                        base_version = int(payload.get('version', 0))
                                        base_weights = payload.get('weights', {})
                                                    
                                        delta = {}
                                        for k, v in global_weights.items():
                                            g = _to_tensor(v)
                                            b = _to_tensor(base_weights.get(k))
                                            if g is not None and b is not None and g.shape == b.shape:
                                                delta[k] = (g - b)
                                            else:
                                                if g is not None:
                                                    delta[k] = g
                                        await upload_delta_to_cloud(base_version, delta, round_count - 1)
                        except Exception as e2:
                            print(f"[Aggregator {aggregator_id}] 上傳 delta 重試仍失敗: {e2}")
                except Exception as e:
                    print(f"[Aggregator {aggregator_id}] Cloud同步失敗: {e}")
                    log_event("cloud_sync_failed", str(e))

                log_training_event_aggregator(
                    'aggregated',
                    {
                        'round_id': round_count - 1,
                        'aggregated_count': len(client_weights_list),
                        'participation_ratio': len(client_weights_list) / max(1, len(round_clients)),
                        'model_version': model_version
                    }
                )
                
                return {
                    "status": "aggregated",
                    "message": f"聚合完成，{len(client_weights_list)}個客戶端參與",
                    "global_performance": global_performance,
                    "participation_ratio": len(client_weights_list) / max(1, len(round_clients)),
                    "next_round": round_count
                }
        
        return {
            "status": "received",
            "message": f"權重已接收，等待聚合",
            "buffer_size": len(client_weights_buffer),
            "expected_clients": len(round_clients),
            "current_round": round_count
        }
        
    except Exception as e:
        print(f"[Aggregator] 處理客戶端 {client_id} 權重時發生錯誤: {str(e)}")
        import traceback as tb_module
        print(f"[Aggregator] 錯誤詳情: {tb_module.format_exc()}")
        raise HTTPException(status_code=500, detail=f"權重處理失敗: {str(e)}")

@app.get("/get_global_weights")
async def get_global_weights():
    """獲取全局權重"""
                   
    if global_weights is None:
        initialize_global_weights()
    
    if global_weights is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "全局權重初始化失敗", "available": False}
        )
    
                              
    json_weights = {}
    if global_weights:
        for key, value in global_weights.items():
            if isinstance(value, torch.Tensor):
                                      
                if value.device.type != 'cpu':
                    value = value.cpu()
                json_weights[key] = value.numpy().tolist()
            elif isinstance(value, (int, float, bool)):
                          
                json_weights[key] = value
            elif isinstance(value, (list, tuple)):
                         
                json_weights[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
            else:
                                 
                try:
                    json_weights[key] = float(value)
                except (ValueError, TypeError):
                    print(f"[Aggregator]  無法序列化權重 {key}: {type(value)}")
                    json_weights[key] = 0.0
    
                 
    json_server_weights = {}
    if server_global_weights:
        for key, value in server_global_weights.items():
            if isinstance(value, torch.Tensor):
                if value.device.type != 'cpu':
                    value = value.cpu()
                json_server_weights[key] = value.numpy().tolist()
            elif isinstance(value, (int, float, bool)):
                json_server_weights[key] = value
            elif isinstance(value, (list, tuple)):
                json_server_weights[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
            else:
                try:
                    json_server_weights[key] = float(value)
                except (ValueError, TypeError):
                    print(f"[Aggregator]  無法序列化伺服器權重 {key}: {type(value)}")
                    json_server_weights[key] = 0.0
    
                   
    weights_info = {
        "status": "success",
        "available": True,
        "global_weights": json_weights,
        "weights_count": len(global_weights) if global_weights else 0,
        "model_version": model_version,
        "server_weights": json_server_weights if json_server_weights else None,
        "server_model_version": server_model_version,
        "round_count": round_count,
        "aggregator_id": aggregator_id,
        "timestamp": time.time()
    }
    
    return JSONResponse(content=weights_info)

@app.get("/aggregation_status")
async def get_aggregation_status():
    """獲取聚合狀態"""
    try:
        status = {
            'federated_round': {
                'current_round': round_count,
                'selected_clients': round_clients,
                'buffer_size': len(client_weights_buffer),
                'min_clients_required': 2,                      
                'round_start_time': round_start_time,
                'elapsed_time': time.time() - round_start_time if round_start_time else 0,
                'timeout_clients': [cid for cid, timeout in client_timeout_status.items() if timeout]
            },
            'global_weights': {
                'available': global_weights is not None,
                'model_version': model_version
            },
            'timeout_config': {
                'aggregation_timeout': AGGREGATION_TIMEOUT,
                'partial_aggregation_enabled': PARTIAL_AGGREGATION_ENABLED,
                'min_partial_ratio': MIN_PARTIAL_RATIO
            }
        }
        
        return convert_numpy_values(status)
        
    except Exception as e:
        error_msg = f"獲取聚合狀態時發生錯誤: {str(e)}"
        print(f"[Aggregator {aggregator_id}] {error_msg}")
        log_event("aggregation_status_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/enhanced_status")
async def get_enhanced_status():
    """獲取增強狀態信息"""
    status = {
        "aggregator_id": aggregator_id,
        "round_count": round_count,
        "buffer_size": len(client_weights_buffer),
        "global_weights_available": global_weights is not None,
        "model_info": {
            "version": model_version,
            "available": global_weights is not None
        },
        "training_phase": is_training_phase,
        "selected_clients": round_clients,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return convert_numpy_values(status)

@app.get("/federated_status")
async def get_federated_status():
    """獲取聯邦學習狀態 - 客戶端需要的端點"""
    try:
                                         
        print(f"[Aggregator {aggregator_id}] federated_status 查詢: round_count={round_count}, selected_clients={round_clients}, buffer_size={len(client_weights_buffer)}")
        
        status = {
            "aggregator_id": aggregator_id,
            "round_count": round_count,
            "current_round": round_count,
            "buffer_size": len(client_weights_buffer),
            "global_weights_available": global_weights is not None,
            "model_info": {
                "version": model_version,
                "available": global_weights is not None
            },
            "training_phase": is_training_phase,
            "selected_clients": round_clients,
            "min_clients_required": 2,                      
            "aggregation_timeout": AGGREGATION_TIMEOUT,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        result = convert_numpy_values(status)
        selected = result.get('selected_clients', []) if isinstance(result, dict) else []
        print(f"[Aggregator {aggregator_id}]  federated_status 返回: round={round_count}, selected_count={len(round_clients)}, selected={selected}")
        return result
    except Exception as e:
        error_msg = f"獲取聯邦學習狀態時發生錯誤: {str(e)}"
        print(f"[Aggregator {aggregator_id}] {error_msg}")
        log_event("federated_status_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/register_client")
async def register_client(client_id: int = Form(...)):
    """註冊客戶端"""
    try:
        print(f"[Aggregator {aggregator_id}]  客戶端 {client_id} 註冊")
        
                       
        assigned_clients = get_assigned_clients()
        if client_id in assigned_clients:
            print(f"[Aggregator {aggregator_id}]  客戶端 {client_id} 註冊成功")
            return {
                "status": "success",
                "message": f"客戶端 {client_id} 註冊成功",
                "aggregator_id": aggregator_id,
                "assigned": True
            }
        else:
            print(f"[Aggregator {aggregator_id}]  客戶端 {client_id} 不在分配列表中")
            return {
                "status": "success",
                "message": f"客戶端 {client_id} 註冊成功（但不在分配列表中）",
                "aggregator_id": aggregator_id,
                "assigned": False
            }
        
    except Exception as e:
        error_msg = f"客戶端註冊失敗: {str(e)}"
        print(f"[Aggregator {aggregator_id}] {error_msg}")
        log_event("client_registration_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/current_round")
async def get_current_round():
    """獲取當前輪次"""
    return {
        "current_round": round_count,
        "aggregator_id": aggregator_id
    }

@app.post("/sync_state")
async def sync_state(client_id: int = Form(...), last_confirmed_round: int = Form(...)):
    """同步客戶端狀態（回傳當前輪次與選中客戶端）"""
    return {
        "status": "success",
        "aggregator_id": aggregator_id,
        "current_round": round_count,
        "last_confirmed_round": last_confirmed_round,
        "selected_clients": list(round_clients) if round_clients else []
    }

@app.post("/reset_round")
async def reset_round(target_round: int = Form(...)):
    """重置輪次"""
    global round_count, round_clients, client_weights_buffer, round_start_time, client_timeout_status, last_completed_round, is_training_phase
    
    print(f"[Aggregator {aggregator_id}]  重置輪次: {round_count} -> {target_round}")
    
                       
    round_count = target_round
    last_completed_round = target_round - 1               
    round_clients = []
    client_weights_buffer.clear()
    client_data_sizes.clear()
    round_start_time = None
    client_timeout_status.clear()
    is_training_phase = False
    
                  
    if hasattr(app.state, 'not_selected_streak'):
        app.state.not_selected_streak = {}
    if hasattr(app.state, 'last_selected_clients'):
        app.state.last_selected_clients = []
    if hasattr(app.state, 'training_results_buffer'):
        app.state.training_results_buffer.clear()
    
             
    training_flow = config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')
    if training_flow == 'train_then_select':
        round_clients = get_assigned_clients()
    else:
        round_clients = select_clients_for_round(target_round, total_clients=len(get_assigned_clients()))
    
    print(f"[Aggregator {aggregator_id}]  輪次重置完成，選中客戶端: {round_clients}")
    
          
    log_event("round_reset", f"重置到輪次 {target_round}")
    
    return {
            "status": "success",
        "message": f"輪次重置為 {target_round}",
                "current_round": round_count,
        "selected_clients": round_clients
    }

@app.post("/report_availability")
async def report_availability(
    client_id: int = Form(...),
    cpu_usage: float = Form(...),
    memory_usage: float = Form(...),
    battery_level: float = Form(...)
):
    """報告客戶端可用性"""
    try:
        print(f"[Aggregator {aggregator_id}] 客戶端 {client_id} 可用性報告:")
        print(f"  - CPU使用率: {cpu_usage:.2f}%")
        print(f"  - 內存使用率: {memory_usage:.2f}%")
        print(f"  - 電池電量: {battery_level:.2f}%")
        
        return {
            "status": "success",
            "message": "可用性報告已接收",
            "client_id": client_id
        }
        
    except Exception as e:
        error_msg = f"處理可用性報告失敗: {str(e)}"
        print(f"[Aggregator {aggregator_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/receive_global_weights")
async def receive_global_weights(
    weights: UploadFile = File(...),
    global_version: int = Form(...),
    broadcast_type: str = Form("periodic")
):
    """🔧 新增：接收 Cloud Server 廣播的全局權重（分層非同步 FL 核心）"""
    global server_global_weights, server_model_version
    
    try:
        print(f"[Aggregator {aggregator_id}] 收到 Cloud Server 廣播的全局權重 (版本: {global_version}, 類型: {broadcast_type})")
        
                
        weights_data = pickle.loads(weights.file.read())
        
        if isinstance(weights_data, dict) and ('server_weights' in weights_data or 'global_weights' in weights_data):
                          
            new_server_weights = weights_data.get('server_weights', weights_data.get('global_weights'))
            broadcast_version = weights_data.get('global_version', global_version)
            broadcast_timestamp = weights_data.get('timestamp', time.time())
        else:
                         
            new_server_weights = weights_data
            broadcast_version = global_version
            broadcast_timestamp = time.time()
        
                  
        if broadcast_version > server_model_version:
                                      
            server_global_weights = new_server_weights
            server_model_version = broadcast_version
            
            print(f"[Aggregator {aggregator_id}]  成功更新伺服器模型權重: 版本 {server_model_version}")
            log_event("server_weights_updated", f"version={server_model_version},broadcast_type={broadcast_type}")
            
            return {
                "status": "success",
                "message": f"伺服器模型權重已更新到版本 {server_model_version}",
                "aggregator_id": aggregator_id,
                "new_version": server_model_version
            }
        else:
            print(f"[Aggregator {aggregator_id}]  廣播版本 {broadcast_version} 不新於當前版本 {server_model_version}，跳過更新")
            return {
                "status": "skipped",
                "message": f"版本 {broadcast_version} 不新於當前版本 {server_model_version}",
                "aggregator_id": aggregator_id,
                "current_version": server_model_version
            }
        
    except Exception as e:
        error_msg = f"處理廣播全局權重失敗: {str(e)}"
        print(f"[Aggregator {aggregator_id}] {error_msg}")
        log_event("broadcast_receive_error", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/select_clients_after_training")
async def select_clients_after_training(
    client_id: int = Form(...),
    training_results: str = Form(...),
    round_id: int = Form(...)
):
    """訓練後選擇客戶端 - 用於train_then_select流程"""
    try:
        print(f"[Aggregator {aggregator_id}] 收到客戶端 {client_id} 訓練後選擇請求")
        
                
        try:
            import json
            results_data = json.loads(training_results)
        except json.JSONDecodeError:
                                   
            try:
                results_data = pickle.loads(training_results.encode())
            except Exception as e:
                print(f"[Aggregator {aggregator_id}] 訓練結果解析失敗: {e}")
                results_data = {}
        
                      
        record_client_performance(client_id, results_data, round_id)
        
                
        if not hasattr(app.state, 'training_results_buffer'):
            app.state.training_results_buffer = {}
        
        app.state.training_results_buffer[client_id] = {
            'results': results_data,
            'round_id': round_id,
            'timestamp': time.time()
        }
        
        print(f"[Aggregator {aggregator_id}]  客戶端 {client_id} 訓練結果已接收")
        print(f"[Aggregator {aggregator_id}] 當前訓練結果緩衝區: {len(app.state.training_results_buffer)} 個客戶端")
        
                                        
        current_count = len(app.state.training_results_buffer)
        total_expected = max(1, len(round_clients))
        min_partial_ratio = MIN_PARTIAL_RATIO if 'MIN_PARTIAL_RATIO' in globals() else 0.3
        min_required = max(1, min(total_expected, int(math.ceil(min_partial_ratio * total_expected))))

        if current_count >= min_required:
            print(f"[Aggregator {aggregator_id}]  收到 {current_count}/{total_expected} 份訓練結果（門檻 {min_required}），開始選擇最佳客戶端")
            
                         
            selected_clients = select_best_clients_after_training()
            
            return {
                "status": "selection_complete",
                "message": "客戶端選擇完成",
                "selected_clients": selected_clients,
                "total_clients": current_count
            }
        else:
            return {
                "status": "waiting",
                "message": f"等待更多客戶端完成訓練 ({current_count}/{total_expected})",
                "selected_clients": [],
                "total_clients": current_count,
                "min_required": min_required
            }
        
    except Exception as e:
        error_msg = f"處理訓練後選擇請求失敗: {str(e)}"
        print(f"[Aggregator {aggregator_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

def select_best_clients_after_training():
    """基於訓練結果選擇最佳客戶端（簡化版）"""
    if not hasattr(app.state, 'training_results_buffer') or not app.state.training_results_buffer:
        return []
    
                      
    selected_clients = list(app.state.training_results_buffer.keys())
    app.state.training_results_buffer.clear()
    
    print(f"[Aggregator {aggregator_id}]  選擇客戶端: {selected_clients}")
    return selected_clients

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="啟動聚合器 (修復版本)")
    parser.add_argument("--aggregator_id", type=int, default=0, help="聚合器ID")
    parser.add_argument("--port", type=int, default=8000, help="端口號")
    
    args = parser.parse_args()

             
    aggregator_id = args.aggregator_id
    aggregator_port = args.port
    
    print(f"[Aggregator {aggregator_id}]  啟動聚合器 (端口: {args.port})")
    print(f"[Aggregator {aggregator_id}] 配置信息:")
    print(f"  - 聚合超時: {AGGREGATION_TIMEOUT}s")
    print(f"  - 最小客戶端數: {config.AGGREGATION_CONFIG.get('min_clients_for_aggregation', 3)}")
    print(f"  - 訓練流程: {config.FEDERATED_CONFIG.get('training_flow', 'select_then_train')}")
    print(f"  - 平滑因子: {config.FEDERATED_CONFIG.get('aggregation', {}).get('smoothing_factor', 0.8)}")
    
          
    uvicorn.run(app, host="0.0.0.0", port=args.port)
