#!/usr/bin/env python3
import asyncio
import subprocess
import time
import json
import os
import signal
import sys
import aiohttp
import contextlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Optional, Tuple
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import torch as _torch_mod
    _torch_lib = os.path.join(os.path.dirname(_torch_mod.__file__), "lib")
    if os.path.isdir(_torch_lib):
        _prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _torch_lib + (os.pathsep + _prev if _prev else "")
        print("已設定 LD_LIBRARY_PATH 含 torch lib")
except Exception:
    pass
import importlib
CONFIG_MODULE = os.environ.get("CONFIG_MODULE", "config_fixed")
try:
    config = importlib.import_module(CONFIG_MODULE)
    print(f"使用配置模組: {CONFIG_MODULE}")
except Exception as e:
    print(f"載入配置模組 {CONFIG_MODULE} 失敗，回退到 config_fixed: {e}")
    import config_fixed as config
def _get_fl_python() -> str:
    return os.environ.get("FL_PYTHON", os.environ.get("PYTHON_EXECUTABLE", sys.executable))
def _get_fl_base_dir() -> str:
    raw = os.environ.get("FL_BASE_DIR", os.environ.get("EXPERIMENT_BASE_DIR", "")).strip()
    if raw:
        return os.path.abspath(raw)
    return os.path.dirname(os.path.abspath(__file__))
def _normalize_health_url(url: str) -> str:
    if not url:
        return ""
    clean = url.rstrip('/')
    if clean.endswith('/health'):
        return clean
    return f"{clean}/health"
def _get_aggregator_ports() -> List[int]:
    agg_cfg = getattr(config, 'NETWORK_CONFIG', {}).get('aggregators', {})
    ports = list(agg_cfg.get('ports', []))
    base_port = int(agg_cfg.get('base_port', 8000))
    num_aggregators = int(getattr(config, 'NUM_AGGREGATORS', 1))
    if len(ports) < num_aggregators:
        if ports:
            start_port = ports[-1] + 1
        else:
            start_port = base_port
        initial_port_count = len(ports)
        for i in range(initial_port_count, num_aggregators):
            ports.append(start_port + (i - initial_port_count))
    return ports[:num_aggregators]
def _resolve_aggregator_for_client(client_id: int) -> Tuple[int, str]:
    num_aggs = max(1, int(getattr(config, 'NUM_AGGREGATORS', 1)))
    agg_id = client_id % num_aggs
    ports = _get_aggregator_ports() or [8000 + i for i in range(num_aggs)]
    port = ports[min(len(ports) - 1, agg_id)]
    host = getattr(config, 'NETWORK_CONFIG', {}).get('aggregators', {}).get('host', '127.0.0.1')
    return agg_id, f"http://{host}:{port}"
class FixedExperimentMonitor:
    def __init__(self, result_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir_abs = result_dir if os.path.isabs(result_dir) else os.path.join(base_dir, result_dir)
        result_dir_abs = os.path.abspath(result_dir_abs)
        self.result_dir = result_dir_abs
        self.processes = {}
        self.monitoring_active = False
        self.start_time = time.time()
        self.training_start_time = datetime.now().isoformat()
        self.experiment_completed = False
        self.stop_reason = None
        os.makedirs(self.result_dir, exist_ok=True)
        os.environ['EXPERIMENT_DIR'] = self.result_dir
        os.environ['FORCE_CPU'] = '1'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        scaled_default = os.path.join(config.DATA_PATH, 'global_test_scaled.csv')
        raw_default = os.path.join(config.DATA_PATH, 'global_test.csv')
        env_existing = os.environ.get('GLOBAL_TEST_PATH')
        if env_existing:
            global_test_path = env_existing
        else:
            if os.path.exists(scaled_default):
                global_test_path = scaled_default
            else:
                global_test_path = raw_default
        os.environ['GLOBAL_TEST_PATH'] = global_test_path
        os.environ.setdefault('CLIENT_EVAL_SOURCE', 'local_split')
        os.environ.setdefault('CLIENT_LOCAL_EVAL_RATIO', '0.2')
        self.save_experiment_config()
        self.comm_monitor = None
        try:
            from utils.fl_comprehensive_evaluator import FLComprehensiveEvaluator
            num_clients = getattr(config, 'NUM_CLIENTS', 60)
            self.evaluator = FLComprehensiveEvaluator(self.result_dir, num_clients)
            self.evaluator_initialized = True
            print(f"全面評估器已初始化（{num_clients} 個客戶端）")
        except Exception as e:
            print(f"評估器初始化失敗: {e}")
            self.evaluator = None
            self.evaluator_initialized = False
    def set_stop_reason(self, reason: str) -> None:
        if not reason:
            return
        self.stop_reason = reason
        try:
            os.makedirs(self.result_dir, exist_ok=True)
            reason_path = os.path.join(self.result_dir, "stop_reason.txt")
            ts = datetime.now().isoformat()
            with open(reason_path, 'w', encoding='utf-8') as f:
                f.write(f"stop_at={ts}\nreason={reason}\n")
        except Exception as e:
            print(f"記錄停止原因失敗: {e}")
    def save_experiment_config(self):
        config_file = os.path.join(self.result_dir, "experiment_config.txt")
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(f"實驗時間: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"實驗類型: 修復版多聚合器聯邦學習\n")
            f.write(f"實驗目錄: {self.result_dir}\n")
            f.write(f"環境變量: EXPERIMENT_DIR = {os.environ.get('EXPERIMENT_DIR', '未設置')}\n")
            f.write(f"總輪數: {config.FEDERATED_CONFIG['rounds']}\n")
            f.write(f"客戶端數量: {config.NUM_CLIENTS}\n")
            f.write(f"聚合器數量: {config.NUM_AGGREGATORS}\n")
            f.write(f"評估頻率: 每 {config.EVAL_EVERY_ROUNDS} 輪完整評估\n")
            f.write(f"參與率: {config.CLIENT_SELECTION_CONFIG['base_participation_ratio']}\n")
            f.write(f"最小客戶端數: {config.AGGREGATION_CONFIG['min_clients_for_aggregation']}\n")
            f.write(f"學習率: {config.CLIENT_LR}\n")
            f.write(f"批次大小: {config.BATCH_SIZE}\n")
            f.write(f"本地訓練輪數: {config.LOCAL_EPOCHS}\n")
            f.write(f"數據路徑: {config.DATA_PATH}\n")
            f.write(f"訓練流程: {config.FEDERATED_CONFIG['training_flow']}\n")
            kd_config = config.MODEL_CONFIG.get('knowledge_distillation', {})
            f.write(f"知識蒸餾 KD Warmup: {kd_config.get('kd_warmup_rounds', 3)} 輪\n")
            f.write(f"服務器學習率: {config.SERVER_LR}\n")
            try:
                import re
                cloud_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_server_fixed.py")
                if os.path.exists(cloud_server_path):
                    with open(cloud_server_path, 'r', encoding='utf-8') as cs_file:
                        cs_content = cs_file.read()
                        match = re.search(r'max_grad_norm\s*=\s*([0-9.]+)', cs_content)
                        if match:
                            max_grad_norm_value = match.group(1)
                            f.write(f"梯度裁剪閾值: {max_grad_norm_value} (Server端)\n")
                        else:
                            f.write(f"梯度裁剪閾值: 200.0 (Server端，無法讀取)\n")
                else:
                    f.write(f"梯度裁剪閾值: 200.0 (Server端，文件不存在)\n")
            except Exception as e:
                f.write(f"梯度裁剪閾值: 200.0 (Server端，讀取失敗: {e})\n")
            f.write(f"緊急修復功能:\n")
            f.write(f"  - 聚合等待時間: 20秒 → 120秒\n")
            f.write(f"  - 最小客戶端數: 2 → 3 (確保聚合穩定性)\n")
            f.write(f"  - 本地訓練輪數: {config.LOCAL_EPOCHS} (當前配置)\n")
            f.write(f"  - 批次大小: 64 (當前配置)\n")
            f.write(f"  - 數據使用比例: 100% (當前配置)\n")
            try:
                import re
                cloud_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_server_fixed.py")
                if os.path.exists(cloud_server_path):
                    with open(cloud_server_path, 'r', encoding='utf-8') as cs_file:
                        cs_content = cs_file.read()
                        match = re.search(r'max_grad_norm\s*=\s*([0-9.]+)', cs_content)
                        if match:
                            max_grad_norm_value = match.group(1)
                            f.write(f"  - 梯度裁剪: {max_grad_norm_value} (Server端，已優化)\n")
                        else:
                            f.write(f"  - 梯度裁剪: 200.0 (Server端，已優化)\n")
                else:
                    f.write(f"  - 梯度裁剪: 200.0 (Server端，已優化)\n")
            except Exception:
                f.write(f"  - 梯度裁剪: 200.0 (Server端，已優化)\n")
            f.write(f"  - 啟用快速模式\n")
            f.write(f"  - 禁用未被選中客戶端訓練 (避免版本不同步)\n")
            f.write(f"  - 客戶端等待行為控制 (最多領先5輪)\n")
            f.write(f"  - 使用{config.MODEL_CONFIG.get('type', 'DNN').upper()}模型\n")
            f.write(f"  - 大幅強化少數類別權重\n")
            f.write(f"  - 啟用餘弦退火學習率調度器\n")
            f.write(f"  - 提高客戶端參與率到80%\n")
            f.write(f"  - 禁用早停以確保充分訓練\n")
            f.write(f"  - 使用Focal Loss處理類別不平衡\n")
            f.write(f"  - 改善輪次同步機制\n")
            f.write(f"  - 優化聚合條件檢查\n")
            f.write(f"  - 改進客戶端選擇邏輯\n")
            f.write(f"  - 修復日誌文件位置問題\n")
            f.write(f"  - 確保環境變量正確設置\n")
    def add_process(self, name, process, port=None):
        self.processes[name] = {
            'process': process,
            'port': port,
            'start_time': time.time(),
            'status': 'running'
        }
        print(f"添加進程: {name} (PID: {process.pid})")
    def check_process_health(self):
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            actual_processes = {
                'cloud_server': 0,
                'aggregator': 0,
                'client': 0
            }
            for line in lines:
                if 'cloud_server_fixed.py' in line:
                    actual_processes['cloud_server'] += 1
                elif 'aggregator_fixed.py' in line:
                    actual_processes['aggregator'] += 1
                elif 'uav_client_fixed.py' in line:
                    actual_processes['client'] += 1
            for name, info in self.processes.items():
                if 'cloud_server' in name:
                    info['status'] = 'running' if actual_processes['cloud_server'] > 0 else 'stopped'
                elif 'aggregator' in name:
                    info['status'] = 'running' if actual_processes['aggregator'] > 0 else 'stopped'
                elif 'client' in name:
                    current_time = time.time()
                    if current_time - info['start_time'] < 10:
                        info['status'] = 'running'
                    elif info['process'].poll() is not None:
                        info['status'] = 'stopped'
                    else:
                        info['status'] = 'running'
                else:
                    if info['process'].poll() is not None:
                        info['status'] = 'stopped'
                    else:
                        info['status'] = 'running'
            stopped_clients = [name for name, info in self.processes.items()
                             if 'client' in name and info['status'] == 'stopped']
            if stopped_clients:
                self._check_and_restart_processes()
            return all(info['status'] == 'running' for info in self.processes.values())
        except Exception as e:
            print(f"進程檢查異常: {e}")
            for name, info in self.processes.items():
                if info['process'].poll() is not None:
                    info['status'] = 'stopped'
                else:
                    info['status'] = 'running'
            return all(info['status'] == 'running' for info in self.processes.values())
    def _check_and_restart_processes(self):
        if self.experiment_completed:
            return
        for name, info in self.processes.items():
            if info['status'] == 'stopped':
                print(f"檢測到進程 {name} 已停止，嘗試重啟...")
                try:
                    if 'cloud_server' in name:
                        cmd = [_get_fl_python(), os.path.join(_get_fl_base_dir(), "cloud_server_fixed.py"), "--port", str(info['port'])]
                        env = os.environ.copy()
                        env['EXPERIMENT_DIR'] = self.result_dir
                        new_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=_get_fl_base_dir())
                        info['process'] = new_process
                        info['start_time'] = time.time()
                        info['status'] = 'running'
                        print(f"重啟 Cloud Server 成功 (PID: {new_process.pid})")
                    elif 'aggregator' in name:
                        parts = name.split('_')
                        if len(parts) >= 2:
                            agg_id = parts[1]
                            cmd = [_get_fl_python(), os.path.join(_get_fl_base_dir(), "aggregator_fixed.py"), "--aggregator_id", agg_id, "--port", str(info['port'])]
                            env = os.environ.copy()
                            env['EXPERIMENT_DIR'] = self.result_dir
                            new_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=_get_fl_base_dir())
                            info['process'] = new_process
                            info['start_time'] = time.time()
                            info['status'] = 'running'
                            print(f"重啟聚合器 {agg_id} 成功 (PID: {new_process.pid})")
                    elif 'client' in name:
                        parts = name.split('_')
                        if len(parts) >= 2:
                            client_id = parts[1]
                            aggregator_id, aggregator_url = _resolve_aggregator_for_client(int(client_id))
                            cloud_url = config.NETWORK_CONFIG['cloud_server']['url']
                            cmd = [
                                _get_fl_python(), os.path.join(_get_fl_base_dir(), "uav_client_fixed.py"),
                                "--client_id", str(client_id),
                                "--aggregator_url", aggregator_url,
                                "--cloud_url", cloud_url,
                                "--result_dir", self.result_dir
                            ]
                            env = os.environ.copy()
                            env['EXPERIMENT_DIR'] = self.result_dir
                            new_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=_get_fl_base_dir())
                            info['process'] = new_process
                            info['start_time'] = time.time()
                            info['status'] = 'running'
                            print(f"重啟客戶端 {client_id} 成功 (PID: {new_process.pid})")
                except Exception as e:
                    print(f"重啟進程 {name} 失敗: {e}")
                    info['status'] = 'failed'
    def log_status(self):
        status = {
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time,
            "training_start_time": self.training_start_time,
            "processes": []
        }
        for name, info in self.processes.items():
            process_info = {
                "name": name,
                "status": info['status'],
                "uptime": time.time() - info['start_time'],
                "port": info['port']
            }
            status["processes"].append(process_info)
        status["monitoring_active"] = self.monitoring_active
        os.makedirs(self.result_dir, exist_ok=True)
        status_file = os.path.join(self.result_dir, "experiment_status.json")
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        elapsed_time = status['elapsed_time']
        print(f"\n進程狀態 (運行時間: {elapsed_time:.1f}秒, {elapsed_time/60:.1f}分鐘):")
        cloud_processes = [p for p in status["processes"] if 'cloud_server' in p['name']]
        agg_processes = [p for p in status["processes"] if 'aggregator' in p['name']]
        client_processes = [p for p in status["processes"] if 'client' in p['name']]
        if cloud_processes:
            cloud_status = "正常" if all(p["status"] == "running" for p in cloud_processes) else "異常"
            print(f"   Cloud Server: {cloud_status} ({len(cloud_processes)} 個進程)")
        if agg_processes:
            agg_status = "正常" if all(p["status"] == "running" for p in agg_processes) else "異常"
            print(f"   Aggregator: {agg_status} ({len(agg_processes)} 個進程)")
        if client_processes:
            running_clients = sum(1 for p in client_processes if p["status"] == "running")
            min_healthy_clients = max(1, int(len(client_processes) * 0.5))
            client_status = "正常" if running_clients >= min_healthy_clients else "異常"
            print(f"   Client: {client_status} ({running_clients}/{len(client_processes)} 個進程)")
        total_processes = len(status["processes"])
        running_processes = sum(1 for p in status["processes"] if p["status"] == "running")
        health_ratio = running_processes / total_processes if total_processes > 0 else 0
        excellent_threshold = 0.9
        good_threshold = 0.7
        if health_ratio >= excellent_threshold:
            health_status = "健康"
        elif health_ratio >= good_threshold:
            health_status = "部分正常"
        else:
            health_status = "異常"
        print(f"   系統狀態: {health_status} ({running_processes}/{total_processes})")
    async def monitor_experiment(self):
        self.monitoring_active = True
        print("開始監控實驗...")
        best_metric = -1.0
        best_round = 0
        no_improve_rounds = 0
        es_patience = getattr(config, 'CONVERGENCE_CONFIG', {}).get('patience', 50)
        es_min_delta = getattr(config, 'CONVERGENCE_CONFIG', {}).get('min_improvement', 0.001)
        es_min_rounds = getattr(config, 'CONVERGENCE_CONFIG', {}).get('min_rounds', 0)
        experiment_cfg = getattr(config, 'EXPERIMENT_CONFIG', {}) or {}
        early_stop_enabled = bool(experiment_cfg.get('early_stopping', True))
        drop_cfg = getattr(config, 'DROP_GUARD_CONFIG', {}) or {}
        drop_enabled = bool(drop_cfg.get('enabled', False))
        drop_min_rounds = int(drop_cfg.get('min_rounds', 0))
        drop_threshold = float(drop_cfg.get('drop_threshold', 0.0))
        drop_patience = int(drop_cfg.get('drop_patience', 0))
        drop_streak = 0
        while self.monitoring_active:
            try:
                all_healthy = self.check_process_health()
                if not all_healthy:
                    print("檢測到進程異常，但繼續監控...")
                training_status = await self.get_training_status()
                if early_stop_enabled:
                    early_stop_file = os.path.join(self.result_dir, "EARLY_STOPPED.txt")
                    if os.path.exists(early_stop_file):
                        try:
                            with open(early_stop_file, 'r', encoding='utf-8') as f:
                                early_stop_content = f.read().strip()
                            print(f"\n檢測到 Cloud Server 早停標記：{early_stop_content}")
                            self.set_stop_reason(f"cloud_early_stop_flag:{early_stop_content}")
                            self.experiment_completed = True
                            self.monitoring_active = False
                            break
                        except Exception as e:
                            print(f"讀取早停文件失敗: {e}")
                try:
                    exp_dir = self.result_dir
                    client_dirs = [d for d in os.listdir(exp_dir) if d.startswith('uav')]
                    metric_values = []
                    for cd in client_dirs:
                        curve_path = os.path.join(exp_dir, cd, f"{cd}_curve.csv")
                        if os.path.exists(curve_path):
                            try:
                                import pandas as pd
                                df = pd.read_csv(curve_path)
                                if 'round' in df.columns and not df.empty:
                                    if 'joint_f1' in df.columns:
                                        metric_values.append(df['joint_f1'].iloc[-1])
                                    elif 'f1_score' in df.columns:
                                        metric_values.append(df['f1_score'].iloc[-1])
                                    elif 'acc' in df.columns:
                                        metric_values.append(df['acc'].iloc[-1])
                                    elif 'accuracy' in df.columns:
                                        metric_values.append(df['accuracy'].iloc[-1])
                            except Exception:
                                pass
                    if len(metric_values) >= 3:
                        current_metric = sum(metric_values) / len(metric_values)
                        agg_rounds = [v.get('round_count', 0) for v in training_status.get('aggregators', {}).values() if 'error' not in v]
                        current_global_round = min(agg_rounds) if agg_rounds else 0
                        if current_metric > best_metric + es_min_delta:
                            best_metric = current_metric
                            best_round = current_global_round
                            no_improve_rounds = 0
                        else:
                            no_improve_rounds = (no_improve_rounds + 1) if current_global_round > best_round else no_improve_rounds
                        if not early_stop_enabled:
                            pass
                        elif current_global_round < es_min_rounds:
                            continue
                        performance_threshold = 0.3
                        if early_stop_enabled and es_patience > 0 and no_improve_rounds >= es_patience and best_metric >= performance_threshold:
                            print(f"\n早停觸發：{es_patience} 輪未提升 (best={best_metric:.6f})，準備停止實驗")
                            self.set_stop_reason(
                                f"monitor_early_stop:best={best_metric:.6f},patience={es_patience}"
                            )
                            try:
                                os.makedirs(self.result_dir, exist_ok=True)
                                status_file = os.path.join(self.result_dir, "experiment_status.json")
                                training_end_time = datetime.now().isoformat()
                                training_elapsed_seconds = int(time.time() - self.start_time)
                                if os.path.exists(status_file):
                                    with open(status_file, 'r', encoding='utf-8') as f:
                                        status = json.load(f)
                                else:
                                    status = {}
                                status["training_end_time"] = training_end_time
                                status["training_elapsed_seconds"] = training_elapsed_seconds
                                with open(status_file, 'w', encoding='utf-8') as f:
                                    json.dump(status, f, indent=2, ensure_ascii=False)
                            except Exception as _werr:
                                print(f"記錄訓練結束時間失敗: {_werr}")
                            self.experiment_completed = True
                            self.monitoring_active = False
                        elif early_stop_enabled and es_patience > 0 and no_improve_rounds >= es_patience and best_metric < performance_threshold:
                            print(f"\n性能過低 (best={best_metric:.6f} < {performance_threshold})，繼續訓練...")
                            no_improve_rounds = 0
                        if drop_enabled and current_global_round >= drop_min_rounds and best_metric > 0:
                            if current_metric < best_metric - drop_threshold:
                                drop_streak += 1
                            else:
                                drop_streak = 0
                            if drop_patience > 0 and drop_streak >= drop_patience:
                                print(f"\n嚴重下滑觸發停止：best={best_metric:.6f}, current={current_metric:.6f}")
                                self.set_stop_reason(
                                    f"drop_guard:best={best_metric:.6f},current={current_metric:.6f},"
                                    f"threshold={drop_threshold},patience={drop_patience}"
                                )
                                self.experiment_completed = True
                                self.monitoring_active = False
                except Exception as _es_err:
                    print(f"早停檢查異常: {_es_err}")
                try:
                    configured_max_rounds = getattr(config, 'MAX_ROUNDS', None)
                    if configured_max_rounds is None:
                        configured_max_rounds = getattr(config, 'FEDERATED_CONFIG', {}).get('rounds', None)
                    if configured_max_rounds is None:
                        configured_max_rounds = getattr(config, 'FEDERATED_CONFIG', {}).get('max_rounds', None)
                    if configured_max_rounds is None:
                        configured_max_rounds = getattr(config, 'CONVERGENCE_CONFIG', {}).get('max_rounds', 70)
                    print(f"[Monitor] 檢查停止條件：最大輪數={configured_max_rounds}")
                    agg_rounds = []
                    for agg_name, agg_status in training_status.get('aggregators', {}).items():
                        if 'error' not in agg_status:
                            round_count = agg_status.get('round_count', 0)
                            agg_rounds.append(round_count)
                            print(f"[Monitor]   {agg_name}: 第{round_count}輪")
                    if len(agg_rounds) >= 1:
                        min_agg_round = min(agg_rounds)
                        print(f"[Monitor]   最小聚合器輪次={min_agg_round}, 最大輪數={configured_max_rounds}")
                        if min_agg_round >= configured_max_rounds:
                            print(f"\n已達到最大輪數 ({configured_max_rounds})，準備自動停止實驗")
                            self.set_stop_reason(f"max_rounds_reached:{configured_max_rounds}")
                            try:
                                stop_flag = os.path.join(self.result_dir, "stop.broadcast")
                                with open(stop_flag, 'w', encoding='utf-8') as f:
                                    f.write(f"stop_at={datetime.now().isoformat()}\nmax_rounds={configured_max_rounds}\n")
                                print(f"已廣播停止訊號: {stop_flag}")
                            except Exception as _stop_err:
                                print(f"廣播停止訊號失敗: {_stop_err}")
                            try:
                                cloud_url = getattr(config, 'NETWORK_CONFIG', {}).get('cloud_server', {}).get('url', 'http://127.0.0.1:8083')
                                health_url = f"{cloud_url.rstrip('/')}/health"
                                async with aiohttp.ClientSession() as session:
                                    try:
                                        async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                                            if resp.status == 200:
                                                print(f"雲端伺服器健康檢查通過 (status={resp.status})")
                                            else:
                                                print(f"雲端伺服器健康檢查異常 (status={resp.status})")
                                    except Exception as health_err:
                                        print(f"雲端伺服器健康檢查失敗: {health_err}")
                            except Exception as check_err:
                                print(f"檢查雲端伺服器狀態失敗: {check_err}")
                            done_flag = os.path.join(self.result_dir, "experiment_completed.flag")
                            with open(done_flag, 'w', encoding='utf-8') as f:
                                f.write(f"completed_at={datetime.now().isoformat()}\nmax_rounds={configured_max_rounds}\n")
                            try:
                                os.makedirs(self.result_dir, exist_ok=True)
                                status_file = os.path.join(self.result_dir, "experiment_status.json")
                                training_end_time = datetime.now().isoformat()
                                training_elapsed_seconds = int(time.time() - self.start_time)
                                if os.path.exists(status_file):
                                    with open(status_file, 'r', encoding='utf-8') as f:
                                        status = json.load(f)
                                else:
                                    status = {}
                                status["training_end_time"] = training_end_time
                                status["training_elapsed_seconds"] = training_elapsed_seconds
                                try:
                                    cloud_log_path = os.path.join(self.result_dir, "cloud_server_log.csv")
                                    if os.path.exists(cloud_log_path):
                                        import pandas as pd
                                        df = pd.read_csv(cloud_log_path)
                                        if 'round' in df.columns and not df.empty:
                                            last_round = int(df['round'].max())
                                            status["last_aggregation_round"] = last_round
                                            print(f"最後一次聚合輪次: {last_round}")
                                except Exception as round_err:
                                    print(f"讀取最後聚合輪次失敗: {round_err}")
                                with open(status_file, 'w', encoding='utf-8') as f:
                                    json.dump(status, f, indent=2, ensure_ascii=False)
                            except Exception as _werr:
                                print(f"記錄訓練結束時間失敗: {_werr}")
                            self.experiment_completed = True
                            self.monitoring_active = False
                except Exception as _auto_stop_err:
                    print(f"自動停止檢查異常: {_auto_stop_err}")
                self.log_status()
                await self.display_training_progress(training_status)
                monitoring_interval = 10
                if self.monitoring_active:
                    await asyncio.sleep(monitoring_interval)
            except KeyboardInterrupt:
                print("\n收到中斷信號，停止監控...")
                break
            except Exception as e:
                print(f"監控異常: {e}")
                await asyncio.sleep(10)
        self.monitoring_active = False
    async def get_training_status(self):
        import aiohttp
        status = {
            'aggregators': {},
            'clients': {},
            'timestamp': time.time()
        }
        try:
            aggregator_ports = _get_aggregator_ports()
            num_aggs = int(getattr(config, 'NUM_AGGREGATORS', 2))
            if len(aggregator_ports) < num_aggs:
                base_port = getattr(config, 'NETWORK_CONFIG', {}).get('aggregators', {}).get('base_port', 8000)
                for i in range(len(aggregator_ports), num_aggs):
                    aggregator_ports.append(base_port + i)
        except Exception:
            base_port = getattr(config, 'NETWORK_CONFIG', {}).get('aggregators', {}).get('base_port', 8000)
            num_aggs = int(getattr(config, 'NUM_AGGREGATORS', 2))
            aggregator_ports = [base_port + i for i in range(num_aggs)]
        checked_ports = set()
        for i, port in enumerate(aggregator_ports):
            if port in checked_ports:
                continue
            checked_ports.add(port)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{port}/federated_status", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            agg_id = data.get('aggregator_id', i)
                            status['aggregators'][f'agg_{agg_id}'] = {
                                'round_count': data.get('round_count', 0),
                                'buffer_size': data.get('buffer_size', 0),
                                'selected_clients': data.get('selected_clients', []),
                                'training_phase': data.get('training_phase', False)
                            }
            except Exception as e:
                status['aggregators'][f'agg_{i}'] = {'error': str(e)}
        base_port = getattr(config, 'NETWORK_CONFIG', {}).get('aggregators', {}).get('base_port', 8000)
        max_port_to_check = min(base_port + 10, 8010)
        for port in range(base_port, max_port_to_check + 1):
            if port in checked_ports:
                continue
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{port}/federated_status", timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            data = await response.json()
                            agg_id = data.get('aggregator_id', port - base_port)
                            if f'agg_{agg_id}' not in status['aggregators']:
                                status['aggregators'][f'agg_{agg_id}'] = {
                                    'round_count': data.get('round_count', 0),
                                    'buffer_size': data.get('buffer_size', 0),
                                    'selected_clients': data.get('selected_clients', []),
                                    'training_phase': data.get('training_phase', False)
                                }
                            checked_ports.add(port)
            except Exception:
                pass
        return status
    async def display_training_progress(self, training_status):
        elapsed = time.time() - self.start_time
        import os
        import sys
        print('\n' * 3)
        separator = '=' * 60
        print(f"{separator}")
        print(f"聯邦學習訓練進度 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{separator}")
        print("聚合器狀態:")
        connected_aggregators = 0
        aggregators = training_status.get('aggregators', {})
        for agg_name, agg_status in aggregators.items():
            if 'error' in agg_status:
                print(f"   {agg_name}: {agg_status['error']}")
            else:
                round_count = agg_status.get('round_count', 0)
                buffer_size = agg_status.get('buffer_size', 0)
                selected = agg_status.get('selected_clients', [])
                training = agg_status.get('training_phase', False)
                print(f"   {agg_name}: 第{round_count}輪, 緩衝區{buffer_size}個權重, 選中客戶端{selected}")
                connected_aggregators += 1
        print("\n客戶端訓練狀態:")
        try:
            client_count = int(getattr(config, 'NUM_CLIENTS', 10))
        except Exception:
            client_count = 10
        client_dirs = [f"uav{i}" for i in range(client_count)]
        active_clients = 0
        completed_clients = 0
        experiment_dir = self.result_dir
        for client_dir in client_dirs:
            client_id = client_dir.replace('uav', '')
            curve_file = os.path.join(experiment_dir, client_dir, f"{client_dir}_curve.csv")
            if os.path.exists(curve_file):
                try:
                    with open(curve_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last_line = lines[-1].strip().split(',')
                            if len(last_line) >= 4:
                                round_num = last_line[0]
                                loss = last_line[1]
                                if len(last_line) >= 5 and last_line[2] == last_line[3]:
                                    accuracy = last_line[2]
                                else:
                                    accuracy = last_line[2]
                                try:
                                    loss_float = float(loss)
                                    acc_float = float(accuracy)
                                    print(f"   客戶端{client_id}: 第{round_num}輪, 損失={loss_float:.6f}, 準確率={acc_float:.6f}")
                                    completed_clients += 1
                                except (ValueError, TypeError):
                                    print(f"   客戶端{client_id}: 第{round_num}輪, 損失={loss}, 準確率={accuracy}")
                                    completed_clients += 1
                            else:
                                print(f"   客戶端{client_id}: 正在初始化...")
                        else:
                            print(f"   客戶端{client_id}: 等待開始...")
                except Exception as e:
                    print(f"   客戶端{client_id}: 讀取錯誤 - {str(e)[:30]}")
            else:
                training_log_file = os.path.join(experiment_dir, f"training_log_client_{client_id}.csv")
                if os.path.exists(training_log_file):
                    try:
                        with open(training_log_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                last_line = lines[-1].strip().split(',')
                                if len(last_line) >= 6:
                                    timestamp = last_line[0]
                                    event = last_line[2]
                                    round_id = last_line[3]
                                    loss = last_line[4]
                                    acc = last_line[5]
                                    try:
                                        loss_float = float(loss)
                                        acc_float = float(acc)
                                        print(f"   客戶端{client_id}: 事件={event}, 輪次={round_id}, 損失={loss_float:.6f}, 準確率={acc_float:.6f}")
                                        completed_clients += 1
                                    except (ValueError, TypeError):
                                        print(f"   客戶端{client_id}: 事件={event}, 輪次={round_id}, 損失={loss}, 準確率={acc}")
                                        completed_clients += 1
                                else:
                                    print(f"   客戶端{client_id}: 正在初始化...")
                            else:
                                print(f"   客戶端{client_id}: 等待開始...")
                    except Exception as e:
                        print(f"   客戶端{client_id}: 讀取錯誤 - {str(e)[:30]}")
                else:
                    print(f"   客戶端{client_id}: 未啟動")
            active_clients += 1
        print(f"\n整體進度:")
        try:
            client_total = int(getattr(config, 'NUM_CLIENTS', 10))
        except Exception:
            client_total = 10
        try:
            actual_agg_ports = _get_aggregator_ports()
            agg_total = len(actual_agg_ports)
            if connected_aggregators > agg_total:
                agg_total = connected_aggregators
        except Exception:
            try:
                agg_total = int(getattr(config, 'NUM_AGGREGATORS', 2))
            except Exception:
                agg_total = 2
        print(f"   活躍客戶端: {active_clients}/{client_total}")
        print(f"   完成訓練客戶端: {completed_clients}/{client_total}")
        print(f"   連接聚合器: {connected_aggregators}/{agg_total}")
        sys.stdout.flush()
        total_buffer = sum(agg.get('buffer_size', 0) for agg in aggregators.values() if 'error' not in agg)
        if total_buffer > 0:
            print(f"   等待聚合權重: {total_buffer}個")
        health_score = 0
        if connected_aggregators >= 1:
            health_score += 1
        min_completed_ratio = 0.15
        min_completed_clients = max(1, int(client_total * min_completed_ratio))
        if completed_clients >= min_completed_clients:
            health_score += 1
        if active_clients >= client_total:
            health_score += 1
        if health_score == 3:
            health_status = "健康"
        elif health_score >= 2:
            health_status = "部分正常"
        else:
            health_status = "異常"
        print(f"   實驗狀態: {health_status} ({health_score}/3)")
        separator = '=' * 60
        print(f"{separator}")
        sys.stdout.flush()
    def collect_round_metrics(self, round_id: int):
        if not self.evaluator_initialized or self.evaluator is None:
            return
        try:
            import pandas as pd
            client_metrics = {}
            computation_times = []
            all_participating_clients = set()
            total_data_size = 0
            communication_bytes = 0
            num_clients = getattr(config, 'NUM_CLIENTS', 60)
            num_aggregators = getattr(config, 'NUM_AGGREGATORS', 4)
            for client_id in range(num_clients):
                curve_file = os.path.join(
                    self.result_dir,
                    f'uav{client_id}',
                    f'uav{client_id}_curve.csv'
                )
                if os.path.exists(curve_file):
                    try:
                        df = pd.read_csv(curve_file)
                        round_data = df[df['round'] == round_id]
                        if not round_data.empty:
                            row = round_data.iloc[-1]
                            client_metrics[client_id] = {
                                'acc': float(row.get('acc', 0)),
                                'f1': float(row.get('joint_f1', row.get('f1_score', 0))),
                                'loss': float(row.get('loss', 0)),
                            }
                            if 'val_acc' in row:
                                client_metrics[client_id]['val_acc'] = float(row['val_acc'])
                            if 'val_loss' in row:
                                client_metrics[client_id]['val_loss'] = float(row['val_loss'])
                            if 'train_time_ms' in row and pd.notna(row['train_time_ms']):
                                computation_times.append(float(row['train_time_ms']) / 1000.0)
                            all_participating_clients.add(client_id)
                            if 'data_size' in row and pd.notna(row['data_size']):
                                data_size = int(row['data_size'])
                                total_data_size += data_size
                                estimated_weight_size = data_size * 1024
                                communication_bytes += estimated_weight_size
                    except Exception as e:
                        print(f"讀取客戶端 {client_id} 指標失敗: {e}")
            participation_data = None
            for agg_id in range(num_aggregators):
                participation_file = os.path.join(
                    self.result_dir,
                    f'aggregator_{agg_id}_participation.csv'
                )
                if os.path.exists(participation_file):
                    try:
                        df_participation = pd.read_csv(participation_file)
                        round_participation = df_participation[df_participation['round_id'] == round_id]
                        if not round_participation.empty:
                            row = round_participation.iloc[-1]
                            participation_data = {
                                'participating_count': int(row.get('num_buffered', 0)),
                                'total_count': int(row.get('num_selected', 0)),
                                'participation_ratio': float(row.get('participation_ratio', 0)),
                                'client_ids': [int(x) for x in str(row.get('client_ids', '')).split(',') if x.strip().isdigit()],
                                'data_sizes': [int(x) for x in str(row.get('data_sizes', '')).split(',') if x.strip().isdigit()]
                            }
                            all_participating_clients.update(participation_data['client_ids'])
                            if participation_data['data_sizes']:
                                total_data_size = sum(participation_data['data_sizes'])
                            break
                    except Exception as e:
                        pass
            if participation_data is None and all_participating_clients:
                participation_data = {
                    'participating_count': len(all_participating_clients),
                    'total_count': num_clients,
                    'participation_ratio': len(all_participating_clients) / max(1, num_clients),
                    'client_ids': list(all_participating_clients),
                    'data_sizes': []
                }
            if participation_data:
                self.evaluator.record_participation_rate(
                    round_id=round_id,
                    participating_clients=participation_data['client_ids'],
                    total_clients=participation_data.get('total_count', num_clients)
                )
            confidence_scores_data = {}
            for agg_id in range(num_aggregators):
                participation_file = os.path.join(
                    self.result_dir,
                    f'aggregator_{agg_id}_participation.csv'
                )
                if os.path.exists(participation_file):
                    try:
                        df_participation = pd.read_csv(participation_file)
                        round_participation = df_participation[df_participation['round_id'] == round_id]
                        if not round_participation.empty:
                            row = round_participation.iloc[-1]
                            if 'confidence_scores' in row and pd.notna(row['confidence_scores']):
                                confidence_str = str(row['confidence_scores'])
                                client_ids_list = participation_data['client_ids'] if participation_data else []
                                confidence_list = [float(x) for x in confidence_str.split(',') if x.strip().replace('.', '').isdigit()]
                                for idx, client_id in enumerate(client_ids_list):
                                    if idx < len(confidence_list):
                                        confidence_scores_data[client_id] = confidence_list[idx]
                            break
                    except Exception as e:
                        pass
            if confidence_scores_data:
                for client_id, confidence in confidence_scores_data.items():
                    self.evaluator.record_confidence_scores(
                        round_id=round_id,
                        client_id=client_id,
                        confidence_score=confidence
                    )
            weight_stats_data = None
            for agg_id in range(num_aggregators):
                participation_file = os.path.join(
                    self.result_dir,
                    f'aggregator_{agg_id}_participation.csv'
                )
                if os.path.exists(participation_file):
                    try:
                        df_participation = pd.read_csv(participation_file)
                        round_participation = df_participation[df_participation['round_id'] == round_id]
                        if not round_participation.empty:
                            row = round_participation.iloc[-1]
                            if 'weight_norm' in row and pd.notna(row['weight_norm']):
                                weight_stats_data = {
                                    'mean': float(row.get('weight_mean', 0.0)),
                                    'std': float(row.get('weight_std', 0.0)),
                                    'norm': float(row.get('weight_norm', 0.0))
                                }
                            break
                    except Exception as e:
                        pass
            if participation_data:
                try:
                    import torch
                    client_weight_stats_dict = {}
                    for agg_id in range(num_aggregators):
                        participation_file = os.path.join(
                            self.result_dir,
                            f'aggregator_{agg_id}_participation.csv'
                        )
                        if os.path.exists(participation_file):
                            try:
                                df_participation = pd.read_csv(participation_file)
                                round_participation = df_participation[df_participation['round_id'] == round_id]
                                if not round_participation.empty:
                                    row = round_participation.iloc[-1]
                                    if 'client_weight_norms' in row and pd.notna(row['client_weight_norms']):
                                        client_ids_list = participation_data.get('client_ids', [])
                                        norms_str = str(row['client_weight_norms'])
                                        means_str = str(row.get('client_weight_means', ''))
                                        stds_str = str(row.get('client_weight_stds', ''))
                                        norms_list = [float(x) for x in norms_str.split(',') if x.strip().replace('.', '').replace('-', '').isdigit()]
                                        means_list = [float(x) for x in means_str.split(',') if x.strip().replace('.', '').replace('-', '').isdigit()] if means_str else []
                                        stds_list = [float(x) for x in stds_str.split(',') if x.strip().replace('.', '').replace('-', '').isdigit()] if stds_str else []
                                        for idx, client_id in enumerate(client_ids_list):
                                            if idx < len(norms_list):
                                                client_weight_stats_dict[client_id] = {
                                                    'weight_norm': norms_list[idx],
                                                    'weight_mean': means_list[idx] if idx < len(means_list) else 0.0,
                                                    'weight_std': stds_list[idx] if idx < len(stds_list) else 0.0
                                                }
                                    break
                            except Exception as e:
                                pass
                    if client_weight_stats_dict:
                        import numpy as np
                        client_norms = [stats['weight_norm'] for stats in client_weight_stats_dict.values()]
                        weight_divergence = float(np.std(client_norms)) if len(client_norms) > 1 else 0.0
                        global_divergence = None
                        if weight_stats_data and weight_stats_data.get('norm', 0) > 0:
                            global_norm = weight_stats_data['norm']
                            divergences = [abs(norm - global_norm) for norm in client_norms]
                            global_divergence = float(np.mean(divergences)) if divergences else 0.0
                        simplified_global_weights = {
                            'weight_norm': weight_stats_data.get('norm', 0.0) if weight_stats_data else 0.0,
                            'weight_mean': weight_stats_data.get('mean', 0.0) if weight_stats_data else 0.0,
                            'weight_std': weight_stats_data.get('std', 0.0) if weight_stats_data else 0.0
                        }
                        simplified_client_weights = {}
                        for client_id, stats in client_weight_stats_dict.items():
                            simplified_client_weights[client_id] = {
                                'weight_norm': stats['weight_norm'],
                                'weight_mean': stats['weight_mean'],
                                'weight_std': stats['weight_std']
                            }
                        self.evaluator.record_weight_divergence(
                            round_id=round_id,
                            client_weights=simplified_client_weights,
                            global_weights=simplified_global_weights
                        )
                    elif weight_stats_data and weight_stats_data.get('std', 0) > 0:
                        simplified_global_weights = {
                            'weight_norm': weight_stats_data.get('norm', 0.0),
                            'weight_mean': weight_stats_data.get('mean', 0.0),
                            'weight_std': weight_stats_data.get('std', 0.0)
                        }
                        simplified_client_weights = {}
                        for client_id in participation_data.get('client_ids', []):
                            simplified_client_weights[client_id] = simplified_global_weights.copy()
                        self.evaluator.record_weight_divergence(
                            round_id=round_id,
                            client_weights=simplified_client_weights,
                            global_weights=simplified_global_weights
                        )
                except Exception as e:
                    print(f"記錄權重差異度失敗: {e}")
            if communication_bytes > 0 or total_data_size > 0:
                upload_bytes = communication_bytes
                download_bytes = int(communication_bytes * 1.5 / max(1, len(all_participating_clients))) if all_participating_clients else 0
                total_comm_bytes = upload_bytes + download_bytes * len(all_participating_clients)
                communication_cost = {
                    'upload_bytes': upload_bytes,
                    'download_bytes': download_bytes * len(all_participating_clients) if all_participating_clients else 0,
                    'total_bytes': total_comm_bytes,
                    'upload_mb': upload_bytes / (1024 * 1024),
                    'download_mb': (download_bytes * len(all_participating_clients)) / (1024 * 1024) if all_participating_clients else 0,
                    'total_mb': total_comm_bytes / (1024 * 1024),
                    'num_clients': len(all_participating_clients)
                }
            else:
                communication_cost = None
            if client_metrics:
                accs = [m['acc'] for m in client_metrics.values()]
                f1s = [m['f1'] for m in client_metrics.values()]
                global_metrics = {
                    'acc': sum(accs) / len(accs) if accs else 0,
                    'f1': sum(f1s) / len(f1s) if f1s else 0,
                }
                computation_cost = None
                if computation_times:
                    computation_cost = {
                        'total_time': sum(computation_times),
                        'avg_time': sum(computation_times) / len(computation_times),
                        'min_time': min(computation_times),
                        'max_time': max(computation_times),
                        'num_clients': len(computation_times)
                    }
                if computation_times:
                    self.evaluator.record_timing_stats(
                        round_id=round_id,
                        training_time=sum(computation_times) / len(computation_times) if computation_times else None,
                        total_round_time=None
                    )
                self.evaluator.record_round_metrics(
                    round_id=round_id,
                    client_metrics=client_metrics,
                    global_metrics=global_metrics,
                    communication_cost=communication_cost,
                    computation_cost=computation_cost
                )
                if client_metrics:
                    losses = [m['loss'] for m in client_metrics.values()]
                    avg_loss = sum(losses) / len(losses) if losses else 0
                    self.evaluator.record_loss_components(
                        round_id=round_id,
                        ce_loss=avg_loss,
                        total_loss=avg_loss
                    )
        except Exception as e:
            print(f"收集輪次 {round_id} 指標失敗: {e}")
            import traceback
            traceback.print_exc()
    def stop_monitoring(self):
        self.monitoring_active = False
        print("停止監控")
async def wait_for_service_ready(url, timeout=120, service_name="服務"):
    import aiohttp
    health_url = _normalize_health_url(url)
    print(f"等待 {service_name} 就緒: {health_url}")
    start_time = time.time()
    check_count = 0
    last_progress_time = 0
    while time.time() - start_time < timeout:
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        elapsed = time.time() - start_time
                        print(f"{service_name} 已就緒 (耗時 {elapsed:.1f}s)")
                        return True
        except asyncio.TimeoutError:
            check_count += 1
            elapsed = time.time() - start_time
            if elapsed - last_progress_time >= 5:
                print(f"{service_name} 檢查中... ({elapsed:.0f}s/{timeout}s)")
                last_progress_time = elapsed
        except Exception as e:
            check_count += 1
            elapsed = time.time() - start_time
            if elapsed - last_progress_time >= 5:
                error_msg = str(e)[:50] if len(str(e)) > 50 else str(e)
                print(f"{service_name} 檢查中... ({elapsed:.0f}s/{timeout}s) - {error_msg}")
                last_progress_time = elapsed
        await asyncio.sleep(2)
    elapsed = time.time() - start_time
    print(f"{service_name} 啟動超時 ({elapsed:.0f}s/{timeout}s) - {health_url}")
    return False
async def cloud_health_watchdog(cloud_url: str, process_ref, ready_event: asyncio.Event, monitor: FixedExperimentMonitor, check_interval: float = 10.0, restart_backoff: float = 5.0):
    import aiohttp
    health_url = _normalize_health_url(cloud_url)
    first_ok_reported = False
    success_streak = 0
    failure_streak = 0
    success_threshold = 2
    failure_threshold = 4
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        success_streak += 1
                        failure_streak = 0
                        if not first_ok_reported:
                            print(f"[Start]  Cloud 健康：{health_url} 200 OK")
                            first_ok_reported = True
                        if success_streak >= success_threshold:
                            ready_event.set()
                        await asyncio.sleep(check_interval)
                        continue
                    else:
                        failure_streak += 1
                        print(f"[Start]  Cloud 健康檢查 HTTP {resp.status} (連續失敗 {failure_streak}/{failure_threshold})")
        except Exception as e:
            failure_streak += 1
            error_msg = str(e) if e else "未知錯誤"
            print(f"[Start]  Cloud 健康檢查失敗: {error_msg} (連續失敗 {failure_streak}/{failure_threshold})", flush=True)
        if failure_streak < failure_threshold:
            await asyncio.sleep(check_interval)
            continue
        if not hasattr(cloud_health_watchdog, 'restart_count'):
            cloud_health_watchdog.restart_count = 0
        if ready_event.is_set():
            print(f"[Start]  Cloud 健康檢查連續失敗 {failure_threshold} 次，暫不暫停輪次（僅警告）", flush=True)
            proc = process_ref.get('proc')
            if proc:
                if proc.poll() is None:
                    print(f"[Start]  Cloud Server 進程仍在運行（PID: {proc.pid}）", flush=True)
                else:
                    print(f"[Start]  Cloud Server 進程已退出（退出碼: {proc.returncode}）", flush=True)
        success_streak = 0
        cloud_health_watchdog.restart_count += 1
        if cloud_health_watchdog.restart_count > 5:
            print(f"[Start]  Cloud Server 重啟次數過多 ({cloud_health_watchdog.restart_count})，暫停重啟")
            await asyncio.sleep(30)
            cloud_health_watchdog.restart_count = 0
            continue
        try:
            proc = process_ref.get('proc')
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    await asyncio.sleep(5)
                    if proc.poll() is None:
                        proc.kill()
                        await asyncio.sleep(2)
                except Exception:
                    pass
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', cloud_port))
                sock.close()
                if result == 0:
                    try:
                        subprocess.run(f"lsof -ti:{cloud_port} | xargs kill -9",
                                      shell=True, timeout=5, check=False)
                        await asyncio.sleep(2)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        print(f"[Start] 嘗試重啟 Cloud Server（第 {cloud_health_watchdog.restart_count} 次）...")
        try:
            BASE_DIR = _get_fl_base_dir()
            cloud_port = config.NETWORK_CONFIG['cloud_server']['port']
            cmd = [_get_fl_python(), os.path.join(BASE_DIR, "cloud_server_fixed.py"), "--port", str(cloud_port)]
            env = os.environ.copy()
            env['EXPERIMENT_DIR'] = monitor.result_dir
            cloud_log_path = os.path.join(monitor.result_dir, "cloud_server.out")
            cloud_log = open(cloud_log_path, 'a', buffering=1)
            proc = subprocess.Popen(cmd, cwd=BASE_DIR, env=env, stdout=cloud_log, stderr=cloud_log, start_new_session=True)
            process_ref['proc'] = proc
            monitor.add_process("cloud_server", proc, port=cloud_port)
            print("[Start]  等待 Cloud Server 啟動完成...")
            await asyncio.sleep(25)
        except Exception as e:
            print(f"[Start]  重啟 Cloud 失敗: {e}")
            await asyncio.sleep(max(5.0, restart_backoff))
            continue
        failure_streak = 0
        ok = await wait_for_service_ready(health_url, timeout=180, service_name="Cloud Server")
        if ok:
            success_streak = success_threshold
            ready_event.set()
            print("[Start]  Cloud 重啟並就緒")
            cloud_health_watchdog.restart_count = 0
        else:
            print("[Start]  Cloud 未就緒（端口綁定失敗或啟動錯誤），將繼續監控與重試")
            proc = process_ref.get('proc')
            if proc and proc.poll() is not None:
                print(f"[Start]  Cloud Server 進程已退出，退出碼: {proc.returncode}")
                try:
                    if proc.stdout:
                        output = proc.stdout.read().decode('utf-8', errors='ignore')
                        if output:
                            print(f"[Start]  Cloud Server 錯誤輸出:\n{output[-500:]}")
                except Exception:
                    pass
        await asyncio.sleep(max(1.0, restart_backoff))
async def check_aggregation_progress(aggregator_urls, round_id, timeout=300):
    import aiohttp
    print(f"檢查第{round_id}輪聚合進度...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                all_ready = True
                for i, url in enumerate(aggregator_urls):
                    try:
                        async with session.get(f"{url}/aggregation_status", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                data = await response.json()
                                current_round = data.get('federated_round', {}).get('current_round', 0)
                                buffer_size = data.get('federated_round', {}).get('buffer_size', 0)
                                if current_round >= round_id:
                                    print(f"聚合器 {i} 已推進到第 {current_round} 輪")
                                else:
                                    print(f"聚合器 {i} 當前輪次: {current_round}, 緩衝區: {buffer_size}")
                                    all_ready = False
                            else:
                                all_ready = False
                    except Exception as e:
                        print(f"聚合器 {i} 檢查失敗: {e}")
                        all_ready = False
                if all_ready:
                    print(f"所有聚合器已推進到第 {round_id} 輪")
                    return True
        except Exception as e:
            print(f"聚合進度檢查異常: {e}")
        await asyncio.sleep(10)
    print(f"聚合進度檢查超時")
    return False
async def start_round_training(aggregator_urls, global_round_id):
    import aiohttp
    print(f"啟動第 {global_round_id} 輪訓練...")
    success_count = 0
    total_aggregators = len(aggregator_urls)
    for i, url in enumerate(aggregator_urls):
        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('round_id', str(global_round_id))
                async with session.post(f"{url}/start_federated_round", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        selected_clients = result.get('selected_clients', [])
                        print(f"聚合器 {i} 輪次啟動成功: {selected_clients}")
                        print(f"   選中客戶端數量: {len(selected_clients)}")
                        success_count += 1
                    else:
                        print(f"聚合器 {i} 輪次啟動失敗: HTTP {response.status}")
                        response_text = await response.text()
                        print(f"   錯誤詳情: {response_text}")
        except Exception as e:
            print(f"聚合器 {i} 輪次啟動異常: {e}")
    print(f"訓練啟動結果: {success_count}/{total_aggregators} 個聚合器成功啟動")
    return success_count == total_aggregators
async def fetch_aggregators_rounds(aggregator_urls, max_retries=3, retry_delay=1.5):
    import aiohttp
    rounds: List[Optional[int]] = []
    buffers: List[Optional[int]] = []
    ack_rounds: List[Optional[int]] = []
    pending_cloud_counts: List[Optional[int]] = []
    query_failed: List[bool] = []
    if not hasattr(fetch_aggregators_rounds, 'query_failure_history'):
        fetch_aggregators_rounds.query_failure_history = {}
    async with aiohttp.ClientSession() as session:
        for idx, url in enumerate(aggregator_urls):
            success = False
            last_exception = None
            for retry in range(max_retries):
                try:
                    timeout = aiohttp.ClientTimeout(total=15, connect=5)
                    async with session.get(f"{url}/federated_status", timeout=timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            rounds.append(int(data.get('round_count', 0)))
                            buffers.append(int(data.get('buffer_size', 0)))
                            ack_rounds.append(int(data.get('last_cloud_ack_round', 0)))
                            pending = data.get('pending_cloud_rounds', [])
                            if isinstance(pending, list):
                                pending_cloud_counts.append(len(pending))
                            else:
                                pending_cloud_counts.append(0)
                            success = True
                            query_failed.append(False)
                            if idx in fetch_aggregators_rounds.query_failure_history:
                                fetch_aggregators_rounds.query_failure_history[idx] = 0
                            break
                        else:
                            last_exception = f"HTTP {resp.status}"
                except asyncio.TimeoutError:
                    last_exception = "Timeout"
                    if retry < max_retries - 1:
                        await asyncio.sleep(retry_delay * (retry + 1))
                except Exception as e:
                    last_exception = str(e)[:50]
                    if retry < max_retries - 1:
                        await asyncio.sleep(retry_delay * (retry + 1))
            if not success:
                if idx not in fetch_aggregators_rounds.query_failure_history:
                    fetch_aggregators_rounds.query_failure_history[idx] = 0
                fetch_aggregators_rounds.query_failure_history[idx] += 1
                if fetch_aggregators_rounds.query_failure_history[idx] >= 5:
                    rounds.append(None)
                    buffers.append(None)
                    ack_rounds.append(None)
                    pending_cloud_counts.append(None)
                    query_failed.append(True)
                    if fetch_aggregators_rounds.query_failure_history[idx] == 5:
                        print(f"聚合器 {idx} 查詢連續失敗 5 次，標記為查詢失敗（可能網絡問題）")
                else:
                    rounds.append(0)
                    buffers.append(0)
                    ack_rounds.append(0)
                    pending_cloud_counts.append(0)
                    query_failed.append(True)
    return rounds, buffers, ack_rounds, pending_cloud_counts, query_failed
async def coordinator_loop(aggregator_urls, start_round=1, experiment_start_time=None, monitor=None):
    import math
    target_round = start_round
    last_started_round = 0
    last_force_advance_time = 0
    force_advance_cooldown = 60
    last_metrics_collection_round = start_round - 1
    print("啟動輪次協調器（簡化版：核心邏輯）...")
    cloud_port = config.NETWORK_CONFIG['cloud_server']['port']
    cloud_ready = globals().get('_cloud_ready_event')
    if cloud_ready is None:
        cloud_ready = asyncio.Event()
        globals()['_cloud_ready_event'] = cloud_ready
    while True:
        try:
            if not cloud_ready.is_set():
                print("等待 Cloud 健康就緒後再協調輪次...")
                await asyncio.sleep(2)
                continue
            rounds, buffers, ack_rounds, pending_cloud, query_failed = await fetch_aggregators_rounds(aggregator_urls)
            if not rounds or all(r is None for r in rounds):
                print("所有聚合器查詢失敗，等待後重試...")
                await asyncio.sleep(5)
                continue
            comparison_rounds: List[int] = []
            for idx, r in enumerate(rounds):
                if r is None:
                    comparison_rounds.append(0)
                    continue
                ack_value = ack_rounds[idx] if idx < len(ack_rounds) and ack_rounds[idx] is not None else 0
                if ack_value and ack_value > 0:
                    comparison_rounds.append(ack_value)
                else:
                    comparison_rounds.append(r if r is not None else 0)
            valid_comparison_rounds = [r for r in comparison_rounds if r is not None and r > 0]
            num_aggs = len(valid_comparison_rounds) if valid_comparison_rounds else len(comparison_rounds)
            threshold = max(1, math.ceil(0.75 * num_aggs))
            min_r = int(min(valid_comparison_rounds)) if valid_comparison_rounds else 0
            max_r = int(max(valid_comparison_rounds)) if valid_comparison_rounds else 0
            if not hasattr(coordinator_loop, 'reset_cooldown'):
                coordinator_loop.reset_cooldown = {}
            current_time = time.time()
            safe_max_r = max_r if max_r is not None else 0
            safe_min_r = min_r if min_r is not None else 0
            if safe_max_r > 0 and safe_min_r >= 0 and safe_max_r - safe_min_r > 30:
                for idx, r in enumerate(comparison_rounds):
                    if r is None or r <= 0:
                        continue
                    if idx in coordinator_loop.reset_cooldown:
                        if current_time - coordinator_loop.reset_cooldown[idx] < 60:
                            continue
                    if safe_max_r - r > 30:
                        try:
                            async with aiohttp.ClientSession() as session:
                                data = aiohttp.FormData()
                                valid_rounds = [cr for cr in comparison_rounds if cr is not None and cr > 0]
                                if len(valid_rounds) > 0:
                                    median_r = sorted(valid_rounds)[len(valid_rounds) // 2]
                                    safe_target = max(safe_min_r + 5, min(median_r - 2, r + 8))
                                else:
                                    safe_target = max(safe_min_r + 5, r + 5)
                                data.add_field('target_round', str(safe_target))
                                timeout = aiohttp.ClientTimeout(total=10)
                                async with session.post(f"{aggregator_urls[idx]}/reset_round", data=data, timeout=timeout) as resp:
                                    if resp.status == 200:
                                        coordinator_loop.reset_cooldown[idx] = current_time
                                        print(f"已重置極度落後聚合器 {idx}: {r} -> {safe_target} (差距={safe_max_r-r}輪)")
                        except Exception as e:
                            print(f"重置聚合器 {idx} 失敗: {e}")
            if ack_rounds and len(ack_rounds) > 0:
                reference_rounds = []
                for idx in range(len(rounds)):
                    if query_failed[idx] and rounds[idx] is None:
                        continue
                    ack_val = ack_rounds[idx] if idx < len(ack_rounds) and ack_rounds[idx] is not None else 0
                    round_val = rounds[idx] if idx < len(rounds) and rounds[idx] is not None else 0
                    ack_val = int(ack_val) if ack_val is not None else 0
                    round_val = int(round_val) if round_val is not None else 0
                    if ack_val > 0:
                        reference_rounds.append(max(ack_val, round_val - 2))
                    elif round_val > 0:
                        reference_rounds.append(max(0, round_val - 5))
                if len(reference_rounds) > 0:
                    sorted_refs = sorted(reference_rounds)
                    median_ref = int(sorted_refs[len(sorted_refs) // 2])
                    safe_target_round = target_round if target_round is not None else 0
                    if median_ref > safe_target_round + 10:
                        new_target = max(safe_target_round + 1, median_ref - 1)
                        print(f"聚合器參考輪次遠超目標（差距={median_ref - safe_target_round}輪），追趕到 {new_target}")
                        target_round = new_target
                        last_started_round = max(0, new_target - 1)
            safe_target_round = target_round if target_round is not None else 0
            ack_completed = []
            abnormal_aggregators = []
            for idx in range(len(ack_rounds)):
                if query_failed[idx] and rounds[idx] is None:
                    if not hasattr(coordinator_loop, 'abnormal_agg_history'):
                        coordinator_loop.abnormal_agg_history = {}
                    if idx not in coordinator_loop.abnormal_agg_history:
                        coordinator_loop.abnormal_agg_history[idx] = 0
                    coordinator_loop.abnormal_agg_history[idx] += 1
                    if coordinator_loop.abnormal_agg_history[idx] >= 5:
                        if idx not in abnormal_aggregators:
                            abnormal_aggregators.append(idx)
                            print(f"標記聚合器 {idx} 為異常（查詢失敗，持續{coordinator_loop.abnormal_agg_history[idx]}次）")
                    continue
                ack_val = ack_rounds[idx] if idx < len(ack_rounds) and ack_rounds[idx] is not None else 0
                round_val = rounds[idx] if idx < len(rounds) and rounds[idx] is not None else 0
                ack_val = int(ack_val) if ack_val is not None else 0
                round_val = int(round_val) if round_val is not None else 0
                if ack_val == 0 and round_val == 0:
                    if not hasattr(coordinator_loop, 'abnormal_agg_history'):
                        coordinator_loop.abnormal_agg_history = {}
                    if idx not in coordinator_loop.abnormal_agg_history:
                        coordinator_loop.abnormal_agg_history[idx] = 0
                    coordinator_loop.abnormal_agg_history[idx] += 1
                    if coordinator_loop.abnormal_agg_history[idx] >= 5:
                        if idx not in abnormal_aggregators:
                            abnormal_aggregators.append(idx)
                            print(f"標記聚合器 {idx} 為異常（ACK=0, 實際輪次=0，持續{coordinator_loop.abnormal_agg_history[idx]}次）")
                elif ack_val == 0 and round_val > 0:
                    if hasattr(coordinator_loop, 'abnormal_agg_history') and idx in coordinator_loop.abnormal_agg_history:
                        coordinator_loop.abnormal_agg_history[idx] = 0
                else:
                    if hasattr(coordinator_loop, 'abnormal_agg_history') and idx in coordinator_loop.abnormal_agg_history:
                        coordinator_loop.abnormal_agg_history[idx] = 0
                if query_failed[idx]:
                    continue
                safe_target_round_int = int(safe_target_round) if safe_target_round is not None else 0
                if ack_val is not None and ack_val >= safe_target_round_int:
                    ack_completed.append(idx)
                elif round_val is not None and round_val >= safe_target_round_int + 2:
                    ack_completed.append(idx)
                elif idx in abnormal_aggregators:
                    pass
            failed_count = sum(1 for f in query_failed if f)
            effective_aggs = num_aggs - len(abnormal_aggregators) - failed_count
            effective_aggs = max(1, effective_aggs)
            ack_reached = len(ack_completed)
            valid_pending = [pc for pc in pending_cloud if pc is not None]
            has_pending = any(pc > 0 for pc in valid_pending)
            pending_count = sum(valid_pending)
            avg_pending_per_agg = pending_count / num_aggs if num_aggs > 0 else 0.0
            allow_with_pending = (avg_pending_per_agg < 1.0) and (ack_reached >= max(1, num_aggs - 1))
            required_aggs = max(1, math.ceil(0.75 * effective_aggs)) if effective_aggs > 0 else 1
            safe_target_round = target_round if target_round is not None else 0
            is_first_round = safe_target_round == 1 and last_started_round == 0
            rounds_reached = sum(1 for idx in range(len(rounds))
                                if idx < len(rounds) and (r := rounds[idx]) is not None
                                and r >= safe_target_round
                                and not query_failed[idx])
            ack_all_zero = ack_reached == 0
            allow_by_rounds = (rounds_reached >= required_aggs and (safe_target_round <= 10 or ack_all_zero))
            ahead_aggs = [idx for idx in range(len(rounds))
                         if idx < len(rounds) and (r := rounds[idx]) is not None
                         and r >= safe_target_round + 2
                         and not query_failed[idx]]
            ahead_count = len(ahead_aggs)
            ahead_ratio = ahead_count / effective_aggs if effective_aggs > 0 else 0
            force_advance_condition = (ahead_ratio >= 0.5 or ahead_count >= 2) and ahead_count > 0
            current_time = time.time()
            time_since_last_force = current_time - last_force_advance_time
            can_force_advance = time_since_last_force >= force_advance_cooldown
            if force_advance_condition and not can_force_advance and time_since_last_force >= 120:
                can_force_advance = True
            force_advance = force_advance_condition and (safe_target_round <= 10 or ack_all_zero) and can_force_advance
            valid_rounds: List[int] = [r for idx in range(len(rounds))
                           if idx < len(rounds) and (r := rounds[idx]) is not None
                           and not query_failed[idx]]
            round_sync_ok = True
            if len(valid_rounds) >= 2:
                import statistics
                round_std = statistics.stdev(valid_rounds) if len(valid_rounds) > 1 else 0.0
                if round_std > 5.0:
                    print(f"聚合器輪次不同步（標準差={round_std:.2f}），但允許推進（輪次: {valid_rounds}）")
            if (ack_reached >= required_aggs or is_first_round or allow_by_rounds or force_advance) and effective_aggs > 0 and (not has_pending or allow_with_pending) and safe_target_round > last_started_round:
                if monitor is not None and safe_target_round > last_metrics_collection_round:
                    for collect_round in range(last_metrics_collection_round + 1, safe_target_round + 1):
                        monitor.collect_round_metrics(collect_round)
                    last_metrics_collection_round = safe_target_round
                next_round = safe_target_round + 1
                ok = await start_round_training(aggregator_urls, next_round)
                if ok:
                    last_started_round = safe_target_round
                    target_round = next_round
                    if is_first_round:
                        print(f"第一輪啟動，推進到第{target_round}輪")
                    elif force_advance:
                        ahead_rounds: List[int] = [r for idx in range(len(rounds))
                                        if idx < len(rounds) and (r := rounds[idx]) is not None
                                        and r >= safe_target_round + 2
                                        and not query_failed[idx]]
                        last_force_advance_time = current_time
                        print(f"檢測到{ahead_count}個聚合器（{ahead_ratio*100:.1f}%）實際輪次領先目標2輪以上（目標={safe_target_round}，實際={max(ahead_rounds) if ahead_rounds else 'N/A'}），強制推進到第{target_round}輪（ACK: {ack_reached}/{required_aggs}）")
                    elif allow_by_rounds:
                        print(f"聚合器實際輪次達到第{safe_target_round}輪，推進到第{target_round}輪（ACK: {ack_reached}/{required_aggs}）")
                    else:
                        print(f"所有聚合器雲端 ACK 達到第{safe_target_round}輪，推進到第{target_round}輪")
            elif has_pending and not allow_with_pending:
                pending_list = [f"agg{i}:{pc}" for i, pc in enumerate(pending_cloud) if pc is not None and pc > 0]
                print(f"檢測到 pending 上傳，暫停推進: {', '.join(pending_list)} (總pending={pending_count}, 平均={avg_pending_per_agg:.1f}/聚合器, ACK達到={ack_reached}/{num_aggs})")
            elif has_pending and allow_with_pending:
                pending_list = [f"agg{i}:{pc}" for i, pc in enumerate(pending_cloud) if pc is not None and pc > 0]
                print(f"檢測到少量 pending 上傳，但允許推進: {', '.join(pending_list)} (總pending={pending_count}, 平均={avg_pending_per_agg:.1f}/聚合器, ACK達到={ack_reached}/{num_aggs})")
            elif ack_reached < required_aggs:
                safe_target_round = target_round if target_round is not None else 0
                if not can_force_advance and force_advance_condition:
                    remaining_cooldown = force_advance_cooldown - time_since_last_force
                    print(f"強制推進冷卻中（還需{remaining_cooldown:.1f}秒），等待 ACK: {ack_reached}/{required_aggs} 個聚合器達到目標輪次 {safe_target_round}")
                elif abnormal_aggregators:
                    print(f"等待 ACK: {ack_reached}/{required_aggs} 個有效聚合器達到目標輪次 {safe_target_round} (異常聚合器: {abnormal_aggregators})")
                else:
                    print(f"等待 ACK: {ack_reached}/{required_aggs} 個聚合器達到目標輪次 {safe_target_round}")
            round_interval = 120
            await asyncio.sleep(round_interval)
        except asyncio.CancelledError:
            print("輪次協調器結束")
            break
        except Exception as e:
            print(f"協調器異常: {e}")
            await asyncio.sleep(5)
def check_and_clear_port_conflicts():
    print("檢查端口衝突...")
    ports_to_check = [
        config.NETWORK_CONFIG['cloud_server']['port'],
        *config.NETWORK_CONFIG['aggregators']['ports']
    ]
    conflicts_found = False
    for port in ports_to_check:
        try:
            result = subprocess.run(['lsof', '-i', f':{port}'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                print(f"端口 {port} 被占用:")
                print(f"   {result.stdout.strip()}")
                conflicts_found = True
                lines = result.stdout.strip().split('\n')[1:]
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            pid = parts[1]
                            try:
                                print(f"終止進程 PID {pid} (端口 {port})")
                                subprocess.run(['kill', '-9', pid], timeout=5)
                            except Exception as e:
                                print(f"終止進程 {pid} 失敗: {e}")
            else:
                print(f"端口 {port} 可用")
        except Exception as e:
            print(f"檢查端口 {port} 失敗: {e}")
    if conflicts_found:
        print("等待端口釋放...")
        time.sleep(3)
        for port in ports_to_check:
            try:
                result = subprocess.run(['lsof', '-i', f':{port}'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    print(f"端口 {port} 仍然被占用")
                else:
                    print(f"端口 {port} 已釋放")
            except Exception as e:
                print(f"重新檢查端口 {port} 失敗: {e}")
    else:
        print("所有端口都可用")
    print("端口衝突檢查完成\n")
async def main():
    print("啟動修復版聯邦學習實驗")
    print("=" * 50)
    check_and_clear_port_conflicts()
    if 'EXPERIMENT_NAME' in os.environ:
        experiment_name = os.environ['EXPERIMENT_NAME']
        print(f"使用批次腳本指定的實驗名稱: {experiment_name}")
    else:
        try:
            if sys.version_info >= (3, 9):
                taiwan_tz = ZoneInfo("Asia/Taipei")
            else:
                try:
                    import pytz
                    taiwan_tz = pytz.timezone("Asia/Taipei")
                except ImportError:
                    raise ImportError("pytz not available")
            timestamp = datetime.now(taiwan_tz).strftime("%Y%m%d_%H%M%S")
        except Exception:
            from datetime import timedelta
            timestamp = (datetime.now() + timedelta(hours=8)).strftime("%Y%m%d_%H%M%S")
        experiment_name = f"tokyo_drone_fixed_{timestamp}"
        print(f"使用時間戳實驗名稱: {experiment_name}")
    BASE_DIR = _get_fl_base_dir()
    result_dir_name = os.environ.get("RESULT_DIR_NAME", "result")
    result_dir = os.path.abspath(os.path.join(BASE_DIR, result_dir_name, experiment_name))
    print(f"使用結果目錄: {result_dir_name}/ (完整路徑: {result_dir})")
    os.environ['EXPERIMENT_DIR'] = result_dir
    print(f"設置環境變量 EXPERIMENT_DIR = {result_dir}")
    monitor = FixedExperimentMonitor(result_dir)
    try:
        try:
            data_dir = getattr(config, 'DATA_PATH', '')
            os.makedirs(result_dir, exist_ok=True)
            src_scaler = os.path.join(data_dir, 'scaler.pkl')
            dst_scaler = os.path.join(result_dir, 'scaler.pkl')
            if os.path.exists(src_scaler) and not os.path.exists(dst_scaler):
                import shutil
                shutil.copyfile(src_scaler, dst_scaler)
                print(f"已同步 scaler.pkl -> {dst_scaler}")
            src_feat = os.path.join(data_dir, 'feature_cols.json')
            dst_feat = os.path.join(result_dir, 'feature_cols.json')
            if os.path.exists(src_feat) and not os.path.exists(dst_feat):
                import shutil
                shutil.copyfile(src_feat, dst_feat)
                print(f"已同步 feature_cols.json -> {dst_feat}")
        except Exception as _e:
            print(f"同步 scaler/feature_cols 失敗: {_e}")
        print("\n1. 啟動 Cloud Server...")
        cloud_server_cmd = [
            _get_fl_python(), os.path.join(BASE_DIR, "cloud_server_fixed.py"),
            "--port", str(config.NETWORK_CONFIG['cloud_server']['port'])
        ]
        cloud_env = os.environ.copy()
        cloud_env['EXPERIMENT_DIR'] = result_dir
        data_path = getattr(config, 'DATA_PATH', '')
        cloud_env['DATA_PATH'] = data_path
        scaled_test = os.path.join(data_path, 'global_test_scaled.csv')
        raw_test = os.path.join(data_path, 'global_test.csv')
        cloud_env['GLOBAL_TEST_PATH'] = scaled_test if os.path.exists(scaled_test) else raw_test
        for k in ['LABEL_COUNTS_PATH', 'DATASET_ROOT', 'GLOBAL_TEST_CSV']:
            cloud_env.pop(k, None)
        cloud_env['IGNORE_PERSISTED_STATE'] = '1'
        if 'FEDAVG_BASELINE' not in cloud_env or cloud_env.get('FEDAVG_BASELINE', '').strip() == '':
            cloud_env['FEDAVG_BASELINE'] = '0'
        key_env_vars = ['FEDAVG_BASELINE', 'SERVER_EPOCHS']
        print("Cloud Server 環境變數配置:")
        for key in key_env_vars:
            value = cloud_env.get(key, '未設置')
            print(f"  - {key} = {value}")
        cloud_log_path = os.path.join(result_dir, "cloud_server.out")
        cloud_log = open(cloud_log_path, 'a', buffering=1)
        cloud_server_process = subprocess.Popen(
            cloud_server_cmd,
            stdout=cloud_log,
            stderr=cloud_log,
            env=cloud_env,
            cwd=BASE_DIR
        )
        monitor.add_process("cloud_server", cloud_server_process, config.NETWORK_CONFIG['cloud_server']['port'])
        cloud_url = config.NETWORK_CONFIG['cloud_server']['url']
        print("等待 Cloud Server 啟動...")
        await asyncio.sleep(15)
        if cloud_server_process.poll() is not None:
            print(f"Cloud Server 進程已退出，退出碼: {cloud_server_process.returncode}")
            print("檢查 Cloud Server 日誌...")
            try:
                with open(cloud_log_path, 'r') as f:
                    log_content = f.read()
                    if log_content:
                        print(f"Cloud Server 日誌:\n{log_content}")
                    else:
                        print("Cloud Server 日誌為空")
            except Exception as e:
                print(f"無法讀取 Cloud Server 日誌: {e}")
            raise Exception("Cloud Server 啟動失敗")
        await asyncio.sleep(10)
        if cloud_server_process.poll() is not None:
            print(f"Cloud Server 進程在等待期間退出，退出碼: {cloud_server_process.returncode}")
            raise Exception("Cloud Server 在啟動期間崩潰")
        try:
            import requests
            max_retries = 6
            for retry in range(max_retries):
                try:
                    response = requests.get(f"{cloud_url}/health", timeout=15)
                    if response.status_code == 200:
                        print(f"Cloud Server 已就緒 (嘗試 {retry+1}/{max_retries})")
                        break
                    else:
                        if retry < max_retries - 1:
                            print(f"Cloud Server 響應異常 ({response.status_code})，等待重試...")
                            await asyncio.sleep(5)
                        else:
                            print(f"Cloud Server 響應異常: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    if retry < max_retries - 1:
                        print(f"Cloud Server 健康檢查失敗，等待重試... ({retry+1}/{max_retries})")
                        await asyncio.sleep(5)
                    else:
                        print(f"Cloud Server 健康檢查失敗: {e}")
                        print("繼續啟動其他組件...")
        except Exception as e:
            print(f"Cloud Server 健康檢查異常: {e}")
            print("繼續啟動其他組件...")
        sys.stdout.flush()
        cloud_ready_event = asyncio.Event()
        globals()['_cloud_ready_event'] = cloud_ready_event
        cloud_proc_ref = { 'proc': cloud_server_process }
        asyncio.create_task(cloud_health_watchdog(cloud_url, cloud_proc_ref, cloud_ready_event, monitor))
        print("\n2. 啟動聚合器...")
        aggregator_urls = []
        available_ports = _get_aggregator_ports()
        for i in range(config.NUM_AGGREGATORS):
            port = available_ports[i]
            aggregator_cmd = [
                _get_fl_python(), os.path.join(BASE_DIR, "aggregator_fixed.py"),
                "--aggregator_id", str(i),
                "--port", str(port)
            ]
            agg_env = os.environ.copy()
            agg_env['EXPERIMENT_DIR'] = result_dir
            agg_env['DATA_PATH'] = data_path
            agg_env['GLOBAL_TEST_PATH'] = cloud_env.get('GLOBAL_TEST_PATH', '')
            agg_env['AGG_ROUND_SYNC'] = 'on'
            agg_env.pop('CLOUD_HEALTH_URL', None)
            agg_env['IGNORE_PERSISTED_STATE'] = '1'
            if i == 0:
                key_env_vars = ['FEDAVG_BASELINE', 'SERVER_EPOCHS']
                print("Aggregator 環境變數配置:")
                for key in key_env_vars:
                    value = agg_env.get(key, '未設置')
                    print(f"  - {key} = {value}")
            agg_log_path = os.path.join(result_dir, f"agg{i}.out")
            agg_log = open(agg_log_path, 'a', buffering=1)
            aggregator_process = subprocess.Popen(
                aggregator_cmd,
                stdout=agg_log,
                stderr=agg_log,
                env=agg_env,
                cwd=BASE_DIR
            )
            monitor.add_process(f"aggregator_{i}", aggregator_process, port)
            aggregator_urls.append(f"http://127.0.0.1:{port}")
        for i, url in enumerate(aggregator_urls):
            if not await wait_for_service_ready(url, timeout=180, service_name=f"聚合器 {i}"):
                raise Exception(f"聚合器 {i} 啟動失敗")
        print("\n重置聚合器輪次到第1輪...")
        for i, url in enumerate(aggregator_urls):
            try:
                async with aiohttp.ClientSession() as session:
                    data = aiohttp.FormData()
                    data.add_field('target_round', '1')
                    async with session.post(f"{url}/reset_round", data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"聚合器 {i} 輪次重置成功: {result.get('message', '')}")
                        else:
                            print(f"聚合器 {i} 輪次重置失敗: HTTP {response.status}")
            except Exception as e:
                print(f"聚合器 {i} 輪次重置異常: {e}")
        print("\n3. 啟動客戶端...")
        for i in range(config.NUM_CLIENTS):
            client_dir = os.path.join(result_dir, f"uav{i}")
            os.makedirs(client_dir, exist_ok=True)
            client_log_path = os.path.join(client_dir, f"client_{i}.log")
            _, aggregator_url = _resolve_aggregator_for_client(i)
            cloud_url = config.NETWORK_CONFIG['cloud_server']['url']
            client_cmd = [
                _get_fl_python(), os.path.join(BASE_DIR, "uav_client_fixed.py"),
                "--client_id", str(i),
                "--aggregator_url", aggregator_url,
                "--cloud_url", cloud_url,
                "--result_dir", result_dir
            ]
            client_env = os.environ.copy()
            client_env['EXPERIMENT_DIR'] = result_dir
            client_env['DATA_PATH'] = data_path
            client_env['GLOBAL_TEST_PATH'] = cloud_env.get('GLOBAL_TEST_PATH', '')
            client_env['IGNORE_PERSISTED_STATE'] = '1'
            if 'CLIENT_KD_ENABLED' in os.environ:
                client_env['CLIENT_KD_ENABLED'] = os.environ['CLIENT_KD_ENABLED']
            if client_env.get('CLIENT_KD_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on'):
                if client_env.get('FEDAVG_BASELINE', '').strip() == '1':
                    print(f"警告：CLIENT_KD_ENABLED=1 但 FEDAVG_BASELINE=1，這會導致 GKD 被關閉")
                    print(f"   建議：如果要用 GKD，請不要設置 FEDAVG_BASELINE=1")
            if i == 0:
                key_env_vars = ['FEDAVG_BASELINE', 'CLIENT_KD_ENABLED', 'SERVER_EPOCHS']
                print("Client 環境變數配置:")
                for key in key_env_vars:
                    value = client_env.get(key, '未設置')
                    print(f"  - {key} = {value}")
            client_log = open(client_log_path, 'a', buffering=1)
            client_process = subprocess.Popen(
                client_cmd,
                stdout=client_log,
                stderr=client_log,
                env=client_env,
                cwd=BASE_DIR
            )
            monitor.add_process(f"client_{i}", client_process)
        print("等待客戶端啟動...")
        await asyncio.sleep(15)
        print("檢查客戶端啟動狀態...")
        for i in range(config.NUM_CLIENTS):
            client_name = f"client_{i}"
            if client_name in monitor.processes:
                process = monitor.processes[client_name]['process']
                if process.poll() is not None:
                    print(f"客戶端 {i} 啟動失敗，嘗試重啟...")
                    monitor._check_and_restart_processes()
        print("客戶端啟動檢查完成")
        print("\n4. 觸發第一輪訓練...")
        if not await start_round_training(aggregator_urls, 1):
            print("訓練啟動失敗，但繼續監控...")
        print("\n5. 開始實驗監控...")
        print("系統將持續運行，按 Ctrl+C 停止...")
        coordinator_task = asyncio.create_task(coordinator_loop(aggregator_urls, start_round=1, experiment_start_time=monitor.start_time, monitor=monitor))
        try:
            await monitor.monitor_experiment()
        finally:
            coordinator_task.cancel()
            with contextlib.suppress(Exception):
                await coordinator_task
        if monitor.experiment_completed:
            print("\n觸發自動停止條件，準備關閉所有進程...")
    except KeyboardInterrupt:
        print("\n收到中斷信號，正在停止實驗...")
        try:
            monitor.set_stop_reason("keyboard_interrupt")
        except Exception:
            pass
    except Exception as e:
        print(f"\n實驗異常: {e}")
        try:
            monitor.set_stop_reason(f"exception:{e}")
        except Exception:
            pass
    finally:
        monitor.stop_monitoring()
        if not getattr(monitor, "stop_reason", None):
            try:
                monitor.set_stop_reason("unknown_stop")
            except Exception:
                pass
        print("\n終止所有進程...")
        for name, info in monitor.processes.items():
            try:
                info['process'].terminate()
                print(f"終止進程: {name}")
            except Exception as e:
                print(f"終止進程 {name} 失敗: {e}")
        await asyncio.sleep(5)
        for name, info in monitor.processes.items():
            try:
                if info['process'].poll() is None:
                    info['process'].kill()
                    print(f"強制終止進程: {name}")
            except Exception as e:
                print(f"強制終止進程 {name} 失敗: {e}")
        print("\n保存通訊統計...")
        if monitor.comm_monitor is not None:
            try:
                comm_stats = monitor.comm_monitor.save_statistics()
                print(f"通訊統計已保存")
                print(f"   - 總通訊次數: {comm_stats['data_transfer_stats']['total_communications']}")
                print(f"   - 總數據傳輸: {comm_stats['data_transfer_stats']['total_bytes_uploaded_mb'] + comm_stats['data_transfer_stats']['total_bytes_downloaded_mb']:.2f} MB")
                print(f"   - 雲端連接數: {comm_stats['connection_stats']['cloud_connections']}")
            except Exception as e:
                print(f"保存通訊統計失敗: {e}")
        else:
            print("通訊監控器未啟用（communication_monitor.py 已移除）")
        print(f"\n實驗結束，結果保存在: {result_dir}")
        print("緊急修復配置總結 (解決卡在第三輪問題):")
        print("  - 聚合等待時間: 20秒 → 120秒")
        print("  - 最小客戶端數: 2 → 3 (確保聚合穩定性)")
        print("  - 本地訓練輪數: 3 → 2 (平衡訓練效果與速度)")
        print("  - 批次大小: 64 → 128 (加快訓練速度)")
        print("  - 數據使用比例: 100% → 50% (加快訓練速度)")
        print("  - 梯度裁剪: 0.01 → 1.0 (放寬限制)")
        print("  - 啟用快速模式")
        print(f"  - 模型: {config.MODEL_CONFIG.get('type', 'DNN').upper()}")
        print("  - 類別權重: 大幅強化少數類別")
        print("  - 參與率: 80% (提高)")
if __name__ == "__main__":
    print("緊急修復聯邦學習實驗啟動 (解決卡在第三輪問題)")
    print("=" * 60)
    print("主要修復:")
    print("  - 聚合等待時間: 20秒 → 120秒")
    print("  - 最小客戶端數: 2 → 3 (確保聚合穩定性)")
    print("  - 本地訓練輪數: 3 → 2 (平衡訓練效果與速度)")
    print("  - 批次大小: 64 → 128 (加快訓練速度)")
    print("  - 數據使用比例: 100% → 50% (加快訓練速度)")
    print("  - 梯度裁剪: 0.01 → 1.0 (放寬限制)")
    print("  - 啟用快速模式")
    print(f"  - 使用{config.MODEL_CONFIG.get('type', 'DNN').upper()}模型")
    print("  - 大幅強化少數類別權重")
    print("  - 啟用餘弦退火學習率調度器")
    print("  - 提高客戶端參與率到80%")
    print("  - 禁用早停以確保充分訓練")
    print("  - 使用Focal Loss處理類別不平衡")
    print("=" * 60)
    def signal_handler(signum, frame):
        print("\n收到信號，正在退出...")
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    asyncio.run(main())
