#!/usr/bin/env python3

import asyncio
import aiohttp
import requests
import pickle
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union, Protocol
from abc import ABC, abstractmethod
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
import uvicorn
from pathlib import Path

logger = logging.getLogger(__name__)

class _HasFastAPI(Protocol):
    app: FastAPI
    name: str

class BaseAPI(ABC):
    def __init__(self, name: str):
        self.name = name
        self.app = FastAPI(title=f"{name} API", version="1.0.0")
        self.setup_routes()
    
    @abstractmethod
    def setup_routes(self):
        pass
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)

class HealthCheckMixin:
    def setup_health_check(self: _HasFastAPI) -> None:
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "name": self.name,
                "timestamp": time.time()
            }

class ErrorHandlerMixin:
    def setup_error_handlers(self: _HasFastAPI) -> None:
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            logger.error(f"全局異常處理: {exc}")
            return {
                "error": str(exc),
                "status": "error",
                "timestamp": time.time()
            }

class ClientAPI(BaseAPI, HealthCheckMixin, ErrorHandlerMixin):
    def __init__(self, client_id: int, config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        super().__init__(f"Client {client_id}")
        self.setup_health_check()
        self.setup_error_handlers()
    
    def setup_routes(self):
        @self.app.get("/status")
        async def get_status():
            return {
                "client_id": self.client_id,
                "status": "active",
                "config": self.config
            }
        
        @self.app.post("/train")
        async def start_training():
            try:
                return {"status": "training_started", "client_id": self.client_id}
            except Exception as e:
                logger.error(f"訓練啟動失敗: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/weights")
        async def get_weights():
            try:
                return {"status": "weights_retrieved", "client_id": self.client_id}
            except Exception as e:
                logger.error(f"權重獲取失敗: {e}")
                raise HTTPException(status_code=500, detail=str(e))


class AggregatorAPI(BaseAPI, HealthCheckMixin, ErrorHandlerMixin):
    def __init__(self, aggregator_id: int, config: Dict[str, Any]):
        self.aggregator_id = aggregator_id
        self.config = config
        self.client_weights_buffer = {}
        self.round_count = 0
        super().__init__(f"Aggregator {aggregator_id}")
        self.setup_health_check()
        self.setup_error_handlers()
    
    def setup_routes(self):
        @self.app.get("/status")
        async def get_status():
            return {
                "aggregator_id": self.aggregator_id,
                "status": "active",
                "round_count": self.round_count,
                "buffer_size": len(self.client_weights_buffer)
            }
        
        @self.app.post("/start_federated_round")
        async def start_federated_round(round_id: int = Form(...)):
            try:
                self.round_count = round_id
                self.client_weights_buffer.clear()
                
                logger.info(f"[Aggregator {self.aggregator_id}] 開始第{round_id}輪聯邦學習")
                
                return {
                    "round_id": round_id,
                    "aggregator_id": self.aggregator_id,
                    "status": "round_started"
                }
            except Exception as e:
                logger.error(f"啟動聯邦學習輪次失敗: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/upload_federated_weights")
        async def upload_federated_weights(
            client_id: int = Form(...),
            data_size: int = Form(...),
            weights: UploadFile = File(...),
            round_id: int = Form(...)
        ):
            try:
                if round_id != self.round_count:
                    raise HTTPException(status_code=400, detail=f"輪次不匹配: {round_id} != {self.round_count}")
                
                weights_data = pickle.loads(weights.file.read())
                
                self.client_weights_buffer[client_id] = {
                    'weights': weights_data,
                    'data_size': data_size
                }
                
                logger.info(f"[Aggregator {self.aggregator_id}] 收到客戶端 {client_id} 的權重")
                
                return {
                    "status": "weights_received",
                    "client_id": client_id,
                    "buffer_size": len(self.client_weights_buffer)
                }
            except Exception as e:
                logger.error(f"權重上傳處理失敗: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/get_global_weights")
        async def get_global_weights():
            try:
                empty_weights = {}
                weights_bytes = pickle.dumps(empty_weights)
                
                return Response(
                    content=weights_bytes,
                    media_type="application/octet-stream"
                )
            except Exception as e:
                logger.error(f"全局權重獲取失敗: {e}")
                raise HTTPException(status_code=500, detail=str(e))

class CloudAPI(BaseAPI, HealthCheckMixin, ErrorHandlerMixin):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_weights = None
        self.aggregation_count = 0
        super().__init__("Cloud Server")
        self.setup_health_check()
        self.setup_error_handlers()
    
    def setup_routes(self):
        @self.app.get("/status")
        async def get_status():
            return {
                "status": "active",
                "aggregation_count": self.aggregation_count,
                "has_global_weights": self.global_weights is not None
            }
        
        @self.app.post("/upload_weights")
        async def upload_weights(
            aggregator_id: int = Form(...),
            data_size: int = Form(...),
            weights: UploadFile = File(...)
        ):
            try:
                upload_data = pickle.loads(weights.file.read())
                
                self.aggregation_count += 1
                
                logger.info(f"[Cloud Server] 收到聚合器 {aggregator_id} 的權重")
                
                return {
                    "status": "weights_received",
                    "aggregator_id": aggregator_id,
                    "aggregation_count": self.aggregation_count
                }
            except Exception as e:
                logger.error(f"權重上傳處理失敗: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/get_global_weights")
        async def get_global_weights():
            try:
                if self.global_weights is None:
                    raise HTTPException(status_code=404, detail="全局權重尚未初始化")
                
                weights_bytes = pickle.dumps(self.global_weights)
                
                return Response(
                    content=weights_bytes,
                    media_type="application/octet-stream"
                )
            except Exception as e:
                logger.error(f"全局權重獲取失敗: {e}")
                raise HTTPException(status_code=500, detail=str(e))


class CommunicationClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if self.session is None:
            raise RuntimeError("Session not initialized")
        async with self.session.get(url, **kwargs) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise HTTPException(status_code=response.status, detail=await response.text())
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, 
                  files: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if self.session is None:
            raise RuntimeError("Session not initialized")
        if files:
            form_data = aiohttp.FormData()
            if data:
                for key, value in data.items():
                    form_data.add_field(key, str(value))
            
            for key, file_info in files.items():
                if isinstance(file_info, (list, tuple)):
                    file_obj, filename, content_type = file_info
                    form_data.add_field(key, file_obj, filename=filename, content_type=content_type)
                else:
                    form_data.add_field(key, file_info)
            
            async with self.session.post(url, data=form_data, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(status_code=response.status, detail=await response.text())
        else:
            async with self.session.post(url, json=data, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(status_code=response.status, detail=await response.text())


class SyncCommunicationClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = requests.get(url, timeout=self.timeout, **kwargs)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, 
             files: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        if files:
            response = requests.post(url, data=data, files=files, timeout=self.timeout, **kwargs)
        else:
            response = requests.post(url, json=data, timeout=self.timeout, **kwargs)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)


def create_api(api_type: str, **kwargs) -> BaseAPI:
    if api_type.lower() == 'client':
        return ClientAPI(**kwargs)
    elif api_type.lower() == 'aggregator':
        return AggregatorAPI(**kwargs)
    elif api_type.lower() == 'cloud':
        return CloudAPI(**kwargs)
    else:
        raise ValueError(f"不支持的API類型: {api_type}")


def create_client(base_url: str, async_mode: bool = True, **kwargs):
    if async_mode:
        return CommunicationClient(base_url, **kwargs)
    else:
        return SyncCommunicationClient(base_url, **kwargs)


if __name__ == "__main__":
    config = {"test": "config"}
    client_api = create_api('client', client_id=1, config=config)
    print(f"客戶端API創建成功: {client_api.name}")
    aggregator_api = create_api('aggregator', aggregator_id=0, config=config)
    print(f"聚合器API創建成功: {aggregator_api.name}")
    cloud_api = create_api('cloud', config=config)
    print(f"雲端API創建成功: {cloud_api.name}")
    client = create_client("http://localhost:8000")
    print(f"通信客戶端創建成功: {client.base_url}") 