# H3-FedUAV: Hierarchical Federated Learning for UAV Network Intrusion Detection

A three-tier hierarchical federated learning framework for UAV Network Intrusion Detection Systems (NIDS).

This project aims to build a stable and extensible Client–Aggregator–Cloud federated learning pipeline that addresses real-world issues such as non-IID data, class imbalance, and potentially malicious clients.

---

## Overview

- **Problem setting**: In low-altitude wireless networks (LAWN), UAV-side NIDS must learn collaboratively across distributed nodes where data cannot be centralized. Network conditions can be unstable, and client data distributions often differ significantly.
- **Approach**: This framework provides a Client–Aggregator–Cloud architecture and integrates components such as FedProx, knowledge distillation, prototype learning, GAN-based augmentation, and regional aggregation to improve stability and detection performance under heterogeneity and adversarial conditions.

This repository is primarily a runnable application (you can start an experiment from the command line), and it can also be partially reused as a library via the `models/`, `utils/`, and `communication/` modules.

---

## Installation

### 1. Get the code

Place this project in any directory on your machine (examples below assume `~/uav/github`). If you use Git:

```bash
git clone <your-repo-url> uav
cd uav/github
```


### 2. Create an environment and install dependencies

Recommended requirements:

- **Python**: 3.8+ (recommended 3.10+)
- **PyTorch**: 2.x (CUDA-compatible build or CPU build)
- Other dependencies: see `requirements.txt`

Example using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install PyTorch first using the official instructions, then run `pip install -r requirements.txt` (you may remove `torch`-related lines in `requirements.txt` if needed).

---

## Usage

### 1. Data and preprocessing

The default dataset is **CICIDS2017** (84 features, 5 classes: BENIGN, DDoS, DoS_Hulk, DoS_Slowhttptest, DoS_slowloris).

1. Prepare the raw dataset according to the paths expected by `preprocess_data.py`.
2. Run preprocessing to generate per-client processed CSV files:

```bash
python preprocess_data.py
```

After preprocessing, you should see a directory such as `processed_data/` under the project. The runtime will read from the path specified by `DATA_PATH` in `config_fixed.py`.

### 2. Basic launch flow

The standard flow is to start three roles (Cloud, Aggregator, Clients). They can run in different terminals and/or on different machines:

1. Start the Cloud server:

```bash
python cloud_server_fixed.py
```

2. Start an Aggregator (one or more):

```bash
python aggregator_fixed.py --aggregator_id 0
```

3. Start UAV Clients (typically multiple terminals or machines):

```bash
python uav_client_fixed.py \
    --client_id 0 \
    --aggregator_url http://127.0.0.1:8000 \
    --cloud_url http://127.0.0.1:9000 \
    --result_dir ./result/uav0
```

The exact ports and URLs must match your network configuration in `config_fixed.py`.

4. Optional: use the experiment orchestrator

If you want a single script to coordinate the experiment:

```bash
python start_fixed_experiment.py
```

This script starts Cloud/Aggregators/Clients based on configuration and monitors progress.

### 3. Inputs and outputs

- **Inputs**
  - Per-client processed CSV files. The root path is configured by `config_fixed.py: DATA_PATH`.
  - CLI arguments such as `--client_id`, `--aggregator_url`, `--cloud_url`, `--result_dir`.
  - Optional environment variables such as `FORCE_CPU`, `GAN_AUG_*`, `CLIENT_EVAL_SOURCE` to override parts of the configuration.

- **Outputs**
  - **Training and validation logs**: each client writes artifacts under its `result_dir`, including `uav{client_id}_curve.csv`, logs, and confusion-matrix CSVs.
  - **Cloud and aggregator logs**: training progress and global metrics in their respective output directories.
  - **Model artifacts**: client encoder weights (e.g. `client_encoder_weights.pt`), global weights, dynamic class-weight files, etc.

At a high level, you only need to:

1. Prepare the dataset and run preprocessing.
2. Configure `config_fixed.py`.
3. Start Cloud / Aggregator / Clients.

The system will run multiple federated rounds and produce reports and model artifacts under `result`.


---

## Configuration reference (important options)

Most configuration lives in `config_fixed.py`:

- **Data and paths**
  - **DATA_PATH**: root directory of processed per-client data.
  - **LOG_DIR**: root directory for experiment outputs (curves, confusion matrices, encoder weights, etc.).

- **Model**
  - **MODEL_CONFIG**
    - `type`: `dnn` / `cnn` / `transformer`, etc.
    - `input_dim`: feature dimension (84 for CICIDS2017).
    - `num_classes`: number of classes (default 5).
    - other fields such as `hidden_dims`, `dropout_rate`, `use_batch_norm`, `use_residual`.

- **Training**
  - **LEARNING_RATE**, **BATCH_SIZE**, **LOCAL_EPOCHS**, **MAX_ROUNDS**
  - **LOSS_CONFIG**: class-weight strategies, label smoothing, etc.
  - **FEDPROX_CONFIG**: enable/disable FedProx, `mu`, and optional round-dependent scheduling.
  - **PROTOTYPE_LOSS_CONFIG**: enable/disable prototype loss and its weight.

- **Federation and networking**
  - **NUM_CLIENTS**, **NUM_AGGREGATORS** and client–aggregator assignment logic.
  - cloud/aggregator HTTP endpoints and ports.

- **Advanced features**
  - **GAN_AUGMENTATION_CONFIG**: enable/disable GAN augmentation, target label, augmentation ratio, etc.
  - **DYNAMIC_CLASS_WEIGHTING**: file name and path for loading dynamic class weights produced by the cloud.
  - **CLIENT_KD_CONFIG**: client-side knowledge distillation settings.

For a quick “first successful run”, verify at minimum:

- `DATA_PATH` points to the processed data directory
- `MODEL_CONFIG['input_dim']` and `num_classes` match your dataset
- `NUM_CLIENTS` matches the number of clients you actually start

For more details, see `ARCHITECTURE.md` and `GITHUB_CORE_FILES.md`.

---

## Project layout (high level)

```text
.
├── start_fixed_experiment.py      Experiment orchestration script
├── config_fixed.py                Global configuration
├── cloud_server_fixed.py          Cloud server
├── aggregator_fixed.py            Regional aggregator
├── uav_client_fixed.py            UAV client
├── preprocess_data.py             Data preprocessing
├── models/
│   ├── dnn.py                     DNN model
│   ├── fedprox.py                 FedProx utilities
│   ├── heterogeneous_fl.py        Heterogeneous FL model
│   ├── regional_aggregation.py    Regional aggregation strategy
│   └── weight_transfer.py         DNN to H2FL weight transfer tool
├── communication/
│   └── base_api.py                FastAPI/HTTP communication base
├── utils/
│   ├── fl_comprehensive_evaluator.py  FL evaluation and reports
│   ├── experiment_grouping.py         Multi-experiment aggregation utilities
│   ├── confidence_score.py            Confidence scoring utilities
│   └── cuda_memory_manager.py         CUDA memory helper
└── requirements.txt               Python dependencies
```





