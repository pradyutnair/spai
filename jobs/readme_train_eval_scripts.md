# SPAI Training and Evaluation Workflow

## ⚙️ Initial Setup

Follow these steps once to prepare the system.

### 1. Create the `.env` File

In the root of your project repository (e.g., `~/DL2/spai/`), create a file named `.env`.

> 🛡️ **Security Warning**: Add `.env` to your `.gitignore` file immediately to prevent accidentally committing your secrets.

**`.env` template:**
```dotenv
# Neptune credentials - DO NOT COMMIT TO GIT
NEPTUNE_API_TOKEN="your-long-neptune-api-token"
NEPTUNE_PROJECT="your-neptune-project-name"
```

### 2. Make Launchers Executable

Give execute permissions to both launcher scripts:
```bash
chmod +x jobs/train/run_trains_launch.sh
chmod +x jobs/eval/run_evals_launch.sh
```

---

## 🚀 Training Workflow

### 1. Configure Training Runs
Open `jobs/train/run_trains_launch.sh` and edit the **`SCRIPT CONFIGURATION`** section.

-   **Job Control & Slurm**: Adjust `MAX_JOBS`, `PARTITION`, `TIME_LIMIT`, etc.
-   **Python Script Parameters**: Set default training hyperparameters like `BATCH_SIZE`.
-   **Experiment Definitions**: Define your training runs by adding key-value pairs to the `CONFIGS` and `DATASETS` arrays. The key (e.g., `"clip_cross_attn_after_sca"`) is the short name you'll use from the command line.

### 2. Launch Training Jobs

Navigate to your project root (`~/DL2/spai`) and run the launcher.

**Run all defined training jobs:**
```bash
./jobs/train/run_trains_launch.sh
```

**Run jobs for a specific configuration on all datasets:**
```bash
# The argument must be a key from the CONFIGS array
./jobs/train/run_trains_launch.sh clip_cross_attn_after_sca
```

**Run jobs for a specific dataset with all configs:**
```bash
# The argument must be a key from the DATASETS array
./jobs/train/run_trains_launch.sh chameleon
```

**Run a specific, targeted training job:**
```bash
./jobs/train/run_trains_launch.sh clip_cross_attn_after_sca chameleon
```

---

## 🔬 Evaluation Workflow

### 1. Configure Evaluation Runs
Open `jobs/eval/run_evals_launch.sh` and edit the **`SCRIPT CONFIGURATION`** section.

-   **Job Control & Slurm**: Adjust `MAX_JOBS`, `PARTITION`, etc.
-   **Python Script Parameters**: Set default evaluation parameters like `BATCH_SIZE`.
-   **Experiment Definitions**: Define your evaluation runs by adding key-value pairs to the `MODELS` and `TEST_SETS` arrays. The key is the short name you'll use from the command line.

### 2. Launch Evaluation Jobs

Navigate to your project root (`~/DL2/spai`) and run the launcher.

**Run all defined evaluation jobs:**
```bash
./jobs/eval/run_evals_launch.sh
```

**Run evaluations for a specific model on all test sets:**
```bash
# The argument must be a key from the MODELS array
./jobs/eval/run_evals_launch.sh clip_cross_attn_after_sca_chameleon
```

**Run evaluations on a specific test set for all models:**
```bash
# The argument must be a key from the TEST_SETS array
./jobs/eval/run_evals_launch.sh dalle3
```

---

## 📊 Monitoring All Jobs

-   **Check the Slurm Queue**: See your currently running or pending jobs.
    ```bash
    squeue -u $USER
    ```
-   **Check the Output Logs**: Stdout from each job are saved in the `jobs/out_files_train/` and `jobs/out_files_eval/` directories.
-   **Check the Results**:
    -   **Training artifacts** (checkpoints, etc.) are saved in subfolders within `/scratch-shared/dl2_spai_models/finetune/`.
    -   **Evaluation artifacts** are saved in subfolders within the `output/` directory.