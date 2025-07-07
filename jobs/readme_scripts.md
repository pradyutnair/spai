# Evaluation

## ⚙️ Setup and Configuration

Follow these steps to get the system ready for use.

### 1. Create the `.env` File

In the root directory, create a file named `.env`. This file will store your Neptune credentials.

**`.env` template:**
```dotenv
# Neptune credentials - DO NOT COMMIT TO GIT
NEPTUNE_API_TOKEN="your-long-neptune-api-token"
NEPTUNE_PROJECT="your-neptune-project-name"
```

### 2. Configure the Launcher Script

Open `run_evals_launch.sh` and edit the **`SCRIPT CONFIGURATION`** section at the top.

-   **Job Control**: Set `MAX_JOBS` to limit the number of concurrently queued jobs.
-   **Slurm Parameters**: Adjust `PARTITION`, `TIME_LIMIT`, `MEMORY`, etc., to match your cluster's requirements.
-   **Python Script Parameters**: Change default values for `BATCH_SIZE`, `NUM_WORKERS`, etc.
-   **Experiment Definitions**: Add new models, test sets, or config files to the `MODELS`, `TEST_SETS`, and `CONFIGS` associative arrays. The key (e.g., `"dalle2"`) is what you'll use in the command line.

### 3. Make the Launcher Executable

Before running the script for the first time, you need to give it execute permissions:

```bash
chmod +x run_evals_launch.sh
```

## 💡 Usage

You run all commands from the directory containing `run_evals_launch.sh`. The script's behavior changes based on the command-line arguments you provide.

### Run All Experiments
To submit a job for every defined model against every defined test set:
```bash
./run_evals_launch.sh
```

### Run on a Specific Model
To evaluate a single model against all available test sets:
```bash
# The argument must be a key from the MODELS array
./run_evals_launch.sh clip_cross_attn_after_sca_chameleon
```

### Run on a Specific Test Set
To evaluate all models against a single test set:
```bash
# The argument must be a key from the TEST_SETS array
./run_evals_launch.sh dalle3
```

### Run a Specific Model-Test Combination
To run a single, targeted evaluation:
```bash
./run_evals_launch.sh clip_cross_attn_after_sca_chameleon dalle3
```

### Run Multiple Specific Selections
You can combine multiple model and test set keys. The script will generate jobs for every valid combination of the specified arguments.
```bash
# Runs `clip_...` on `dalle3` and `sdxl`.
# Runs `convnext_...` on `dalle3` and `sdxl`.
./run_evals_launch.sh clip_cross_attn_after_sca_chameleon convnext_cross_attn_after_sca_chameleon dalle3 sdxl
```

## 📊 Monitoring Jobs

-   **Check the Slurm Queue**: See your currently running or pending jobs.
    ```bash
    squeue -u $USER
    ```
-   **Check the Output Logs**: The stdout and stderr from each job are saved in the `jobs/out_files_eval/` directory. The files are named according to the job name and ID, for example:
    -   `eval_clip_cross_attn_after_sca_chameleon_dalle3_12345.out`
    -   `eval_clip_cross_attn_after_sca_chameleon_dalle3_12345.err`

-   **Check the Python Results**: The actual evaluation artifacts (CSVs, images, etc.) are saved in the `output/` directory, organized by timestamp, model, and test set.