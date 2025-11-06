# Google Cloud Parallelization Setup

This document describes how to set up and use Google Cloud Compute Engine for parallel HAL benchmark execution.

## What We Built

The HAL harness now supports running evaluations on Google Cloud VMs using the `--google-cloud` flag. This enables:
- **Parallel execution** across 10-100+ VMs simultaneously
- **10-100x speedup** compared to local execution
- **Auto-cleanup** of VMs after completion
- **Cost-effective** execution (~$0.04 per task)

## Setup Steps Completed

### 1. Google Cloud Authentication

```bash
# Logged into Google Cloud
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project vitaly-gcp
```

### 2. Enabled Required APIs

```bash
gcloud services enable compute.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable storage.googleapis.com
```

### 3. Created Artifact Registry for Docker Images

```bash
gcloud artifacts repositories create hal-artifacts \
  --repository-format=docker \
  --location=us-central1 \
  --project=vitaly-gcp
```

### 4. Created Cloud Storage Bucket for Results

```bash
gsutil mb -p vitaly-gcp -l us-central1 gs://hal-results
```

### 5. Configured Docker for Artifact Registry

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 6. Built and Pushed Docker Image

```bash
cd hal-harness
docker build -t us-central1-docker.pkg.dev/vitaly-gcp/hal-artifacts/hal-agent-runner:latest \
  -f hal/utils/docker/Dockerfile hal/utils/docker/

docker push us-central1-docker.pkg.dev/vitaly-gcp/hal-artifacts/hal-agent-runner:latest
```

### 7. Set Up SSH Keys (Already Existed)

- Private key: `~/.ssh/hal_gcp`
- Public key: `~/.ssh/hal_gcp.pub`

### 8. Installed Python Dependencies

```bash
pip install google-cloud-compute google-cloud-storage paramiko
```

### 9. Configured Environment Variables

Added to `hal-harness/.env`:

```bash
GCP_PROJECT_ID=vitaly-gcp
GCP_ZONE=us-central1-a
GCP_REGION=us-central1
ARTIFACT_REGISTRY=hal-artifacts
BUCKET_NAME=hal-results
SSH_PUBLIC_KEY_PATH=~/.ssh/hal_gcp.pub
SSH_PRIVATE_KEY_PATH=~/.ssh/hal_gcp
```

## Implementation Details

### Files Created/Modified

1. **`hal/utils/gcp_utils.py`** - Google Cloud VM management
   - VirtualMachineManager class
   - VM lifecycle operations
   - File transfer via SSH
   - Command execution on VMs

2. **`hal/utils/gcp_runner.py`** - Parallel execution runner
   - GCPRunner class
   - Parallel task distribution
   - Result aggregation

3. **`hal/cli.py`** - Added `--google-cloud` flag
   - New CLI option
   - Validation logic

4. **`hal/agent_runner.py`** - GCP runner integration
   - use_google_cloud parameter
   - Runner selection logic

## Usage

### Test with Single Task

```bash
hal-eval --benchmark usaco \
  --agent_dir agents/usaco_example_agent/ \
  --agent_function main.run \
  --agent_name "Test Run" \
  --google-cloud \
  --max_concurrent 1 \
  --max_tasks 1 \
  -A model_name=gpt-4o-mini-2024-07-18
```

### Full Parallel Run

```bash
hal-eval --benchmark usaco \
  --agent_dir agents/usaco_example_agent/ \
  --agent_function main.run \
  --agent_name "USACO Full Run" \
  --google-cloud \
  --max_concurrent 50 \
  -A model_name=gpt-4o-2024-11-20
```

### With High Concurrency

```bash
hal-eval --benchmark swebench_verified_mini \
  --agent_dir agents/swebench_example_agent/ \
  --agent_function main.run \
  --agent_name "SWE-bench Run" \
  --google-cloud \
  --max_concurrent 100 \
  -A model_name=gpt-4o-2024-11-20
```

## Architecture

```
hal-eval --google-cloud
    ↓
AgentRunner (GCPRunner)
    ↓
Create N GCE VMs (parallel)
    ↓
Each VM: 
    1. Pull Docker image from Artifact Registry
    2. Receive task data via SSH
    3. Execute agent in Docker container
    4. Save results to /tmp/hal/output
    ↓
Main process collects results
    ↓
VMs deleted automatically
    ↓
Results aggregated and saved
```

## Cost Estimates

### Small Benchmark (10 tasks, 5 min each)
- **10 VMs × 5 minutes = ~50 minutes compute**
- **Cost: ~$0.50** (standard VMs) or **~$0.05** (preemptible)

### Medium Benchmark (50 tasks, 10 min each)
- **50 VMs × 10 minutes = ~10 minutes wall-clock time**
- **Cost: ~$2** (standard) or **~$0.20** (preemptible)

### Large Benchmark (200 tasks, 15 min each)
- **50 VMs × 60 minutes = ~1 hour wall-clock**
- **Cost: ~$15** (standard) or **~$1.50** (preemptible)

## Monitoring & Debugging

### Check Running VMs

```bash
gcloud compute instances list --project=vitaly-gcp
```

### View VM Logs

```bash
gcloud compute instances get-serial-port-output <vm-name> --zone=us-central1-a
```

### View Results

```bash
# Results are saved to:
hal-harness/results/<benchmark>/<run_id>/

# Also check Cloud Storage:
gsutil ls gs://hal-results/
```

### List Docker Images

```bash
gcloud artifacts docker images list us-central1-docker.pkg.dev/vitaly-gcp/hal-artifacts/hal-agent-runner
```

## Troubleshooting

### VMs stuck in "creating" state
- Check quota limits: `gcloud compute project-info describe --project=vitaly-gcp`
- Check serial logs for startup errors

### Docker pull failures
- Re-authenticate: `gcloud auth configure-docker us-central1-docker.pkg.dev`
- Verify image exists: `gcloud artifacts docker images list`

### SSH connection failures
- Verify key permissions: `chmod 600 ~/.ssh/hal_gcp`
- Check firewall rules allow SSH (port 22)

### Permission errors
- Ensure Compute Engine API is enabled
- Check IAM roles for service account

## Cost Optimization Tips

1. **Use preemptible instances** (91% cost savings)
2. **Auto-scaling** - VMs created/destroyed on-demand
3. **Regional compute** - Choose closest zone
4. **Commit to sustained use** - For long-running benchmarks

## Next Steps

1. Test with a single task to verify setup
2. Run small benchmarks (10-50 tasks)
3. Scale up to full runs with high concurrency
4. Monitor costs and optimize VM sizes
5. Enable preemptible instances for cost savings

## Configuration Summary

**Project**: vitaly-gcp  
**Zone**: us-central1-a  
**Artifact Registry**: hal-artifacts  
**Storage Bucket**: hal-results  
**SSH Keys**: ~/.ssh/hal_gcp  
**Docker Image**: hal-agent-runner:latest  

## Related Files

- Strategy doc: `../incompetagents/GOOGLE_CLOUD_PARALLELIZATION.md`
- Quick start: `../incompetagents/GCP_QUICKSTART.md`
- Setup script: `../incompetagents/setup_gcp_parallelization.sh`

