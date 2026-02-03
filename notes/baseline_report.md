# Baseline Evaluation Report

**Generated:** 2026-01-27T15:18:00

## Executive Summary

This baseline experiment attempted to run 20 SWE-bench Lite instances using the mini-swe-agent with gpt-5-mini model. **All instances failed during environment setup** due to Docker image pull timeouts (120s default timeout is insufficient for multi-GB SWE-bench images).

**Key Finding:** This is an infrastructure issue, not a model/algorithm issue. The solution is to either:
1. Pre-pull Docker images before running experiments
2. Increase Docker startup timeout in environment configuration
3. Use Modal sandbox environment (swerex_modal) which handles image caching

## Configuration

```yaml
model: openai/gpt-5-mini
subset: lite
split: test
slice: 0:20
workers: 4
environment: docker
config_file: src/minisweagent/config/extra/swebench_baseline.yaml
```

**Run Command:**
```bash
NO_PROXY="localhost,127.0.0.1" uv run mini-extra swebench \
  --config src/minisweagent/config/extra/swebench_baseline.yaml \
  --subset lite --split test --slice 0:20 \
  -w 4 -o results/baseline_lite_test_0_20
```

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Instances | 20 |
| Resolved | 0 |
| Failed | 20 |
| Resolution Rate | 0.0% |
| Avg Turns | 0.0 |
| Avg Tool Calls | 0.0 |
| Total Cost | $0.0000 |

**Note:** All failures occurred at environment setup stage, before any model interaction.

## Failure Breakdown

| Failure Type | Count | Percentage | Description |
|--------------|-------|------------|-------------|
| docker_timeout | 19 | 95.0% | Docker image pull exceeded 120s timeout |
| docker_error | 1 | 5.0% | Docker daemon returned error (exit status 125) |

## Root Cause Analysis

### Docker Image Pull Timeout

SWE-bench Docker images are large (2-5GB each). The default 120-second timeout is insufficient for pulling these images on first run.

**Evidence:**
```
subprocess.TimeoutExpired: Command '['docker', 'run', '-d', '--name',
'minisweagent-18ad3cc8', '-w', '/testbed', '--rm',
'docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest',
'sleep', '2h']' timed out after 120 seconds
```

### Solutions

1. **Pre-pull Images** (Recommended for local development):
   ```bash
   # Get list of required images for your slice
   docker pull docker.io/swebench/sweb.eval.x86_64.django_1776_django-11039:latest
   # ... repeat for all instances
   ```

2. **Increase Timeout** (Modify docker.py):
   Change `timeout=120` to `timeout=600` in `_start_container()` method.

3. **Use Modal Environment** (Recommended for production):
   ```yaml
   environment:
     environment_class: swerex_modal
   ```
   Modal caches images and handles startup more gracefully.

## Sample Failure Cases

### docker_timeout

**Instance:** `astropy__astropy-12907`
- Exit status: `TimeoutExpired`
- Turns: 0
- Tool calls: 0
- Trajectory file: `results/baseline_lite_test_0_20/astropy__astropy-12907/astropy__astropy-12907.traj.json`

**Instance:** `django__django-11039`
- Exit status: `TimeoutExpired`
- Turns: 0
- Tool calls: 0
- Trajectory file: `results/baseline_lite_test_0_20/django__django-11039/django__django-11039.traj.json`

### docker_error

**Instance:** `astropy__astropy-14365`
- Exit status: `CalledProcessError`
- Turns: 0
- Tool calls: 0
- Error: Docker daemon returned exit status 125 (typically indicates image not found or permission issue)

## Instance Details

| Instance ID | Status | Turns | Tool Calls | Cost |
|-------------|--------|-------|------------|------|
| astropy__astropy-12907 | TimeoutExpired | 0 | 0 | $0.0000 |
| astropy__astropy-14182 | TimeoutExpired | 0 | 0 | $0.0000 |
| astropy__astropy-14365 | CalledProcessError | 0 | 0 | $0.0000 |
| astropy__astropy-14995 | TimeoutExpired | 0 | 0 | $0.0000 |
| astropy__astropy-6938 | TimeoutExpired | 0 | 0 | $0.0000 |
| astropy__astropy-7746 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-10914 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-10924 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11001 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11019 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11039 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11049 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11099 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11133 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11179 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11283 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11422 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11564 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11583 | TimeoutExpired | 0 | 0 | $0.0000 |
| django__django-11620 | TimeoutExpired | 0 | 0 | $0.0000 |

## Next Steps

1. **Immediate:** Pre-pull Docker images for target instances
2. **Short-term:** Re-run baseline with pre-pulled images to get valid trajectory data
3. **Medium-term:** Consider switching to Modal environment for better scalability

## Appendix: Output Files

```
results/baseline_lite_test_0_20/
├── minisweagent.log           # Detailed run log
├── preds.json                 # Predictions/submissions
├── summary.json               # JSON summary statistics
├── exit_statuses_*.yaml       # Exit status breakdown
├── run.log                    # Console output log
└── <instance_id>/
    └── <instance_id>.traj.json  # Per-instance trajectory
```
