# Examples

## Basic submission

Submit a training script to Azure ML:

=== "CLI"

    ```bash
    submit-aml --script train.py
    ```

    With a specific experiment name and run name:

    ```bash
    submit-aml \
        --script train.py \
        --experiment-name my-experiment \
        --run-name "baseline-run"
    ```

    Pass extra arguments to the script after `--`:

    ```bash
    submit-aml --script train.py -- --learning-rate 1e-4 --batch-size 32
    ```

=== "Python"

    ```python
    from submit_aml import submit_to_aml

    submit_to_aml(script_path="train.py")
    ```

    With a specific experiment name and run name:

    ```python
    submit_to_aml(
        script_path="train.py",
        experiment_name="my-experiment",
        run_name="baseline-run",
    )
    ```

    Pass extra arguments to the script:

    ```python
    submit_to_aml(
        script_path="train.py",
        script_args=["--learning-rate", "1e-4", "--batch-size", "32"],
    )
    ```

## Choosing a compute target

=== "CLI"

    ```bash
    submit-aml \
        --script train.py \
        --compute-target gpu-v100-cluster \
        --num-nodes 1
    ```

=== "Python"

    ```python
    submit_to_aml(
        script_path="train.py",
        compute_target="gpu-v100-cluster",
        num_nodes=1,
    )
    ```

## Multi-node training

### MPI (default)

When `--num-gpus` is **not** set, MPI distribution is used:

=== "CLI"

    ```bash
    submit-aml \
        --script train.py \
        --compute-target gpu-cluster \
        --num-nodes 4
    ```

=== "Python"

    ```python
    submit_to_aml(
        script_path="train.py",
        compute_target="gpu-cluster",
        num_nodes=4,
    )
    ```

### PyTorch distributed

Set `--num-gpus` to enable
[`PyTorchDistribution`](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.pytorchdistribution?view=azure-python):

=== "CLI"

    ```bash
    submit-aml \
        --script train.py \
        --compute-target gpu-cluster \
        --num-nodes 2 \
        --num-gpus 4
    ```

=== "Python"

    ```python
    submit_to_aml(
        script_path="train.py",
        compute_target="gpu-cluster",
        num_nodes=2,
        num_gpus=4,
    )
    ```

This configures 4 processes per node across 2 nodes (8 GPUs total).

## Sweep jobs

Run a grid sweep over hyperparameters:

=== "CLI"

    ```bash
    submit-aml \
        --script train.py \
        --sweep "learning_rate=[1e-4,2e-4,5e-4]" \
        --sweep "seed=[0,1,2]"
    ```

    Limit concurrent trials:

    ```bash
    submit-aml \
        --script train.py \
        --sweep "learning_rate=[1e-4,2e-4,5e-4]" \
        --max-concurrent-trials 3
    ```

=== "Python"

    ```python
    from azure.ai.ml.sweep import Choice
    from submit_aml import submit_to_aml

    submit_to_aml(
        script_path="train.py",
        sweep_inputs={
            "learning_rate": Choice(values=[1e-4, 2e-4, 5e-4]),
            "seed": Choice(values=[0, 1, 2]),
        },
        sweep_max_concurrent_trials=3,
    )
    ```

## Data mounting

Datasets are passed to the job as
[`Input`](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.input?view=azure-python)
objects.

=== "CLI"

    Mount a dataset:

    ```bash
    submit-aml \
        --script train.py \
        --mount "data=MY-DATASET:2"
    ```

    Download a dataset:

    ```bash
    submit-aml \
        --script train.py \
        --download "data=MY-DATASET"
    ```

    Use outputs from a previous job:

    ```bash
    submit-aml \
        --script evaluate.py \
        --mount "checkpoint=job_dir:my-training-job:models/best.pth"
    ```

=== "Python"

    ```python
    submit_to_aml(
        script_path="train.py",
        datasets_mount=["data=MY-DATASET:2"],
    )

    # Or download instead of mount
    submit_to_aml(
        script_path="train.py",
        datasets_download=["data=MY-DATASET"],
    )

    # Use outputs from a previous job
    submit_to_aml(
        script_path="evaluate.py",
        datasets_mount=["checkpoint=job_dir:my-training-job:models/best.pth"],
    )
    ```

Configure an output datastore:

=== "CLI"

    ```bash
    submit-aml \
        --script train.py \
        --output "results=mydatastore/experiment-outputs"
    ```

=== "Python"

    ```python
    submit_to_aml(
        script_path="train.py",
        datasets_output=["results=mydatastore/experiment-outputs"],
    )
    ```

## Environment management

### Docker build context (default)

By default, `submit-aml` builds a Docker context from your project's
`pyproject.toml`, `uv.lock`, and `.python-version`:

=== "CLI"

    ```bash
    submit-aml --script train.py
    ```

    Install specific dependency groups:

    ```bash
    submit-aml \
        --script train.py \
        --dependency-group train \
        --dependency-group data
    ```

=== "Python"

    ```python
    submit_to_aml(script_path="train.py")

    # With specific dependency groups
    submit_to_aml(
        script_path="train.py",
        dependency_groups=["train", "data"],
    )
    ```

### Custom Docker image

You can use any image, including ones from the
[Azure ML containers repo](https://github.com/Azure/AzureML-Containers):

=== "CLI"

    ```bash
    submit-aml \
        --script train.py \
        --docker-image "myregistry.azurecr.io/training:latest" \
        --no-build-context
    ```

=== "Python"

    ```python
    submit_to_aml(
        script_path="train.py",
        base_docker_image="myregistry.azurecr.io/training:latest",
        build_docker_context=False,
    )
    ```

### Existing Azure ML environment

=== "CLI"

    ```bash
    submit-aml --script train.py --aml-environment "my-curated-env"
    ```

=== "Python"

    ```python
    submit_to_aml(
        script_path="train.py",
        aml_environment="my-curated-env",
    )
    ```

### Conda environment

=== "CLI"

    ```bash
    submit-aml --script train.py --conda-env-file environment.yml
    ```

=== "Python"

    ```python
    from pathlib import Path

    submit_to_aml(
        script_path="train.py",
        conda_env_file=Path("environment.yml"),
        build_docker_context=False,
    )
    ```

## Setting environment variables

=== "CLI"

    ```bash
    submit-aml \
        --script train.py \
        --set "WANDB_API_KEY=abc123" \
        --set "NCCL_DEBUG=INFO"
    ```

=== "Python"

    ```python
    submit_to_aml(
        script_path="train.py",
        environment_variables={
            "WANDB_API_KEY": "abc123",
            "NCCL_DEBUG": "INFO",
        },
    )
    ```

## Debugging

Enable remote debugging with `debugpy`:

=== "CLI"

    ```bash
    submit-aml --script train.py --debug
    ```

=== "Python"

    ```python
    submit_to_aml(script_path="train.py", debug=True)
    ```

This installs `debugpy`, starts the script with a debug listener on port 5678,
and adds a VS Code service for remote connection.

## Dry run

Preview the job configuration without submitting:

=== "CLI"

    ```bash
    submit-aml --script train.py --dry-run
    ```

=== "Python"

    ```python
    submit_to_aml(script_path="train.py", dry_run=True)
    ```

## Stream logs

Submit and wait for the job to complete, streaming logs:

=== "CLI"

    ```bash
    submit-aml --script train.py --stream-logs
    ```

=== "Python"

    ```python
    submit_to_aml(script_path="train.py", wait_for_completion=True)
    ```

