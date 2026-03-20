# submit-aml

Submit jobs to [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/) with minimal friction.

`submit-aml` wraps the [azure-ai-ml](https://pypi.org/project/azure-ai-ml/) SDK
and provides two interfaces — a **CLI** and a **Python API** — for
submitting training jobs, managing environments, mounting data, running
sweeps, and more.

## Key features

- **One-command job submission** — submit training scripts to Azure ML with
  sensible defaults and minimal boilerplate.
- **CLI and Python API** — use whichever interface fits your workflow.
- **Flexible environment management** — use Docker images, build contexts with
  `uv` dependency resolution, conda environment files, or existing Azure ML
  environments.
- **Data mounting and downloading** — attach Azure ML datasets and job outputs
  as inputs, and configure output datastores.
- **Hyperparameter sweeps** — define grid sweeps inline with a concise
  `parameter=[value1,value2]` syntax.
- **Multi-node distributed training** — scale to multiple nodes with MPI or
  PyTorch distributed, including GPU-aware configuration.
- **Built-in services** — enable TensorBoard and VS Code remote debugging
  directly from CLI flags.
- **Layered configuration** — set defaults via a TOML config file, environment
  variables, or CLI flags, with clear precedence rules.

## Quick start

Install:

```bash
uv tool install submit-aml
```

=== "CLI"

    ```bash
    submit-aml \
        --script train.py \
        --compute-target my-gpu-cluster \
        -- --learning-rate 1e-4
    ```

=== "Python"

    ```python
    from submit_aml import submit_to_aml

    submit_to_aml(
        compute_target="my-gpu-cluster",
        script_path="train.py",
        script_args=["--learning-rate", "1e-4"],
    )
    ```

## Next steps

- [Configuration](configuration.md) — set up your config file and environment
  variables.
- [CLI Reference](cli-reference.md) — full list of all CLI options.
- [Python API](api/index.md) — API reference for programmatic usage.
- [Examples](examples.md) — common usage patterns and recipes.
- [Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
  — official Azure Machine Learning docs.

