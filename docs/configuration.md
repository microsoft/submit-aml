# Configuration

`submit-aml` uses a layered configuration system. Values are resolved in the
following precedence order (highest to lowest):

1. **CLI flags** — explicitly passed on the command line.
2. **Environment variables** — prefixed with `SUBMIT_AML_`.
3. **Config file** — a TOML file at `~/.config/submit-aml/config.toml`.
4. **Package defaults** — built-in fallback values.

## Config file

The config file lives at:

```
~/.config/submit-aml/config.toml
```

### Template

Use this template as a starting point:

```toml
# Default workspace used when --workspace is not passed
default_workspace = "my-workspace"

[workspaces.my-workspace]
subscription_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
resource_group = "my-resource-group"

[workspaces.other-workspace]
subscription_id = "yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy"
resource_group = "other-rg"

[compute]
compute_target = "my-gpu-cluster"
num_nodes = 1
docker_shared_memory_gb = 256

[environment]
docker_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"

[command]
command_prefix = "uv run --no-default-groups"
executable = "python"

[tensorboard]
tensorboard_dir = "logs/tensorboard"
enable_tensorboard = true
```

### Config sections

#### Top-level keys

| Key | Description |
|---|---|
| `default_workspace` | Name of the workspace profile used when `--workspace` is not passed |

#### Workspace profiles

Workspace profiles let you define per-workspace `subscription_id` and
`resource_group` values so you don't need to pass them every time. Add
`[workspaces.<name>]` sections to your config file:

```toml
[workspaces.my-workspace]
subscription_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
resource_group = "my-resource-group"

[workspaces.other-workspace]
subscription_id = "yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy"
resource_group = "other-rg"
```

When you pass `--workspace my-workspace` **without** `--subscription` or
`--resource-group`, the tool looks up the matching profile and fills in the
missing values automatically:

```bash
# Uses subscription_id and resource_group from [workspaces.my-workspace]
submit-aml --workspace my-workspace --script train.py
```

If `--workspace` is not passed, the tool uses `default_workspace` from the
config file.

Profile values are only used when the corresponding CLI flag is **not**
provided. Explicit flags always take precedence.

#### `[compute]`

| Key | Default | Description |
|---|---|---|
| `compute_target` | — | Azure ML compute target name |
| `num_nodes` | `1` | Number of compute nodes |
| `docker_shared_memory_gb` | `256` | Shared memory for Docker container (GB) |

#### `[environment]`

| Key | Default | Description |
|---|---|---|
| `docker_image` | `mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04` | Base Docker image (see [Azure ML containers](https://github.com/Azure/AzureML-Containers)) |

#### `[command]`

| Key | Default | Description |
|---|---|---|
| `command_prefix` | `uv run --no-default-groups` | Prefix prepended to the command |
| `executable` | `python` | Executable to run the script |

#### `[tensorboard]`

| Key | Default | Description |
|---|---|---|
| `tensorboard_dir` | `logs/tensorboard` | Directory for TensorBoard logs |
| `enable_tensorboard` | `true` | Enable TensorBoard service |

## Environment variables

All config keys can be overridden via environment variables using the
`SUBMIT_AML_` prefix. The variable name is derived from the config key in
uppercase:

| Config key | Environment variable |
|---|---|
| `default_workspace` | `SUBMIT_AML_DEFAULT_WORKSPACE` |
| `compute_target` | `SUBMIT_AML_COMPUTE_TARGET` |
| `num_nodes` | `SUBMIT_AML_NUM_NODES` |
| `docker_image` | `SUBMIT_AML_DOCKER_IMAGE` |
| `command_prefix` | `SUBMIT_AML_COMMAND_PREFIX` |
| `executable` | `SUBMIT_AML_EXECUTABLE` |

Example:

```bash
export SUBMIT_AML_DEFAULT_WORKSPACE="my-workspace"
export SUBMIT_AML_COMPUTE_TARGET="gpu-cluster-v100"
submit-aml --script train.py
```

## CLI flags

CLI flags always take the highest precedence. See the full list in the
[CLI Reference](cli-reference.md).

```bash
submit-aml --workspace my-workspace --compute-target gpu-cluster --script train.py
```

## Precedence example

Given:

- Config file sets `compute_target = "cpu-cluster"`
- Environment variable `SUBMIT_AML_COMPUTE_TARGET=gpu-cluster`
- CLI flag `--compute-target a100-cluster`

The resolved value is `a100-cluster` (CLI flag wins).

