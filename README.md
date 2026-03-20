# submit-aml

A CLI tool to submit jobs to Azure Machine Learning.

## Quick start

```shell
uv tool install submit-aml
```

```shell
submit-aml \
    --subscription "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" \
    --resource-group "my-resource-group" \
    --workspace "my-workspace" \
    --compute-target "my-gpu-cluster" \
    --script run.py \
        arg1 \
        arg2
```

## Configuration

Azure ML defaults can be set via a config file at `~/.config/submit-aml/config.toml` or environment variables prefixed `SUBMIT_AML_`. See the [documentation](https://microsoft.github.io/submit-aml) for details.
