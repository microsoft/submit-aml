# Python API

`submit-aml` can be used as a Python library. The main entry point is
[`submit_to_aml`][submit_aml.aml.submit_to_aml]:

```python
from submit_aml import submit_to_aml

job = submit_to_aml(
    workspace_name="my-ml-workspace",
    compute_target="gpu-v100x4",
    script_path="train.py",
    script_args=["--epochs", "50", "--lr", "1e-4"],
    num_nodes=2,
    num_gpus=4,
)
```

The workspace's `subscription_id` and `resource_group` are resolved
automatically from the
[config file](configuration.md#workspace-profiles)
when a matching profile exists.

::: submit_aml.aml.submit_to_aml
