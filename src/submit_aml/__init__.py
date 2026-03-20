"""``submit-aml`` — submit jobs to Azure Machine Learning.

This package provides both a CLI tool and a Python API for submitting jobs
to Azure Machine Learning. See the
`documentation <https://microsoft.github.io/submit-aml>`_ for usage details.

Example::

    from submit_aml import submit_to_aml

    submit_to_aml(
        workspace_name="my-ml-workspace",
        compute_target="gpu-v100x4",
        script_path="train.py",
    )
"""

__version__ = "0.1.0"

from .aml import submit_to_aml

__all__ = [
    "submit_to_aml",
]
