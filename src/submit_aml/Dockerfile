# This file is a template that needs to be provided the base Docker image and the uv sync command
FROM {base_docker_image}
{docker_run}
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG WORKSPACE="/workspace"
WORKDIR $WORKSPACE

# Copy files required to create the locked environment
# We change the owner to 1000:1000 to avoid permission issues when running on Singularity
COPY --chown=1000:1000 pyproject.toml uv.lock .python-version $WORKSPACE/

# - Enforce some location for the virtual environment
# - Do not check whether the uv.lock file is up to date when running `uv sync` and `uv run`.
#     Make sure it is up to date before submitting!
# - Install Python binaries in a known location so we can then give permissions to the user
#      running the container. This is important for Singularity compatibility.
# - Use only managed Python packages, i.e. no system Python packages.
ENV UV_PROJECT_ENVIRONMENT="$WORKSPACE/.venv" \
    UV_FROZEN=true \
    UV_PYTHON_INSTALL_DIR="/python" \
    UV_PYTHON_PREFERENCE="only-managed"

# Create virtual environment and install dependencies
RUN {uv_sync_command} \
    && chown -R 1000:1000 $WORKSPACE \
    && chmod -R go+rwX $WORKSPACE

# Uncomment the following line to disable network access for uv, so that the environment
# does not change at runtime.
# Currently disabled so that the build requirements are installed at runtime.
# ENV UV_OFFLINE=1

# Make sure the printed output is not buffered so that it appears in real-time and is
# not interrupted if a different process is killed.
# More info at https://yakshaving.eu/devopssec/virtualization/container/multiple-run-commanas-opposed-to-a-single-chained-run-command/
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
