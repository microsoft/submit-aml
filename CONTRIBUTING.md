# Contributing

## Development setup

```shell
git clone https://github.com/microsoft/submit-aml
cd submit-aml
uv sync --all-groups
```

## Running checks

Using [mise](https://mise.jdx.dev/):

```shell
mise run check   # run all checks
mise run test    # tests only
mise run lint    # linting only
mise run format  # formatting check
mise run types   # type checking
```

Or directly:

```shell
uv run tox          # run all checks
uv run tox -e pytest # tests only
```

## Pre-commit hooks

```shell
uv run prek install
```

## Documentation

```shell
mise run docs-serve  # local preview
mise run docs-build  # build static site
```

## Pull requests

Keep PRs small and focused. Include a description of **what** changed and **why**.

## Code style

- **Formatting and linting:** [ruff](https://docs.astral.sh/ruff/)
- **Type checking:** [ty](https://docs.astral.sh/ty/)
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
