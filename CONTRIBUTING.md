# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Development setup

```shell
git clone https://github.com/microsoft/submit-aml
cd submit-aml
uv sync --all-groups
uv run prek install --install-hooks
```

## Code style

- **Formatting and linting:** [ruff](https://docs.astral.sh/ruff/)
- **Type checking:** [ty](https://docs.astral.sh/ty/)
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## Documentation

If you want to check the rendered documentation before pushing:

```shell
mise run docs-serve  # local preview
mise run docs-build  # build static site
```

## Running checks

Check that everything passes before pushing a commit:

```shell
uv run tox           # run all checks
```

## Pull requests

- Before creating a PR, open an issue describing what you tried, what you expected and what you got.
- Keep PRs small and focused.
- Include a description of **what** changed and **why**.
- The PR title should look something like this: "Fix environment hash generation", and not "fixed hash" or "Replaces str with int".
- The branch name should start with the issue number. For example, `27-fix-hash`.
