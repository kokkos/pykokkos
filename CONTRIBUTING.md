# Contributing to PyKokkos

Thank you for your interest in contributing to PyKokkos! This guide covers everything you need to know to get a pull request reviewed and merged smoothly.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Opening Issues](#opening-issues)
- [Pull Requests](#pull-requests)
  - [Title](#title)
  - [Description](#description)
  - [Code Formatting](#code-formatting)
  - [Testing](#testing)
  - [Examples](#examples)

---

## Getting Started

Before contributing, please:

1. Check the [open issues](https://github.com/kokkos/pykokkos/issues) to see if your bug or feature is already being tracked.
2. For significant changes, open an issue first to discuss the approach before writing code. This avoids wasted effort if the direction needs to change.
3. Fork the repository and create a branch with a short, descriptive name (e.g., `fix/type-error-handling` or `feat/sparse-matrix-support`).

---

## Opening Issues

Use the issue templates provided in this repository. Choose **Bug Report** for unexpected behavior or **Feature Request** for new functionality.

---

## Pull Requests

### Title

Titles must follow the `module: description` format, where `module` is the part of the codebase being changed. Keep the description short and in the imperative mood.

**Examples:**

| ✅ Good | ❌ Avoid |
|---|---|
| `core: raise TypeError for unsupported types` | `Fixed a bug` |
| `views: add support for rank-3 layouts` | `Updates to the views module` |
| `ci: pin black version to 25.12.0` | `black` |

[!TIP]
If you are not sure what module name to choose, just add the name of the Lowest Common Ancestor (LCA) directory of the changed file (for example, you changed `pykokkos/core/compiler` and `pykokkos/core/visitor/visitor_util.py`. The LCA here is `core`---use it as a module name.
Avoid changes that involve multiple directories under `pykokkos/` (e.g., `pykokkos/interface` and `pykokkos/core`) in one PR.

### Description

Every pull request must include a description that gives reviewers and future contributors enough context to understand the goal and major changes **before** reading the code. A good description covers:

- **What** changed and **why**
- Any relevant design decisions or trade-offs
- A link to the related issue, if one exists

To automatically close a linked issue when the PR is merged, include `closes #XXX` in the description, replacing `XXX` with the issue number.

### Code Formatting

PyKokkos enforces consistent formatting through [CI checks](https://github.com/kokkos/pykokkos/blob/main/.github/workflows/formatting.yml). All code must be formatted with [`black`](https://black.readthedocs.io/) at the pinned version before opening a pull request:

```bash
pip install black==25.12.0
black .
```

PRs that fail the formatting check will not be reviewed until the check passes.

### Testing

Most changes should be accompanied by unit tests. Please follow these guidelines:

- **Extend existing test files** in the `test` directory wherever possible. Adding a new test file should be a last resort.
- **If you have access to a GPU, we recommend running all tests locally, as our CI does not have GPU runners. This helps us identify existing issues and support more GPU models. The command to run the suite locally is `python runtests.py`

### Examples

Pull requests that introduce significant new functionality should include a working example in the `examples/pykokkos` directory. Examples should be self-contained and demonstrate the intended use case.

---

## Questions?

If you are unsure about any of the above or need guidance on where to start, feel free to ask a question in the [Kokkos slack](https://kokkosteam.slack.com/archives/C035BTCFPSS) **#pykokkos** channel.
