# PyKokkos Contributor Guide

## Pull Requests
If you are looking to submit a pull request, please follow the guidelines below

### name
Pull requests names should be formatted as `module: feature`.
E.g., if the PR catches an unsupported type, the title should be
`core: raise TypeError for type 'garbage'`.

### description
Pull requests should have a relevant description
expanding on the name; reviewers and future contributors
should be able to understand the goal and major changes in
the pull request before looking at the code itself.
If the pull request closes issue #XXX, the description
should contain the phrase `closes #XXX`. This links the
issue to the pull request and automatically closes the issue
when the pull request is merged.

### content

#### code formatting
The (continuous integration status)[https://github.com/kokkos/pykokkos/blob/main/.github/workflows/formatting.yml]
enforces a strong formatting rule. Please make sure your code is formatted with `black==25.12.0` before opening
a pull request.

#### testing
Most features should be accompanied with unit tests.
Please look to extend existing test files in the `test` directory,
adding a new file should be a last resort.

Please make sure testing passses locally with the
`python3 runtests.py` command.

#### examples
Large feature changes should be accompanied by
relevant examples in the `examples/pykokkos` directory.
