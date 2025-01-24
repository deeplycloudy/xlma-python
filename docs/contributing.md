# Contributor's Guide
---

Discussion of contributions takes place in [GitHub's issues](https://github.com/deeplycloudy/xlma-python/issues) and [pull requests](https://github.com/deeplycloudy/xlma-python/pulls) sections of the xlma-python repository. 

To create a "developer install" of xlma-python:

```sh
git clone https://github.com/deeplycloudy/xlma-python.git
cd xlma-python
pip install -e .
```

This will allow you to edit files in the xlma-python directory on your local machine, and the `pip install -e .` command will allow those changes to be applied to your python environment immediately.


xlma-python is built using [setuptools](https://setuptools.pypa.io/en/latest/). The library is entirely pure python, so no additional compliation steps are required for installation.

Automated testing is included to prevent future code contributions from breaking compatibility with previous versions of the library. Tests are stored in the `tests/` directory. pytest is used for the basic test architecture, with the `pytest-mpl` plugin providing image differences. See the `tests/truth/images/` directory for baseline images. `pytest-cov` is used for coverage checking, results are uploaded to [CodeCov](https://codecov.io).

Documentation is built by [mkdocstring-python](https://mkdocstrings.github.io/python/). See the `mkdocs.yml` file for configuration and the `docs/reference/` directory for page layout.