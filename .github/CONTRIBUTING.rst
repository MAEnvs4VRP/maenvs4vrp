Contributing to MAEnvs4VRP
==============================

Thank you for your interest in contributing to our project! We welcome contributions from anyone and appreciate even the smallest fixes. Please read this document to understand how to contribute to **MAEnvs4VRP**.

Getting Started
---------------

To get started with contributing, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Install the necessary dependencies.
3. Create a new branch for your changes: git checkout -b my-branch-name
4. Make your desired changes or additions.
5. Run the tests to ensure everything is working as expected: pytest tests
6. Commit your changes: git commit -m "Descriptive commit message"
7. Push to your branch: git push origin my-branch-name
8. Submit a pull request to the ``main`` branch of the original repository.

Code Style
----------

In general, we follow the `Google Style Guide <https://google.github.io/styleguide/pyguide.html>`_ . We use  `conventional commit messages <https://www.conventionalcommits.org/en/v1.0.0/>`_ for commit messages.  Consistent code style improves readability and makes collaboration easier.

To enforce this, we use **pre-commit** to automatically run:

- **Black** (code formatter)
- **Ruff** (linter and auto-fixer)

on every commit.

Pre-commit Setup
----------------

The project is configured to use **pre-commit**, which ensures that all code meets quality standards before being committed.

If you don’t have it installed, install it with::

   pip install pre-commit

Then install the Git hooks::

    pre-commit install

You should see::

    pre-commit installed at .git/hooks/pre-commit


Running Pre-commit Manually
---------------------------

You can run all hooks manually with::

    pre-commit run --all-files

Acknowledgements
---------------------------

We adapted these contributing guidelines from `this repo <https://github.com/ai4co/rl4co/blob/main/.github/CONTRIBUTING.md>`_

Licensing
---------------------------

By contributing to MAEnvs4VRP, you agree that your contributions will be licensed under the `LICENSE <https://github.com/ricgama/maenvs4vrp/blob/master/LICENSE>`_  file of the project.
