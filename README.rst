**********
MAEnvs4VRP
**********

Multi Agent Environments for Vehicle Routing Problems

MAEnvs4VRP is a library made up of multi-agent environments for simulating classic vehicle routing problems.

`Documentation <https://maenvs4vrp.readthedocs.io/en/latest/>`_ | `Install <#install>`_ | `Quickstart Notebook <https://maenvs4vrp.readthedocs.io/en/latest/content/start.html>`_ | `Train Your Model <#training>`_ | `Paper <https://arxiv.org/abs/2411.14411>`_

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Google Colab Badge
    :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/1.0.0-quickstart-cvrptw.ipynb

What's NEW!
=====================

- Added six new environments: **DVRPTW**, **DSVRPTW**, **MTVRP**, **MTDVRP**, **GMTVRP**, and **GMTDVRP**  
- Introduced three new hands-on Jupyter notebook tutorials  
- Integrated plotting tools for visualization and analysis 

Environments
============

.. list-table:: Available Vehicle Routing Environments:
   :widths: 25 5 5
   :header-rows: 1

   * - Evironment
     - Source
     - Description
   * - CVRPSTW (Capacitated Vehicle Routing Problem with Soft Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/cvrpstw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/cvrpstw/cvrpstw.html>`_
   * - CVRPTW (Capacitated Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/cvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/cvrptw/cvrptw.html>`_
   * - DSVRPTW (Dynamic Stochastic Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/dsvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/dsvrptw/dsvrptw.html>`_
   * - DVRPTW (Dynamic Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/dvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/dsvrptw/dvrptw.html>`_
   * - GMTDVRP (General Multi-Tasking Depot Vehicle Routing Problems)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/gmtdvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/gmtdvrp/gmtdvrp.html>`_
   * - GMTVRP (General Multi-Tasking Vehicle Routing Problems)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/gmtvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/gmtvrp/gmtvrp.html>`_
   * - MDVRPTW (Multi-Depot Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/mdvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/mdvrptw/mdvrptw.html>`_
   * - MTDVRP (Multi-Tasking Depot Vehicle Routing Problems)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/mtdvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/mtdvrp/mtdvrp.html>`_
   * - MTVRP (Multi-Tasking Vehicle Routing Problems)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/mtvrp>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/mtvrp/mtvrp.html>`_
   * - PCVRPTW (Prize Collecting Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/pcvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/pcvrptw/pcvrptw.html>`_
   * - PDPTW (Pickup and Delivery Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/pdptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/pdptw/pdptw.html>`_
   * - SDVRPTW (Split Delivery Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/sdvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/sdvrptw/sdvrptw.html>`_
   * - TOPTW (Team Orienteering Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/toptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/toptw/toptw.html>`_

Install
==========

For a clean setup, isolate library dependencies using a virtual environment. The library requires Python 3.11 or higher for installation, and it has been tested and confirmed stable with Python 3.13.5.
To create an isolated environment with conda:

.. code:: shell

    conda create --name maenvs4vrp python=3.13.5
    conda activate maenvs4vrp

To install MAENVS4VRP locally on your machine:

.. code:: shell

    git clone https://github.com/ricgama/maenvs4vrp.git && cd maenvs4vrp
    pip install -e .

Getting Started
===================

We've prepared five hands-on notebooks that walk you through the library's different functionalities and environments. Feel free to explore them and use them as a starting point for your own experiments.

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Notebook
     - Description
     - Colab
   * - `01: Quickstart <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/1.0.0_quickstart_cvrptw.html>`_
     - Learning MAEnvs4VRP basic usage.
     - |colab-quickstart|
   * - `02: MAEnvs4VRP library <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/2.0.0_maenvs4vrp_exploration_and_challenges.html>`_
     - Exploring MAEnvs4VRP library with challenges.
     - |colab-challenges|
   * - `03: Multi-Tasking Environments <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/3.0.0_multitask_environments.html>`_
     - Exploring MAEnvs4VRP multi-tasking environments.
     - |colab-multitask|
   * - `04: Stochastic Environments <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/4.0.0_maenvs4vrp_stochastic_environments.ipynb>`_
     - Adapting MAEnvs4VRP deterministic environments into stochastic ones.
     - |colab-stochastic|
   * - `05: PyVRP <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/5.0.0_PyVRP_cvrptw_solver.ipynb>`_
     - Exploring PyVRP on MAEnvs4VRP instances and environments.
     - |colab-PyVRP|

.. |colab-quickstart| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/1.0.0-quickstart-cvrptw.ipynb
.. |colab-challenges| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/2.0.0_maenvs4vrp_exploration_and_challenges.ipynb
.. |colab-multitask| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/3.0.0_multitask_environments.ipynb
.. |colab-stochastic| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/4.0.0_maenvs4vrp_stochastic_environments.ipynb
.. |colab-PyVRP| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/5.0.0_PyVRP_cvrptw_solver.ipynb

Training
=============

Two baseline models are available, which can be trained with:

.. code-block:: python

    python maenvs4vrp/learning/mardam/train_mardam.py --vrp_env toptw --num_agents 5 --num_nodes 51  --val_set servs_50_agents_5 --selection stime

.. code-block:: python

    python maenvs4vrp/learning/madyam/train_madyam.py --vrp_env toptw --num_agents 5 --num_nodes 51  --val_set servs_50_agents_5 --selection stime

Unit Testing
=================

You can run unit tests and assure compatibility across environments, with customizable parameters, using:

.. code-block:: python

    pytest --device cpu --batch 1 --num_agents 2 --num_nodes 11

Directory Tree Structure
===========================

.. code:: text

    ├───maenvs4vrp
    │   ├───core
    │   ├───environments
    │   │   ├───dvrptw
    │   │   ├───dsvrptw
    │   │   ├───cvrpstw
    │   │   ├───cvrptw
    │   │   ├───gmtdvrp
    │   │   ├───gmtvrp
    │   │   ├───mdvrptw
    │   │   ├───mtdvrp
    │   │   ├───mtvrp
    │   │   ├───pcvrptw
    │   │   ├───pdptw
    │   │   ├───sdvrptw
    │   │   ├───toptw
    │   ├───learning
    │   │   ├───madyam
    │   │   ├───mardam
    │   ├───notebooks
    │   ├───utils
    ├───tests
    │   ├───unit
    │   │   ├───environments

Citation
===============

To credit the library in your publications, use this citation:

.. code-block:: bibtex

    @article{gama2024maenvs4vrp,
      title={Multi-Agent Environments for Vehicle Routing Problems},
      author={Ricardo Gama and Daniel Fuertes and Carlos R. del-Blanco and Hugo L. Fernandes},
      year={2024},
      journal={arXiv preprint arXiv:2411.14411},
      note={\url{https://github.com/ricgama/maenvs4vrp}}
      url={https://arxiv.org/abs/2411.14411},
    }

Contributors
==================

.. image:: https://contrib.rocks/image?repo=ricgama/maenvs4vrp
   :alt: Contributors
   :target: https://github.com/ricgama/maenvs4vrp/graphs/contributors

Contributing
============
We welcome contributions to **MAEnvs4VRP**!  
If you’d like to use this library in your academic research/industry projects, or if you have suggestions, feature requests, or any feedback, we’d be happy to hear from you.  

You can **open an issue** here on GitHub or **drop us an email** — we’d love to collaborate and improve the project together.


Acknowledgements
=================
MAEnvs4VRP has been inspired by, and benefits from, the ideas and tooling of the broader open-source community. In particular, we would like to thank `PettingZoo <https://www.pettingzoo.ml/>`_, 
`Flatland <https://github.com/flatland-association/flatland-rl/>`_, `MARDAM <https://gitlab.inria.fr/gbono/mardam>`_, `RL4CO <https://rl4co.readthedocs.io/en/latest//>`_, `RoutFinder <https://github.com/ai4co/routefinder/tree/main//>`_, `PyVRP <https://pyvrp.org//>`_ .


TODO List
=================

1. confirm tensor dimensions and standardize (e.g. change depot_idx from (batch, 1) to (batch) , dones, etc.)
2. include cur_node_idx in old envs
3. include check_solution_validity method
4. include new envs in docs
5. review key names (try to standardize names with those used in RL4CO).
