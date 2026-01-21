=====================
Unit Testing
=====================

The library includes several tests to verify that it is running correctly.  
Before simulating problems, it is recommended to run the available unit tests, located in the directory:

``tests/unit/environments``

We provide the following types of tests:

Check Multi-Task Solution Test
==============================

This script runs tests for **Multi-Tasking environments**, validating solution correctness under multiple conditions:

* **Random instances:** Tests across batches with different variants generated through random attribute sampling.
* **All variants:** Creates and validates instances for every available environment variant.
* **Different agent selectors:** Tests all agent selectors — ``AgentSelector``, ``RandomSelector``, and ``SmallestTimeAgentSelector``.
* **Different configurations:** Evaluates combinations of various numbers of agents and nodes.

Configuration values can be customized using the following command-line parameters:

* ``--device`` can be ``CPU`` or ``GPU``.
* ``--batch`` can be any integer.
* ``--num_agents`` can be a single or multiple integer values. To define n values, use the parameter n times.
* ``--num_nodes`` can be a signle or multiple integer values. To define n values, use the parameter n times.

To execute the test, run:

.. code-block:: bash

    pytest --device cpu --batch 1 --num_agents 2 --num_nodes 11 --num_nodes 13 check_solution_mt_test.py


Check Solution Test
===================

This script performs solution validation for **non–Multi-Tasking environments**:

* **Random instances:** Tests random variants generated across batches.
* **Different agent selectors:** Includes ``AgentSelector``, ``RandomSelector``, and ``SmallestTimeAgentSelector``.
* **Different configurations:** Runs across multiple agent and node configurations.

Configuration values can be customized using the following command-line parameters:

* ``--device`` can be ``CPU`` or ``GPU``.
* ``--batch`` can be any integer.
* ``--num_agents`` can be a single or multiple integer values. To define n values, use the parameter n times.
* ``--num_nodes`` can be a signle or multiple integer values. To define n values, use the parameter n times.

To execute the test, run:

.. code-block:: bash

    pytest --device cpu --batch 1 --num_agents 2 --num_nodes 11 --num_nodes 13 check_solution_test.py


Reset Test
==========

The **Reset Test** ensures that environments reset and behave correctly under different configurations.  
It includes three types of checks:

* **Reset tests:** Verify that both benchmarking and standard instances reset without errors.
* **Observation tests:** Confirm that environment observations are correctly produced.
* **Agent iteration tests:** Validate that agent selection and iteration work as expected for all selectors and generator types.

To execute the test, run:

.. code-block:: bash

    pytest reset_test.py


Reset Benchmarking Test
=======================

Validates the correct behavior of **benchmarking resets**.

To execute the test, run:

.. code-block:: bash

    pytest reset_bench_test.py


Seed Test
=========

The **Seed Test** verifies reproducibility and consistency across environments:

* **Seed consistency:** Checks whether identical seeds yield identical outputs, and different seeds yield different ones.
* **Instance generator:** Validates equivalence between benchmarking and standard instance generators.

To execute the test, run:

.. code-block:: bash

    pytest seed_test.py


Seed Benchmarking Test
======================

Validates seed reproducibility for **benchmarking environments**.

To execute the test, run:

.. code-block:: bash

    pytest seed_bench_test.py
