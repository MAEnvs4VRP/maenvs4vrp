from setuptools import find_packages, setup

requirements = [
    "numpy==2.3.1",
    "tqdm==4.66.1",
    "pandas==2.3.0",
    "ipykernel",
    "ipywidgets",
    "matplotlib>=3.10.5",
    "jupyter==1.0.0",
    "torch==2.7.0",
    "tensorboard==2.13.0",
    "scipy",
    "wandb",
    "huggingface_hub",
    "ml_collections",
    "pytest==7.4.3",
    "tensordict==0.9.1",
    "sphinx-math-dollar==1.2.1",
    "nbsphinx==0.9.5",
    "furo==2024.8.6",
    "sphinx-copybutton==0.5.2",
]
setup(
    name="maenvs4vrp",
    packages=find_packages(),
    python_requires=">=3.11, <4",
    install_requires=requirements,
    version="0.2.0",
    description="Multi Agent Environments for Vehicle Routing Problems",
    author="mustelideos",
    license="",
)
