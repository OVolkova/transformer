from setuptools import setup

setup(
    name="tmodels",
    version="0.0.1",
    author="Olga Volkova",
    packages=["tmodels"],
    description="A PyTorch re-implementation of Transformer model and experiments with its variants",
    license="MIT",
    install_requires=[
        "torch==2.0.0",
    ],
)
