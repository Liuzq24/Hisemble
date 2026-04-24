from setuptools import setup, find_packages

setup(
    name="hisemble",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "scanpy",
        "igraph",
        "leidenalg",
        "pandas"
    ],
)