from setuptools import setup, find_packages

setup(
    name="math-reasoning-arabic-grpo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "PyYAML>=6.0.0",
    ],
) 