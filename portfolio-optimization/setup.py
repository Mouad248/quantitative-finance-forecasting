from setuptools import setup, find_packages

setup(
    name="quantopt",
    version="1.0.0",
    description=(
        "Institutional MPT portfolio optimizer: Black-Litterman returns, "
        "factor-model risk, CVaR optimization, risk parity, "
        "and walk-forward backtesting."
    ),
    author="Anonymous",
    author_email="",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.1",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "matplotlib>=3.8",
        "seaborn>=0.13",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "black>=23.0",
            "mypy>=1.5",
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
