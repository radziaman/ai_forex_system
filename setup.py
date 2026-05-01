from setuptools import setup, find_packages

setup(
    name="rts_ai_fx_trading",
    version="3.0",
    description="RTS - AI FX Trading System with ML and risk management",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.3.0",
        "numpy>=1.26.4,<2.0.0",
        "scikit-learn>=1.6.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "ta-lib>=0.6.0",
        "yfinance>=1.2.0",
        "ccxt>=4.5.0",
        "tensorflow>=2.15.0,<2.16.0",
    ],
)
