from setuptools import setup, find_packages

setup(
    name="ai_forex_system",
    version="3.0",
    description="AI-powered Forex trading system with ML and risk management",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.3.0",
        "numpy>=2.0.0",
        "scikit-learn>=1.6.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "ta-lib>=0.6.0",
        "yfinance>=1.2.0",
        "ccxt>=4.5.0",
    ],
)
