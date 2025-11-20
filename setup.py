"""
Setup script for ConsistencyAI.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="consistencyAI",
    version="2.0.0",
    author="Peter Banyas, Shristi Sharma, Alistair Simmons, Atharva Vispute",
    description="A benchmark for evaluating LLM consistency across demographics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/banyasp/consistencyAI",  
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "requests>=2.28.0",
        "certifi>=2022.12.7",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.1.0",
        "sentence-transformers>=2.2.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "plotly>=5.0.0",
        "nest_asyncio>=1.6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
    keywords="llm benchmark ai consistency bias fairness nlp",
    project_urls={
        "Documentation": "https://github.com/banyasp/consistencyAI#readme",
        "Source": "https://github.com/banyasp/consistencyAI",
        "Bug Reports": "https://github.com/banyasp/consistencyAI/issues",
    },
)

