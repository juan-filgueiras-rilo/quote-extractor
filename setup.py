from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="factual-qa",
    version="1.0.0",
    author="FactualQA-System",
    author_email="",
    description="A configurable system for processing HotpotQA dataset with LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jfilgueiras/factual-qa",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "huggingface": [
            "torch>=1.11.0",
            "transformers>=4.20.0",
            "accelerate>=0.12.0",
            "bitsandbytes>=0.35.0",
        ],
        "ollama": [
            "ollama>=0.1.0",
        ],
        "openai": [
            "openai>=1.0.0",
        ],
        "all": [
            "torch>=1.11.0",
            "transformers>=4.20.0", 
            "accelerate>=0.12.0",
            "bitsandbytes>=0.35.0",
            "ollama>=0.1.0",
            "openai>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "inference=src.cli.inference:main",
            "train=src.cli.train:main",
        ],
    },
    include_package_data=True,
)