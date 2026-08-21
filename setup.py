from setuptools import setup, find_packages

setup(
    name="multilingual_asr_pipeline",
    version="1.0.0",
    description="Pipeline de nettoyage de données ASR multilingue prêt pour la production (WAXAL dataset)",
    author="MLOps & Speech Engineering Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "datasets>=2.14.0,<3.0.0",
        "soundfile>=0.12.1",
        "pydub>=0.25.1",
        "webrtcvad>=2.0.10",
        "setuptools<70",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0", "jupyter"],
        "music": ["tensorflow>=2.12.0", "tensorflow-hub>=0.13.0"],
    },
    entry_points={
        "console_scripts": [
            "waxal-asr-clean=core.main:main",
        ],
    },
)
