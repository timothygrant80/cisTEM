from setuptools import setup, find_packages

setup(
    name="tm_post",
    version="0.1",
    author="Kexin Zhang",
    description="Postprocessing utilities for 2D template matching",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "mrcfile",
        "joblib",
        "tqdm",
        "scikit-image",
    ],
    entry_points={
        "console_scripts": [
            "filter-particles = cli.filter_particles:main",
            "extract-particles = cli.extract_particles:main",
        ],
    },
)