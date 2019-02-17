import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="linefinder",
    version="0.9.2",
    author="Zach Hafen",
    author_email="zachary.h.hafen@gmail.com",
    description="A tool for finding and classifying the worldlines of Lagrangian parcels of mass, in the context of hydrodynamic simulations of galaxy formation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhafen/linefinder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas>=0.20.3',
        'mock>=2.0.0',
        'numpy>=1.15.4',
        'pytest>=3.4.0',
        'Jug>=1.6.7',
        'setuptools>=28.8.0',
        'palettable>=3.1.1',
        'matplotlib>=2.0.2',
        'h5py>=2.7.0',
        'GitPython>=2.1.11',
        'scipy>=1.2.0',
        'verdict>=1.0',
        'galaxy-dive>=0.9.1'
    ],
)
