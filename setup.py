import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="linefinder",
    version="0.8.1",
    author="Zach Hafen",
    author_email="zachary.h.hafen@gmail.com",
    description="A tool for finding and classifying the worldlines of Lagrangian parcels of mass, in the context of hydrodynamic simulations of galaxy formation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhafen/linefinder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy'>=1.14.5,
        'pandas'>=0.20.3,
        'mock'>=2.0.0,
        'pytest'>=3.4.0,
        'Jug'>=1.6.7,
        'setuptools'>=28.8.0,
        'matplotlib'>=2.0.2,
        'h5py'>=2.7.0,
        'galaxy_dive'>=0.8.1.2,
        'scipy'>=1.1.0,
    ],
)
