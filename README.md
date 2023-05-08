# linefinder
A tool for finding the worldlines of Lagrangian parcels of mass, in the context of hydrodynamic simulations of galaxy formation.

Fundamental parts of this code are based on previous work by Daniel Anglés-Alcázar.

## Installation

As of this moment, there are three repositories linefinder depends on that are not up-to-date on pip.
These must be installed from source as follows,
```
git clone git@github.com:zhafen/trove.git
cd trove
pip install -e .

git clone git@github.com:zhafen/galaxy-dive.git
cd galaxy-dive
pip install -e .

git clone git@github.com:ageller/Firefly.git
cd Firefly
pip install -e .
```

Then install linefinder from source.
```
git clone git@github.com:zhafen/linefinder.git
cd linefinder
pip install -e .
```

Note that this will install the latest commit, which may not be as reliable as the latest release.

## Using Data Products

Please [see the wiki for information on how to work with data products produced by the code](https://github.com/zhafen/linefinder/wiki).
