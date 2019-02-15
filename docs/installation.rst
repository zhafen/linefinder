Installation
============

Installation of Linefinder is straightforward using pip. ::

   pip install linefinder --user

Note that the ``--user`` command is automatically included because it's assumed you're using a cluster.

.. NOTE::
   Linefinder uses both Python and HDF5.
   On clusters these modules usually have to be loaded before installation and use, e.g.
   ``module load python3`` and ``module load phdf5`` on Stampede.
   Linefinder works with both Python 2 and Python 3, but it's recommended to use Python 3.
   (Don't forget to only have one Python module loaded at once.)
