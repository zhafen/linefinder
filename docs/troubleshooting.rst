Troubleshooting
===============

Common problems, and how to handle them.

HDF5 Import Errors
------------------

A common import error is: ::

    ImportError: libhdf5.so.10: cannot open shared object file: No such file or directory

To solve this ensure you have the right HDF5 library loaded.
For example in the case of Python3 on Stampede 2 you probably need ``module load phdf5`` instead of ``module load hdf5``.

Firefly Fails to Install
------------------------

If Firefly fails to install (as part of the visualization step), there's a good chance it's because you haven't set up an automatic connection to GitHub using ssh.
`Follow GitHub's tutorial to do so. <https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/>`_
Alternatively you can turn the Firefly install off (``install_firefly=False`` as one of the ``visualization_kwargs``) and clone it yourself.
Afterwards change ``firefly_dir`` in ``export_to_firefly_kwargs`` to the location of the Firefly repository you clone.
