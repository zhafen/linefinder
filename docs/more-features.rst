More Features
=============

Creating Your Own Halo Files
----------------------------

Linefinder can currently support halo files from either `AHF <http://popia.ft.uam.es/AHF/>`_ or `Rockstar <https://bitbucket.org/gfcstanford/rockstar>`_.
If you want to run linefinder on a FIRE simulation that doesn't have halo files I recommend using `AHF wrapper <https://bitbucket.org/zhafen/ahf_wrapper/src>`_, which makes it easier to run AHF.

The Linefinder Config File
--------------------------

The config file contains defaults that Linefinder falls back to when they aren't specified in the job script.
This is useful when running Linefinder for a variety of simulations, a variety of parameters, or both.
The config file is useful when you start to use linefinder more extensively.

The config file is located at e.g. ``your/linefinder/dir/config.py``.
If you specified the install directory (`as recommended <https://zhafen.github.io/linefinder/docs/html/installation.html>`_), then the file will be in that directory.
An overview of each section in the config file is described below, with descriptions of each variable in-line in the config file itself.

1. **Global Default Parameters:**
A number of choices are made when linking particles to galaxies or classifying particles.
The most important choices are set here as default parameters.

2. **Miscellanious Values:**
Constants that aren't important for the physics, but are still necessary for linefinder to run.

3. **Simulation Information:**
Information about the simulation sample being analyzed.

4. **System Information:**
Information about the computing environment.

Advanced Job Submission
-----------------------

In many cases you may want to run Linefinder not just on a single simulation, but on a number of simulations.
Further, you might want to systematically vary the parameters used with Linefinder.
Linefinder offers options to do this automatically with minimal monitoring by the user.
However these options require additional explanation, and I've postponed creating a simplified example until there's actual demand.
If you're actually interested, `contact me <zachary.h.hafen@gmail.com>`_, and I'd be happy to put together an example.
