Quickstart
==========

This page is to help you to start tracking particles once you've `installed linefinder <https://zhafen.github.io/linefinder/docs/html/installation.html>`_.
Please `contact me <mailto:zachary.h.hafen@gmail.com>`_ if you find anything confusing or have any questions.
If you find a bug please `open an issue <https://github.com/zhafen/linefinder/issues/new/choose>`_.

Example Files
-------------

If you have access to the FIRE simulation data on Stampede 2 you can follow along step-by-step to make sure everything is working as expected.
Enter the following commands to download an example job script, submission script, and list of IDs (for the fiducial ``m12i_res7100`` simulation): ::

    curl -LO https://raw.githubusercontent.com/zhafen/linefinder/master/linefinder/job_scripts/linefinder_example.py
    curl -LO https://raw.githubusercontent.com/zhafen/linefinder/master/linefinder/job_scripts/submit_linefinder.sh
    curl -Lo ids_full_m12i_example.hdf5 https://github.com/zhafen/linefinder/blob/master/linefinder/job_scripts/ids_full_m12i_example.hdf5?raw=true

Running Linefinder
------------------

On an Interactive Node
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to run Linefinder is on an interactive node.
On Stampede 2, for example, you can start an interactive node with e.g. ``idev``.
I recommend using ``idev -m 120 -p skx-dev`` to request two hours on a Skylake development node (which has more memory than a KNL node).
To start a single core working on your Linefinder job, simply enter on the commandline ::

    jug execute linefinder_example.py &

.. NOTE::
   For this to work the jug executable must be on your ``$PATH``.
   If you installed Linefinder on a cluster this likely means adding ``~/.local/bin`` to your ``$PATH``,
   e.g. by adding ``export PATH=$PATH:$HOME/.local/bin`` to your ``~/.bashrc``.
   Alternatively you can refer to the executable directly.

If you're using Linefinder to do particle tracking on many snapshots of a high resolution simulation chances are this will take longer than preferred.
Fortunately Linefinder is parallelized using `Jug <https://jug.readthedocs.io/en/latest/>`_ (see also :ref:`a-note-about-jug`).
To start more cores running Linefinder just reenter the line above, or better yet do a loop in bash, e.g. ::

    for i in $(seq 5) ; do jug execute linefinder_example.py & done

Each core will then be responsible for one snapshot at a time.
Deciding how many cores to use (in the above example I use 5) is almost always a function of the memory availble to a node because each core needs a snapshot worth of memory.
If you run out of memory and your job crashes you will need to restart linefinder (see :ref:`restarting-a-job`).

As a Batch Job
~~~~~~~~~~~~~~

Linefinder can also easily be run as a batch job.
This is important for when you want to use more than a single node to speed up a particle tracking calculation.
To submit as a batch job simply modify `submit_linefinder.sh <https://github.com/zhafen/linefinder/blob/master/linefinder/job_scripts/submit_linefinder.sh>`_ according to your cluster and preferences, then submit with e.g. ::

    sbatch submit_linefinder.sh linefinder_example.py 6

In this example the above command will run ``linefinder_example.py`` on a single node using 6 cores.
Reentering the above command will do the same on another node.

.. TIP::
   Tired of having to ssh into the cluster multiple times when you want to have multiple windows up (e.g. one interactive node running linefinder, one node for submitting jobs, etc)?
   Try using `tmux <https://github.com/tmux/tmux>`_!
   Using tmux will also allow your interactive jobs to keep going even when your connection breaks!

.. NOTE::
   Resubmitting the same job *does not* cause conflicts, but just speeds up the job by throwing more nodes at it.

The most computationally intensive parts of particle tracking are embarrassingly parallel, so please feel free to use a number of nodes to greatly speed up the work.
Particle tracking isn't *that* expensive, so about ten nodes is probably sufficient, and will complete most runs in about 10 minutes.

.. _restarting-a-job:

Restarting a Job
~~~~~~~~~~~~~~~~

Sometimes jobs crash.
Maybe too many cores were used per node and you ran out of memory, maybe you ran out of time, etc.
When this happens you'll want to restart the Linefinder run.
This is simple and involves two steps:

1. Deleting all the jugdata in your output directory (e.g. ``rm -r path/to/output/*jugdata``).
2. Turning off the parts of your job that have already completed.

Jug communicates through the filesystem (see :ref:`a-note-about-jug`), so (1) is necessary to get a fresh start.
(2) is necessary to prevent redoing work (and also crashing when Linefinder tries to save a file where one already exists), and is as simple as adding the argument e.g. ``run_tracking = False,`` in ``linefinder.run_linefinder_jug()`` in your job script.

.. _a-note-about-jug:

A Note About Jug
~~~~~~~~~~~~~~~~

Linefinder is parallelized using `Jug <https://jug.readthedocs.io/en/latest/>`_ (see :ref:`a-note-about-jug`).
The most noteable thing about Jug is that it communicates between processes using the file system.
The main benefit to this is easy communication between multiple nodes, even allowing the user to add more nodes as they become available/are needed.
All the communications for Jug are kept in `jugdata` folders, tagged using the same tag used for a job, e.g. `m12i_example.jugdata`.
To learn more, see `the official docs <https://jug.readthedocs.io/en/latest/>`_

