Installing PYFAB
================

Prerequisites
~~~~~~~~~~~~~

`Fabber <https://fabber_core.readthedocs.io>`_ must be installed. The easiest way
to ensure this is to install the latest version of 
`FSL <https://fsl.fmrib.ox.ac.uk/fsl/>`_.

Installation using pip
~~~~~~~~~~~~~~~~~~~~~~

Installation using pip should be as simple as::

    pip install pyfab

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

The ``fabber.mvn`` module requires the ``fslpy`` library. However we do not choose to
depend on this library. If you want to use the MVN functionality you will need to 
install it::

    pip install fslpy

 
