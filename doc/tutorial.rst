PYFAB tutorial
==============

This tutorial demonstrates basic use of the PYFAB API. For more detailed information see the
API reference.

Creating the API interface object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A command line interface object is created by::

    from fabber import FabberCl
    fab = FabberCl()

A shared library interface object is created by::

    from fabber import FabberShlib
    fab = FabberShlib()

Since both objects implement the same interface, for the remainder of this tutorial we will
not specify which is being used.

Querying models and methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~


