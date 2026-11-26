Cancellation and logging
========================

Canceling long jobs
--------------------

Extracting imagery for thousands of assets, or running a local model over
them, can take hours. Every long-running component accepts a
``cancel_event``, a standard :class:`threading.Event`, and checks it between
assets, tiles or batches. When you set the event, from another thread, a GUI
button or a signal handler, the job stops at the next check and raises
:class:`~rapidtools.core.OperationCanceled`. Everything finished up to that
point is kept.

.. code-block:: python

   import threading
   from rapidtools import AerialImageryExtractor, AssetAnalyzer, OperationCanceled, Pipeline

   stop = threading.Event()
   pipeline = Pipeline(
       [
           AerialImageryExtractor('ortho.tif', save_directory='crops', cancel_event=stop),
           AssetAnalyzer(model, prompt='...', cancel_event=stop),
       ],
       cancel_event=stop,
   )

   try:
       buildings = pipeline.run(buildings)
   except OperationCanceled:
       print('Stopped by the user; partial results are on the assets.')

Give the pipeline and its steps the same event. The pipeline then stops
between steps, and each step stops partway through its own work. The
exception is a subclass of :class:`RuntimeError`, so a generic handler still
catches it, while a specific one can tell a stop the user asked for from a
real failure. To make your own components stoppable in the same way, call
:func:`~rapidtools.core.raise_if_canceled` in their loops.

Logging
-------

Importing rAPIdtools leaves your logging configuration alone. To see progress
messages in a script or notebook, call
:func:`~rapidtools.config.configure_logging` once:

.. code-block:: python

   import rapidtools as rt
   rt.configure_logging()          # INFO to stdout
   rt.configure_logging('DEBUG')

After that call the ``rapidtools`` logger no longer passes messages up to the
root logger, so each message prints once even if your application has set up
root logging too. If you would rather send the library's messages through
your own handlers, skip the call.

HTTP sessions
-------------

All network components use :func:`~rapidtools.config.get_configured_session`,
a :class:`requests.Session` with retries, exponential back-off and a default
timeout. If you make your own calls to the same services, use it too, so
they fail and retry the same way:

.. code-block:: python

   from rapidtools.config import get_configured_session
   session = get_configured_session(retries=3, backoff_factor=0.5)
