Cancellation and logging
========================

Cancelling long jobs
--------------------

Extracting imagery for thousands of assets or running a local model over
them takes minutes to hours. Every long-running component accepts a
``cancel_event``, a standard :class:`threading.Event`, and checks it between
assets, tiles or batches. Setting the event from another thread, a GUI
button or a signal handler stops the job at the next boundary and raises
:class:`~rapidtools.core.OperationCancelled`, while keeping everything
produced so far.

.. code-block:: python

   import threading
   from rapidtools import AerialImageryExtractor, AssetAnalyzer, OperationCancelled, Pipeline

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
   except OperationCancelled:
       print('Stopped by the user; partial results are on the assets.')

Sharing one event between the pipeline and its steps means the pipeline
stops between steps and each step stops within itself. The exception derives
from :class:`RuntimeError`, so generic handlers still catch it while specific
ones can tell a user-requested stop from a failure. Your own components can
join the scheme with :func:`~rapidtools.core.raise_if_cancelled`.

Logging
-------

Importing rAPIdtools never touches logging configuration. To see progress
messages in a script or notebook, call
:func:`~rapidtools.config.configure_logging` once:

.. code-block:: python

   import rapidtools as rt
   rt.configure_logging()          # INFO to stdout
   rt.configure_logging('DEBUG')

The ``rapidtools`` logger then stops propagating to the root logger, so
messages print once even when your application configured root logging.
Skip the call if you would rather route the library's messages through your
own handlers.

HTTP sessions
-------------

Network components share :func:`~rapidtools.config.get_configured_session`,
a :class:`requests.Session` with retries, exponential back-off and a
default timeout. Use it for your own calls to the same providers so they
behave consistently:

.. code-block:: python

   from rapidtools.config import get_configured_session
   session = get_configured_session(retries=3, backoff_factor=0.5)
