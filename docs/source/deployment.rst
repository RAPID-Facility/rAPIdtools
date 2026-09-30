Deploying a shared server
=========================

This page sets up one ``rapidtools-gui`` server that colleagues use from
their browsers over HTTPS. It runs the GUI in Docker behind the Caddy web
server, which obtains the certificate by itself, and keeps imagery, results
and caches on the host disk. The files live under ``deploy/`` in the
repository. The same steps work on a machine in your lab, a university VM or
a cloud instance; the last section covers the cloud specifics.

What you need
-------------

- A Linux machine with Docker and the Docker Compose plugin. For local
  models (Gemma 4, Qwen, Llama) and fast SAM 3 detection it needs an NVIDIA
  GPU with at least 16 GB of memory, the NVIDIA driver and the `NVIDIA
  Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.
  Without a GPU the cloud model backends (Gemini, Claude, OpenAI, Qwen and
  Muse Spark through their APIs) still work and SAM 3 runs slowly on the
  CPU.
- A DNS name pointing at the machine (for example ``gui.example.edu``) and
  ports 80 and 443 reachable by your colleagues. Port 8765 stays closed;
  Caddy is the only thing exposed.
- Optionally an SMTP relay for email notifications.

One-time setup
--------------

1. Get the code onto the machine and prepare the folders that will hold
   imagery and results:

   .. code-block:: bash

      git clone https://github.com/RAPID-Facility/rAPIdtools.git
      cd rAPIdtools
      sudo mkdir -p /srv/rapidtools/data /srv/rapidtools/output
      sudo chown -R $USER /srv/rapidtools

2. Copy the environment template and fill it in:

   .. code-block:: bash

      cp deploy/.env.example deploy/.env
      python -c "import secrets; print(secrets.token_urlsafe(18))"   # a token
      nano deploy/.env

   ``RAPIDTOOLS_DOMAIN`` is the DNS name, ``RAPIDTOOLS_GUI_TOKEN`` the
   secret colleagues must present, ``DATA_DIR`` and ``OUTPUT_DIR`` the two
   folders above. Fill the ``RAPIDTOOLS_SMTP_*`` lines to enable email
   notifications; leave them empty otherwise. On a machine without a GPU
   uncomment the two CPU lines at the end.

3. Build and start. With a GPU:

   .. code-block:: bash

      docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml \
          --env-file deploy/.env up -d --build

   Without one:

   .. code-block:: bash

      docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build

   The first build downloads PyTorch and the package dependencies (several
   gigabytes for the GPU image) and takes a few minutes. ``docker compose
   logs -f gui`` shows the server starting; the first request builds the
   RAPID route database from Mapillary in two to three minutes, after which
   it is cached in the ``rapidtools-cache`` volume for good.

4. Open ``https://<domain>/?token=<your token>`` and check that the map
   loads. That link is what you send to colleagues; the token can also be
   typed on the sign-in page.

Day to day
----------

- **Everyone shares one workspace and one job queue.** One detection or
  inference runs at a time; a second person's request is refused until it
  finishes. For a team that runs analyses at the same time, start a second
  GUI service on another port in the compose file with its own token and
  add it to the Caddyfile under a second path or name.
- **Imagery colleagues load** must sit under ``DATA_DIR``; the file browser
  cannot leave it. Results, crops and downloaded basemaps land under
  ``OUTPUT_DIR``. Both are plain folders on the host, so copy files in and
  out as usual.
- **Model weights and the route database** persist in the
  ``rapidtools-cache`` volume, so restarts do not download them again.
- **Notifications**: with the SMTP lines filled in, users can ask for an
  email when a long run ends; webhooks (Slack, Teams) work without any
  setup. Browser alerts work because the site is served over HTTPS. Links
  in messages use ``RAPIDTOOLS_PUBLIC_URL``, which the compose file sets to
  the domain.
- **Updating**: ``git pull`` then the same ``up -d --build`` command.
  Restarting clears the loaded image and assets from memory but keeps every
  result file.
- **Rotating the token**: change it in ``deploy/.env`` and restart the
  ``gui`` service.

A Windows machine in the lab
----------------------------

On a Windows workstation that colleagues reach over the local network, skip
Docker and Caddy: run the console script in a conda environment, keep it
alive with Task Scheduler, and open the port in the Windows firewall.

1. Install, in an Anaconda Prompt (the CUDA build of PyTorch is picked up
   automatically on a machine with an NVIDIA driver):

   .. code-block:: bat

      conda create -n rapidtools python=3.11 -y
      conda activate rapidtools
      pip install "git+https://github.com/RAPID-Facility/rAPIdtools.git@main"
      mkdir D:\rapidtools\data D:\rapidtools\output

2. Try it once by hand, with a token you generate:

   .. code-block:: bat

      python -c "import secrets; print(secrets.token_urlsafe(18))"
      set RAPIDTOOLS_GUI_TOKEN=<paste the token>
      rapidtools-gui --host 0.0.0.0 --port 8765 --no-browser ^
          --data-root D:\rapidtools\data --output-dir D:\rapidtools\output

   The window prints share links such as ``http://LAB-PC-07:8765/?token=...``
   using the machine name and its LAN address. Open one from another
   computer on the network to check it works. The first request builds the
   RAPID route database from Mapillary (two to three minutes, then cached
   under ``%USERPROFILE%\.cache\rapidtools``).

3. Open the port: in *Windows Defender Firewall with Advanced Security*
   add an inbound rule for TCP port 8765, restricted to the *Domain* and
   *Private* profiles so it is not reachable from a public network, or in
   an administrator prompt:

   .. code-block:: bat

      netsh advfirewall firewall add rule name="rapidtools GUI" dir=in action=allow protocol=TCP localport=8765 profile=domain,private

4. Keep it running: save a batch file such as
   ``D:\rapidtools\start-gui.bat``:

   .. code-block:: bat

      @echo off
      call %USERPROFILE%\anaconda3\Scripts\activate.bat rapidtools
      set RAPIDTOOLS_GUI_TOKEN=<token>
      set RAPIDTOOLS_SMTP_HOST=smtp.example.edu
      set RAPIDTOOLS_SMTP_USER=me@example.edu
      set RAPIDTOOLS_SMTP_PASSWORD=<password>
      set RAPIDTOOLS_SMTP_FROM=me@example.edu
      rapidtools-gui --host 0.0.0.0 --port 8765 --no-browser --data-root D:\rapidtools\data --output-dir D:\rapidtools\output >> D:\rapidtools\gui.log 2>&1

   then in Task Scheduler create a task that runs it *At startup* (or *At
   log on*), with *Run whether user is logged on or not* and *Restart the
   task if it fails* enabled. Under *Power Options* set the machine never
   to sleep, or the server disappears with it. The SMTP lines are optional.

Notes for this setup:

- Colleagues connect over plain HTTP on the intranet. The token still
  protects the app, but browser alerts need HTTPS, so users get email or
  webhook notifications instead. To add HTTPS, install the Windows build
  of Caddy with a Caddyfile of ``LAB-PC-07 { tls internal reverse_proxy
  127.0.0.1:8765 }``; colleagues then trust Caddy's local certificate
  once.
- Paths in the GUI are Windows paths under the data folder, for example
  ``D:\rapidtools\data\eaton\ortho.tif``. Colleagues can drop files in
  through a network share of that folder.
- One job at a time, as on any other server. The GPU is shared with
  whatever else runs on the workstation, so close other GPU work before
  large inference runs.

Running without Docker
----------------------

The container only wraps the console script. On a machine where you would
rather use a conda environment, install the package and run:

.. code-block:: bash

   RAPIDTOOLS_GUI_TOKEN=<token> RAPIDTOOLS_SMTP_HOST=... rapidtools-gui \
       --host 127.0.0.1 --port 8765 --no-browser \
       --data-root /srv/rapidtools/data --output-dir /srv/rapidtools/output \
       --public-url https://gui.example.edu/

then point Caddy or nginx at ``127.0.0.1:8765`` for HTTPS. A systemd unit
with ``Restart=always`` keeps it up across reboots.

On a cloud machine
------------------

Everything above applies; three things differ.

- **Choose the machine for the models you will use.** An instance with one
  24 GB card (an L4 or A10G) runs the local Gemma 4 models in 4-bit and SAM
  3 comfortably. A 2 to 4 vCPU machine without a GPU is enough when
  colleagues bring their own cloud model keys. Give it a static address so
  the DNS name keeps working after a stop and start.
- **Stop it when nobody uses it.** GPU instances bill by the hour whether
  busy or not. A stopped instance only pays for its disk and comes back
  with everything intact. Use the provider's instance schedule for working
  hours, or install ``deploy/idle-shutdown.sh`` as a cron job on the host:
  it asks the GUI whether a job is running and whether anyone has been
  using the page, and shuts the machine down after ``IDLE_MINUTES`` (30 by
  default) of silence. Starting it again is a click in the cloud console or
  a scheduled start.
- **Disk**: keep ``DATA_DIR``, ``OUTPUT_DIR`` and the Docker volumes on a
  persistent disk that survives the instance, and snapshot it now and then.

Security notes
--------------

The token is the only protection and everyone shares it. Serve the site
over HTTPS only (the compose setup does), keep port 8765 closed, rotate the
token when someone leaves, and prefer the institution's network or a VPN
for the address. API keys pasted into the page are sent to the server for
the duration of a job and are not stored.
