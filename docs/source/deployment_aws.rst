Deploying on AWS
================

The ``deploy/aws/`` folder of the repository turns an AWS account into a
working ``rapidtools-gui`` server with one CloudFormation template. You get
an HTTPS address that colleagues open in their browsers, protected by an
access token, on a machine that stops itself when nobody uses it. No DNS
name, SSH key or DevOps experience is needed. For a machine you already
have, see :doc:`deployment`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Purpose
   * - ``rapidtools-gui.yaml``
     - The CloudFormation template. Everything is in this one file.
   * - ``launch.sh``
     - Command-line launcher: checks your account, creates or updates the
       stack, prints the share link.
   * - ``teardown.sh``
     - Removes the stack and tells you what is left.
   * - ``bootstrap.sh``
     - What runs on the machine. It is embedded in the template and
       installed there as ``rapidtools-admin``.
   * - ``build-template.py``
     - Re-embeds ``bootstrap.sh`` into the template after an edit.

What gets created
-----------------

- One EC2 machine in its own small network (VPC), running the GUI and the
  Caddy web server in Docker. Caddy obtains the HTTPS certificate by itself.
- A fixed public address (Elastic IP). Without a DNS name of your own the
  server is reachable as ``https://rapidtools-<address with dashes>.sslip.io/``.
  `sslip.io <https://sslip.io>`_ is a free public DNS service that maps such
  names to the address inside them, which is enough for a real certificate.
- Two encrypted disks: a system disk (operating system and container image)
  and a data disk mounted on ``/srv/rapidtools`` that holds imagery
  (``data/``), results (``output/``), model weights and the route database
  (``cache/``) and the certificates (``caddy/``).
- The access token, stored encrypted in Systems Manager Parameter Store. It
  never appears in the stack outputs.
- An idle timer on the machine that stops it after 60 minutes without a job
  or a visitor, and a wake link that starts it again.
- Two small Lambda functions: one stores the secrets and picks the machine
  image when the stack is created, one serves the wake link.

Nothing in the template is shared with other stacks, so you can run several
copies (for example one per project) under different stack names.

Before you start
----------------

An AWS account
~~~~~~~~~~~~~~

Sign in to the `AWS console <https://console.aws.amazon.com/>`_ with a user
that may create EC2, IAM, Lambda and CloudFormation resources (an
administrator of the account can). If your university manages the account,
ask whether public Lambda function URLs are allowed. If they are not, set
*Create a wake link* to ``no``.

Choose a region
~~~~~~~~~~~~~~~

Pick the region closest to your users that offers GPU machines, for example
``us-west-2`` (Oregon) or ``us-east-1`` (N. Virginia). The region selector is
at the top right of the console. Everything below happens in that one
region.

Choose a machine
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 14 32

   * - Machine type
     - GPU
     - vCPU / memory
     - Approx. USD per hour
     - Use it for
   * - ``g5.xlarge`` (default)
     - A10G, 24 GB
     - 4 / 16 GiB
     - 1.01
     - Local Gemma 4 in 4-bit, SAM 3
   * - ``g5.2xlarge``
     - A10G, 24 GB
     - 8 / 32 GiB
     - 1.21
     - The same with large rasters
   * - ``g5.4xlarge``
     - A10G, 24 GB
     - 16 / 64 GiB
     - 1.62
     - Very large orthomosaics
   * - ``g6.xlarge``
     - L4, 24 GB
     - 4 / 16 GiB
     - 0.80
     - Cheaper, somewhat slower card
   * - ``g6.2xlarge``
     - L4, 24 GB
     - 8 / 32 GiB
     - 0.98
     -
   * - ``g6e.xlarge``
     - L40S, 48 GB
     - 4 / 32 GiB
     - 1.86
     - Larger local models
   * - ``g4dn.xlarge``
     - T4, 16 GB
     - 4 / 16 GiB
     - 0.53
     - Lowest-cost GPU, small models
   * - ``g4dn.2xlarge``
     - T4, 16 GB
     - 8 / 32 GiB
     - 0.75
     -
   * - ``t3.large``
     - none
     - 2 / 8 GiB
     - 0.08
     - Users bring cloud model API keys
   * - ``t3.xlarge``
     - none
     - 4 / 16 GiB
     - 0.17
     -
   * - ``m7i.large``
     - none
     - 2 / 8 GiB
     - 0.10
     -
   * - ``m7i.xlarge``
     - none
     - 4 / 16 GiB
     - 0.20
     -

Prices are approximate Linux on-demand rates for ``us-east-1`` and
``us-west-2`` as published by AWS on 2026-11-10. They differ by region and
change over time: check the `EC2 pricing page
<https://aws.amazon.com/ec2/pricing/on-demand/>`_.

The hourly price is only charged while the machine runs. What you pay all
month, also while it is stopped, is approximately:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Item
     - Approx. cost
   * - Disks (gp3), per GiB per month
     - 0.08 USD. Defaults: 150 + 100 GiB on a GPU machine (about 20 USD),
       40 + 100 GiB on a CPU machine (about 11 USD)
   * - Public IPv4 address
     - 0.005 USD per hour, about 3.65 USD per month
   * - Parameter Store, Lambda, logs
     - effectively zero at this scale
   * - Snapshot kept after teardown
     - about 0.05 USD per GiB actually used, per month

Example: a ``g5.xlarge`` used 20 hours a month costs about 20 USD of compute
plus about 24 USD for the disks and the address. Data transfer out to the
internet (downloading results) is billed by AWS separately at its usual
rates. See `EBS pricing <https://aws.amazon.com/ebs/pricing/>`_ and `VPC
pricing <https://aws.amazon.com/vpc/pricing/>`_ for the current numbers.

GPU machines only: check the quota
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New AWS accounts cannot start GPU machines. The limit is the service quota
**Running On-Demand G and VT instances**, counted in vCPUs, and it starts at
0. You need 4 for an ``xlarge``, 8 for a ``2xlarge``, 16 for a ``4xlarge``.

1. Open `Service Quotas, Amazon EC2
   <https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA>`_
   and make sure the region at the top right is the one you chose.
2. Look at **Applied account-level quota value**. If it is at least the
   number you need, you are done.
3. Otherwise choose **Request increase at account level**, enter 8 (room for
   one ``2xlarge``), and submit. Small requests are often approved within
   minutes to a few hours. Some are reviewed by AWS Support and take one to
   two business days. You get an email either way.

From the command line:

.. code-block:: bash

   aws service-quotas get-service-quota --region us-west-2 \
       --service-code ec2 --quota-code L-DB2E81BA --query Quota.Value
   aws service-quotas request-service-quota-increase --region us-west-2 \
       --service-code ec2 --quota-code L-DB2E81BA --desired-value 8

While you wait you can launch a CPU machine (``t3.large``) under a different
stack name and try the GUI with a cloud model API key.

Optional things to have ready
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A **Hugging Face token** (``hf_...``) of an account that accepted the
  licences of the gated SAM 3 and Gemma weights. Without one the package
  falls back to its shared token.
- Your own **DNS name**, if you prefer ``gui.example.edu`` over the sslip.io
  name.
- An **S3 bucket** for moving imagery in and results out.
- **SMTP** details if users should get an email when a long job ends.

Path A: the console, step by step
---------------------------------

1. Download ``deploy/aws/rapidtools-gui.yaml`` from the repository to your
   computer.
2. In the AWS console open **CloudFormation**, check the region, and choose
   **Create stack**, **With new resources (standard)**.
3. Choose **Choose an existing template**, **Upload a template file**, pick
   the file, and press **Next**.
4. **Stack name**: ``rapidtools-gui`` (lower-case letters, digits and
   dashes).
5. Fill in the parameters. The defaults work. The ones worth a look:

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Parameter
        - Meaning
      * - Machine type
        - From the table above.
      * - Data disk size
        - Room for imagery, results and model weights. It can be enlarged
          later, not shrunk.
      * - Network allowed to open the GUI
        - ``0.0.0.0/0`` for anywhere, or your campus range, or
          ``your.address/32``.
      * - Access token
        - Leave blank to have one generated.
      * - Your own DNS name
        - Leave blank for the automatic sslip.io name.
      * - Stop after this many idle minutes
        - 60 by default. 0 keeps the machine running and billing.
      * - Container image tag, Git branch
        - Leave as they are.
      * - Hugging Face token
        - Optional, see above.

6. Press **Next**. On the *Configure stack options* page scroll down, tick
   **I acknowledge that AWS CloudFormation might create IAM resources**, and
   press **Next**, then **Submit**.
7. Wait until the stack status is **CREATE_COMPLETE**. The stack waits for
   the GUI to answer, so this takes as long as the machine needs to set
   itself up: expect roughly 10 to 15 minutes when the container image is
   published, and 25 to 40 minutes when the machine has to build the image
   from source (the first release of the image had not been published when
   this was written, so the first launches build). The **Events** tab shows
   progress. The stack gives up after 60 minutes.
8. Open the **Outputs** tab:

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Output
        - Use
      * - ``GuiUrl``
        - The address of the GUI.
      * - ``TokenConsoleLink``
        - Opens the token in Parameter Store. Press **Show decrypted
          value**.
      * - ``GetTokenCommand``
        - Prints the token. Paste it into CloudShell (the terminal icon in
          the console's top bar).
      * - ``WakeLink``
        - Starts the machine after it stopped. Bookmark it.
      * - ``ConnectCommand``, ``StartCommand``, ``StopCommand``
        - Ready-made commands.
      * - ``ElasticIp``
        - The fixed address, for your own DNS record.
      * - ``BootstrapResult``
        - What the machine reported, including whether HTTPS was ready.

9. The share link is ``GuiUrl`` followed by ``?token=`` and the token, for
   example ``https://rapidtools-203-0-113-10.sslip.io/?token=AbC...``. Open
   it. That link is what you send to colleagues. The token can also be typed
   on the sign-in page.

Path B: the command line
------------------------

You need the `AWS CLI version 2
<https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_
with working credentials (``aws configure``, or ``aws sso login`` in an
organisation). AWS CloudShell in the console already has both.

.. code-block:: bash

   git clone https://github.com/RAPID-Facility/rAPIdtools.git
   cd rAPIdtools

   # GPU machine, reachable from anywhere, token generated:
   deploy/aws/launch.sh --region us-west-2

   # Small CPU machine that only your current network can reach:
   deploy/aws/launch.sh --region us-west-2 --cpu --my-ip

   # With a Hugging Face token and a bucket:
   HF_TOKEN=hf_xxx deploy/aws/launch.sh --region us-west-2 --bucket my-rapid-data

The script checks the GPU quota, picks an Availability Zone that offers the
machine type, asks for confirmation, waits for the stack and prints the share
link, the wake link and the command that opens a shell. Run
``deploy/aws/launch.sh --help`` for all options. Running it again with one
option changes that setting of the existing stack and leaves the rest alone.

Without the script:

.. code-block:: bash

   aws cloudformation deploy --region us-west-2 --stack-name rapidtools-gui \
       --template-file deploy/aws/rapidtools-gui.yaml --capabilities CAPABILITY_IAM \
       --parameter-overrides InstanceType=g5.xlarge AllowedCidr=0.0.0.0/0
   aws cloudformation describe-stacks --region us-west-2 --stack-name rapidtools-gui \
       --query "Stacks[0].Outputs" --output table

First login
-----------

Open the share link and check that the map loads. The very first request
builds the RAPID route database from Mapillary in two to three minutes, after
which it is cached on the data disk. The first use of a local model downloads
its weights (several gigabytes) to the data disk as well.

If the browser says the site cannot be reached or shows a certificate
warning in the first minutes, wait a little and reload: Caddy may still be
getting the certificate. See `Troubleshooting`_ if it persists.

Getting a shell on the machine
------------------------------

No SSH key is needed. In the console open **EC2**, **Instances**, select the
machine, press **Connect**, choose the **Session Manager** tab and press
**Connect**. From your computer, with the `Session Manager plugin
<https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html>`_
installed, run the ``ConnectCommand`` output:

.. code-block:: bash

   aws ssm start-session --region us-west-2 --target i-0123456789abcdef0

On the machine everything is done with ``sudo rapidtools-admin``:

.. code-block:: bash

   sudo rapidtools-admin status          # containers, GPU, disk space, share link
   sudo rapidtools-admin token           # print the share link
   sudo rapidtools-admin logs gui        # follow the GUI log (Ctrl+C to leave)
   sudo rapidtools-admin restart         # restart the containers
   sudo rapidtools-admin boot            # apply changed stack settings or secrets now
   sudo rapidtools-admin update          # get the newest image or rebuild, then restart
   sudo rapidtools-admin rotate-token    # new token
   sudo rapidtools-admin pull-data       # S3 bucket -> /srv/rapidtools/data
   sudo rapidtools-admin push-results    # /srv/rapidtools/output -> S3 bucket

A single command can also be run without opening a shell:

.. code-block:: bash

   aws ssm send-command --region us-west-2 --instance-ids i-0123456789abcdef0 \
       --document-name AWS-RunShellScript \
       --parameters 'commands=["rapidtools-admin status"]' \
       --query Command.CommandId --output text
   aws ssm get-command-invocation --region us-west-2 --instance-id i-0123456789abcdef0 \
       --command-id <the id printed above> --query StandardOutputContent --output text

Uploading imagery
-----------------

The GUI's file browser shows ``/srv/rapidtools/data`` on the machine (as
``/data``). Three ways to put files there:

- **Through an S3 bucket (recommended).** Give the stack the name of a
  bucket you own (parameter *S3 bucket for imagery and results*, or
  ``launch.sh --bucket NAME``). Upload your files under the ``data/``
  prefix, in the S3 console or with
  ``aws s3 sync ./eaton s3://my-rapid-data/data/eaton/``. Then run
  ``sudo rapidtools-admin pull-data`` on the machine. The machine may read
  and write that one bucket and nothing else.
- **With scp**, if you launched with an EC2 key pair and an SSH network:
  ``scp -i key.pem ortho.tif ubuntu@<ElasticIp>:/tmp/``, then in a shell
  ``sudo mv /tmp/ortho.tif /srv/rapidtools/data/``.
- **Directly from the internet**, in a shell on the machine:
  ``sudo curl -L -o /srv/rapidtools/data/ortho.tif <url>``.

Basemap tiles and street-level imagery need no upload: the GUI downloads
them itself.

Getting results out
-------------------

Results, crops and downloaded basemaps are under ``/srv/rapidtools/output``.
The GUI offers the GeoJSON results for download in the browser. For the
crops and everything else at once run ``sudo rapidtools-admin push-results``,
which copies the folder to ``s3://<bucket>/output/``, and download from
there (``aws s3 sync s3://my-rapid-data/output/ ./results/``).

Stopping and starting
---------------------

A stopped machine keeps both disks, its address and its name, and costs only
the storage and the address.

- **By itself.** After *Stop after this many idle minutes* without a running
  job and without a browser using the page, the machine shuts down. A
  running job always keeps it up. An open browser tab counts as use, so close
  the tab when you are done.
- **Wake link.** Opening the ``WakeLink`` output shows a **Start the
  server** button when the machine is stopped, waits while it boots (two to
  three minutes) and then forwards to the GUI. Colleagues can use it too.
  They still need the token.
- **By hand.** In the EC2 console select the machine and use **Instance
  state**, **Stop instance** or **Start instance**, or run the
  ``StopCommand`` and ``StartCommand`` outputs.
- **Every day at a fixed hour.** Set *Also stop every day at this hour* as a
  safety net. This stop does not wait for a running job.

After a start everything comes back by itself. A stopped GPU machine
sometimes cannot be started because AWS has no free capacity of that type in
the zone at that moment. Try again a few minutes later.

Changing settings
-----------------

Change parameters in the CloudFormation console (**Update stack**, **Use
existing template**, edit the values) or by running ``launch.sh`` again with
the option. CloudFormation applies the change as follows:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Change
     - Effect
   * - Allowed network, SSH network, wake link, daily stop
     - Applied at once, no interruption.
   * - Machine type, idle minutes, image tag, Git reference, DNS name,
       bucket, SMTP settings
     - The machine is stopped and started (a few minutes). It then applies
       the new settings by itself.
   * - Data disk size (larger only)
     - Enlarged in place. The file system grows at the next start or
       ``sudo rapidtools-admin boot``. AWS allows one size change per disk
       every six hours.
   * - Access token, Hugging Face token, SMTP password
     - Stored at once. They take effect at the next start or
       ``sudo rapidtools-admin boot``.
   * - Availability Zone, key pair, snapshot, system disk size
     - Cannot be changed on a running stack. Create a new stack (see
       below).

Switching between CPU and GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The machine image is chosen once, when the stack is created: a plain Ubuntu
image for CPU types and the AWS Deep Learning image with the NVIDIA driver
for GPU types. Changing the machine type within the same family is a normal
update. To move from a CPU type to a GPU type (or back), remove the stack
keeping the snapshot and launch a new one from it:

.. code-block:: bash

   deploy/aws/teardown.sh --region us-west-2            # prints the snapshot id
   deploy/aws/launch.sh --region us-west-2 --instance-type g5.xlarge \
       --snapshot snap-0123456789abcdef0

The address, and with it the sslip.io name, the token and the wake link, are
new after that.

Updating to a new version
-------------------------

- Newest published image, or newest commit of the Git reference when the
  machine builds from source: ``sudo rapidtools-admin update``. Results and
  caches are kept. The loaded image and assets are cleared from memory.
- A specific release: set *Container image tag* (for example ``0.3.0``) in
  the stack. The machine restarts with it.
- The machine never updates by itself: a stop and start keeps the version
  it has.

Rotating the token
------------------

Do this when someone leaves the group. On the machine:

.. code-block:: bash

   sudo rapidtools-admin rotate-token                     # random new token
   sudo rapidtools-admin rotate-token my-new-token-2027   # or one you choose

The new token is stored in Parameter Store and the GUI restarts with it.
Everyone has to sign in again with the new share link. Alternatively set
*Access token* in the stack and restart the machine.

Backups
-------

The data disk is the only thing worth backing up. Take a snapshot whenever
you like:

.. code-block:: bash

   aws ec2 create-snapshot --region us-west-2 --volume-id <DataVolumeId output> \
       --description "rapidtools data $(date +%F)"

or in the console under **EC2**, **Volumes**, select the volume named
``<stack>-data``, **Actions**, **Create snapshot**. Snapshots are
incremental and billed for the data they hold. For automatic daily snapshots
use `Amazon Data Lifecycle Manager
<https://docs.aws.amazon.com/ebs/latest/userguide/snapshot-lifecycle.html>`_.
Copying results to S3 with ``push-results`` is a second, independent copy.

Removing everything
-------------------

.. code-block:: bash

   deploy/aws/teardown.sh --region us-west-2                    # keep a snapshot of the data disk
   deploy/aws/teardown.sh --region us-west-2 --delete-snapshot  # keep nothing

or in the console: **CloudFormation**, select the stack, **Delete**.

Deleting the stack removes the machine, the system disk, the address, the
network, the functions and the stored secrets. **The data disk is saved as a
snapshot first, and that snapshot stays and is billed until you delete it**
(EC2, **Snapshots**). This is deliberate: results survive an accidental
deletion. A new stack can start from the snapshot (*Restore the data disk
from this snapshot*). An S3 bucket you supplied is never touched.

Troubleshooting
---------------

**The stack stays in CREATE_IN_PROGRESS for a long time.** The last
resource, ``Ready``, waits for the machine to report that the GUI answers. A
source build of the CUDA image takes 15 to 25 minutes on its own. To watch,
open a shell (see above; Session Manager works as soon as the machine is up)
and run

.. code-block:: bash

   sudo tail -f /var/log/rapidtools-bootstrap.log

or, without a shell, read the machine's console output:

.. code-block:: bash

   aws ec2 get-console-output --region us-west-2 --instance-id <InstanceId> \
       --latest --output text | tail -n 60

**The stack failed and rolled back.** The **Events** tab shows the reason
next to ``Ready`` (the last lines of the bootstrap log) or next to the
resource that failed. A rollback deletes the machine, so to investigate
launch again with rollback turned off: ``launch.sh`` does that for you, and
in the console it is *Stack failure options*, **Preserve successfully
provisioned resources**. Delete the failed stack before the next attempt.

**"You have requested more vCPU capacity than your current vCPU limit of
0".** The GPU quota. See `GPU machines only: check the quota`_.

**"Insufficient capacity" or "not supported in your requested Availability
Zone".** AWS has no free machine of that type in the zone. Delete the failed
stack and launch again with another zone in *Availability Zone* (for
example ``us-west-2b`` instead of ``us-west-2a``), another machine type, or
later. ``launch.sh`` picks a zone that offers the type.

**The certificate is not issued (browser warning, or** ``BootstrapResult``
**says "certificate pending").** Caddy keeps retrying by itself. Check that
port 80 is open to the world (the template does that; a network firewall in
front of your account might not), that a DNS name of your own really points
at the ``ElasticIp`` (``nslookup gui.example.edu``), and read the log with
``sudo rapidtools-admin logs caddy``. After fixing DNS run
``sudo rapidtools-admin restart``.

**The sslip.io name does not resolve.** Some campus and corporate DNS
resolvers block names that point at addresses embedded in them (DNS
rebinding protection). Test with
``nslookup rapidtools-203-0-113-10.sslip.io`` and compare with
``nslookup rapidtools-203-0-113-10.sslip.io 1.1.1.1``. If only the second
works, ask your network team to allow ``sslip.io``, or use a DNS name of
your own (parameter *Your own DNS name*).

**The page loads but says sign in required.** Use the full share link with
``?token=``, or type the token on the sign-in page. Print the current one
with ``GetTokenCommand``.

**The wake link shows "Forbidden".** The account does not allow public
Lambda function URLs. Start the machine from the EC2 console instead and set
*Create a wake link* to ``no``.

**The machine stops although I am working.** Only requests from a browser
with the page open count as use. A job started and left running keeps the
machine up until it ends. Raise *Stop after this many idle minutes* if
needed.

**The machine never stops.** A browser tab left open on the GUI counts as
use. Also, GUI versions older than the idle counter (``idle_s`` in
``/api/state``) fall back to "no web request at all for the idle period",
which a tab left open defeats in the same way.

**Out of disk space.** ``sudo rapidtools-admin status`` shows both disks.
The data disk fills with imagery, results and model weights: enlarge *Data
disk size* in the stack, or delete old folders under
``/srv/rapidtools/output`` and unused weights under
``/srv/rapidtools/cache/huggingface``. The system disk fills with old
container images after several updates: run ``sudo docker image prune -a``.

**A local model fails with a CUDA out-of-memory error.** The card is too
small for that model. Use the 4-bit option, a smaller model, or a machine
type with more GPU memory.

Security notes
--------------

- The token is the only protection of the GUI and everyone shares it. Limit
  *Network allowed to open the GUI* to your campus or VPN range when you
  can, and rotate the token when someone leaves.
- Port 443 is open to the allowed network only. Port 80 is open to the
  world because the certificate authority has to reach it. It serves
  nothing but the redirect to HTTPS. Port 22 is closed unless you supply a
  key pair and an SSH network. Port 8765 is never exposed.
- Shell access goes through Session Manager, which is controlled by your AWS
  sign-in and leaves an audit trail in the account.
- The access token, the Hugging Face token and the SMTP password are kept in
  Parameter Store as encrypted values. They are not written into the
  machine's user data and not shown in the stack outputs or events. The
  machine may read them and nothing else in Parameter Store.
- Both disks are encrypted. The machine's metadata service accepts only
  token-based requests (IMDSv2) and cannot be reached from inside the
  containers.
- The wake link is public but unguessable. All it can do is start this one
  stopped machine after a button press. It cannot create machines, change
  the machine type or read anything. The worst case of a leaked link is that
  the machine runs for the idle period each time someone presses the button.
  Turn it off with *Create a wake link* ``no``.
- The sslip.io name depends on a DNS service run by volunteers, and the
  name becomes public through certificate transparency logs like every
  HTTPS host name. For a long-lived server use a DNS name of your own.
- API keys users paste into the page are sent to the server for the
  duration of a job and are not stored.

How it works
------------

``rapidtools-gui.yaml`` creates the network, the address, the disks, the
roles and the machine. The machine's user data writes the stack settings to
``/etc/rapidtools/stack.env``, unpacks ``bootstrap.sh`` (stored compressed in
the template) and runs it. The script installs Docker and the AWS CLI where
they are missing, formats and mounts the data disk, reads the secrets from
Parameter Store, writes ``/opt/rapidtools/.env`` and the compose files, pulls
``ghcr.io/rapid-facility/rapidtools-gui`` or, when that tag is not
published, clones the repository at the Git reference and builds the image,
starts the containers, installs the idle timer, waits until the GUI answers
on ``/api/state`` and then tells CloudFormation that the stack is complete.
It runs again on every boot, so settings changed in the stack are applied
after the restart CloudFormation performs. Everything it does is logged to
``/var/log/rapidtools-bootstrap.log``.

To change what runs on the machine, edit ``bootstrap.sh``, run
``python deploy/aws/build-template.py`` and update the stack with the new
template. ``python deploy/aws/build-template.py --extract`` prints the script
a template carries.
