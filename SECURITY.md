# Security Policy

rAPIdtools processes geospatial data with local and hosted machine-learning
models, downloads imagery from third-party services, and ships a browser-based
server (`rapidtools-gui`) that teams may run on a shared machine. We take
reports about any of these seriously.

## Supported versions

Security fixes are released for the latest minor version on PyPI. Older
releases do not receive patches; please upgrade.

| Version | Supported |
| --- | --- |
| 0.2.x | Yes |
| < 0.2 | No |

## Reporting a vulnerability

**Please do not report security problems through public issues, pull requests
or discussions.**

Use GitHub's private vulnerability reporting instead:
[Report a vulnerability](https://github.com/RAPID-Facility/rAPIdtools/security/advisories/new).
The report is visible only to the maintainers, and GitHub lets us work on a fix
with you in a private fork before anything is published.

If you cannot use GitHub, email <uwrapid@uwrapid.org> with "rAPIdtools
security" in the subject line.

Please include:

- the rapidtools version (`pip show rapidtools`) and how it was installed;
- the component involved (for example the GUI server, a model wrapper, an
  imagery client, or the deploy files);
- steps to reproduce, or a proof of concept, and the impact you believe it has;
- whether the problem is already public anywhere.

### What to expect

- We acknowledge reports within 5 working days.
- We aim to confirm the problem and agree on severity within 2 weeks, and to
  release a fix for confirmed high-severity issues within 30 days of
  confirmation. We will tell you if we need longer.
- We publish a GitHub Security Advisory with credit to the reporter (unless you
  prefer to stay anonymous) once a fixed release is available, and note the fix
  in `CHANGELOG.md`.

We ask that you give us reasonable time to fix a problem before disclosing it
publicly, and that you do not access or modify data that is not yours while
investigating.

## Scope

In scope:

- The `rapidtools` Python package published on PyPI and the code in this
  repository.
- The `rapidtools-gui` server, including its access token, `--data-root`
  sandbox, file-serving endpoints and notification features.
- The deployment files under `deploy/` (Docker, Compose, Caddy).
- Handling of credentials: provider API keys, Mapillary access tokens and
  Hugging Face tokens.

Out of scope:

- Vulnerabilities in third-party services the package talks to (Google,
  Bing, Mapillary, Hugging Face, the model providers). Report those to the
  service.
- Issues that require an attacker to already have the GUI access token or
  shell access to the host.
- Model output quality, prompt injection against vision-language models, or
  the correctness of damage assessments. These are research questions, not
  vulnerabilities, and belong in a normal issue.

## Security notes for users

- **The GUI token protects the application, not the network.** The server
  speaks plain HTTP. When sharing it, bind it on a trusted network or VPN, or
  put it behind the TLS proxy in `deploy/`. Always start a shared server with
  `--token auto` (or `RAPIDTOOLS_GUI_TOKEN`) and `--data-root`; without a token
  every endpoint is open to anyone who can reach the port.
- **Keep credentials out of files you commit.** Supply API keys through
  environment variables (`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `MODEL_API_KEY`, `DASHSCOPE_API_KEY`) or pass them at run
  time. If a key ever lands in a commit, revoke it with the provider first;
  deleting the file does not remove it from history.
- **Keys pasted into the GUI** are sent to the server for the duration of a job
  and are not written to disk, but they do travel over whatever transport the
  server uses. Use HTTPS for a shared server.
- **Stay current.** `pip install --upgrade rapidtools` picks up fixes; the
  changelog lists security-relevant changes under `Fixed`.
