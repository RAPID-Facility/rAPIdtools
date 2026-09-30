# Copyright (c) 2025 The University of Washington
#
# This file is part of rapidtools.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# rapidtools. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 09-30-2026
"""
Tell the user when a long GUI job has finished.

The GUI runs detection, imagery downloads and inference on a background
thread and keeps the results in the server, so a user can close the tab and
come back later. This module adds the "come back" signal: an email or a
webhook (Slack, Teams, or any endpoint that accepts JSON) sent when the job
ends, from settings the user gives while the job is running.

Email needs an SMTP relay, configured on the server side through
environment variables or the matching ``rapidtools-gui`` flags:

==========================  ==========================================
Variable                    Meaning
==========================  ==========================================
``RAPIDTOOLS_SMTP_HOST``    Relay host name (email is off when unset)
``RAPIDTOOLS_SMTP_PORT``    Port, default 587 (465 with ``SMTP_SSL``)
``RAPIDTOOLS_SMTP_USER``    Login, optional
``RAPIDTOOLS_SMTP_PASSWORD``  Password, optional
``RAPIDTOOLS_SMTP_FROM``    Sender address, defaults to the user
``RAPIDTOOLS_SMTP_SSL``     ``1`` for implicit TLS instead of STARTTLS
``RAPIDTOOLS_PUBLIC_URL``   Link put in messages, defaults to the bound URL
==========================  ==========================================

Webhooks need nothing: the URL is posted a JSON document with the job name,
status, message, duration, result files and the link back to the GUI, plus
a ``text`` field so Slack and Teams incoming webhooks render it directly.

Example:
    >>> from rapidtools.gui.notify import NotificationConfig, Notifier
    >>> notifier = Notifier(NotificationConfig(smtp_host=''))
    >>> notifier.email_available
    False
    >>> Notifier.classify('someone@example.org')
    'email'
    >>> Notifier.classify('https://hooks.slack.com/services/T/B/x')
    'webhook'
"""

from __future__ import annotations

import logging
import os
import re
import smtplib
import ssl
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


@dataclass
class NotificationConfig:
    """
    Where notifications are sent from.

    Attributes:
        smtp_host (str): SMTP relay; empty disables email.
        smtp_port (int): Relay port.
        smtp_user (str): Login, empty for an open relay.
        smtp_password (str): Password for ``smtp_user``.
        smtp_from (str): Sender address; defaults to ``smtp_user``.
        smtp_ssl (bool): Use implicit TLS (port 465) instead of STARTTLS.
        public_url (str): The link put in messages; the server fills it with
            its own address when empty.

    Example:
        >>> cfg = NotificationConfig.from_env({'RAPIDTOOLS_SMTP_HOST': 'mail.x.org'})
        >>> cfg.smtp_host, cfg.smtp_port
        ('mail.x.org', 587)
    """

    smtp_host: str = ''
    smtp_port: int = 587
    smtp_user: str = ''
    smtp_password: str = ''
    smtp_from: str = ''
    smtp_ssl: bool = False
    public_url: str = ''

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **overrides: Any):
        """
        Read the ``RAPIDTOOLS_SMTP_*`` variables, then apply ``overrides``.

        Args:
            env (dict[str, str] | None): Environment to read; ``os.environ``
                by default.
            **overrides: Field values that win over the environment (falsy
                values are ignored so CLI defaults do not mask variables).

        Returns:
            NotificationConfig: The merged configuration.
        """
        env = os.environ if env is None else env
        values: dict[str, Any] = {
            'smtp_host': env.get('RAPIDTOOLS_SMTP_HOST', ''),
            'smtp_port': int(env.get('RAPIDTOOLS_SMTP_PORT') or 0) or 0,
            'smtp_user': env.get('RAPIDTOOLS_SMTP_USER', ''),
            'smtp_password': env.get('RAPIDTOOLS_SMTP_PASSWORD', ''),
            'smtp_from': env.get('RAPIDTOOLS_SMTP_FROM', ''),
            'smtp_ssl': env.get('RAPIDTOOLS_SMTP_SSL', '') in ('1', 'true', 'yes'),
            'public_url': env.get('RAPIDTOOLS_PUBLIC_URL', ''),
        }
        for key, value in overrides.items():
            if value:
                values[key] = value
        if not values['smtp_port']:
            values['smtp_port'] = 465 if values['smtp_ssl'] else 587
        return cls(**values)


@dataclass
class JobSummary:
    """
    What a finished job looks like to the recipient.

    Attributes:
        name (str): Job name as shown in the GUI.
        status (str): ``'done'``, ``'error'`` or ``'cancelled'``.
        message (str): The one-line outcome.
        duration_s (float): Wall-clock time of the job.
        results (list[dict[str, str]]): ``{'label', 'path'}`` result files.
        url (str): Link back to the GUI.
    """

    name: str
    status: str
    message: str
    duration_s: float = 0.0
    results: list[dict[str, str]] = field(default_factory=list)
    url: str = ''

    @property
    def subject(self) -> str:
        """Subject line, e.g. ``'rapidtools: Inference finished'``."""
        verb = {'done': 'finished', 'error': 'failed', 'cancelled': 'was cancelled'}
        return f'rapidtools: {self.name} {verb.get(self.status, self.status)}'

    @property
    def text(self) -> str:
        """Plain-text body shared by email and webhooks."""
        minutes, seconds = divmod(int(self.duration_s), 60)
        lines = [self.message, '', f'Run time: {minutes} min {seconds} s.']
        if self.results:
            lines += ['', 'Result files:']
            lines += [f'  - {r["label"]}: {r["path"]}' for r in self.results]
        if self.url:
            lines += ['', f'Open the results: {self.url}']
        return '\n'.join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON document posted to webhooks."""
        return {
            'source': 'rapidtools',
            'job': self.name,
            'status': self.status,
            'message': self.message,
            'duration_s': round(self.duration_s, 1),
            'results': list(self.results),
            'url': self.url,
            'text': f'{self.subject}\n{self.text}',
        }


class Notifier:
    """
    Send job summaries by email or webhook.

    Args:
        config (NotificationConfig): SMTP relay and link settings.
        post (Callable | None): Replacement for ``requests.post`` (tests).

    Example:
        >>> notifier = Notifier(NotificationConfig())
        >>> notifier.validate_target('not an address')
        Traceback (most recent call last):
        ...
        ValueError: Enter an email address or a webhook URL (https://...).
    """

    def __init__(
        self,
        config: NotificationConfig | None = None,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or NotificationConfig.from_env()
        self._post = post

    @property
    def email_available(self) -> bool:
        """``True`` when an SMTP relay is configured."""
        return bool(self.config.smtp_host)

    @staticmethod
    def classify(target: str) -> str:
        """
        ``'email'``, ``'webhook'`` or ``''`` for an unrecognised target.

        Example:
            >>> Notifier.classify('a@b.co'), Notifier.classify('ftp://x')
            ('email', '')
        """
        target = (target or '').strip()
        if _EMAIL_RE.match(target):
            return 'email'
        if target.startswith(('https://', 'http://')):
            return 'webhook'
        return ''

    def validate_target(self, target: str) -> str:
        """
        Check that ``target`` can be delivered to and return its kind.

        Raises:
            ValueError: For an unrecognised target, or an email address when
                no SMTP relay is configured.
        """
        kind = self.classify(target)
        if not kind:
            raise ValueError('Enter an email address or a webhook URL (https://...).')
        if kind == 'email' and not self.email_available:
            raise ValueError(
                'Email is not set up on this server: start it with '
                '--smtp-host (or set RAPIDTOOLS_SMTP_HOST). A webhook URL '
                'works without any setup.'
            )
        return kind

    # ------------------------------------------------------------ channels
    def send_email(self, to: str, summary: JobSummary) -> None:
        """
        Send ``summary`` to ``to`` through the configured relay.

        Raises:
            RuntimeError: When no relay is configured.
            smtplib.SMTPException, OSError: On delivery failure.
        """
        cfg = self.config
        if not cfg.smtp_host:
            raise RuntimeError('No SMTP relay is configured.')
        message = EmailMessage()
        message['Subject'] = summary.subject
        message['From'] = (
            cfg.smtp_from or cfg.smtp_user or f'rapidtools@{cfg.smtp_host}'
        )
        message['To'] = to
        message.set_content(summary.text)
        if cfg.smtp_ssl:
            client = smtplib.SMTP_SSL(
                cfg.smtp_host,
                cfg.smtp_port,
                timeout=30,
                context=ssl.create_default_context(),
            )
        else:
            client = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30)
        with client:
            if not cfg.smtp_ssl:
                client.ehlo()
                if client.has_extn('starttls'):
                    client.starttls(context=ssl.create_default_context())
                    client.ehlo()
            if cfg.smtp_user:
                client.login(cfg.smtp_user, cfg.smtp_password)
            client.send_message(message)

    def send_webhook(self, url: str, summary: JobSummary) -> None:
        """
        POST ``summary`` as JSON to ``url``.

        Raises:
            requests.RequestException: On a network error or non-2xx reply.
        """
        if self._post is None:
            import requests

            self._post = requests.post
        response = self._post(url, json=summary.to_dict(), timeout=30)
        response.raise_for_status()

    def send(self, target: str, summary: JobSummary) -> str:
        """
        Deliver ``summary`` to ``target`` and describe what was done.

        Returns:
            str: A log line such as ``'Emailed a@b.org.'``.

        Raises:
            ValueError: If the target is unusable.
            Exception: Whatever the channel raised.
        """
        kind = self.validate_target(target)
        if kind == 'email':
            self.send_email(target.strip(), summary)
            return f'Notification emailed to {target.strip()}.'
        self.send_webhook(target.strip(), summary)
        return 'Notification posted to the webhook.'

    def send_in_background(
        self,
        target: str,
        summary: JobSummary,
        on_result: Callable[[bool, str], None],
    ) -> threading.Thread:
        """
        Run :meth:`send` on a thread and report through ``on_result``.

        Args:
            target (str): Email address or webhook URL.
            summary (JobSummary): What to send.
            on_result (Callable[[bool, str], None]): Called with ``(ok,
                message)`` when delivery finishes.

        Returns:
            threading.Thread: The started thread.
        """

        def runner() -> None:
            try:
                on_result(True, self.send(target, summary))
            except Exception as exc:  # noqa: BLE001 - reported to the GUI log
                logger.warning(f'Notification failed: {exc}')
                on_result(False, f'Notification to {target.strip()} failed: {exc}')

        thread = threading.Thread(target=runner, name='notify', daemon=True)
        thread.start()
        return thread
