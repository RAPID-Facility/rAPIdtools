"""Tests for rapidtools.gui.notify: job summaries, email and webhook delivery."""

import threading

import pytest

from rapidtools.gui.notify import JobSummary, NotificationConfig, Notifier


def test_config_from_env_and_overrides():
    env = {
        'RAPIDTOOLS_SMTP_HOST': 'mail.example.org',
        'RAPIDTOOLS_SMTP_USER': 'u',
        'RAPIDTOOLS_SMTP_PASSWORD': 'p',
        'RAPIDTOOLS_SMTP_SSL': '1',
        'RAPIDTOOLS_PUBLIC_URL': 'https://gui.example.org/',
    }
    cfg = NotificationConfig.from_env(env)
    assert cfg.smtp_host == 'mail.example.org' and cfg.smtp_ssl is True
    assert cfg.smtp_port == 465  # implicit TLS default
    assert cfg.public_url == 'https://gui.example.org/'
    # Overrides win, falsy overrides do not mask the environment:
    cfg = NotificationConfig.from_env(env, smtp_host='relay', smtp_port=None)
    assert cfg.smtp_host == 'relay' and cfg.smtp_user == 'u'
    assert NotificationConfig.from_env({}).smtp_port == 587


def test_summary_subject_and_text():
    summary = JobSummary(
        name='Inference',
        status='done',
        message='Inference finished.',
        duration_s=125,
        results=[{'label': 'Inferred assets', 'path': '/out/a.geojson'}],
        url='http://localhost:8765/',
    )
    assert summary.subject == 'rapidtools: Inference finished'
    text = summary.text
    assert 'Run time: 2 min 5 s.' in text
    assert '  - Inferred assets: /out/a.geojson' in text
    assert 'Open the results: http://localhost:8765/' in text
    assert JobSummary('Detection', 'error', 'boom').subject == (
        'rapidtools: Detection failed'
    )
    doc = summary.to_dict()
    assert doc['source'] == 'rapidtools' and doc['status'] == 'done'
    assert doc['text'].startswith(summary.subject)


def test_classify_and_validate_targets():
    notifier = Notifier(NotificationConfig())
    assert Notifier.classify('a@b.org') == 'email'
    assert Notifier.classify(' https://hooks.slack.com/x ') == 'webhook'
    assert Notifier.classify('nope') == '' and Notifier.classify('') == ''
    assert notifier.email_available is False
    with pytest.raises(ValueError, match='email address or a webhook'):
        notifier.validate_target('nope')
    with pytest.raises(ValueError, match='not set up on this server'):
        notifier.validate_target('a@b.org')
    assert notifier.validate_target('https://x.y/z') == 'webhook'
    assert Notifier(NotificationConfig(smtp_host='m')).validate_target('a@b.org') == (
        'email'
    )


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.context = host, port, context
        self.calls = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append('quit')

    def ehlo(self):
        self.calls.append('ehlo')

    def has_extn(self, name):
        return name == 'starttls'

    def starttls(self, context=None):
        self.calls.append('starttls')

    def login(self, user, password):
        self.calls.append(('login', user, password))

    def send_message(self, message):
        self.calls.append(('send', message))


def test_send_email_starttls_and_ssl(monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr('rapidtools.gui.notify.smtplib.SMTP', FakeSMTP)
    monkeypatch.setattr('rapidtools.gui.notify.smtplib.SMTP_SSL', FakeSMTP)
    summary = JobSummary('Detection', 'done', 'Detection finished.', 3.0)
    notifier = Notifier(
        NotificationConfig(
            smtp_host='mail', smtp_port=587, smtp_user='u', smtp_password='p'
        )
    )
    assert notifier.send('bob@example.org', summary) == (
        'Notification emailed to bob@example.org.'
    )
    smtp = FakeSMTP.instances[-1]
    assert (smtp.host, smtp.port) == ('mail', 587)
    assert smtp.calls[:3] == ['ehlo', 'starttls', 'ehlo']
    assert ('login', 'u', 'p') in smtp.calls
    message = [c for c in smtp.calls if isinstance(c, tuple) and c[0] == 'send'][0][1]
    assert message['To'] == 'bob@example.org' and message['From'] == 'u'
    assert message['Subject'] == 'rapidtools: Detection finished'
    assert 'Detection finished.' in message.get_content()

    ssl_notifier = Notifier(
        NotificationConfig(
            smtp_host='mail', smtp_port=465, smtp_ssl=True, smtp_from='gui@x.org'
        )
    )
    ssl_notifier.send_email('bob@example.org', summary)
    smtp = FakeSMTP.instances[-1]
    assert smtp.port == 465 and smtp.context is not None
    assert 'starttls' not in smtp.calls
    assert smtp.calls[-2][1]['From'] == 'gui@x.org'

    with pytest.raises(RuntimeError):
        Notifier(NotificationConfig()).send_email('bob@example.org', summary)


class FakeResponse:
    def __init__(self, status):
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f'HTTP {self.status}')


def test_send_webhook_and_background_delivery():
    posts = []

    def post(url, json=None, timeout=None):
        posts.append((url, json, timeout))
        return FakeResponse(500 if 'bad' in url else 200)

    notifier = Notifier(NotificationConfig(), post=post)
    summary = JobSummary('Inference', 'done', 'ok', 1.0, url='http://gui/')
    assert notifier.send('https://hooks/x', summary) == (
        'Notification posted to the webhook.'
    )
    assert posts[0][0] == 'https://hooks/x' and posts[0][1]['job'] == 'Inference'
    assert posts[0][1]['url'] == 'http://gui/'

    results = []
    done = threading.Event()

    def on_result(ok, message):
        results.append((ok, message))
        done.set()

    notifier.send_in_background('https://hooks/bad', summary, on_result).join(5)
    assert done.wait(5) and results[-1][0] is False
    assert 'failed' in results[-1][1] and 'HTTP 500' in results[-1][1]
    done.clear()
    notifier.send_in_background('https://hooks/ok', summary, on_result).join(5)
    assert results[-1] == (True, 'Notification posted to the webhook.')
    done.clear()
    notifier.send_in_background('garbage', summary, on_result).join(5)
    assert results[-1][0] is False and 'email address or a webhook' in results[-1][1]
