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
# 11-24-2026

"""
Command-line front end of the JSON job interface.

``rapidtools-job`` (or ``python -m rapidtools.jobs``) runs and checks job
files and prints the interface definition::

    rapidtools-job run job.json [--secrets secrets.json] [--result out.json]
    rapidtools-job validate job.json
    rapidtools-job schema [job|result|secrets]
    rapidtools-job steps [--format json|rst]

Exit status of ``run``: 0 when the job finished, 1 when a step failed, 2 when
the job file is invalid, 3 when the job was canceled (Ctrl+C or SIGTERM).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from typing import Any

from .reference import (
    describe_steps,
    job_schema,
    result_schema,
    secrets_schema,
    steps_rst,
)
from .runner import JobRunner
from .schema import JobError

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INVALID = 2
EXIT_CANCELED = 3


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog='rapidtools-job',
        description='Run, check and describe rapidtools JSON job files.',
    )
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser('run', help='Run a job file.')
    run.add_argument('job', help='Path of the job file.')
    run.add_argument(
        '--secrets',
        help='Secrets file holding the values of {"$secret": ...} references.',
    )
    run.add_argument(
        '--result',
        help='Where to write the result file (default: <job>.result.json).',
    )
    run.add_argument(
        '--log-level',
        default='INFO',
        help='Console log level (default: INFO).',
    )

    validate = commands.add_parser(
        'validate', help='Check a job file without running it.'
    )
    validate.add_argument('job', help='Path of the job file.')
    validate.add_argument('--secrets', help='Secrets file.')
    validate.add_argument(
        '--require-secrets',
        action='store_true',
        help='Fail when a secret reference cannot be resolved on this machine.',
    )

    schema = commands.add_parser('schema', help='Print a JSON Schema.')
    schema.add_argument(
        'document',
        nargs='?',
        default='job',
        choices=('job', 'result', 'secrets'),
        help='Which document to describe (default: job).',
    )

    steps = commands.add_parser('steps', help='Print the catalog of step types.')
    steps.add_argument(
        '--format',
        default='json',
        choices=('json', 'rst'),
        help='Output format (default: json).',
    )
    return parser


def _print_json(document: Any) -> None:
    """Write a JSON document to standard output."""
    print(json.dumps(document, indent=2, ensure_ascii=False))


def _run(args: argparse.Namespace) -> int:
    """Run a job file and translate its outcome into an exit status."""
    from rapidtools.config import configure_logging

    configure_logging(args.log_level.upper())
    runner = JobRunner()

    def stop(_signum: int, _frame: Any) -> None:
        """Cancel cooperatively on the first signal, hard on the second."""
        print('Canceling: the current batch will finish first.', file=sys.stderr)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        runner.cancel()

    # Signal handlers can only be installed from the main thread.
    handled = threading.current_thread() is threading.main_thread()
    previous = {}
    if handled:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
    try:
        result = runner.run(args.job, secrets=args.secrets, result_path=args.result)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    stream = sys.stdout if result.ok else sys.stderr
    print(f'[{result.status}] {result.message}', file=stream)
    if result.result_file is not None:
        print(f'Result file: {result.result_file}', file=stream)
    if result.ok:
        return EXIT_OK
    if result.status == 'canceled':
        return EXIT_CANCELED
    started = any(step.started_at for step in result.steps)
    if isinstance(result.exception, JobError) and not started:
        return EXIT_INVALID
    return EXIT_FAILED


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of ``rapidtools-job``.

    Args:
        argv (list[str] | None): Arguments; defaults to ``sys.argv[1:]``.

    Returns:
        int: The process exit status.

    Example:
        >>> main(['schema', 'secrets'])  # doctest: +ELLIPSIS
        {
          "$schema": "https://json-schema.org/draft/2020-12/schema",
        ...
        0
    """
    args = _parser().parse_args(argv)
    try:
        if args.command == 'run':
            return _run(args)
        if args.command == 'validate':
            job = JobRunner().validate(
                args.job,
                secrets=args.secrets,
                allow_missing_secrets=not args.require_secrets,
            )
            print(f'{args.job}: valid ({len(job.steps)} step(s)).')
        elif args.command == 'schema':
            _print_json(
                {
                    'job': job_schema,
                    'result': result_schema,
                    'secrets': secrets_schema,
                }[args.document]()
            )
        elif args.format == 'rst':
            print(steps_rst(), end='')
        else:
            _print_json(describe_steps())
    except JobError as exc:
        print(f'Invalid job: {exc.describe()}', file=sys.stderr)
        return EXIT_INVALID
    return EXIT_OK


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
