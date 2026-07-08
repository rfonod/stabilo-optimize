# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Optional file logging: duplicate console output to a user-specified log file, and align
stabilo's own internal logging with this project's --verbosity level.

Status messages and warnings/errors go through print()/tqdm.write() (stdout by default);
tqdm's progress bars render to stderr. Teeing only stdout therefore captures every message
the user sees printed, without the noise of the constantly-rewriting progress bars.

Separately, stabilo (the library) configures its own `stabilo.stabilo` logger at import
time, independent of anything in this project (its own logging.StreamHandler, default
level INFO) - so its messages (e.g. "WARNING - Not enough points to estimate the
transformation matrix.") bypass both stabilo-optimize's --verbosity-gated print()s and the
tee above entirely. configure_stabilo_logging reins that in.
"""

import logging
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, TextIO, Union

ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')

# stabilo-optimize --verbosity -> minimum level let through on stabilo's own logger(s).
# 0 (quiet) is fully silent, including stabilo's warnings/errors; higher levels progressively
# match where stabilo-optimize's own output gets more detailed (see benchmark.py's parse_cli_args).
STABILO_LOG_LEVELS = {
    0: logging.CRITICAL + 1,
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
}


def configure_stabilo_logging(verbosity: int) -> None:
    """
    Align stabilo's own internal logging with stabilo-optimize's --verbosity level, and
    route it through the current sys.stdout so tee_stdout_to_file (if active) captures it.
    """
    level = STABILO_LOG_LEVELS.get(verbosity, logging.INFO)
    for name in list(logging.root.manager.loggerDict):
        if name == 'stabilo' or name.startswith('stabilo.'):
            logger = logging.getLogger(name)
            logger.setLevel(level)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setStream(sys.stdout)


class _TeeStream:
    """Duplicates writes to the original stream and a log file (ANSI color codes stripped)."""

    def __init__(self, original: TextIO, log_file: TextIO) -> None:
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._log_file.write(ANSI_ESCAPE_RE.sub('', data))
        return self._original.write(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextmanager
def tee_stdout_to_file(log_filepath: Optional[Union[str, Path]]) -> Iterator[None]:
    """
    Duplicate stdout to `log_filepath` (appended to, ANSI color codes stripped) for the
    duration of the context. A no-op if `log_filepath` is None.
    """
    if log_filepath is None:
        yield
        return

    log_filepath = Path(log_filepath).resolve()
    log_filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving logs to: {log_filepath}")

    with open(log_filepath, 'a') as log_file:
        log_file.write(f"\n{'='*80}\nRun started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n")
        original_stdout = sys.stdout
        sys.stdout = _TeeStream(original_stdout, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
