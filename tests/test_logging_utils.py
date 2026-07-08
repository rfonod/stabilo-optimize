# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import io
import logging
import sys
from pathlib import Path

import pytest

from stabilo_optimize.utils.logging_utils import STABILO_LOG_LEVELS, configure_stabilo_logging, tee_stdout_to_file


@pytest.fixture
def fake_stabilo_logger():
    """A throwaway 'stabilo.*' logger with its own StreamHandler, cleaned up afterward."""
    logger = logging.getLogger('stabilo.faketest')
    handler = logging.StreamHandler(io.StringIO())
    logger.addHandler(handler)
    try:
        yield logger, handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)


def test_tee_stdout_to_file_none_is_a_noop(capsys):
    original_stdout = sys.stdout

    with tee_stdout_to_file(None):
        assert sys.stdout is original_stdout
        print("hello")

    assert sys.stdout is original_stdout
    assert capsys.readouterr().out == "hello\n"


def test_tee_stdout_to_file_writes_console_and_file(tmp_path, capsys):
    log_path = tmp_path / 'logs' / 'run.log'  # parent directory does not exist yet

    with tee_stdout_to_file(log_path):
        print("benchmark started")
        print("\033[91mWarning: something failed\033[0m")

    captured = capsys.readouterr().out
    assert "benchmark started" in captured
    assert "\033[91mWarning: something failed\033[0m" in captured  # console keeps colors

    log_contents = log_path.read_text()
    assert "benchmark started" in log_contents
    assert "Warning: something failed" in log_contents
    assert "\033[91m" not in log_contents and "\033[0m" not in log_contents  # ANSI stripped in file


def test_tee_stdout_to_file_appends_across_calls(tmp_path):
    log_path = tmp_path / 'run.log'

    with tee_stdout_to_file(log_path):
        print("first run")
    with tee_stdout_to_file(log_path):
        print("second run")

    log_contents = log_path.read_text()
    assert "first run" in log_contents
    assert "second run" in log_contents


def test_tee_stdout_to_file_restores_stdout_on_exception(tmp_path):
    original_stdout = sys.stdout
    log_path = tmp_path / 'run.log'

    try:
        with tee_stdout_to_file(log_path):
            raise ValueError("boom")
    except ValueError:
        pass

    assert sys.stdout is original_stdout


def test_tee_stdout_to_file_reports_the_resolved_absolute_path(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with tee_stdout_to_file(Path('relative.log')):
        pass

    assert str((tmp_path / 'relative.log').resolve()) in capsys.readouterr().out


@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_configure_stabilo_logging_sets_the_expected_level(fake_stabilo_logger, verbosity):
    logger, _ = fake_stabilo_logger

    configure_stabilo_logging(verbosity)

    assert logger.level == STABILO_LOG_LEVELS[verbosity]


def test_configure_stabilo_logging_verbosity_0_is_fully_silent(fake_stabilo_logger):
    logger, _ = fake_stabilo_logger

    configure_stabilo_logging(0)

    assert not logger.isEnabledFor(logging.CRITICAL)


def test_configure_stabilo_logging_verbosity_2_allows_warnings(fake_stabilo_logger):
    logger, _ = fake_stabilo_logger

    configure_stabilo_logging(2)

    assert logger.isEnabledFor(logging.WARNING)
    assert not logger.isEnabledFor(logging.INFO)


def test_configure_stabilo_logging_redirects_stream_handler_to_current_stdout(fake_stabilo_logger):
    logger, handler = fake_stabilo_logger
    assert handler.stream is not sys.stdout  # starts pointed at its own throwaway StringIO

    configure_stabilo_logging(3)

    assert handler.stream is sys.stdout


def test_configure_stabilo_logging_does_not_touch_unrelated_loggers():
    other_logger = logging.getLogger('some_other_package')
    other_logger.setLevel(logging.DEBUG)

    configure_stabilo_logging(0)

    assert other_logger.level == logging.DEBUG
