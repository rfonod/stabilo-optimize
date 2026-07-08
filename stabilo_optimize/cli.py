# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
cli.py - Umbrella command-line interface for Stabilo Optimize.

Dispatches the 'stabilo-optimize' console command to the individual stages. Each
subcommand accepts exactly the same arguments as its underlying module; run
'stabilo-optimize <command> --help' for the full per-command reference.

Usage:
    stabilo-optimize <command> [options]

Commands:
    benchmark    : Run a ground-truth-free benchmark from a .json configuration file (or directory of them).
    plot         : Re-plot existing benchmark results without re-running the benchmark.

Options:
    --help, -h     : Show this help message and exit.
    --version, -V  : Show the installed stabilo-optimize version and exit.

Examples:
  1. Run a benchmark, saving plots and a visualization video, overwriting prior results:
        stabilo-optimize benchmark experiments/sample_experiment/simple_benchmark.json -sp -sv -o

  2. Re-plot existing results for a benchmark:
        stabilo-optimize plot experiments/sample_experiment/simple_benchmark.json
"""

import importlib
import sys

from stabilo_optimize import __version__

# Subcommand -> (module with a main() entry point, one-line description).
# Modules are imported lazily so that 'stabilo-optimize --help' stays fast.
COMMANDS = {
    'benchmark': ('stabilo_optimize.benchmark', 'Run a ground-truth-free benchmark from a .json configuration file'),
    'plot': ('stabilo_optimize.utils.plot', 'Re-plot existing benchmark results without re-running the benchmark'),
}


def build_usage() -> str:
    """Build the top-level usage/help message."""
    lines = [
        'usage: stabilo-optimize <command> [options]',
        '',
        'Stabilo Optimize: ground-truth-free benchmarking for the Stabilo stabilization library.',
        '',
        'commands:',
    ]
    width = max(len(name) for name in COMMANDS)
    lines += [f'  {name:<{width}}  {description}' for name, (_, description) in COMMANDS.items()]
    lines += [
        '',
        "Run 'stabilo-optimize <command> --help' for command-specific options.",
    ]
    return '\n'.join(lines)


def main() -> None:
    """Entry point for the 'stabilo-optimize' console command."""
    argv = sys.argv[1:]

    if not argv or argv[0] in ('-h', '--help'):
        print(build_usage())
        return
    if argv[0] in ('-V', '--version'):
        print(f'stabilo-optimize {__version__}')
        return

    command = argv[0]
    if command not in COMMANDS:
        print(f"stabilo-optimize: error: unknown command '{command}'\n\n{build_usage()}", file=sys.stderr)
        sys.exit(2)

    module = importlib.import_module(COMMANDS[command][0])
    sys.argv = [f'stabilo-optimize {command}'] + argv[1:]
    module.main()


if __name__ == '__main__':
    main()
