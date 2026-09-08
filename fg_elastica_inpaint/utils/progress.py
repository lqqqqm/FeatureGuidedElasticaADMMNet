"""Consistent, compact progress bars for training, validation and evaluation."""
import os
import shutil
import sys

from tqdm import tqdm


def progress_bar(iterable, *, desc, cfg, total=None):
    # The launcher captures the pipe and renders CR redraws itself. A direct
    # noninteractive invocation defaults to epoch summaries instead of redraws.
    enabled = cfg.get("logging", {}).get("progress_bar",
        sys.stderr.isatty() or os.environ.get("FG_ELASTICA_PROGRESS") == "1")
    return tqdm(
        iterable, desc=desc, total=total, disable=not enabled,
        mininterval=1.0, miniters=1, leave=True, unit="batch", ascii=" =",
        ncols=max(40, shutil.get_terminal_size(fallback=(110, 30)).columns - 1),
        bar_format="{desc} |{bar}| {percentage:3.0f}% {n_fmt}/{total_fmt} [{rate_fmt}, ETA {remaining}]{postfix}",
    )
