import os
import sys
from collections.abc import Iterable
from typing import TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")

# Force rich to use the un-redirected low-level stdout stream
unsuppressed_console = Console(file=sys.__stdout__, force_terminal=True)


def track(
    sequence: Iterable[T] | range,
    description: str = "Processing...",
    total: int | None = None,
    transient: bool = False,
    bar_width: int = 25,
    disable: bool = False,
) -> Iterable[T]:
    # Disable progress bar if explicitly set, or during pytest runs
    if disable or "PYTEST_CURRENT_TEST" in os.environ:
        yield from sequence
        return

    if total is None and hasattr(sequence, "__len__"):
        total = len(sequence)

    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=bar_width),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ]

    with Progress(
        *columns, console=unsuppressed_console, transient=transient
    ) as progress:
        task_id = progress.add_task(description, total=total)
        for item in sequence:
            yield item
            progress.advance(task_id, 1)
            progress.refresh()  # Force instant render sync


class RichProgress:
    """Compact manually updated progress bar object for background processes."""

    def __init__(
        self,
        description: str = "Processing...",
        total: int | None = None,
        transient: bool = False,
        bar_width: int = 25,
    ):
        self.description = description
        self.total = total
        self.transient = transient
        self.bar_width = bar_width
        self.progress: Progress | None = None
        self.task_id: Any = None

    def __enter__(self):
        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=self.bar_width),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim cyan]{task.fields[metrics]}"),
        ]
        self.progress = Progress(*columns, console=console, transient=self.transient)
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.description, total=self.total, metrics=""
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()

    def update(
        self,
        advance: int = 1,
        description: str | None = None,
        metrics: str | None = None,
    ):
        if self.progress and self.task_id is not None:
            kwargs: dict[str, Any] = {}
            if description is not None:
                kwargs["description"] = description
            if metrics is not None:
                kwargs["metrics"] = metrics

            self.progress.advance(self.task_id, advance=advance)
            if kwargs:
                self.progress.update(self.task_id, **kwargs)
