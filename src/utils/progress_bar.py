import os
import sys
from collections.abc import Iterable
from typing import Any, TypeVar

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

unsuppressed_console = Console(file=sys.__stdout__, force_terminal=True)


def track(
    sequence: Iterable[T] | range,
    description: str = "Processing...",
    total: int | None = None,
    transient: bool = False,
    bar_width: int = 25,
    disable: bool = False,
    complete_on_early_exit: bool = False,
) -> Iterable[T]:
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
        try:
            for item in sequence:
                yield item
                progress.advance(task_id, 1)
                progress.refresh()
        finally:
            # If standard loop did not complete (e.g. break or exception)
            task = progress.tasks[task_id]
            if not task.finished:
                if complete_on_early_exit and task.total is not None:
                    # Snap progress to 100%
                    progress.update(task_id, completed=task.total)
                else:
                    # Alternatively, shrink total to reflect stopped state accurately
                    progress.update(
                        task_id,
                        total=task.completed,
                        description=f"{description} (Stopped)",
                    )


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
        self.progress = Progress(
            *columns, console=unsuppressed_console, transient=self.transient
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.description, total=self.total, metrics=""
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            # If context exits with an exception, mark it as stopped/interrupted visually
            if exc_type is not None and self.task_id is not None:
                task = self.progress.tasks[self.task_id]
                self.progress.update(
                    self.task_id,
                    total=task.completed,
                    description=f"[bold red]{self.description} (Failed)",
                )
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

    def complete(self, description: str | None = None):
        """Force the progress bar to instantly fill to 100% on early exit."""
        if self.progress and self.task_id is not None:
            task = self.progress.tasks[self.task_id]
            target_total = task.total if task.total is not None else task.completed
            desc = description if description is not None else f"{task.description} (Done)"
            self.progress.update(self.task_id, completed=target_total, description=desc)

    def stop_early(self, description: str | None = None):
        """Set total = completed so the bar locks at current progress without errors."""
        if self.progress and self.task_id is not None:
            task = self.progress.tasks[self.task_id]
            desc = description if description is not None else f"{task.description} (Halted)"
            self.progress.update(self.task_id, total=task.completed, description=desc)
