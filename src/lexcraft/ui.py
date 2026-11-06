"""Shared terminal UI helpers (Rich)."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

THEME = Theme(
    {
        "brand": "bold #c9a227",
        "muted": "grey58",
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "key": "bold cyan",
    }
)

console = Console(theme=THEME, highlight=False)
err_console = Console(theme=THEME, stderr=True, highlight=False)

LOGO = "[brand]⚖  lexcraft[/brand]"


def header(subtitle: str) -> None:
    err_console.print(f"{LOGO} [muted]· {subtitle}[/muted]")


def kv_table(rows: list[tuple[str, str]], title: str | None = None) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="key", no_wrap=True)
    t.add_column()
    for k, v in rows:
        t.add_row(k, v)
    if title:
        t.title = title
    return t


def panel(renderable, title: str, style: str = "brand") -> Panel:
    return Panel(
        renderable, title=f"[{style}]{title}[/{style}]", border_style="grey35", expand=False
    )


def fail(message: str, hint: str | None = None) -> None:
    err_console.print(f"[err]✗[/err] {message}")
    if hint:
        err_console.print(f"  [muted]{hint}[/muted]")
