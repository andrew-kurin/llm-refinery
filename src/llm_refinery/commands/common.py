from __future__ import annotations

import shutil

import click


def parse_lm_eval_limit(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise click.BadParameter("must be a positive integer or 'all'") from exc
    if parsed <= 0:
        raise click.BadParameter("must be a positive integer or 'all'")
    return parsed


def table(rows: list[tuple[object, ...]]) -> str:
    widths = [0] * max(len(row) for row in rows)
    rendered = [[str(cell) for cell in row] for row in rows]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    terminal_width = shutil.get_terminal_size((120, 20)).columns
    output_lines: list[str] = []
    for row_index, row in enumerate(rendered):
        cells = []
        for index, cell in enumerate(row):
            width = widths[index]
            max_width = max(12, min(width, terminal_width // len(widths)))
            if len(cell) > max_width:
                cell = cell[: max_width - 1] + "…"
            cells.append(cell.ljust(max_width))
        output_lines.append("  ".join(cells).rstrip())
        if row_index == 0:
            separator_cells = ["-" * min(width, terminal_width // len(widths)) for width in widths]
            output_lines.append("  ".join(separator_cells))
    return "\n".join(output_lines)
