from __future__ import annotations

from typing import Any

import click
import duckdb

from llm_refinery import __version__
from llm_refinery.commands.agent_eval import agent_eval_command
from llm_refinery.commands.dabstep import dabstep_command
from llm_refinery.commands.http_load import http_load_command
from llm_refinery.commands.lm_eval import lm_eval_command
from llm_refinery.commands.quality_compare import quality_compare_command
from llm_refinery.commands.reporting import compare_command, reparse_command, report_command
from llm_refinery.commands.suite import suite_command
from llm_refinery.commands.sweep import bench_command, init_command, plan_command, server_command
from llm_refinery.commands.system import backfill_system_metadata_command
from llm_refinery.commands.targets import target_command
from llm_refinery.core.config import ConfigError
from llm_refinery.utils.terminal import sanitize_terminal_text


class ErrorHandlingGroup(click.Group):
    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (click.exceptions.Exit, click.Abort):
            raise
        except click.ClickException as exc:
            exc.message = sanitize_terminal_text(exc.message)
            raise
        except (ConfigError, OSError, IndexError, RuntimeError, duckdb.Error) as exc:
            raise click.ClickException(sanitize_terminal_text(str(exc))) from exc
        except KeyboardInterrupt as exc:
            raise click.Abort() from exc


@click.group(
    cls=ErrorHandlingGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, "--version", prog_name="llm-refinery")
def main() -> None:
    """Local LLM benchmarking and serving workflow harness."""


main.add_command(agent_eval_command)
main.add_command(backfill_system_metadata_command)
main.add_command(bench_command)
main.add_command(compare_command)
main.add_command(dabstep_command)
main.add_command(http_load_command)
main.add_command(init_command)
main.add_command(lm_eval_command)
main.add_command(plan_command)
main.add_command(quality_compare_command)
main.add_command(reparse_command)
main.add_command(report_command)
main.add_command(server_command)
main.add_command(suite_command)
main.add_command(target_command)


if __name__ == "__main__":
    main()
