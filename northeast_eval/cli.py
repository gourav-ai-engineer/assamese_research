from pathlib import Path
import json
import typer
from rich import print
from .schema import read_jsonl, validate_rows
from .generation import generate_file
from .reporting import score_file

app = typer.Typer(help="Reproducible small-LLM evaluation for Northeast Indian languages.")

@app.command()
def validate(input: str = typer.Option(..., help="Input JSONL file"),
             require_reference: bool = typer.Option(False, help="Require reference text")):
    rows = read_jsonl(input)
    errors = validate_rows(rows, require_reference=require_reference)
    if errors:
        for error in errors:
            print(f"[red]ERROR[/red] {error}")
        raise typer.Exit(1)
    print(f"[green]Valid[/green]: {len(rows)} rows in {input}")

@app.command()
def generate(input: str, output: str, model: str = typer.Option(..., "--model"),
             task: str = "translation", condition: str = "zero_shot",
             limit: int | None = None, max_new_tokens: int = 256,
             revision: str | None = None):
    count = generate_file(input, output, model, task, condition, limit, max_new_tokens, revision)
    print(f"Wrote {count} predictions to {output}")

@app.command()
def score(predictions: str, output: str = "results/metrics.json"):
    result = score_file(predictions, output)
    print(f"Wrote {len(result['aggregates'])} aggregate rows to {output}")

if __name__ == "__main__":
    app()
