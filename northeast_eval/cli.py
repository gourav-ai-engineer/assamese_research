import typer
from rich import print
from .schema import read_jsonl, validate_rows
from .generation import generate_file
from .reporting import score_file
app=typer.Typer(help="Small LLM evaluation for Northeast Indian languages")
@app.command()
def validate(input: str=typer.Option(...), require_reference: bool=False):
    rows=read_jsonl(input); errors=validate_rows(rows,require_reference)
    if errors:
        for e in errors: print(f"[red]ERROR[/red] {e}")
        raise typer.Exit(1)
    print(f"[green]Valid[/green]: {len(rows)} rows")
@app.command()
def generate(input: str, output: str, model: str=typer.Option(...,"--model"),
             task: str="translation", condition: str="zero_shot", limit: int|None=None,
             max_new_tokens: int=256, revision: str|None=None):
    n=generate_file(input,output,model,task,condition,limit,max_new_tokens,revision)
    print(f"Wrote {n} predictions to {output}")
@app.command()
def score(predictions: str, output: str="results/metrics.json"):
    r=score_file(predictions,output)
    print(f"Wrote {len(r['aggregates'])} aggregate rows to {output}")
if __name__=="__main__": app()
