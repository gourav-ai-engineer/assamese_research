import typer
from rich import print

app = typer.Typer(help="Small LLM evaluation for Northeast Indian languages", no_args_is_help=True)
EXP, MODELS = "configs/experiment.yaml", "configs/models.yaml"


@app.command()
def prepare(raw_dir: str = "data/raw/indic-gen-bench", out_dir: str = "data/processed",
            config: str = EXP, download: bool = True):
    """Download IndicGenBench and write frozen eval/exemplar JSONL files."""
    from .data import download_indicgenbench
    from .data import prepare as _prepare
    from .runner import load_config
    cfg = load_config(config, MODELS)
    if download:
        download_indicgenbench(raw_dir)
    s = cfg["sampling"]
    m = _prepare(raw_dir, out_dir, cfg["languages"], cfg["tasks"], s["translation"], s["qa"],
                 s["summarization"], s["seed"], s["exemplar_pool"])
    for name, c in m["files"].items():
        print(f"{name}: eval={c['eval']} exemplars={c['exemplars']} (available {c['available']})")


@app.command("prepare-in22")
def prepare_in22(out_dir: str = "data/processed", n: int = 500, seed: int = 20261010):
    """Optional: Meitei-script Manipuri from IN22-Gen for the script analysis (needs the in22 extra)."""
    from pathlib import Path

    from .data import _write_jsonl, in22_rows
    rows = in22_rows("mni_Mtei")
    for a, b in (("en", "mni_Mtei"), ("mni_Mtei", "en")):
        sel = [r for r in rows if r["source_lang"] == a][:n]
        _write_jsonl(Path(out_dir) / "eval" / f"translation_{a}-{b}.jsonl", sel)
        print(f"translation_{a}-{b}: {len(sel)}")


@app.command()
def validate(input: str = typer.Option(..., "--input"), require_reference: bool = False):
    from .schema import read_jsonl, validate_rows
    rows = read_jsonl(input)
    errors = validate_rows(rows, require_reference)
    if errors:
        for e in errors:
            print(f"[red]ERROR[/red] {e}")
        raise typer.Exit(1)
    print(f"[green]Valid[/green]: {len(rows)} rows")


@app.command()
def run(model: str | None = typer.Option(None, help="Substring filter on model id"), limit: int | None = None,
        data_dir: str = "data/processed", out_dir: str = "results/predictions", config: str = EXP,
        models: str = MODELS):
    """Run the full grid (resumable) for all or one model."""
    from .runner import load_config, run_grid
    run_grid(load_config(config, models), data_dir, out_dir, model, limit)


@app.command()
def generate(input: str = typer.Option(..., "--input"), output: str = typer.Option(..., "--output"),
             model: str = typer.Option(..., "--model"), task: str = "translation",
             condition: str = "zero_shot", exemplars: str | None = None, k: int = 3, limit: int | None = None,
             max_new_tokens: int = 128, revision: str | None = None, family: str | None = None):
    """Generate predictions for a single file."""
    from .generation import generate_file
    n = generate_file(input, output, model, task, condition, limit, max_new_tokens, revision, exemplars, k, family)
    print(f"Wrote {n} predictions to {output}")


@app.command("mt-baseline")
def mt_baseline(backend: str = "indictrans2", limit: int | None = None, data_dir: str = "data/processed",
                out_dir: str = "results/predictions"):
    """Score a dedicated MT system (IndicTrans2/NLLB) on the same FLORES items."""
    from .runner import load_config, run_mt_baseline
    run_mt_baseline(load_config(EXP, MODELS), data_dir, out_dir, backend, limit)


@app.command()
def fertility(data_dir: str = "data/processed", output: str = "results/fertility.csv"):
    """Tokens-per-word of every model tokenizer on each language."""
    from .fertility import run as _run
    from .runner import load_config
    cfg = load_config(EXP, MODELS)
    rows = _run([m["id"] for m in cfg["models"]], cfg["languages"], f"{data_dir}/eval", output)
    print(f"Wrote {len(rows)} rows to {output}")


@app.command()
def score(predictions: str = typer.Option(None, "--predictions", help="Single JSONL file"),
          pred_dir: str = "results/predictions", output: str = typer.Option(None, "--output"),
          resamples: int = 1000):
    """Score one file (JSON) or a whole predictions directory (CSV)."""
    from .reporting import score_dir, score_file
    if predictions:
        r = score_file(predictions, output or "results/metrics.json")
        print(f"Wrote {len(r['aggregates'])} aggregate rows to {output}")
    else:
        rows = score_dir(pred_dir, output or "results/scores.csv", resamples)
        print(f"Wrote {len(rows)} score rows to {output or 'results/scores.csv'}")


@app.command()
def report(scores: str = "results/scores.csv", fertility_csv: str = "results/fertility.csv",
           out_dir: str = "results"):
    """Tables (Markdown + LaTeX) and figures from scores.csv."""
    from .reporting import make_plots, read_csv, write_tables
    rows = read_csv(scores)
    write_tables(rows, f"{out_dir}/tables")
    make_plots(rows, f"{out_dir}/figures", fertility_csv)
    print(f"Tables in {out_dir}/tables, figures in {out_dir}/figures")


@app.command("sample-errors")
def sample_errors(pred_dir: str = "results/predictions", output: str = "results/annotation/sample.csv",
                  per_language: int = 50):
    """Blind sample for manual error-taxonomy annotation."""
    from .reporting import sample_for_annotation
    print(f"Wrote {sample_for_annotation(pred_dir, output, per_language)} rows to {output}")


if __name__ == "__main__":
    app()
