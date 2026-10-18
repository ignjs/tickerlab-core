from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from tickerlab.analysis.engine import analyze_ticker, fuzzy_suggest_symbol, normalize_symbol, validate_yahoo_symbol
from tickerlab.config.models import load_config
from tickerlab.features.builder import build_features
from tickerlab.ingest.service import get_realtime_quote, load_from_csv, load_from_yahoo, load_synthetic
from tickerlab.labels.builder import build_bar_labels, build_event_labels
from tickerlab.models.baseline import fit_logistic
from tickerlab.patterns.structural import scan_structural
from tickerlab.pivots.detector import detect_pivots
from tickerlab.reporting.reporter import (
    create_quote_table,
    print_accessible_report,
    print_metrics,
    print_quote,
)
from tickerlab.stats.metrics import compute_signal_metrics, stability_score
from tickerlab.storage.io import dataset_path, load_frame, save_frame
from tickerlab.utils.dataset import as_xy, event_dataset
from tickerlab.validation.walkforward import walk_forward_indices

app = typer.Typer(help="TickerLab CLI")
features_app = typer.Typer(help="Feature commands")
patterns_app = typer.Typer(help="Pattern commands")
labels_app = typer.Typer(help="Label commands")
validate_app = typer.Typer(help="Validation commands")
score_app = typer.Typer(help="Scoring commands")
report_app = typer.Typer(help="Reporting commands")

app.add_typer(features_app, name="features")
app.add_typer(patterns_app, name="patterns")
app.add_typer(labels_app, name="labels")
app.add_typer(validate_app, name="validate")
app.add_typer(score_app, name="score")
app.add_typer(report_app, name="report")


def _save_core(cache_dir: Path, name: str, symbol: str, timeframe: str, df: pl.DataFrame, export: str = "parquet") -> Path:
    path = dataset_path(cache_dir, name, symbol, timeframe, export)
    save_frame(df, path, export=export)
    return path

@app.command()
def ingest(
    symbol: str = "AAPL",
    timeframe: str = "1d",
    source: str = "synthetic",
    input_path: str | None = None,
    config: str | None = None,
    cache_dir: str = "data",
    export: str = "parquet",
    seed: int | None = None,
    period: str = "max",
):
    cfg = load_config(config)
    source = source or cfg.ingest.source
    input_path = input_path or cfg.ingest.input_path
    seed = seed if seed is not None else cfg.ingest.seed
    period = period or cfg.ingest.period
    
    try:
        bars = analyze_ticker.run_ingest(
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            input_path=input_path,
            cache_dir=Path(cache_dir),
            export=export,
            seed=seed,
            period=period
        )
        typer.echo(f"ingested={bars.height}")
    except Exception as e:
        raise typer.BadParameter(str(e))


@features_app.command("build")
def features_build(symbol: str = "AAPL", timeframe: str = "1d", cache_dir: str = "data", export: str = "parquet"):
    try:
        feats = analyze_ticker.run_features(
            symbol=symbol,
            timeframe=timeframe,
            cache_dir=Path(cache_dir),
            export=export
        )
        typer.echo(f"features={feats.height}")
    except Exception as e:
        raise typer.BadParameter(str(e))


@patterns_app.command("scan")
def patterns_scan(symbol: str = "AAPL", timeframe: str = "1d", engine: str = "structural", cache_dir: str = "data", export: str = "parquet"):
    try:
        piv, events = analyze_ticker.run_patterns(
            symbol=symbol,
            timeframe=timeframe,
            engine=engine,
            cache_dir=Path(cache_dir),
            export=export
        )
        typer.echo(f"events={events.height}")
    except Exception as e:
        raise typer.BadParameter(str(e))


@labels_app.command("build")
def labels_build(symbol: str = "AAPL", timeframe: str = "1d", horizons: str = "1,3,5,10,20,60", cache_dir: str = "data", export: str = "parquet"):
    try:
        lab_bar, lab_event = analyze_ticker.run_labels(
            symbol=symbol,
            timeframe=timeframe,
            horizons=horizons,
            cache_dir=Path(cache_dir),
            export=export
        )
        typer.echo(f"labels_bar={lab_bar.height}")
    except Exception as e:
        raise typer.BadParameter(str(e))


@validate_app.command("walkforward")
def validate_walkforward(
    symbol: str = "AAPL",
    timeframe: str = "1d",
    model: str = "logistic",
    mode: str = "rolling",
    horizon: int = 5,
    cache_dir: str = "data",
    export: str = "parquet",
):
    try:
        result = analyze_ticker.run_validate(
            symbol=symbol,
            timeframe=timeframe,
            model=model,
            mode=mode,
            horizon=horizon,
            cache_dir=Path(cache_dir),
            export=export
        )
        print_metrics(result["summary"])
        typer.echo(f"validation={result['path']} rows={result['rows']}")
    except Exception as e:
        raise typer.BadParameter(str(e))


@score_app.command("now")
def score_now(symbol: str = "AAPL", timeframe: str = "1d", horizon: int = 5, cache_dir: str = "data"):
    feats = load_frame(dataset_path(Path(cache_dir), "features", symbol, timeframe))
    events = load_frame(dataset_path(Path(cache_dir), "events", symbol, timeframe))
    labels_event = load_frame(dataset_path(Path(cache_dir), "labels_event", symbol, timeframe))
    ds = event_dataset(feats, events, labels_event, horizon=horizon)
    X, y, _, sub = as_xy(ds)
    model = fit_logistic(X, y)
    latest = sub.select([c for c in sub.columns if c in ["ret_1", "ret_5", "ema_ratio", "range_pct", "zscore_20", "vol_ratio_20", "rsi_14", "drawdown_20", "realized_vol_20"]]).tail(1).to_numpy()
    p = float(model.predict_proba(latest)[:, 1][0])
    typer.echo(f"score_now={p:.6f}")


@report_app.command("edge")
def report_edge(symbol: str = "AAPL", timeframe: str = "1d", cache_dir: str = "data", format: str = "table"):
    path = dataset_path(Path(cache_dir), "validation_summary", symbol, timeframe)
    if not path.exists():
        typer.secho(
            f"\n⚠️  No se encontraron datos de validación para '{symbol}' (timeframe: {timeframe}).\n",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "Esto ocurre porque aún no se ha ejecutado el pipeline completo para este activo.\n"
            "Para generar el reporte, puedes:\n\n"
            f"  1. Ejecutar el análisis automático completo:\n"
            f"     tickerlab analyze {symbol}\n\n"
            f"  2. O ejecutar el pipeline paso a paso:\n"
            f"     tickerlab ingest --symbol {symbol} --source yahoo\n"
            f"     tickerlab features build --symbol {symbol}\n"
            f"     tickerlab patterns scan --symbol {symbol}\n"
            f"     tickerlab labels build --symbol {symbol}\n"
            f"     tickerlab validate walkforward --symbol {symbol}\n"
            f"     tickerlab report edge --symbol {symbol}\n",
            fg=typer.colors.CYAN,
            err=True,
        )
        raise typer.Exit(code=1)

    summary = load_frame(path)
    if format == "table":
        print_metrics(summary.row(0, named=True))
    elif format in {"csv", "json", "parquet"}:
        out = dataset_path(Path(cache_dir), "report_edge", symbol, timeframe, export=format)
        save_frame(summary, out, export=format)
        typer.echo(f"exported={out}")
    else:
        raise typer.BadParameter("format debe ser table/csv/json/parquet")


@app.command("quote")
def quote(
    ticker: str | None = typer.Argument(None, help="Símbolo del ticker (ej. AAPL, IONQ, IAM.SN, IAM)"),
    symbol: str | None = typer.Option(None, "--symbol", "-s", help="Símbolo del ticker"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Monitorear y refrescar en tiempo real"),
    interval: int = typer.Option(5, "--interval", "-i", help="Intervalo en segundos para modo watch"),
):
    """Muestra la cotización en tiempo real para activos de EE.UU. o Chile."""
    raw_sym = ticker or symbol or "AAPL"
    sym, note = normalize_symbol(raw_sym)
    import time
    from rich.live import Live

    if not watch:
        try:
            if note:
                typer.secho(f"ℹ️  {note}", fg=typer.colors.CYAN)
            q = get_realtime_quote(sym)
            print_quote(q)
        except Exception as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
    else:
        typer.echo(f"Monitoreando {sym} cada {interval}s (presiona Ctrl+C para salir)...")
        try:
            with Live(create_quote_table(get_realtime_quote(sym)), refresh_per_second=1) as live:
                while True:
                    time.sleep(interval)
                    live.update(create_quote_table(get_realtime_quote(sym)))
        except KeyboardInterrupt:
            typer.echo("\nMonitoreo finalizado.")
        except Exception as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)


@app.command("analyze")
def analyze(
    ticker: str | None = typer.Argument(None, help="Símbolo del ticker a analizar (ej. AAPL, IONQ, IAM.SN, IAM)"),
    symbol: str | None = typer.Option(None, "--symbol", "-s", help="Símbolo del ticker"),
    source: str = typer.Option("yahoo", "--source", help="Fuente de datos: yahoo o synthetic"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Temporalidad de las velas (ej. 1d)"),
    period: str = typer.Option("max", "--period", help="Período histórico a descargar (ej. max, 5y, 2y)"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Muestra glosario explicativo para principiantes"),
    cache_dir: str = typer.Option("data", "--cache-dir", help="Directorio de almacenamiento de datos"),
    seed: int | None = typer.Option(None, "--seed", help="Semilla manual si source=synthetic"),
):
    """
    Ejecuta un análisis cuantitativo E2E 'One-Click' y muestra un reporte comprensible y visual.
    """
    from rich.console import Console

    console = Console()
    raw_sym = ticker or symbol or "AAPL"

    with console.status(
        f"[bold green]Analizando {raw_sym.upper()} (ingesta, indicadores, patrones y validación)...[/bold green]"
    ):
        try:
            res = analyze_ticker(
                raw_sym,
                timeframe=timeframe,
                source=source,
                cache_dir=Path(cache_dir),
                period=period,
                seed=seed,
            )
        except ConnectionError:
            typer.secho(
                "\n⚠️  Error de conexión a Internet.\n"
                "No se pudo conectar a Yahoo Finance. Verifica tu conexión de red e intenta nuevamente.\n"
                "Si no tienes conexión, puedes usar datos sintéticos:\n\n"
                f"  tickerlab analyze {raw_sym} --source synthetic\n",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(code=1)
        except TimeoutError:
            typer.secho(
                "\n⚠️  Tiempo de espera agotado.\n"
                "Yahoo Finance no respondió a tiempo. Intenta nuevamente en unos segundos.\n",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(code=1)
        except ValueError as e:
            error_msg = str(e)
            suggestion = fuzzy_suggest_symbol(raw_sym)
            if suggestion:
                error_msg += f"\n\n💡 ¿Quisiste decir '{suggestion}'?"
                if suggestion.endswith(".SN"):
                    error_msg += " (En Chile las acciones llevan el sufijo .SN)"
            typer.secho(f"\n❌ {error_msg}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "network" in error_str or "unreachable" in error_str:
                typer.secho(
                    f"\n⚠️  Error de conexión: {e}\n"
                    "Verifica tu conexión de red o usa datos sintéticos con: --source synthetic\n",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
            else:
                typer.secho(f"\n❌ Error durante el análisis: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    print_accessible_report(
        symbol=res["symbol"],
        quote=res["quote"],
        summary=res["summary"],
        score_now=res["score_now"],
        events_count=res["events_count"],
        explain=explain,
        message=res["message"],
        note=res["note"],
    )
