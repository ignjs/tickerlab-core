from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from tickerlab.features.builder import build_features
from tickerlab.ingest.service import get_realtime_quote, load_from_yahoo, load_synthetic
from tickerlab.labels.builder import build_bar_labels, build_event_labels
from tickerlab.models.baseline import fit_logistic
from tickerlab.patterns.structural import scan_structural
from tickerlab.pivots.detector import detect_pivots
from tickerlab.stats.metrics import compute_signal_metrics, stability_score
from tickerlab.storage.io import dataset_path, save_frame
from tickerlab.utils.dataset import as_xy, event_dataset
from tickerlab.validation.walkforward import walk_forward_indices

# Diccionario de tickers comunes de la Bolsa de Santiago (BCS) para autocorrección amigable
CHILE_STOCKS: dict[str, str] = {
    "IAM": "IAM.SN",
    "CHILE": "CHILE.SN",
    "BCI": "BCI.SN",
    "BSANTANDER": "BSANTANDER.SN",
    "SAN": "BSANTANDER.SN",
    "SQM": "SQM-B.SN",
    "SQMB": "SQM-B.SN",
    "SQM-B": "SQM-B.SN",
    "FALABELLA": "FALABELLA.SN",
    "CENCOSUD": "CENCOSUD.SN",
    "CAP": "CAP.SN",
    "VAPORES": "VAPORES.SN",
    "ENELAM": "ENELAM.SN",
    "ENELCHILE": "ENELCHILE.SN",
    "COLBUN": "COLBUN.SN",
    "COPEC": "COPEC.SN",
    "ANDINA": "ANDINA-B.SN",
    "ANDINAB": "ANDINA-B.SN",
    "CCU": "CCU.SN",
    "ENTEL": "ENTEL.SN",
    "CMPC": "CMPC.SN",
    "ECL": "ECL.SN",
    "SMU": "SMU.SN",
    "ITAUCL": "ITAUCL.SN",
    "PARAUCO": "PARAUCO.SN",
    "AGUAS": "AGUAS-A.SN",
    "AGUAS-A": "AGUAS-A.SN",
}

# Tickers populares de EE.UU. para sugerencias difusas
US_POPULAR: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "IONQ",
    "AMD", "INTC", "NFLX", "DIS", "BA", "JPM", "V", "MA", "PYPL",
    "KO", "PEP", "WMT", "HD", "MCD", "NKE", "COST", "UNH",
]


def _levenshtein(s1: str, s2: str) -> int:
    """Calcula la distancia de edición Levenshtein entre dos cadenas."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def fuzzy_suggest_symbol(symbol: str, max_distance: int = 3) -> str | None:
    """
    Busca el símbolo más cercano (Levenshtein) entre los tickers conocidos.
    Retorna el mejor candidato o None si ninguno está lo suficientemente cerca.
    """
    clean = symbol.strip().upper()
    # Construir pool de candidatos: claves del dict chileno + tickers US populares
    candidates: dict[str, str] = {}
    for key, mapped in CHILE_STOCKS.items():
        candidates[key] = mapped
    for ticker in US_POPULAR:
        candidates[ticker] = ticker

    best_match: str | None = None
    best_dist = max_distance + 1

    for key, resolved in candidates.items():
        dist = _levenshtein(clean, key)
        if dist < best_dist:
            best_dist = dist
            best_match = resolved

    if best_dist <= max_distance and best_match:
        return best_match
    return None


def normalize_symbol(symbol: str) -> tuple[str, str | None]:
    """
    Normaliza el símbolo y detecta si es una acción chilena sin sufijo .SN.
    Retorna (símbolo_normalizado, mensaje_sugerencia_o_None).
    """
    clean = symbol.strip().upper()
    if clean in CHILE_STOCKS:
        mapped = CHILE_STOCKS[clean]
        return mapped, f"Detectado símbolo chileno '{clean}'. Usando automáticamente '{mapped}' (Bolsa de Santiago)."
    return clean, None


def _serialize_chart_data(bars: pl.DataFrame, pivots: pl.DataFrame) -> tuple[list[dict], list[dict]]:
    serialized_bars = (
        bars.select(
            pl.col("ts").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp"),
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
        .to_dicts()
    )
    serialized_pivots = []
    if not pivots.is_empty():
        serialized_pivots = (
            pivots.select(
                pl.col("ts").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("timestamp"),
                pl.col("pivot_type").alias("type"),
                pl.col("price").alias("value"),
            )
            .to_dicts()
        )
    return serialized_bars, serialized_pivots


def validate_yahoo_symbol(symbol: str, timeframe: str = "1d") -> tuple[bool, str | None]:
    """
    Verifica preventivamente que un símbolo exista en Yahoo Finance antes de
    ejecutar el pipeline completo. Retorna (es_válido, mensaje_error_o_None).
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info
        last_price = getattr(fi, "last_price", None)
        if last_price is None or (isinstance(last_price, float) and last_price != last_price):
            suggestion = fuzzy_suggest_symbol(symbol)
            hint = ""
            if suggestion:
                hint = f"\n💡 ¿Quisiste decir '{suggestion}'?"
                if suggestion.endswith(".SN"):
                    hint += " (En Chile las acciones llevan el sufijo .SN en Yahoo Finance)"
            return False, (
                f"No se encontraron datos para el símbolo '{symbol}' en Yahoo Finance.\n"
                f"Verifica que el ticker sea correcto y esté disponible en los mercados soportados."
                f"{hint}"
            )
        return True, None
    except ImportError:
        return False, "yfinance no está instalado. Ejecuta: pip install yfinance"
    except ConnectionError:
        return False, (
            "⚠️  Error de conexión a Internet.\n"
            "No se pudo conectar a Yahoo Finance. Verifica tu conexión de red e intenta nuevamente.\n"
            "Si no tienes conexión, puedes usar datos sintéticos con: --source synthetic"
        )
    except TimeoutError:
        return False, (
            "⚠️  Tiempo de espera agotado.\n"
            "Yahoo Finance no respondió a tiempo. Intenta nuevamente en unos segundos.\n"
            "Alternativamente, usa datos sintéticos con: --source synthetic"
        )
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "network" in error_str or "unreachable" in error_str:
            return False, (
                "⚠️  Error de conexión a Internet.\n"
                f"Detalle: {e}\n"
                "Verifica tu conexión de red o usa datos sintéticos con: --source synthetic"
            )
        return False, f"Error inesperado al validar el símbolo '{symbol}': {e}"


def analyze_ticker(
    symbol: str,
    timeframe: str = "1d",
    source: str = "yahoo",
    cache_dir: Path = Path("data"),
    period: str = "max",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Orquesta el pipeline completo E2E para un símbolo:
    1. Obtiene cotización en tiempo real (si source=yahoo).
    2. Ingesta las barras OHLCV (reales o sintéticas).
    3. Construye features cuantitativos vectorizados.
    4. Detecta extremos locales (pivots) y patrones estructurales.
    5. Genera etiquetas multi-horizonte (forward returns).
    6. Valida el modelo mediante Walk-Forward Out-Of-Sample con purga temporal.
    7. Calcula la inferencia predictiva actual (score now).
    """
    sym, note = normalize_symbol(symbol)

    # 0. Validación preventiva del símbolo (solo para fuente Yahoo)
    if source in {"yahoo", "yfinance"}:
        valid, err_msg = validate_yahoo_symbol(sym)
        if not valid:
            suggestion = fuzzy_suggest_symbol(symbol)
            return {
                "symbol": sym,
                "timeframe": timeframe,
                "note": note,
                "quote": None,
                "bars": [],
                "pivots": [],
                "bars_count": 0,
                "events_count": 0,
                "summary": None,
                "score_now": None,
                "message": err_msg,
            }

    # 1. Cotización en tiempo real
    quote = None
    if source in {"yahoo", "yfinance"}:
        try:
            quote = get_realtime_quote(sym)
        except Exception:
            quote = None

    # 2. Ingesta de barras
    if source in {"yahoo", "yfinance"}:
        bars = load_from_yahoo(sym, timeframe=timeframe, period=period)
    elif source == "synthetic":
        bars = load_synthetic(sym, timeframe=timeframe, seed=seed)
    else:
        raise ValueError(f"Fuente '{source}' no soportada para análisis automático. Usa 'yahoo' o 'synthetic'.")

    save_frame(bars, dataset_path(cache_dir, "bars", sym, timeframe))

    # 3. Features
    feats = build_features(bars)
    save_frame(feats, dataset_path(cache_dir, "features", sym, timeframe))

    # 4. Patrones y Pivots
    piv = detect_pivots(bars, window=3)
    serialized_bars, serialized_pivots = _serialize_chart_data(bars, piv)
    events = scan_structural(piv, tolerance=0.01)
    save_frame(piv, dataset_path(cache_dir, "pivots", sym, timeframe))
    save_frame(events, dataset_path(cache_dir, "events", sym, timeframe))

    # 5. Etiquetas futuras
    lab_bar = build_bar_labels(bars, horizons=[1, 3, 5, 10, 20, 60])
    lab_event = build_event_labels(bars, events, horizons=[1, 3, 5, 10, 20, 60])
    save_frame(lab_bar, dataset_path(cache_dir, "labels_bar", sym, timeframe))
    save_frame(lab_event, dataset_path(cache_dir, "labels_event", sym, timeframe))

    # 6. Validación Walk-Forward
    ds = event_dataset(feats, events, lab_event, horizon=5)
    X, y, fr, sub = as_xy(ds)

    if len(y) < 20:
        return {
            "symbol": sym,
            "timeframe": timeframe,
            "note": note,
            "quote": quote,
            "bars": serialized_bars,
            "pivots": serialized_pivots,
            "bars_count": bars.height,
            "events_count": events.height,
            "summary": None,
            "score_now": None,
            "message": (
                f"Se detectaron {events.height} formaciones técnicas en el histórico. "
                "Se requieren al menos 20 eventos para entrenar y validar un modelo estadístico confiable."
            ),
        }

    folds = list(walk_forward_indices(len(y), mode="rolling", train_size=50, test_size=20, n_splits=5, purge=1))
    fold_rows = []
    aucs = []
    for i, fold in enumerate(folds):
        Xtr, ytr = X[fold.train_idx], y[fold.train_idx]
        Xte, yte = X[fold.test_idx], y[fold.test_idx]
        frte = fr[fold.test_idx]
        clf = fit_logistic(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        m = compute_signal_metrics(yte, prob, frte)
        aucs.append(m["auc"])
        fold_rows.append({"fold": i, **m})

    summary = compute_signal_metrics(y, fit_logistic(X, y).predict_proba(X)[:, 1], fr)
    summary["stability_score"] = stability_score([x for x in aucs if x == x])

    save_frame(pl.DataFrame(fold_rows), dataset_path(cache_dir, "validation_folds", sym, timeframe))
    save_frame(pl.DataFrame([summary]), dataset_path(cache_dir, "validation_summary", sym, timeframe))

    # 7. Score now
    model = fit_logistic(X, y)
    feature_cols = [
        c
        for c in sub.columns
        if c in [
            "ret_1",
            "ret_5",
            "ema_ratio",
            "range_pct",
            "zscore_20",
            "vol_ratio_20",
            "rsi_14",
            "drawdown_20",
            "realized_vol_20",
        ]
    ]
    latest = sub.select(feature_cols).tail(1).to_numpy()
    score_now = float(model.predict_proba(latest)[:, 1][0])

    return {
        "symbol": sym,
        "timeframe": timeframe,
        "note": note,
        "quote": quote,
        "bars": serialized_bars,
        "pivots": serialized_pivots,
        "bars_count": bars.height,
        "events_count": events.height,
        "summary": summary,
        "score_now": score_now,
        "message": None,
    }

def run_ingest(
    symbol: str,
    timeframe: str = "1d",
    source: str = "synthetic",
    input_path: str | None = None,
    cache_dir: Path = Path("data"),
    export: str = "parquet",
    seed: int | None = None,
    period: str = "max",
) -> pl.DataFrame:
    """Orquesta la ingesta de datos y los persiste en disco."""
    from tickerlab.ingest.service import load_from_csv, load_from_yahoo, load_synthetic
    from tickerlab.storage.io import save_dataset

    if source == "csv":
        if not input_path:
            raise ValueError("--input-path es requerido cuando source=csv")
        bars = load_from_csv(input_path, symbol=symbol, timeframe=timeframe)
    elif source in {"yahoo", "yfinance"}:
        bars = load_from_yahoo(symbol=symbol, timeframe=timeframe, period=period)
    else:
        bars = load_synthetic(symbol=symbol, timeframe=timeframe, seed=seed)

    save_dataset(cache_dir, "bars", symbol, timeframe, bars, export=export)
    return bars


def run_features(
    symbol: str,
    timeframe: str = "1d",
    cache_dir: Path = Path("data"),
    export: str = "parquet",
) -> pl.DataFrame:
    """Orquesta la construcción de features y los persiste."""
    from tickerlab.storage.io import load_frame, save_dataset

    bars = load_frame(dataset_path(cache_dir, "bars", symbol, timeframe))
    feats = build_features(bars)
    save_dataset(cache_dir, "features", symbol, timeframe, feats, export=export)
    return feats


def run_patterns(
    symbol: str,
    timeframe: str = "1d",
    engine: str = "structural",
    cache_dir: Path = Path("data"),
    export: str = "parquet",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Orquesta la detección de pivots y patrones."""
    from tickerlab.storage.io import load_frame, save_dataset

    if engine != "structural":
        raise ValueError("Engine soportado en V1 mínima: structural")

    bars = load_frame(dataset_path(cache_dir, "bars", symbol, timeframe))
    piv = detect_pivots(bars, window=3)
    events = scan_structural(piv, tolerance=0.01)

    save_dataset(cache_dir, "pivots", symbol, timeframe, piv, export=export)
    save_dataset(cache_dir, "events", symbol, timeframe, events, export=export)
    return piv, events


def run_labels(
    symbol: str,
    timeframe: str = "1d",
    horizons: str = "1,3,5,10,20,60",
    cache_dir: Path = Path("data"),
    export: str = "parquet",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Orquesta la generación de etiquetas multi-horizonte."""
    from tickerlab.storage.io import load_frame, save_dataset

    hs = [int(x.strip()) for x in horizons.split(",") if x.strip()]
    bars = load_frame(dataset_path(cache_dir, "bars", symbol, timeframe))
    events = load_frame(dataset_path(cache_dir, "events", symbol, timeframe))

    lab_bar = build_bar_labels(bars, horizons=hs)
    lab_event = build_event_labels(bars, events, horizons=hs)

    save_dataset(cache_dir, "labels_bar", symbol, timeframe, lab_bar, export=export)
    save_dataset(cache_dir, "labels_event", symbol, timeframe, lab_event, export=export)
    return lab_bar, lab_event


def run_validate(
    symbol: str,
    timeframe: str = "1d",
    model: str = "logistic",
    mode: str = "rolling",
    horizon: int = 5,
    cache_dir: Path = Path("data"),
    export: str = "parquet",
) -> dict[str, Any]:
    """Orquesta la validación Walk-Forward y persiste el summary."""
    from tickerlab.storage.io import load_frame, save_dataset
    from tickerlab.utils.dataset import as_xy, event_dataset

    if model != "logistic":
        raise ValueError("Modelo soportado en V1 mínima: logistic")
    if mode not in {"rolling", "expanding"}:
        raise ValueError("mode debe ser rolling o expanding")

    feats = load_frame(dataset_path(cache_dir, "features", symbol, timeframe))
    events = load_frame(dataset_path(cache_dir, "events", symbol, timeframe))
    labels_event = load_frame(dataset_path(cache_dir, "labels_event", symbol, timeframe))

    ds = event_dataset(feats, events, labels_event, horizon=horizon)
    X, y, fr, sub = as_xy(ds)

    if len(y) < 20:
        raise ValueError("Muy pocos eventos válidos para walk-forward")

    folds = list(walk_forward_indices(len(y), mode=mode, train_size=50, test_size=20, n_splits=5, purge=1))
    fold_rows = []
    aucs = []
    for i, fold in enumerate(folds):
        Xtr, ytr = X[fold.train_idx], y[fold.train_idx]
        Xte, yte = X[fold.test_idx], y[fold.test_idx]
        frte = fr[fold.test_idx]
        clf = fit_logistic(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        m = compute_signal_metrics(yte, prob, frte)
        aucs.append(m["auc"])
        fold_rows.append({"fold": i, **m})

    summary = compute_signal_metrics(y, fit_logistic(X, y).predict_proba(X)[:, 1], fr)
    summary["stability_score"] = stability_score([x for x in aucs if x == x])

    save_dataset(cache_dir, "validation_folds", symbol, timeframe, pl.DataFrame(fold_rows), export=export)
    save_dataset(cache_dir, "validation_summary", symbol, timeframe, pl.DataFrame([summary]), export=export)

    return {
        "summary": summary,
        "rows": sub.height,
        "path": dataset_path(cache_dir, "validation_summary", symbol, timeframe),
    }


def run_score(
    symbol: str,
    timeframe: str = "1d",
    horizon: int = 5,
    cache_dir: Path = Path("data"),
) -> float:
    """Calcula el score predictivo actual basándose en los datos guardados."""
    from tickerlab.storage.io import load_frame
    from tickerlab.utils.dataset import as_xy, event_dataset

    feats = load_frame(dataset_path(cache_dir, "features", symbol, timeframe))
    events = load_frame(dataset_path(cache_dir, "events", symbol, timeframe))
    labels_event = load_frame(dataset_path(cache_dir, "labels_event", symbol, timeframe))
    ds = event_dataset(feats, events, labels_event, horizon=horizon)
    X, y, _, sub = as_xy(ds)
    model = fit_logistic(X, y)
    feature_cols = [
        c
        for c in sub.columns
        if c in [
            "ret_1",
            "ret_5",
            "ema_ratio",
            "range_pct",
            "zscore_20",
            "vol_ratio_20",
            "rsi_14",
            "drawdown_20",
            "realized_vol_20",
        ]
    ]
    latest = sub.select(feature_cols).tail(1).to_numpy()
    return float(model.predict_proba(latest)[:, 1][0])
