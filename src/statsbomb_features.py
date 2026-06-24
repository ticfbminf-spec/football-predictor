"""
statsbomb_features.py
Carga el resumen ligero de StatsBomb y lo integra como features predictivas.
Mapea nombres de equipos entre StatsBomb y el dataset martj42.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path(__file__).parent.parent / "data" / "processed"

# Mapeo de nombres: StatsBomb -> martj42 (cuando difieren)
TEAM_NAME_MAP = {
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "South Korea": "South Korea",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "China PR": "China",
    "United States": "USA",
    "Côte d'Ivoire": "Ivory Coast",
}


def load_statsbomb_summary() -> pd.DataFrame | None:
    """Carga el resumen por equipo. None si no se ha pre-procesado."""
    path = PROCESSED / "statsbomb_team_stats.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Normalizar nombres al estándar martj42
    df["team"] = df["team"].replace(TEAM_NAME_MAP)
    return df


def get_team_advanced_stats(summary_df: pd.DataFrame, team: str) -> dict:
    """
    Retorna las stats avanzadas de un equipo.
    Si no está en StatsBomb (equipo menor), retorna NaN.
    """
    if summary_df is None:
        return _empty_stats()

    row = summary_df[summary_df["team"] == team]
    if row.empty:
        return _empty_stats()

    r = row.iloc[0]
    return {
        "xg_for": r["avg_xg_for"],
        "xg_against": r["avg_xg_against"],
        "xg_diff": r["xg_diff"],
        "finishing": r["finishing"],
        "shots": r["avg_shots"],
        "pass_accuracy": r["avg_pass_accuracy"],
        "has_data": True,
    }


def _empty_stats() -> dict:
    return {
        "xg_for": np.nan, "xg_against": np.nan, "xg_diff": np.nan,
        "finishing": np.nan, "shots": np.nan, "pass_accuracy": np.nan,
        "has_data": False,
    }


def enrich_features_with_statsbomb(features_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade xG histórico por equipo SIN data leakage.

    Para cada partido, el xG de un equipo se calcula usando SOLO sus partidos
    StatsBomb ANTERIORES a la fecha del partido. Esto evita que el modelo use
    información del futuro (que en la versión anterior inflaba la precisión).

    Equipos sin partidos StatsBomb previos quedan con NaN (el modelo lo maneja).
    """
    cols = ["home_xg_for", "home_xg_against", "home_xg_diff",
            "away_xg_for", "away_xg_against", "away_xg_diff", "xg_diff_matchup"]

    if summary_df is None:
        for col in cols:
            features_df[col] = np.nan
        return features_df

    # Cargar los partidos individuales de StatsBomb (con fecha aproximada por año)
    matches_path = PROCESSED / "statsbomb_matches.parquet"
    if not matches_path.exists():
        # Fallback: usar promedio global (con leakage, pero mejor que nada)
        return _enrich_global(features_df, summary_df, cols)

    sb = pd.read_parquet(matches_path)
    sb["team"] = sb["team"].replace(TEAM_NAME_MAP)
    # Aproximar fecha del partido StatsBomb al 1 de julio de su año
    # (los torneos de selección son a mitad de año; suficiente para ordenar)
    sb["sb_date"] = pd.to_datetime(sb["year"].astype(str) + "-07-01")

    features_df = features_df.copy().sort_values("date").reset_index(drop=True)

    def team_xg_before(team: str, before: pd.Timestamp) -> dict:
        """xG promedio del equipo en torneos StatsBomb anteriores a 'before'."""
        past = sb[(sb["team"] == team) & (sb["sb_date"] < before)]
        if past.empty:
            return {"xgf": np.nan, "xga": np.nan, "xgd": np.nan}
        xgf = past["xg_for"].mean()
        xga = past["xg_against"].mean()
        return {"xgf": xgf, "xga": xga, "xgd": xgf - xga}

    # Cache por (equipo, año) para no recalcular en cada fila
    cache: dict = {}

    def cached_xg(team, date):
        key = (team, date.year)
        if key not in cache:
            cache[key] = team_xg_before(team, date)
        return cache[key]

    h_for, h_ag, h_d, a_for, a_ag, a_d = [], [], [], [], [], []
    for row in features_df.itertuples(index=False):
        hx = cached_xg(row.home_team, row.date)
        ax = cached_xg(row.away_team, row.date)
        h_for.append(hx["xgf"]); h_ag.append(hx["xga"]); h_d.append(hx["xgd"])
        a_for.append(ax["xgf"]); a_ag.append(ax["xga"]); a_d.append(ax["xgd"])

    features_df["home_xg_for"]     = h_for
    features_df["home_xg_against"] = h_ag
    features_df["home_xg_diff"]    = h_d
    features_df["away_xg_for"]     = a_for
    features_df["away_xg_against"] = a_ag
    features_df["away_xg_diff"]    = a_d
    features_df["xg_diff_matchup"] = features_df["home_xg_diff"] - features_df["away_xg_diff"]

    return features_df


def _enrich_global(features_df, summary_df, cols):
    """Fallback con promedio global (mantiene compatibilidad)."""
    stats_lookup = {}
    for _, r in summary_df.iterrows():
        stats_lookup[r["team"]] = {
            "xg_for": r["avg_xg_for"], "xg_against": r["avg_xg_against"], "xg_diff": r["xg_diff"],
        }

    def lookup(team, key):
        return stats_lookup.get(team, {}).get(key, np.nan)

    features_df = features_df.copy()
    for side in ["home", "away"]:
        features_df[f"{side}_xg_for"]     = features_df[f"{side}_team"].map(lambda t: lookup(t, "xg_for"))
        features_df[f"{side}_xg_against"] = features_df[f"{side}_team"].map(lambda t: lookup(t, "xg_against"))
        features_df[f"{side}_xg_diff"]    = features_df[f"{side}_team"].map(lambda t: lookup(t, "xg_diff"))
    features_df["xg_diff_matchup"] = features_df["home_xg_diff"] - features_df["away_xg_diff"]
    return features_df
