"""
fetch_rankings.py
Fuente 4: Rankings FIFA históricos
Uso recomendado: reemplazar fifa_ranking.csv con el dataset de Kaggle:
  → https://www.kaggle.com/datasets/cashncarry/fifaworldranking
  Columnas esperadas: rank_date, country_full, rank
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "raw"


def load_rankings() -> pd.DataFrame:
    """Carga rankings FIFA. Compatible con el dataset de Kaggle."""
    path = RAW_PATH / "fifa_ranking.csv"
    if not path.exists():
        print("⚠️  fifa_ranking.csv no encontrado. Usando rankings estimados.")
        return _generate_estimated_rankings()

    df = pd.read_csv(path, parse_dates=["rank_date"])

    # Normalizar nombres de columnas según distintas versiones del dataset
    col_map = {
        "country_full": "team",
        "country_abrv": "team_abbr",
        "rank": "fifa_rank",
        "total_points": "fifa_points",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "team" not in df.columns:
        # Intentar detectar columna de nombre de equipo
        for c in df.columns:
            if df[c].dtype == object and df[c].nunique() > 50:
                df = df.rename(columns={c: "team"})
                break

    return df.sort_values("rank_date")


def get_ranking_at_date(rankings_df: pd.DataFrame, team: str, date: pd.Timestamp) -> int | None:
    """
    Retorna el ranking FIFA de un equipo en una fecha dada.
    Usa el ranking más reciente anterior a la fecha.
    """
    if rankings_df is None or rankings_df.empty:
        return None

    team_ranks = rankings_df[rankings_df["team"] == team]
    if team_ranks.empty:
        return None

    past = team_ranks[team_ranks["rank_date"] <= date]
    if past.empty:
        # Usar el más antiguo disponible
        return int(team_ranks.iloc[0]["fifa_rank"])

    return int(past.iloc[-1]["fifa_rank"])


def enrich_with_rankings(matches_df: pd.DataFrame, rankings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de ranking FIFA a cada partido:
    - home_fifa_rank, away_fifa_rank
    - rank_diff (positivo = local mejor rankeado)
    """
    if rankings_df is None or rankings_df.empty:
        matches_df["home_fifa_rank"] = np.nan
        matches_df["away_fifa_rank"] = np.nan
        matches_df["rank_diff"] = np.nan
        return matches_df

    home_ranks, away_ranks = [], []

    for _, row in matches_df.iterrows():
        hr = get_ranking_at_date(rankings_df, row["home_team"], row["date"])
        ar = get_ranking_at_date(rankings_df, row["away_team"], row["date"])
        home_ranks.append(hr)
        away_ranks.append(ar)

    matches_df = matches_df.copy()
    matches_df["home_fifa_rank"] = home_ranks
    matches_df["away_fifa_rank"] = away_ranks

    # Diferencia de ranking: negativo = local mejor posicionado (número menor = mejor)
    matches_df["rank_diff"] = matches_df["away_fifa_rank"] - matches_df["home_fifa_rank"]

    return matches_df


def _generate_estimated_rankings() -> pd.DataFrame | None:
    """Placeholder mínimo cuando no hay datos reales."""
    return None
