"""
fetch_historical.py
Fuente 1: Dataset martj42 — 49,000+ partidos internacionales desde 1872
Repositorio: github.com/martj42/international_results
"""
import pandas as pd
import urllib.request
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "raw"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

COMPETITION_MAP = {
    "FIFA World Cup": "Mundial",
    "FIFA World Cup qualification": "Eliminatorias",
    "Copa América": "Copa América",
    "UEFA Euro": "EURO",
    "UEFA Euro qualification": "Eliminatorias EURO",
    "Africa Cup of Nations": "Copa África",
    "African Cup of Nations": "Copa África",
    "African Cup of Nations qualification": "Elim. Copa África",
    "Friendly": "Amistoso",
    "UEFA Nations League": "Nations League",
    "CONMEBOL–UEFA Cup of Champions": "Finalissima",
}

PRIORITY_TOURNAMENTS = [
    "FIFA World Cup",
    "Copa América",
    "UEFA Euro",
    "Africa Cup of Nations",
    "African Cup of Nations",
    "FIFA World Cup qualification",
    "UEFA Euro qualification",
    "African Cup of Nations qualification",
    "UEFA Nations League",
    "Friendly",
]


def download_results(force: bool = False) -> Path:
    """
    Descarga el CSV si:
    - No existe, o
    - force=True, o
    - Tiene más de 24 horas de antigüedad (siempre datos frescos)
    """
    import time
    dest = RAW_PATH / "results.csv"
    needs_update = (
        not dest.exists() or
        force or
        (time.time() - dest.stat().st_mtime) > 86400  # 24 horas
    )
    if needs_update:
        print("⬇️  Actualizando resultados históricos...")
        try:
            urllib.request.urlretrieve(RESULTS_URL, dest)
            print(f"   ✅ Datos actualizados ({dest})")
        except Exception as e:
            if dest.exists():
                print(f"   ⚠️  Error al actualizar, usando caché local: {e}")
            else:
                raise
    return dest


def load_results(min_year: int = 2000) -> pd.DataFrame:
    """
    Carga y limpia el dataset histórico.
    Filtra por año mínimo y clasifica las competencias.
    """
    path = download_results()
    df = pd.read_csv(path, parse_dates=["date"])

    # Filtrar por año mínimo
    df = df[df["date"].dt.year >= min_year].copy()

    # Limpiar scores nulos
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Clasificar competencia
    df["competition_type"] = df["tournament"].map(COMPETITION_MAP).fillna("Otra")

    # Resultado codificado: H=local gana, D=empate, A=visita gana
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    df["result"] = pd.Categorical(
        pd.Series(["H", "D", "A"], dtype=str).iloc[
            pd.cut(pd.Series(range(len(df))), bins=3, labels=False)
        ].values
    )
    df["result"] = "D"
    df.loc[df["home_score"] > df["away_score"], "result"] = "H"
    df.loc[df["home_score"] < df["away_score"], "result"] = "A"

    # Total de goles
    df["total_goals"] = df["home_score"] + df["away_score"]

    # Flag: partido de alta importancia
    df["is_competitive"] = df["tournament"].isin(
        [t for t in PRIORITY_TOURNAMENTS if t != "Friendly"]
    )

    return df.sort_values("date").reset_index(drop=True)


def get_team_list(df: pd.DataFrame) -> list[str]:
    """Retorna lista ordenada de todas las selecciones."""
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    return sorted(teams)


def filter_by_competitions(df: pd.DataFrame, competitions: list[str]) -> pd.DataFrame:
    """Filtra por tipo de competencia."""
    return df[df["competition_type"].isin(competitions)].copy()
