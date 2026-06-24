"""
elo_ratings.py
Calcula ratings Elo para selecciones nacionales usando la metodología
World Football Elo (la que adoptó FIFA en 2018).

Ventaja clave: el Elo se calcula CRONOLÓGICAMENTE recorriendo los partidos
en orden. Para cada partido, el Elo de ambos equipos refleja SOLO los
partidos previos => imposible que haya data leakage.

Referencia: https://www.eloratings.net/about
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path(__file__).parent.parent / "data" / "processed"

# ── Parámetros del sistema (estándar World Football Elo) ──────────────────────
INITIAL_RATING = 1500       # rating inicial de un equipo nuevo
HOME_ADVANTAGE = 100        # ventaja de localía en puntos Elo

# Factor K según importancia del partido (weight index del World Football Elo)
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Copa América": 50,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 40,
    "Africa Cup of Nations": 50,
    "African Cup of Nations": 50,
    "African Cup of Nations qualification": 40,
    "UEFA Nations League": 40,
    "CONCACAF Nations League": 40,
    "Gold Cup": 50,
    "AFC Asian Cup": 50,
    "Confederations Cup": 40,
    "Friendly": 20,
}
DEFAULT_K = 30              # competencias no listadas (continentales menores)


def get_k_factor(tournament: str, goal_diff: int) -> float:
    """
    Factor K ajustado por margen de goles.
    En World Football Elo, ganar por más goles aumenta el cambio de rating.
    """
    base_k = K_BY_TOURNAMENT.get(tournament, DEFAULT_K)

    # Multiplicador por margen de victoria
    if goal_diff <= 1:
        multiplier = 1.0
    elif goal_diff == 2:
        multiplier = 1.5
    elif goal_diff == 3:
        multiplier = 1.75
    else:
        multiplier = 1.75 + (goal_diff - 3) / 8

    return base_k * multiplier


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Probabilidad esperada de que A gane (fórmula Elo estándar).
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_history(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Recorre todos los partidos cronológicamente y calcula el Elo.

    Retorna:
    - df con columnas home_elo, away_elo (el rating ANTES de cada partido)
    - dict final {team: rating} con el Elo actual de cada selección
    """
    df = df.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    home_elos = np.zeros(len(df))
    away_elos = np.zeros(len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        home = row.home_team
        away = row.away_team

        r_home = ratings.get(home, INITIAL_RATING)
        r_away = ratings.get(away, INITIAL_RATING)

        # Guardar el rating ANTES del partido (esto es lo que usa el modelo)
        home_elos[i] = r_home
        away_elos[i] = r_away

        # Resultado real
        hs, as_ = row.home_score, row.away_score
        if hs > as_:
            score_home = 1.0
        elif hs == as_:
            score_home = 0.5
        else:
            score_home = 0.0

        # Esperado (con ventaja de localía, salvo campo neutral)
        ha = 0 if getattr(row, "neutral", False) else HOME_ADVANTAGE
        exp_home = expected_score(r_home + ha, r_away)

        # Factor K
        goal_diff = abs(hs - as_)
        k = get_k_factor(getattr(row, "tournament", "Friendly"), goal_diff)

        # Actualizar ratings
        change = k * (score_home - exp_home)
        ratings[home] = r_home + change
        ratings[away] = r_away - change

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df, ratings


def build_and_save_elo(df: pd.DataFrame) -> dict:
    """
    Calcula el Elo sobre todo el historial y guarda el rating final por equipo.
    Retorna el dict de ratings actuales.
    """
    _, final_ratings = compute_elo_history(df)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    ratings_df = pd.DataFrame(
        [{"team": t, "elo": round(r, 1)} for t, r in final_ratings.items()]
    ).sort_values("elo", ascending=False)
    ratings_df.to_parquet(PROCESSED / "elo_current.parquet", index=False)

    return final_ratings


def load_current_elo() -> dict:
    """Carga el Elo actual por equipo (para predicción de partidos futuros)."""
    path = PROCESSED / "elo_current.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {r["team"]: r["elo"] for _, r in df.iterrows()}


def elo_win_probability(elo_home: float, elo_away: float, neutral: bool = False) -> dict:
    """
    Convierte diferencia de Elo en probabilidades 1X2.
    Usa una curva de empate calibrada empíricamente para fútbol.
    """
    ha = 0 if neutral else HOME_ADVANTAGE
    exp_home = expected_score(elo_home + ha, elo_away)

    # Probabilidad de empate: máxima cuando los equipos son parejos (~28%),
    # decrece cuando hay diferencia grande
    diff = abs((elo_home + ha) - elo_away)
    p_draw = 0.28 * np.exp(-diff / 350)

    # Repartir el resto entre victoria local y visitante según expected
    remaining = 1 - p_draw
    p_home = remaining * exp_home
    p_away = remaining * (1 - exp_home)

    return {"prob_home": p_home, "prob_draw": p_draw, "prob_away": p_away}
