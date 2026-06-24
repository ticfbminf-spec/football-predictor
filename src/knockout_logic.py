"""
knockout_logic.py
Lógica de fases eliminatorias:
  - Detección automática de si un partido puede ir a ET/penales
  - Modelo de tiempo extra (¿se resuelve antes de penales?)
  - Simulación de tanda de penales basada en historial real
"""
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request

RAW_PATH = Path(__file__).parent.parent / "data" / "raw"

# ── Estructura de competencias ─────────────────────────────────────────────────
# competition -> fases que tienen partidos eliminatorios (sin empate posible)
COMPETITION_PHASES = {
    "FIFA World Cup": {
        "group": ["Grupo A", "Grupo B", "Grupo C", "Grupo D",
                  "Grupo E", "Grupo F", "Grupo G", "Grupo H",
                  "Grupo I", "Grupo J", "Grupo K", "Grupo L",
                  "Group Stage", "Fase de Grupos",
                  "Grupo - Jornada 3"],
        "knockout": ["Dieciseisavos de Final", "Round of 32",
                     "Octavos de Final", "Round of 16",
                     "Cuartos de Final", "Quarter-finals",
                     "Semifinal", "Semi-finals",
                     "Tercer Puesto", "Third Place",
                     "Final"],
    },
    "Copa América": {
        "group": ["Grupo A", "Grupo B", "Grupo C", "Grupo D", "Group Stage"],
        "knockout": ["Cuartos de Final", "Quarter-finals",
                     "Semifinal", "Semi-finals",
                     "Tercer Puesto", "Third Place", "Final"],
    },
    "UEFA Euro": {
        "group": ["Grupo A", "Grupo B", "Grupo C", "Grupo D",
                  "Grupo E", "Grupo F", "Group Stage"],
        "knockout": ["Round of 16", "Octavos de Final",
                     "Cuartos de Final", "Quarter-finals",
                     "Semifinal", "Semi-finals",
                     "Tercer Puesto", "Third Place", "Final"],
    },
    "Africa Cup of Nations": {
        "group": ["Group Stage", "Grupo A", "Grupo B", "Grupo C", "Grupo D"],
        "knockout": ["Round of 16", "Quarter-finals", "Semi-finals",
                     "Third Place", "Final"],
    },
    "African Cup of Nations": {
        "group": ["Group Stage"],
        "knockout": ["Round of 16", "Quarter-finals", "Semi-finals",
                     "Third Place", "Final"],
    },
    "UEFA Nations League": {
        "group": ["Group Stage", "Liga A", "Liga B", "Liga C", "Liga D"],
        "knockout": ["Semi-finals", "Final", "Third Place"],
    },
    "Gold Cup": {
        "group": ["Group Stage"],
        "knockout": ["Quarter-finals", "Semi-finals", "Final"],
    },
    "AFC Asian Cup": {
        "group": ["Group Stage"],
        "knockout": ["Round of 16", "Quarter-finals", "Semi-finals", "Final"],
    },
}

# Competencias que SIEMPRE son eliminatorias (sin fase de grupos)
ALWAYS_KNOCKOUT = [
    "FIFA World Cup qualification",  # partido de play-off (no todos)
]

# Competencias que NUNCA van a ET
NEVER_KNOCKOUT = ["Friendly"]

# ── Carga de datos de penales ──────────────────────────────────────────────────
SHOOTOUT_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"


def load_shootouts() -> pd.DataFrame:
    """Descarga y carga el historial de tandas de penales."""
    dest = RAW_PATH / "shootouts.csv"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(SHOOTOUT_URL, dest)
        except Exception:
            return pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])

    return pd.read_csv(dest, parse_dates=["date"])


def compute_penalty_stats(shootouts_df: pd.DataFrame) -> dict:
    """
    Calcula win rate en penales por equipo.
    Retorna dict: team -> {"wins": n, "total": n, "win_rate": float}
    """
    if shootouts_df.empty:
        return {}

    teams = set(shootouts_df["home_team"].tolist() + shootouts_df["away_team"].tolist())
    stats = {}

    for team in teams:
        mask = (shootouts_df["home_team"] == team) | (shootouts_df["away_team"] == team)
        team_sh = shootouts_df[mask]
        wins = int((team_sh["winner"] == team).sum())
        total = len(team_sh)
        stats[team] = {
            "wins": wins,
            "losses": total - wins,
            "total": total,
            "win_rate": wins / total if total > 0 else 0.5,
        }

    return stats


# ── Detección automática de fase ──────────────────────────────────────────────
def is_knockout_phase(competition: str, phase: str | None) -> bool:
    """
    Determina si un partido es eliminatorio (no puede terminar en empate).
    competition: nombre de la competencia
    phase: fase dentro de la competencia (puede ser None)
    """
    if competition in NEVER_KNOCKOUT:
        return False

    if phase is None:
        # Sin fase especificada: asumir que no es eliminatorio por defecto
        return False

    comp_data = COMPETITION_PHASES.get(competition)
    if comp_data is None:
        # Competencia desconocida → asumir que puede ser eliminatorio si phase lo indica
        knockout_keywords = ["final", "semi", "quarter", "cuarto", "octavo",
                             "round of", "knockout", "eliminat", "playoff"]
        return any(kw in phase.lower() for kw in knockout_keywords)

    knockout_phases = comp_data.get("knockout", [])
    return any(kp.lower() in phase.lower() or phase.lower() in kp.lower()
               for kp in knockout_phases)


def get_phases_for_competition(competition: str) -> dict:
    """
    Retorna las fases disponibles para una competencia.
    {"group": [...], "knockout": [...]}
    """
    return COMPETITION_PHASES.get(competition, {
        "group": ["Fase de Grupos"],
        "knockout": ["Octavos", "Cuartos", "Semifinal", "Final"],
    })


def competition_supports_knockout(competition: str) -> bool:
    """¿Esta competencia tiene al menos alguna fase eliminatoria?"""
    if competition in NEVER_KNOCKOUT:
        return False
    return competition in COMPETITION_PHASES or competition not in NEVER_KNOCKOUT


# ── Modelo de tiempo extra ────────────────────────────────────────────────────
# Probabilidad empírica de que un partido eliminatorio empatado en 90'
# se resuelva en tiempo extra (vs ir a penales)
# Basado en datos históricos de Mundiales y Euros
ET_RESOLUTION_RATE = 0.38   # 38% de los empates en EK se resuelven en ET
                             # 62% van a penales

def simulate_extra_time(
    home_team: str,
    away_team: str,
    prob_home: float,
    prob_away: float,
    n_simulations: int = 10_000,
    seed: int = None,
) -> dict:
    """
    Simula el tiempo extra cuando el partido termina empatado en 90'.
    
    Retorna probabilidades de:
    - home gana en ET
    - away gana en ET  
    - va a penales
    """
    rng = np.random.default_rng(seed)

    # Normalizar probabilidades (sin empate en ET — hay golden goal implícito)
    total = prob_home + prob_away
    p_home_et = (prob_home / total) * ET_RESOLUTION_RATE
    p_away_et = (prob_away / total) * ET_RESOLUTION_RATE
    p_penalties = 1 - ET_RESOLUTION_RATE

    return {
        "prob_home_et": p_home_et,
        "prob_away_et": p_away_et,
        "prob_penalties": p_penalties,
    }


# ── Simulación de tanda de penales ────────────────────────────────────────────
DEFAULT_CONVERSION_RATE = 0.756   # tasa histórica de conversión en penales internacionales


def get_team_penalty_rate(team: str, penalty_stats: dict) -> float:
    """
    Retorna la tasa histórica de éxito en tanda de penales de un equipo.
    Si no hay datos suficientes, usa la media global.
    """
    if team not in penalty_stats or penalty_stats[team]["total"] < 3:
        return 0.50   # sin datos suficientes → 50/50

    # Suavizar con prior bayesiano (mean global = 0.5, peso = 5 partidos)
    prior_weight = 5
    prior_mean = 0.50
    team_wins = penalty_stats[team]["wins"]
    team_total = penalty_stats[team]["total"]

    smoothed = (prior_weight * prior_mean + team_wins) / (prior_weight + team_total)
    return smoothed


def simulate_penalty_shootout(
    home_team: str,
    away_team: str,
    penalty_stats: dict,
    n_kicks: int = 5,
    n_simulations: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    Simula una tanda de penales entre dos equipos.
    
    Parámetros:
    - n_kicks: tiros por equipo en la tanda normal (5)
    - n_simulations: Monte Carlo
    
    Retorna:
    - prob_home_wins: probabilidad de que local gane la tanda
    - prob_away_wins
    - expected_kicks: media de tiros totales
    - simulated_shootout: ejemplo de tanda simulada (para mostrar en UI)
    - historical_data: stats reales de cada equipo
    """
    rng = np.random.default_rng(seed)

    home_rate = get_team_penalty_rate(home_team, penalty_stats)
    away_rate = get_team_penalty_rate(away_team, penalty_stats)

    # Tasa de conversión individual (por tiro) basada en historial de equipo
    # El win_rate del equipo refleja su desempeño GLOBAL en tandas,
    # lo traducimos a conversión por tiro asumiendo distribución binomial
    # Calibración empírica: win_rate ≈ f(conv_rate_home - conv_rate_away)
    # Usamos conversión base ajustada por el factor del equipo
    base = DEFAULT_CONVERSION_RATE
    home_conv = base * (0.7 + 0.6 * home_rate)   # escalar entorno a base
    away_conv = base * (0.7 + 0.6 * away_rate)
    home_conv = np.clip(home_conv, 0.5, 0.95)
    away_conv = np.clip(away_conv, 0.5, 0.95)

    home_wins = 0
    away_wins = 0
    total_kicks_list = []

    for _ in range(n_simulations):
        h_goals = 0
        a_goals = 0
        kicks = 0

        # Ronda normal: 5 tiros cada uno
        for i in range(n_kicks):
            h_scored = rng.random() < home_conv
            a_scored = rng.random() < away_conv
            h_goals += h_scored
            a_goals += a_scored
            kicks += 2

        # Sudden death si empate después de 5
        while h_goals == a_goals:
            h_scored = rng.random() < home_conv
            a_scored = rng.random() < away_conv
            kicks += 2
            if h_scored and not a_scored:
                h_goals += 1
            elif a_scored and not h_scored:
                a_goals += 1
            elif h_scored and a_scored:
                h_goals += 1
                a_goals += 1
            # Si ambos fallan → continuar

        if h_goals > a_goals:
            home_wins += 1
        else:
            away_wins += 1

        total_kicks_list.append(kicks)

    prob_home = home_wins / n_simulations
    prob_away = away_wins / n_simulations

    # Simular UNA tanda representativa para mostrar en UI
    example = _simulate_one_shootout(home_team, away_team, home_conv, away_conv,
                                     n_kicks, rng)

    return {
        "prob_home_wins": prob_home,
        "prob_away_wins": prob_away,
        "home_conversion_rate": home_conv,
        "away_conversion_rate": away_conv,
        "expected_kicks": np.mean(total_kicks_list),
        "example_shootout": example,
        "home_historical": penalty_stats.get(home_team, {"total": 0, "wins": 0, "win_rate": 0.5}),
        "away_historical": penalty_stats.get(away_team, {"total": 0, "wins": 0, "win_rate": 0.5}),
        "home_raw_rate": home_rate,
        "away_raw_rate": away_rate,
    }


def _simulate_one_shootout(
    home_team: str,
    away_team: str,
    home_conv: float,
    away_conv: float,
    n_kicks: int,
    rng: np.random.Generator,
) -> dict:
    """Simula una tanda completa para visualización."""
    home_kicks = []
    away_kicks = []
    h_goals = 0
    a_goals = 0

    for i in range(n_kicks):
        h = rng.random() < home_conv
        a = rng.random() < away_conv
        home_kicks.append({"kick": i + 1, "scored": bool(h), "cumulative": h_goals + h})
        away_kicks.append({"kick": i + 1, "scored": bool(a), "cumulative": a_goals + a})
        h_goals += h
        a_goals += a

    # Sudden death si necesario
    sd_kicks = []
    sd_round = 1
    while h_goals == a_goals:
        h = rng.random() < home_conv
        a = rng.random() < away_conv
        sd_kicks.append({
            "round": sd_round,
            "home_scored": bool(h),
            "away_scored": bool(a),
        })
        h_goals += h
        a_goals += a
        sd_round += 1
        if sd_round > 20:  # safeguard
            break

    winner = home_team if h_goals > a_goals else away_team

    return {
        "home_kicks": home_kicks,
        "away_kicks": away_kicks,
        "sudden_death": sd_kicks,
        "home_goals": h_goals,
        "away_goals": a_goals,
        "winner": winner,
    }


# ── Pipeline completo de predicción con knockout ──────────────────────────────
def full_match_prediction(
    base_prediction: dict,
    competition: str,
    phase: str | None,
    is_knockout_override: bool | None,
    penalty_stats: dict,
) -> dict:
    """
    Extiende la predicción base con lógica de ET y penales si aplica.
    
    base_prediction: resultado de model.predict_match()
    is_knockout_override: None = automático, True/False = manual
    """
    result = base_prediction.copy()

    # Determinar si es eliminatorio
    if is_knockout_override is not None:
        is_knockout = is_knockout_override
        result["knockout_source"] = "manual"
    else:
        is_knockout = is_knockout_phase(competition, phase)
        result["knockout_source"] = "automático"

    result["is_knockout"] = is_knockout

    if not is_knockout:
        result["et_penalties"] = None
        return result

    # En partido eliminatorio, el empate lleva a ET/penales
    prob_h = base_prediction["prob_home"]
    prob_d = base_prediction["prob_draw"]
    prob_a = base_prediction["prob_away"]

    home_team = base_prediction["home_team"]
    away_team = base_prediction["away_team"]

    # Simular ET (solo relevante si hay prob de empate)
    et_result = simulate_extra_time(home_team, away_team, prob_h, prob_a)

    # Simular penales
    penalty_result = simulate_penalty_shootout(
        home_team, away_team, penalty_stats, seed=42
    )

    # Probabilidades finales de clasificación (considerando 90' + ET + P)
    # P(home clasifica) = P(gana 90') + P(empata 90') * P(gana ET) + P(empata 90') * P(va penales) * P(gana penales)
    p_home_classifies = (
        prob_h +
        prob_d * et_result["prob_home_et"] +
        prob_d * et_result["prob_penalties"] * penalty_result["prob_home_wins"]
    )
    p_away_classifies = (
        prob_a +
        prob_d * et_result["prob_away_et"] +
        prob_d * et_result["prob_penalties"] * penalty_result["prob_away_wins"]
    )

    result["et_penalties"] = {
        "extra_time": et_result,
        "penalties": penalty_result,
        "prob_home_classifies": p_home_classifies,
        "prob_away_classifies": p_away_classifies,
        "prob_goes_to_et": prob_d,
        "prob_goes_to_penalties": prob_d * et_result["prob_penalties"],
    }

    return result
