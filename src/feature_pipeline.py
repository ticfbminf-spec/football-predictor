"""
feature_pipeline.py
Construye el dataset de features para el modelo predictivo.
Combina historial + rankings + estadísticas rodantes.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed"


def compute_team_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calcula forma reciente de cada equipo (últimos N partidos).
    Retorna el DataFrame original con columnas adicionales de forma.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Desnormalizar: crear vista desde perspectiva de cada equipo
    home_view = df[["date", "home_team", "away_team", "home_score", "away_score", "result", "neutral"]].copy()
    home_view.columns = ["date", "team", "opponent", "goals_for", "goals_against", "match_result", "neutral"]
    home_view["is_home"] = True
    home_view["won"] = home_view["match_result"] == "H"
    home_view["drew"] = home_view["match_result"] == "D"
    home_view["lost"] = home_view["match_result"] == "A"

    away_view = df[["date", "away_team", "home_team", "away_score", "home_score", "result", "neutral"]].copy()
    away_view.columns = ["date", "team", "opponent", "goals_for", "goals_against", "match_result", "neutral"]
    away_view["is_home"] = False
    away_view["won"] = away_view["match_result"] == "A"
    away_view["drew"] = away_view["match_result"] == "D"
    away_view["lost"] = away_view["match_result"] == "H"

    all_matches = pd.concat([home_view, away_view]).sort_values("date").reset_index(drop=True)

    # Calcular stats rodantes por equipo
    form_stats = {}
    for team, group in all_matches.groupby("team"):
        group = group.sort_values("date").reset_index(drop=True)
        form_stats[team] = {
            "dates": group["date"].tolist(),
            "pts": (group["won"] * 3 + group["drew"]).tolist(),
            "gf": group["goals_for"].tolist(),
            "ga": group["goals_against"].tolist(),
        }

    def rolling_form(team: str, before_date: pd.Timestamp, n: int = window):
        """Puntos, GF, GA promedio en los últimos N partidos antes de una fecha."""
        if team not in form_stats:
            return {"form_pts": np.nan, "form_gf": np.nan, "form_ga": np.nan, "form_n": 0}

        dates = form_stats[team]["dates"]
        past_idx = [i for i, d in enumerate(dates) if d < before_date]
        past_idx = past_idx[-n:]  # últimos N

        if not past_idx:
            return {"form_pts": np.nan, "form_gf": np.nan, "form_ga": np.nan, "form_n": 0}

        pts = np.mean([form_stats[team]["pts"][i] for i in past_idx])
        gf = np.mean([form_stats[team]["gf"][i] for i in past_idx])
        ga = np.mean([form_stats[team]["ga"][i] for i in past_idx])
        return {"form_pts": pts, "form_gf": gf, "form_ga": ga, "form_n": len(past_idx)}

    # Aplicar a cada partido
    records = []
    for _, row in df.iterrows():
        h = rolling_form(row["home_team"], row["date"])
        a = rolling_form(row["away_team"], row["date"])
        records.append({
            "home_form_pts": h["form_pts"],
            "home_form_gf": h["form_gf"],
            "home_form_ga": h["form_ga"],
            "away_form_pts": a["form_pts"],
            "away_form_gf": a["form_gf"],
            "away_form_ga": a["form_ga"],
            "form_pts_diff": h["form_pts"] - a["form_pts"] if not np.isnan(h["form_pts"]) and not np.isnan(a["form_pts"]) else np.nan,
        })

    form_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), form_df], axis=1)


def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade estadísticas históricas de enfrentamientos directos (H2H).
    Para cada partido, cuántos ganó local, visita, empates en los últimos 10 H2H.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    h2h_home_wins, h2h_away_wins, h2h_draws, h2h_n = [], [], [], []

    for idx, row in df.iterrows():
        t1, t2, date = row["home_team"], row["away_team"], row["date"]

        # Partidos anteriores entre estos dos equipos (en cualquier dirección)
        mask = (
            ((df["home_team"] == t1) & (df["away_team"] == t2)) |
            ((df["home_team"] == t2) & (df["away_team"] == t1))
        ) & (df["date"] < date)

        past = df[mask].tail(10)  # últimos 10

        if past.empty:
            h2h_home_wins.append(np.nan)
            h2h_away_wins.append(np.nan)
            h2h_draws.append(np.nan)
            h2h_n.append(0)
            continue

        # Contar victorias del equipo local actual
        home_w = ((past["home_team"] == t1) & (past["result"] == "H")).sum() + \
                 ((past["away_team"] == t1) & (past["result"] == "A")).sum()
        away_w = ((past["home_team"] == t2) & (past["result"] == "H")).sum() + \
                 ((past["away_team"] == t2) & (past["result"] == "A")).sum()
        draws = (past["result"] == "D").sum()

        h2h_home_wins.append(home_w)
        h2h_away_wins.append(away_w)
        h2h_draws.append(draws)
        h2h_n.append(len(past))

    df["h2h_home_wins"] = h2h_home_wins
    df["h2h_away_wins"] = h2h_away_wins
    df["h2h_draws"] = h2h_draws
    df["h2h_n"] = h2h_n

    return df


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features de contexto del partido."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Peso de competencia (partidos más importantes tienen mayor peso)
    competition_weight = {
        "FIFA World Cup": 5,
        "Copa América": 4,
        "UEFA Euro": 4,
        "Africa Cup of Nations": 4,
        "African Cup of Nations": 4,
        "FIFA World Cup qualification": 3,
        "UEFA Euro qualification": 2,
        "African Cup of Nations qualification": 2,
        "UEFA Nations League": 2,
        "Friendly": 1,
    }
    df["competition_weight"] = df["tournament"].map(competition_weight).fillna(2)

    # Ventaja de localía (1 si local, 0 si neutral)
    df["home_advantage"] = (~df["neutral"]).astype(int)

    return df


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features que capturan si el partido es entre equipos parejos.
    Estas features mejoran la detección de empates y partidos cerrados.
    Requiere que elo_diff ya esté calculado.
    """
    df = df.copy()

    if "elo_diff" in df.columns:
        df["elo_diff_abs"] = df["elo_diff"].abs()
        # Flag: equipos muy parejos (diferencia < 100 puntos Elo)
        df["elo_similar"] = (df["elo_diff_abs"] < 100).astype(int)

    # Paridad en forma reciente
    if "form_pts_diff" in df.columns:
        df["form_similar"] = df["form_pts_diff"].abs().fillna(0)

    # H2H equilibrado: ¿los enfrentamientos históricos están parejos?
    if all(c in df.columns for c in ["h2h_n", "h2h_home_wins", "h2h_away_wins"]):
        df["h2h_balance"] = df.apply(
            lambda r: 1 if r["h2h_n"] > 0 and abs(r["h2h_home_wins"] - r["h2h_away_wins"]) <= 1
            else 0, axis=1
        )
    return df


def build_feature_dataset(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame = None,
    form_window: int = 5,
    min_year: int = 2000,
) -> pd.DataFrame:
    """
    Pipeline completo: toma partidos crudos y retorna dataset con features.
    """
    print("🔧 Filtrando por año...")
    df = matches_df[matches_df["date"].dt.year >= min_year].copy()

    print("🔧 Añadiendo features contextuales...")
    df = add_contextual_features(df)

    print(f"🔧 Calculando forma reciente (ventana={form_window})...")
    df = compute_team_form(df, window=form_window)

    print("🔧 Calculando H2H...")
    df = compute_h2h(df)

    # Rating Elo (calculado cronológicamente, sin leakage)
    print("🔧 Calculando rating Elo...")
    from src.elo_ratings import compute_elo_history
    df, _ = compute_elo_history(df)

    # Features de partido parejo (requiere Elo y H2H ya calculados)
    print("🔧 Añadiendo features de paridad...")
    df = add_matchup_features(df)

    if rankings_df is not None and not rankings_df.empty:
        print("🔧 Añadiendo rankings FIFA...")
        from src.fetch_rankings import enrich_with_rankings
        df = enrich_with_rankings(df, rankings_df)

    # Features avanzadas de StatsBomb (xG histórico)
    try:
        from src.statsbomb_features import load_statsbomb_summary, enrich_features_with_statsbomb
        sb = load_statsbomb_summary()
        if sb is not None:
            print("🔧 Añadiendo xG histórico (StatsBomb)...")
            df = enrich_features_with_statsbomb(df, sb)
    except Exception as e:
        print(f"   ⚠️  StatsBomb no disponible: {e}")

    print(f"✅ Dataset listo: {len(df)} partidos, {len(df.columns)} columnas")
    return df


def save_processed(df: pd.DataFrame, filename: str = "features.parquet"):
    """Guarda el dataset procesado."""
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_PATH / filename
    df.to_parquet(path, index=False)
    print(f"💾 Guardado en {path}")
    return path


def load_processed(filename: str = "features.parquet") -> pd.DataFrame | None:
    """Carga el dataset procesado si existe."""
    path = PROCESSED_PATH / filename
    if path.exists():
        return pd.read_parquet(path)
    return None
