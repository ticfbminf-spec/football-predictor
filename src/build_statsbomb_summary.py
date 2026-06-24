"""
build_statsbomb_summary.py
Pre-procesa los torneos de élite de StatsBomb Open Data y genera un
resumen LIGERO por selección (no guarda los eventos crudos, que pesan GB).

Output: data/processed/statsbomb_team_stats.parquet
  Una fila por (equipo, torneo) con métricas avanzadas agregadas.

Ejecutar una sola vez (o cuando StatsBomb publique nuevos torneos):
  python src/build_statsbomb_summary.py
"""
import urllib.request
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RAW = Path(__file__).parent.parent / "data" / "raw"

# Torneos de selecciones a procesar: (competition_id, season_id, nombre, año)
NATIONAL_TEAM_COMPS = [
    (43, 106, "FIFA World Cup", 2022),
    (43, 3,   "FIFA World Cup", 2018),
    (55, 282, "UEFA Euro", 2024),
    (55, 43,  "UEFA Euro", 2020),
    (223, 282, "Copa America", 2024),
    (1267, 107, "African Cup of Nations", 2023),
]


def fetch_json(url: str, retries: int = 3):
    """Descarga y parsea JSON con reintentos."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"   ⚠️  Falló {url}: {e}")
                return None


def process_match_events(match_id: int) -> dict:
    """
    Extrae métricas agregadas por equipo de un partido.
    Retorna: {team_name: {xg, shots, passes, ...}}
    """
    events = fetch_json(f"{BASE}/events/{match_id}.json")
    if events is None:
        return {}

    team_stats = {}

    for ev in events:
        team = ev.get("team", {}).get("name")
        if not team:
            continue

        if team not in team_stats:
            team_stats[team] = {
                "xg": 0.0, "shots": 0, "shots_on_target": 0, "goals": 0,
                "passes": 0, "passes_completed": 0,
                "pressures": 0, "tackles": 0, "interceptions": 0,
                "dribbles": 0, "dribbles_completed": 0,
            }

        ev_type = ev.get("type", {}).get("name", "")

        if ev_type == "Shot":
            shot = ev.get("shot", {})
            team_stats[team]["shots"] += 1
            team_stats[team]["xg"] += shot.get("statsbomb_xg", 0)
            outcome = shot.get("outcome", {}).get("name", "")
            if outcome == "Goal":
                team_stats[team]["goals"] += 1
                team_stats[team]["shots_on_target"] += 1
            elif outcome == "Saved":
                team_stats[team]["shots_on_target"] += 1

        elif ev_type == "Pass":
            team_stats[team]["passes"] += 1
            # Pass sin outcome = completado
            if ev.get("pass", {}).get("outcome") is None:
                team_stats[team]["passes_completed"] += 1

        elif ev_type == "Pressure":
            team_stats[team]["pressures"] += 1
        elif ev_type == "Interception":
            team_stats[team]["interceptions"] += 1
        elif ev_type == "Duel":
            if ev.get("duel", {}).get("type", {}).get("name") == "Tackle":
                team_stats[team]["tackles"] += 1
        elif ev_type == "Dribble":
            team_stats[team]["dribbles"] += 1
            if ev.get("dribble", {}).get("outcome", {}).get("name") == "Complete":
                team_stats[team]["dribbles_completed"] += 1

    return team_stats


def build_summary():
    """Procesa todos los torneos y genera el resumen por equipo."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    all_records = []

    for comp_id, season_id, comp_name, year in NATIONAL_TEAM_COMPS:
        print(f"\n🏆 Procesando {comp_name} {year}...")
        matches = fetch_json(f"{BASE}/matches/{comp_id}/{season_id}.json")
        if matches is None:
            continue

        print(f"   {len(matches)} partidos encontrados")

        for i, match in enumerate(matches):
            mid = match["match_id"]
            home = match["home_team"]["home_team_name"]
            away = match["away_team"]["away_team_name"]

            stats = process_match_events(mid)
            if not stats:
                continue

            # Guardar registro por equipo en este partido
            for team, s in stats.items():
                opponent = away if team == home else home
                is_home = team == home
                gf = match["home_score"] if is_home else match["away_score"]
                ga = match["away_score"] if is_home else match["home_score"]

                # xG concedido = xG del oponente
                opp_xg = stats.get(opponent, {}).get("xg", np.nan)

                all_records.append({
                    "team": team,
                    "competition": comp_name,
                    "year": year,
                    "match_id": mid,
                    "opponent": opponent,
                    "goals_for": gf,
                    "goals_against": ga,
                    "xg_for": round(s["xg"], 3),
                    "xg_against": round(opp_xg, 3) if not np.isnan(opp_xg) else np.nan,
                    "shots": s["shots"],
                    "shots_on_target": s["shots_on_target"],
                    "pass_accuracy": round(s["passes_completed"] / s["passes"], 3) if s["passes"] > 0 else np.nan,
                    "pressures": s["pressures"],
                    "interceptions": s["interceptions"],
                    "tackles": s["tackles"],
                })

            if (i + 1) % 10 == 0:
                print(f"   ... {i+1}/{len(matches)} partidos procesados")

    # Guardar dataset de partidos individuales (ligero)
    matches_df = pd.DataFrame(all_records)
    matches_df.to_parquet(PROCESSED / "statsbomb_matches.parquet", index=False)
    print(f"\n💾 {len(matches_df)} registros de partido guardados")

    # Agregar a resumen por equipo (promedio de todas sus apariciones)
    team_summary = matches_df.groupby("team").agg(
        matches=("match_id", "count"),
        avg_xg_for=("xg_for", "mean"),
        avg_xg_against=("xg_against", "mean"),
        avg_goals_for=("goals_for", "mean"),
        avg_goals_against=("goals_against", "mean"),
        avg_shots=("shots", "mean"),
        avg_pass_accuracy=("pass_accuracy", "mean"),
        avg_pressures=("pressures", "mean"),
    ).reset_index()

    # xG diff: capacidad neta (ataque - defensa)
    team_summary["xg_diff"] = team_summary["avg_xg_for"] - team_summary["avg_xg_against"]
    # Eficiencia: goles reales vs esperados (finishing)
    team_summary["finishing"] = team_summary["avg_goals_for"] - team_summary["avg_xg_for"]

    team_summary = team_summary.round(3).sort_values("xg_diff", ascending=False)
    team_summary.to_parquet(PROCESSED / "statsbomb_team_stats.parquet", index=False)
    print(f"💾 Resumen de {len(team_summary)} selecciones guardado")
    print("\nTop 10 selecciones por xG diff:")
    print(team_summary.head(10).to_string(index=False))

    return team_summary


if __name__ == "__main__":
    build_summary()
