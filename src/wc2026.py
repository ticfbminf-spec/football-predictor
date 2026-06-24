"""
wc2026.py
Datos del Mundial 2026: grupos, formato de clasificación, bracket.
48 selecciones, 12 grupos (A-L), Round of 32 → R16 → QF → SF → Final.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ── Grupos del Mundial 2026 ───────────────────────────────────────────────────
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Mapeo de nombres entre martj42 y nombres de pantalla
DISPLAY_NAMES = {
    "Czech Republic": "Chequia",
    "Bosnia and Herzegovina": "Bosnia",
    "United States": "USA",
    "Ivory Coast": "Costa de Marfil",
    "DR Congo": "DR Congo",
    "South Korea": "Corea del Sur",
    "Saudi Arabia": "Arabia Saudí",
    "Cape Verde": "Cabo Verde",
    "New Zealand": "Nueva Zelanda",
}

def display(team: str) -> str:
    return DISPLAY_NAMES.get(team, team)


def get_wc2026_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae todos los partidos del Mundial 2026 del dataset histórico,
    aplicando parches de resultados recientes aún no en martj42."""
    wc = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"].dt.year == 2026)
    ].copy()

    # Aplicar parches de resultados recientes
    try:
        from src.wc2026_patches import apply_patches
        wc = apply_patches(wc)
    except Exception:
        pass

    return wc.sort_values("date").reset_index(drop=True)


def compute_group_standings(wc_df: pd.DataFrame) -> dict:
    """
    Calcula la tabla de posiciones de cada grupo.
    Retorna dict: {grupo: DataFrame con columnas equipo, PJ, G, E, P, GF, GA, DG, Pts}
    """
    standings = {}

    for group, teams in GROUPS.items():
        rows = []
        for team in teams:
            mask = (
                ((wc_df["home_team"] == team) | (wc_df["away_team"] == team)) &
                wc_df["home_score"].notna()
            )
            matches = wc_df[mask]

            pj = g = e = p = gf = ga = 0
            for _, r in matches.iterrows():
                pj += 1
                if r["home_team"] == team:
                    gf += r["home_score"]; ga += r["away_score"]
                    res = r["result"]
                    if res == "H": g += 1
                    elif res == "D": e += 1
                    else: p += 1
                else:
                    gf += r["away_score"]; ga += r["home_score"]
                    res = r["result"]
                    if res == "A": g += 1
                    elif res == "D": e += 1
                    else: p += 1

            rows.append({
                "team": team,
                "display": display(team),
                "PJ": pj, "G": g, "E": e, "P": p,
                "GF": int(gf), "GA": int(ga),
                "DG": int(gf - ga),
                "Pts": g * 3 + e,
            })

        table = pd.DataFrame(rows).sort_values(
            ["Pts", "DG", "GF"], ascending=[False, False, False]
        ).reset_index(drop=True)
        table.index += 1
        standings[group] = table

    return standings


def get_qualified_teams(standings: dict) -> dict:
    """
    Determina quién clasifica según las reglas del Mundial 2026:
    - Top 2 de cada grupo → clasifican directo (24 equipos)
    - 8 mejores terceros → también clasifican (8 equipos)
    Total: 32 equipos para el Round of 32 (dieciseisavos)

    Retorna:
    {grupo: {"1st": equipo, "2nd": equipo, "3rd": equipo, "confirmed": bool}}
    """
    result = {}
    third_place_teams = []

    for group, table in standings.items():
        # Solo confirmado si todos han jugado 3 partidos
        all_played = (table["PJ"] == 3).all()

        first  = table.iloc[0]["team"] if len(table) > 0 else None
        second = table.iloc[1]["team"] if len(table) > 1 else None
        third  = table.iloc[2]["team"] if len(table) > 2 else None

        result[group] = {
            "1st": first,
            "2nd": second,
            "3rd": third,
            "confirmed": all_played,
            "table": table,
        }

        if third and all_played:
            third_row = table.iloc[2]
            third_place_teams.append({
                "team": third,
                "group": group,
                "Pts": third_row["Pts"],
                "DG": third_row["DG"],
                "GF": third_row["GF"],
            })

    # Top 8 terceros
    if third_place_teams:
        third_df = pd.DataFrame(third_place_teams).sort_values(
            ["Pts", "DG", "GF"], ascending=[False, False, False]
        ).head(8)
        qualified_thirds = set(third_df["team"].tolist())
    else:
        qualified_thirds = set()

    for group in result:
        result[group]["3rd_qualified"] = result[group]["3rd"] in qualified_thirds

    return result


def get_remaining_matches(wc_df: pd.DataFrame) -> list:
    """
    Retorna los partidos de grupos que aún no se han jugado.
    """
    played = set(zip(wc_df["home_team"], wc_df["away_team"]))
    remaining = []

    # Jornada 3 programada (25-27 junio 2026)
    jornada3 = [
        # Grupo A
        ("Czech Republic", "Mexico"), ("South Africa", "South Korea"),
        # Grupo B
        ("Switzerland", "Canada"), ("Bosnia and Herzegovina", "Qatar"),
        # Grupo C
        ("Scotland", "Brazil"), ("Morocco", "Haiti"),
        # Grupo D
        ("Turkey", "United States"), ("Paraguay", "Australia"),
        # Grupo E
        ("Curaçao", "Ivory Coast"), ("Ecuador", "Germany"),
        # Grupo F
        ("Japan", "Sweden"), ("Tunisia", "Netherlands"),
        # Grupo G
        ("Egypt", "Iran"), ("New Zealand", "Belgium"),
        # Grupo H
        ("Saudi Arabia", "Cape Verde"), ("Uruguay", "Spain"),
        # Grupo I
        ("France", "Norway"), ("Iraq", "Senegal"),
        # Grupo J
        ("Algeria", "Austria"), ("Jordan", "Argentina"),
        # Grupo K
        ("Portugal", "Uzbekistan"), ("Colombia", "DR Congo"),
        # Grupo L
        ("England", "Ghana"), ("Croatia", "Panama"),
    ]

    for home, away in jornada3:
        if (home, away) not in played:
            remaining.append({"home": home, "away": away, "phase": "Grupo - Jornada 3"})

    return remaining


# ── Estructura del bracket (Round of 32) ────────────────────────────────────
# Según el cuadro oficial de la FIFA para el Mundial 2026
# Los cruces del R32 son fijos según posición en el grupo
R32_BRACKET = [
    # Partido 1A
    {"match": "R32-1", "home_slot": "1A", "away_slot": "3C/D/E"},
    {"match": "R32-2", "home_slot": "1C", "away_slot": "3A/B/F"},
    {"match": "R32-3", "home_slot": "1B", "away_slot": "3A/C/D"},
    {"match": "R32-4", "home_slot": "1D", "away_slot": "3B/E/F"},
    {"match": "R32-5", "home_slot": "1E", "away_slot": "3G/H/I"},
    {"match": "R32-6", "home_slot": "1G", "away_slot": "3E/F/J"},
    {"match": "R32-7", "home_slot": "1F", "away_slot": "3G/H/K"},
    {"match": "R32-8", "home_slot": "1H", "away_slot": "3F/I/J"},
    {"match": "R32-9",  "home_slot": "1I", "away_slot": "3K/L"},
    {"match": "R32-10", "home_slot": "1K", "away_slot": "3I/L"},
    {"match": "R32-11", "home_slot": "1J", "away_slot": "2L"},
    {"match": "R32-12", "home_slot": "1L", "away_slot": "2J"},
    {"match": "R32-13", "home_slot": "2A", "away_slot": "2B"},
    {"match": "R32-14", "home_slot": "2C", "away_slot": "2D"},
    {"match": "R32-15", "home_slot": "2E", "away_slot": "2F"},
    {"match": "R32-16", "home_slot": "2G", "away_slot": "2H"},
]
