"""
wc2026_simulator.py
Simulador Monte Carlo del Mundial 2026.
Simula N torneos completos desde el estado actual (jornada 3 pendiente)
y calcula las probabilidades de cada selección en cada ronda.
"""
import numpy as np
import json
import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
SIM_CACHE = PROCESSED / "wc2026_simulation.json"

# Cruces del R32 según el cuadro oficial FIFA 2026
# slot → (posición_grupo1, posición_grupo2)
# Simplificado: top 2 de cada grupo en cruces cruzados
R32_MATCHUPS = [
    ("1A", "2B"), ("1B", "2A"),
    ("1C", "2D"), ("1D", "2C"),
    ("1E", "2F"), ("1F", "2E"),
    ("1G", "2H"), ("1H", "2G"),
    ("1I", "2J"), ("1J", "2I"),
    ("1K", "2L"), ("1L", "2K"),
    # Los 8 mejores terceros van contra los 8 primeros peor posicionados
    # Simplificado por ahora
]

# Orden de selección de mejores terceros según FIFA
# Los 8 mejores terceros de los grupos A-L
THIRD_PLACE_ORDER = ['A','B','C','D','E','F','G','H','I','J','K','L']


def simulate_match_result(prob_home: float, prob_draw: float, prob_away: float) -> str:
    """Simula el resultado de un partido."""
    probs = np.array([prob_home, prob_draw, prob_away])
    probs = np.clip(probs, 0.001, 1.0)
    probs /= probs.sum()
    return np.random.choice(['H', 'D', 'A'], p=probs)


def elo_win_prob(elo_home: float, elo_away: float) -> float:
    """Probabilidad de que el local gane según Elo."""
    return 1 / (1 + 10 ** ((elo_away - elo_home) / 400))


def simulate_knockout_match(home: str, away: str, elo_map: dict) -> str:
    """
    Simula un partido eliminatorio (sin empate posible).
    Usa Elo para determinar el ganador si va a penales.
    """
    h_elo = elo_map.get(home, 1500)
    a_elo = elo_map.get(away, 1500)
    p_home = elo_win_prob(h_elo, a_elo)
    return home if np.random.random() < p_home else away


def run_simulation(
    groups: dict,
    current_pts: dict,
    current_gf: dict,
    current_ga: dict,
    j3_probs: dict,
    elo_map: dict,
    n_sim: int = 3000,
    seed: int = None,
) -> dict:
    """
    Simula el Mundial completo N veces.

    Args:
        groups: {grupo: [equipos]}
        current_pts/gf/ga: estado actual tras jornadas 1-2
        j3_probs: {(home, away): (p_home, p_draw, p_away)} para partidos de J3
        elo_map: {team: elo_rating}
        n_sim: número de simulaciones

    Returns:
        {team: {champion, final, sf, qf, r16, r32, qualified}} probabilidades
    """
    if seed is not None:
        np.random.seed(seed)

    all_teams = [t for g in groups.values() for t in g]
    reach = {t: {
        'qualified': 0, 'r32': 0, 'r16': 0,
        'qf': 0, 'sf': 0, 'final': 0, 'champion': 0
    } for t in all_teams}

    for _ in range(n_sim):
        # ── Simular jornada 3 ─────────────────────────────────────────────────
        sim_pts = current_pts.copy()
        sim_gf = current_gf.copy()
        sim_ga = current_ga.copy()

        for (h, a), (ph, pd, pa) in j3_probs.items():
            r = simulate_match_result(ph, pd, pa)
            if r == 'H':
                sim_pts[h] = sim_pts.get(h, 0) + 3
                sim_gf[h] = sim_gf.get(h, 0) + 1
                sim_ga[a] = sim_ga.get(a, 0) + 1
            elif r == 'D':
                sim_pts[h] = sim_pts.get(h, 0) + 1
                sim_pts[a] = sim_pts.get(a, 0) + 1
                sim_gf[h] = sim_gf.get(h, 0) + 1
                sim_gf[a] = sim_gf.get(a, 0) + 1
                sim_ga[h] = sim_ga.get(h, 0) + 1
                sim_ga[a] = sim_ga.get(a, 0) + 1
            else:
                sim_pts[a] = sim_pts.get(a, 0) + 3
                sim_gf[a] = sim_gf.get(a, 0) + 1
                sim_ga[h] = sim_ga.get(h, 0) + 1

        # ── Determinar clasificados ────────────────────────────────────────────
        group_order = {}  # grupo → [1°, 2°, 3°, 4°]
        for g, teams in groups.items():
            sorted_teams = sorted(teams, key=lambda t: (
                -sim_pts.get(t, 0),
                -(sim_gf.get(t, 0) - sim_ga.get(t, 0)),
                -sim_gf.get(t, 0),
                -elo_map.get(t, 1500)  # Elo como desempate final
            ))
            group_order[g] = sorted_teams

        # Top 2 de cada grupo clasifican
        qualified_slots = {}  # "1A", "2B", etc.
        thirds = []
        for g, order in group_order.items():
            qualified_slots[f'1{g}'] = order[0]
            qualified_slots[f'2{g}'] = order[1]
            reach[order[0]]['qualified'] += 1
            reach[order[1]]['qualified'] += 1
            thirds.append({
                'team': order[2], 'group': g,
                'pts': sim_pts.get(order[2], 0),
                'gd': sim_gf.get(order[2], 0) - sim_ga.get(order[2], 0),
                'gf': sim_gf.get(order[2], 0),
            })

        # 8 mejores terceros también clasifican
        thirds_sorted = sorted(thirds, key=lambda x: (-x['pts'], -x['gd'], -x['gf']))
        for i, t in enumerate(thirds_sorted[:8]):
            qualified_slots[f'3{t["group"]}'] = t['team']
            reach[t['team']]['qualified'] += 1

        # ── Simular R32 (dieciseisavos) ────────────────────────────────────────
        # Bracket oficial FIFA 2026 (simplificado: cruces por posición de grupo)
        r32_winners = []
        r32_pairs = [
            (qualified_slots.get('1A'), qualified_slots.get('2B')),
            (qualified_slots.get('1B'), qualified_slots.get('2A')),
            (qualified_slots.get('1C'), qualified_slots.get('2D')),
            (qualified_slots.get('1D'), qualified_slots.get('2C')),
            (qualified_slots.get('1E'), qualified_slots.get('2F')),
            (qualified_slots.get('1F'), qualified_slots.get('2E')),
            (qualified_slots.get('1G'), qualified_slots.get('2H')),
            (qualified_slots.get('1H'), qualified_slots.get('2G')),
            (qualified_slots.get('1I'), qualified_slots.get('2J')),
            (qualified_slots.get('1J'), qualified_slots.get('2I')),
            (qualified_slots.get('1K'), qualified_slots.get('2L')),
            (qualified_slots.get('1L'), qualified_slots.get('2K')),
        ]
        # Los 8 terceros llenan los 4 partidos restantes del R32
        t3_slots = [t['team'] for t in thirds_sorted[:8]]
        top1_slots = [qualified_slots.get(f'1{g}') for g in ['I','J','K','L']]
        for i, (t1, t3) in enumerate(zip(top1_slots, t3_slots[:4])):
            if t1 and t3:
                r32_pairs.append((t1, t3))
        # 4 cruces más con los terceros restantes
        for t1, t3 in zip(top1_slots[4:] if len(top1_slots)>4 else [], t3_slots[4:]):
            if t1 and t3:
                r32_pairs.append((t1, t3))

        for h, a in r32_pairs:
            if h and a:
                w = simulate_knockout_match(h, a, elo_map)
                r32_winners.append(w)
                reach[w]['r32'] += 1

        # ── R16 ───────────────────────────────────────────────────────────────
        r16_winners = []
        for i in range(0, len(r32_winners) - 1, 2):
            w = simulate_knockout_match(r32_winners[i], r32_winners[i+1], elo_map)
            r16_winners.append(w)
            reach[w]['r16'] += 1

        # ── QF ────────────────────────────────────────────────────────────────
        qf_winners = []
        for i in range(0, len(r16_winners) - 1, 2):
            w = simulate_knockout_match(r16_winners[i], r16_winners[i+1], elo_map)
            qf_winners.append(w)
            reach[w]['qf'] += 1

        # ── SF ────────────────────────────────────────────────────────────────
        sf_winners = []
        for i in range(0, len(qf_winners) - 1, 2):
            w = simulate_knockout_match(qf_winners[i], qf_winners[i+1], elo_map)
            sf_winners.append(w)
            reach[w]['sf'] += 1

        # ── Final ─────────────────────────────────────────────────────────────
        if len(sf_winners) >= 2:
            champion = simulate_knockout_match(sf_winners[0], sf_winners[1], elo_map)
            reach[champion]['final'] += 1
            reach[champion]['champion'] += 1

    # Convertir a probabilidades
    return {t: {k: round(v / n_sim, 4) for k, v in counts.items()}
            for t, counts in reach.items()}


def load_simulation() -> dict | None:
    """Carga la última simulación guardada."""
    if not SIM_CACHE.exists():
        return None
    try:
        return json.loads(SIM_CACHE.read_text())
    except Exception:
        return None


def save_simulation(results: dict):
    """Guarda los resultados de la simulación."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SIM_CACHE.write_text(json.dumps(results))


def get_top_contenders(results: dict, n: int = 16) -> list[dict]:
    """Retorna los N favoritos ordenados por probabilidad de campeonato."""
    sorted_teams = sorted(results.items(), key=lambda x: -x[1].get('champion', 0))
    return [
        {'team': team, **probs}
        for team, probs in sorted_teams[:n]
    ]
