"""
value_betting.py
Análisis de valor en cuotas: compara las probabilidades del modelo
con las probabilidades implícitas del mercado de apuestas.

Valor positivo = el modelo cree que hay más probabilidad de lo que el mercado paga.
Esto es lo que usan los apostadores profesionales (value betting).

Fórmula: Valor = (Prob_modelo - Prob_implícita_mercado)
Kelly criterion: f = (p*b - (1-p)) / b  donde b = cuota - 1
"""
import numpy as np


def remove_margin(odds_home: float, odds_draw: float, odds_away: float) -> tuple:
    """
    Elimina el margen de la casa de apuestas (overround).
    Convierte cuotas decimales a probabilidades implícitas reales.
    """
    total = 1/odds_home + 1/odds_draw + 1/odds_away
    p_home = (1/odds_home) / total
    p_draw = (1/odds_draw) / total
    p_away = (1/odds_away) / total
    margin = total - 1  # margen típico 5-8%
    return p_home, p_draw, p_away, margin


def calculate_value(
    model_prob: float,
    market_odds: float,
    kelly_fraction: float = 0.25,  # Kelly fraccionado (más conservador)
) -> dict:
    """
    Calcula el valor de una apuesta.
    
    Args:
        model_prob: probabilidad del modelo (0-1)
        market_odds: cuota decimal del mercado (ej: 2.50)
        kelly_fraction: fracción de Kelly a usar (0.25 = quarter Kelly)
    
    Returns:
        {value, edge, kelly_stake, ev, recommendation}
    """
    # Probabilidad implícita del mercado (sin margen)
    implied_prob = 1 / market_odds
    
    # Edge: cuánto más cree el modelo vs el mercado
    edge = model_prob - implied_prob
    
    # Expected Value: cuánto se gana en promedio por cada unidad apostada
    ev = model_prob * (market_odds - 1) - (1 - model_prob)
    
    # Kelly criterion (stake óptimo como % del bankroll)
    b = market_odds - 1
    kelly = (model_prob * b - (1 - model_prob)) / b if b > 0 else 0
    kelly_stake = max(0, kelly * kelly_fraction)
    
    # Recomendación
    if edge > 0.05 and ev > 0.05:
        recommendation = "✅ Valor alto"
    elif edge > 0.02 and ev > 0.02:
        recommendation = "🟡 Valor moderado"
    elif edge > 0:
        recommendation = "⬜ Valor marginal"
    else:
        recommendation = "❌ Sin valor"
    
    return {
        "model_prob": model_prob,
        "implied_prob": implied_prob,
        "market_odds": market_odds,
        "edge": edge,
        "ev": ev,
        "kelly_stake": kelly_stake,
        "recommendation": recommendation,
    }


def analyze_match_value(
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    odds_home: float = None,
    odds_draw: float = None,
    odds_away: float = None,
    market_probs: dict = None,
) -> dict:
    """
    Analiza el valor en las 3 apuestas de un partido (1X2).
    
    Puede recibir cuotas decimales O probabilidades de mercado directamente.
    Returns análisis completo de valor para local, empate y visitante.
    """
    # Obtener probabilidades del mercado
    if market_probs:
        mkt_ph = market_probs.get("prob_home", 0)
        mkt_pd = market_probs.get("prob_draw", 0)
        mkt_pa = market_probs.get("prob_away", 0)
        # Cuotas implícitas (con margen del 5%)
        margin = 1.05
        odds_h = margin / mkt_ph if mkt_ph > 0 else 99
        odds_d = margin / mkt_pd if mkt_pd > 0 else 99
        odds_a = margin / mkt_pa if mkt_pa > 0 else 99
    elif odds_home and odds_draw and odds_away:
        mkt_ph, mkt_pd, mkt_pa, _ = remove_margin(odds_home, odds_draw, odds_away)
        odds_h, odds_d, odds_a = odds_home, odds_draw, odds_away
    else:
        return None

    return {
        "home": calculate_value(prob_home, odds_h),
        "draw": calculate_value(prob_draw, odds_d),
        "away": calculate_value(prob_away, odds_a),
        "best_value": max(
            [("home", prob_home, odds_h), ("draw", prob_draw, odds_d), ("away", prob_away, odds_a)],
            key=lambda x: x[1] * (x[2] - 1) - (1 - x[1])  # mayor EV
        )[0],
        "market_margin": (1/odds_h + 1/odds_d + 1/odds_a - 1) if odds_home else 0,
    }


def format_value_summary(analysis: dict, home_team: str, away_team: str) -> str:
    """Genera un resumen legible del análisis de valor."""
    if not analysis:
        return "Sin datos de cuotas disponibles"
    
    lines = []
    for side, team in [("home", home_team), ("draw", "Empate"), ("away", away_team)]:
        v = analysis[side]
        if v["edge"] > 0.02:
            lines.append(
                f"{v['recommendation']} {team}: modelo {v['model_prob']:.0%} "
                f"vs mercado {v['implied_prob']:.0%} "
                f"(edge +{v['edge']:.0%}, EV {v['ev']:+.2f})"
            )
    
    return " | ".join(lines) if lines else "Sin valor detectado en este partido"
