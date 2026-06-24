"""
odds_features.py
Obtiene cuotas de apuestas pre-partido para selecciones desde The Odds API.

Free tier: 500 requests/mes — suficiente para el Mundial 2026 completo
  (104 partidos × 1 request = 104 requests en todo el torneo)

Registro: https://the-odds-api.com (gratis, sin tarjeta de crédito)
El API key se configura en Streamlit Cloud como variable de entorno:
  ODDS_API_KEY = "tu_key_aquí"

Las cuotas son el feature de mayor impacto potencial porque incorporan
información que el modelo no puede calcular: lesiones, planteles,
motivación táctica, condiciones meteorológicas, forma de individuos clave.
"""
import os
import json
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
ODDS_CACHE = PROCESSED / "odds_cache.json"

# Mapeo de nombres entre The Odds API y el dataset martj42
ODDS_TO_MARTJ42 = {
    "USA": "United States",
    "Ivory Coast": "Ivory Coast",
    "South Korea": "South Korea",
    "Czech Republic": "Czech Republic",
    "DR Congo": "DR Congo",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "New Zealand": "New Zealand",
    "Saudi Arabia": "Saudi Arabia",
    "Cape Verde Islands": "Cape Verde",
    "Curaçao": "Curaçao",
}


def get_api_key() -> str | None:
    """Lee la API key desde variable de entorno o archivo local."""
    # En Streamlit Cloud: configurar en Settings → Secrets
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key
    # Localmente: archivo .env o directamente
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def fetch_wc2026_odds(api_key: str) -> dict:
    """
    Obtiene cuotas 1X2 de los próximos partidos del Mundial 2026.
    Retorna dict: {(home, away): {prob_h, prob_d, prob_a, bookmaker, updated}}
    Consume 1 request de la cuota mensual.
    """
    base_url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_2026/odds"
    params = urllib.parse.urlencode({
        "apiKey": api_key,
        "regions": "eu",           # cuotas europeas (más completas)
        "markets": "h2h",          # 1X2
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    })

    try:
        url = f"{base_url}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "football-predictor/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())

        # Procesar respuesta
        results = {}
        for match in data:
            home_raw = match.get("home_team", "")
            away_raw = match.get("away_team", "")
            home = ODDS_TO_MARTJ42.get(home_raw, home_raw)
            away = ODDS_TO_MARTJ42.get(away_raw, away_raw)

            # Usar promedio de todos los bookmakers disponibles
            ph_list, pd_list, pa_list = [], [], []

            for bk in match.get("bookmakers", []):
                for market in bk.get("markets", []):
                    if market["key"] == "h2h":
                        odds = {o["name"]: o["price"] for o in market["outcomes"]}
                        oh = odds.get(home_raw, odds.get(home))
                        oa = odds.get(away_raw, odds.get(away))
                        # Odds incluyen home, away y Draw
                        od = next((odds[k] for k in odds if "draw" in k.lower()), None)

                        if oh and od and oa:
                            # Convertir a probabilidades implícitas (sin margen)
                            total = 1/oh + 1/od + 1/oa
                            ph_list.append((1/oh) / total)
                            pd_list.append((1/od) / total)
                            pa_list.append((1/oa) / total)

            if ph_list:
                results[(home, away)] = {
                    "prob_home": float(np.mean(ph_list)),
                    "prob_draw": float(np.mean(pd_list)),
                    "prob_away": float(np.mean(pa_list)),
                    "n_bookmakers": len(ph_list),
                    "commence_time": match.get("commence_time", ""),
                    "updated": datetime.now(timezone.utc).isoformat(),
                }

        # Guardar en caché
        cache_data = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "matches": {f"{h}|{a}": v for (h, a), v in results.items()}
        }
        PROCESSED.mkdir(parents=True, exist_ok=True)
        ODDS_CACHE.write_text(json.dumps(cache_data, indent=2))
        print(f"   ✅ {len(results)} partidos con cuotas descargados")
        return results

    except Exception as e:
        print(f"   ⚠️  Error al obtener cuotas: {e}")
        return {}


def load_odds_cache() -> dict:
    """Carga el caché local de cuotas."""
    if not ODDS_CACHE.exists():
        return {}
    try:
        data = json.loads(ODDS_CACHE.read_text())
        results = {}
        for key, v in data.get("matches", {}).items():
            h, a = key.split("|", 1)
            results[(h, a)] = v
        return results
    except Exception:
        return {}


def get_match_odds(home: str, away: str, odds_cache: dict) -> dict | None:
    """
    Retorna las cuotas de un partido específico del caché.
    Prueba ambas direcciones (home/away pueden estar invertidos).
    """
    if (home, away) in odds_cache:
        return odds_cache[(home, away)]
    # Probar si están invertidos
    if (away, home) in odds_cache:
        inv = odds_cache[(away, home)]
        return {
            "prob_home": inv["prob_away"],
            "prob_draw": inv["prob_draw"],
            "prob_away": inv["prob_home"],
            "n_bookmakers": inv["n_bookmakers"],
            "inverted": True,
        }
    return None


def blend_elo_with_odds(
    elo_probs: dict,
    odds: dict | None,
    elo_weight: float = 0.45,
) -> dict:
    """
    Combina las probabilidades del modelo Elo/XGBoost con las del mercado.
    Peso por defecto: 45% modelo + 55% mercado.

    El mercado incorpora info que el modelo no tiene (lesiones, planteles).
    Estudios muestran que el mercado es más predictivo que cualquier modelo
    de datos históricos para partidos individuales.

    Args:
        elo_probs: {"prob_home": x, "prob_draw": x, "prob_away": x}
        odds: dict de cuotas o None si no disponible
        elo_weight: peso del modelo (0-1). Default 0.45 basado en literatura.

    Returns:
        Probabilidades fusionadas.
    """
    if odds is None:
        return elo_probs

    mkt_weight = 1 - elo_weight
    blended = {
        "prob_home": elo_weight * elo_probs["prob_home"] + mkt_weight * odds["prob_home"],
        "prob_draw": elo_weight * elo_probs["prob_draw"] + mkt_weight * odds["prob_draw"],
        "prob_away": elo_weight * elo_probs["prob_away"] + mkt_weight * odds["prob_away"],
    }

    # Normalizar
    total = sum(blended.values())
    blended = {k: v/total for k, v in blended.items()}
    blended["source"] = "modelo + mercado"
    blended["mkt_prob_home"] = odds["prob_home"]
    blended["mkt_prob_draw"] = odds["prob_draw"]
    blended["mkt_prob_away"] = odds["prob_away"]
    blended["n_bookmakers"] = odds.get("n_bookmakers", 0)

    return blended


def odds_cache_is_fresh(max_hours: float = 4.0) -> bool:
    """Verifica si el caché de cuotas es reciente (default: 4 horas)."""
    if not ODDS_CACHE.exists():
        return False
    try:
        data = json.loads(ODDS_CACHE.read_text())
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        return age_hours < max_hours
    except Exception:
        return False
