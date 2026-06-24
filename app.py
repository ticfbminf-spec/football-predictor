"""
app.py — Football Predictor
App Streamlit mobile-first para predicción de partidos de selecciones.
Deploy gratuito en Streamlit Community Cloud.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.fetch_historical import load_results, get_team_list
from src.feature_pipeline import build_feature_dataset, load_processed, save_processed
from src.model import train, load_models, predict_match
from src.statsbomb_features import load_statsbomb_summary
from src.elo_ratings import load_current_elo, build_and_save_elo
from src.wc2026 import (get_wc2026_matches, compute_group_standings,
                        get_qualified_teams, get_remaining_matches,
                        GROUPS, display as wc_display)
from src.wc2026_patches import PATCHES, apply_patches
from src.wc2026_simulator import run_simulation, save_simulation, load_simulation, get_top_contenders
from src.value_betting import analyze_match_value, calculate_value
from src.odds_features import (get_api_key, fetch_wc2026_odds, load_odds_cache,
                                get_match_odds, blend_elo_with_odds, odds_cache_is_fresh)
from src.knockout_logic import (
    load_shootouts, compute_penalty_stats,
    get_phases_for_competition, competition_supports_knockout,
    is_knockout_phase, full_match_prediction, COMPETITION_PHASES,
)

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0d1117; color: #e6edf3; }

  .hero {
    background: linear-gradient(135deg, #1a2744 0%, #0d1117 60%, #1a1a2e 100%);
    border: 1px solid #21d060; border-radius: 16px;
    padding: 1.5rem 1.2rem; margin-bottom: 1.5rem; text-align: center;
  }
  .hero h1 { font-size: 2rem; font-weight: 800; color: #21d060; margin: 0; letter-spacing: -1px; }
  .hero p { color: #8b949e; font-size: 0.85rem; margin: 0.3rem 0 0; }

  .pred-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1.2rem; margin: 0.6rem 0;
  }
  .pred-card.highlight { border-color: #21d060; background: #0d2818; }
  .pred-card.et-card   { border-color: #d29922; background: #1a1500; }
  .pred-card.pen-card  { border-color: #58a6ff; background: #0d1a2e; }

  .mini-metric {
    background: #21262d; border-radius: 8px;
    padding: 0.5rem 0.8rem; text-align: center;
    font-family: 'JetBrains Mono', monospace;
  }
  .mini-metric .val { font-size: 1.4rem; font-weight: 700; color: #21d060; }
  .mini-metric .lbl { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; }

  .stButton > button {
    background: #21d060 !important; color: #0d1117 !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; padding: 0.7rem 1.5rem !important;
    font-size: 1rem !important; width: 100% !important;
  }

  .kick-row {
    display: flex; align-items: center; gap: 0.4rem;
    padding: 0.2rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
  }
  .kick-dot { font-size: 1.1rem; }
  .section-title {
    font-size: 0.75rem; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.1em; margin: 1.2rem 0 0.5rem;
  }
  .phase-badge {
    display: inline-block; font-size: 0.75rem; font-weight: 600;
    padding: 0.15rem 0.6rem; border-radius: 20px; margin-left: 0.4rem;
  }
  .phase-ko  { background: #f85149; color: white; }
  .phase-grp { background: #21262d; color: #8b949e; }

  .h2h-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.4rem 0; border-bottom: 1px solid #21262d; font-size: 0.85rem;
  }
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📥 Actualizando datos...", ttl=86400)  # refresca cada 24h
def get_raw_data():
    return load_results(min_year=1990)

@st.cache_data(show_spinner="⚙️ Construyendo features...", ttl=86400)
def get_features(raw_df):
    cached = load_processed()
    if cached is not None:
        return cached
    features = build_feature_dataset(raw_df, form_window=5, min_year=2000)
    save_processed(features)
    return features

@st.cache_resource(show_spinner="🤖 Entrenando modelo híbrido...")
def get_trained_model(features_df):
    clf, reg, feature_cols, clf_elite, elite_cols, reg_h, reg_a = load_models()
    if clf is None:
        metrics = train(features_df)
        return (metrics["clf"], metrics["reg"], metrics["feature_cols"],
                metrics["clf_elite"], metrics["elite_cols"],
                metrics.get("reg_h"), metrics.get("reg_a"), metrics)
    return clf, reg, feature_cols, clf_elite, elite_cols, None

@st.cache_data(show_spinner="📊 Cargando xG StatsBomb...")
def get_statsbomb():
    return load_statsbomb_summary()

@st.cache_data(show_spinner="⚡ Calculando rating Elo...", ttl=86400)
def get_elo(raw_df):
    # Recalcula el Elo con todo el historial (sin leakage) y lo guarda
    build_and_save_elo(raw_df)
    return load_current_elo()

@st.cache_data(show_spinner="📋 Cargando historial de penales...")
def get_penalty_stats():
    sh = load_shootouts()
    return compute_penalty_stats(sh), sh


# ── Cargar ────────────────────────────────────────────────────────────────────
raw_df    = get_raw_data()
features  = get_features(raw_df)
clf, reg, feature_cols, clf_elite, elite_cols, reg_h, reg_a, train_metrics = get_trained_model(features)
statsbomb_summary = get_statsbomb()
elo_map = get_elo(raw_df)

@st.cache_data(show_spinner="💰 Cargando cuotas...", ttl=14400)  # refresca cada 4h
def get_odds():
    """Carga cuotas del caché local o las descarga si hay API key."""
    api_key = get_api_key()
    if api_key and not odds_cache_is_fresh():
        return fetch_wc2026_odds(api_key)
    return load_odds_cache()

odds_cache = get_odds()
penalty_stats, shootouts_df = get_penalty_stats()
team_list = get_team_list(raw_df)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚽ Football Predictor</h1>
  <p>Resultado · Goles · Tiempo extra · Penales</p>
</div>
""", unsafe_allow_html=True)

n_matches = len(raw_df)
n_teams   = len(team_list)
acc       = f"{train_metrics['result_accuracy']:.0%}" if train_metrics else "—"

c1, c2, c3 = st.columns(3)
for col, val, lbl in [(c1, f"{n_matches:,}", "partidos"), (c2, str(n_teams), "selecciones"), (c3, acc, "precisión")]:
    col.markdown(f'<div class="mini-metric"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_wc, tab_sim, tab_value, tab_rank, tab_stats, tab_h2h, tab_info = st.tabs(["🎯 Predecir", "🌎 Mundial 2026", "🎲 Simulador", "💰 Valor", "🏆 Ranking", "📊 Estadísticas", "⚔️ H2H", "ℹ️ Info"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="section-title">Equipos</div>', unsafe_allow_html=True)

    home_team = st.selectbox("🏠 Local", team_list, index=team_list.index("Brazil") if "Brazil" in team_list else 0)
    away_team = st.selectbox("✈️ Visita", team_list, index=team_list.index("Argentina") if "Argentina" in team_list else 1)

    st.markdown('<div class="section-title">Competencia</div>', unsafe_allow_html=True)

    competition_list = list(COMPETITION_PHASES.keys()) + ["Friendly"]
    competition = st.selectbox("🏆 Competencia", competition_list)

    # Fases de la competencia
    phases_data = get_phases_for_competition(competition)
    all_phases  = phases_data.get("group", []) + phases_data.get("knockout", [])
    knockout_phases = set(phases_data.get("knockout", []))

    phase = st.selectbox(
        "📍 Fase",
        ["— (no especificar)"] + all_phases,
        help="Selecciona la fase para activar lógica de tiempo extra y penales automáticamente",
    )
    phase_val = None if phase == "— (no especificar)" else phase

    # Badge visual de fase
    if phase_val:
        is_ko_auto = is_knockout_phase(competition, phase_val)
        badge_cls  = "phase-ko"  if is_ko_auto else "phase-grp"
        badge_txt  = "⚡ Eliminatoria" if is_ko_auto else "● Grupos"
        st.markdown(f'Fase detectada: <span class="phase-badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)

    neutral = st.toggle("Campo neutral", value=False)

    # Override manual
    with st.expander("⚙️ Ajuste manual de eliminatoria"):
        override_opts = {"Automático (según fase)": None, "Forzar SÍ eliminatoria": True, "Forzar NO eliminatoria": False}
        override_key  = st.radio("Modo eliminatoria", list(override_opts.keys()), horizontal=True)
        knockout_override = override_opts[override_key]

    st.markdown("<br>", unsafe_allow_html=True)

    if home_team == away_team:
        st.warning("Elige dos equipos distintos.")
    else:
        if st.button("⚡ Predecir partido"):
            with st.spinner("Calculando predicción..."):
                base_pred = predict_match(
                    home_team=home_team, away_team=away_team,
                    df=features, competition=competition,
                    neutral=neutral, clf=clf, reg=reg, feature_cols=feature_cols,
                    clf_elite=clf_elite, elite_cols=elite_cols,
                    statsbomb_summary=statsbomb_summary,
                    reg_h=reg_h, reg_a=reg_a,
                )
                full_pred = full_match_prediction(
                    base_prediction=base_pred,
                    competition=competition,
                    phase=phase_val,
                    is_knockout_override=knockout_override,
                    penalty_stats=penalty_stats,
                )

            if "error" in full_pred:
                st.error(full_pred["error"])
            else:
                # ── Resultado en 90' ──────────────────────────────────────────
                result_labels = {"H": f"Gana {home_team}", "D": "Empate", "A": f"Gana {away_team}"}
                result_colors = {"H": "#21d060", "D": "#d29922", "A": "#f85149"}
                res   = full_pred["predicted_result"]
                color = result_colors[res]

                is_ko = full_pred["is_knockout"]
                ko_src = full_pred["knockout_source"]

                # Marcador predicho (si hay regresores de goles separados)
                gh = full_pred.get("predicted_home_goals")
                ga = full_pred.get("predicted_away_goals")
                score_str = f"{gh:.1f} – {ga:.1f}" if gh is not None else f"{full_pred['predicted_total_goals']} goles"

                gh_str = f"{gh:.1f}" if gh else "—"
                ga_str = f"{ga:.1f}" if ga else "—"
                ko_badge = "⚡ <span style='color:#f85149'>Eliminatoria</span> &nbsp;·&nbsp;" if is_ko else ""

                st.markdown(f"""
                <div class="pred-card highlight">
                  <div style="font-size:0.75rem;color:#8b949e">RESULTADO 90 MINUTOS</div>
                  <div style="font-size:1.8rem;font-weight:800;color:{color};margin:0.2rem 0">{result_labels[res]}</div>
                  <div style="display:flex;justify-content:space-around;margin:0.5rem 0;text-align:center">
                    <div>
                      <div style="font-size:1.3rem;font-weight:700;color:#21d060">{gh_str}</div>
                      <div style="font-size:0.7rem;color:#8b949e">xGoles {home_team[:12]}</div>
                    </div>
                    <div style="font-size:1rem;color:#484f58;padding-top:0.4rem">–</div>
                    <div>
                      <div style="font-size:1.3rem;font-weight:700;color:#f85149">{ga_str}</div>
                      <div style="font-size:0.7rem;color:#8b949e">xGoles {away_team[:12]}</div>
                    </div>
                  </div>
                  <div style="color:#8b949e;font-size:0.78rem;text-align:center">
                    {ko_badge}
                    Ensemble XGBoost+Poisson &nbsp;·&nbsp; <span style="opacity:0.5">{ko_src}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Barras de probabilidad 90'
                fig = go.Figure()
                labels = [f"🏠 {home_team}", "Empate", f"✈️ {away_team}"]
                values = [full_pred["prob_home"], full_pred["prob_draw"], full_pred["prob_away"]]
                colors_bar = ["#21d060", "#d29922", "#f85149"]

                fig.add_trace(go.Bar(
                    x=labels, y=[v * 100 for v in values],
                    marker_color=colors_bar,
                    text=[f"{v:.0%}" for v in values],
                    textposition="outside",
                    textfont=dict(color="#e6edf3", size=14),
                ))
                fig.update_layout(
                    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    font_color="#e6edf3", showlegend=False,
                    margin=dict(t=20, b=10, l=10, r=10), height=200,
                    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # ── Badge de cuotas del mercado ───────────────────────────────
                odds_b = full_pred.get("odds_blended")
                if odds_b:
                    n_bk = odds_b.get("n_bookmakers", 0)
                    st.markdown(f"""
                    <div style="text-align:center;margin:0.3rem 0">
                      <span style="background:#1a1a2e;border:1px solid #58a6ff;color:#58a6ff;
                                   font-size:0.7rem;font-weight:600;padding:0.2rem 0.7rem;border-radius:20px">
                        💰 Cuotas del mercado integradas ({n_bk} casas de apuestas) · 55% peso
                      </span>
                    </div>
                    <div class="pred-card" style="margin:0.4rem 0;padding:0.7rem">
                      <div style="font-size:0.72rem;color:#8b949e;margin-bottom:0.4rem">COMPARATIVA: MODELO vs MERCADO</div>
                      <div style="display:flex;justify-content:space-around;font-family:monospace;font-size:0.82rem">
                        <div style="text-align:center">
                          <div style="color:#484f58;font-size:0.65rem">MODELO</div>
                          <div>{home_team[:10]}: <strong style="color:#21d060">{odds_b.get("mkt_prob_home",0):.0%}</strong></div>
                          <div>Empate: <strong style="color:#d29922">{odds_b.get("mkt_prob_draw",0):.0%}</strong></div>
                          <div>{away_team[:10]}: <strong style="color:#f85149">{odds_b.get("mkt_prob_away",0):.0%}</strong></div>
                        </div>
                        <div style="color:#484f58;font-size:1.2rem;padding-top:0.5rem">⟷</div>
                        <div style="text-align:center">
                          <div style="color:#484f58;font-size:0.65rem">MERCADO</div>
                          <div>{home_team[:10]}: <strong style="color:#21d060">{odds_b.get("mkt_prob_home",0):.0%}</strong></div>
                          <div>Empate: <strong style="color:#d29922">{odds_b.get("mkt_prob_draw",0):.0%}</strong></div>
                          <div>{away_team[:10]}: <strong style="color:#f85149">{odds_b.get("mkt_prob_away",0):.0%}</strong></div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                                # ── Comparación de Elo ────────────────────────────────────────
                h_elo = elo_map.get(home_team, 1500)
                a_elo = elo_map.get(away_team, 1500)
                elo_fav = home_team if h_elo > a_elo else away_team
                elo_gap = abs(h_elo - a_elo)

                st.markdown('<div class="section-title">Rating Elo (fuerza del equipo)</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="pred-card">
                  <div style="display:flex;justify-content:space-around;text-align:center;align-items:center">
                    <div>
                      <div style="font-size:1.5rem;font-weight:800;color:{'#21d060' if h_elo>=a_elo else '#8b949e'}">{h_elo:.0f}</div>
                      <div style="font-size:0.72rem;color:#8b949e">🏠 {home_team}</div>
                    </div>
                    <div style="font-size:0.8rem;color:#484f58">
                      Δ {elo_gap:.0f}<br>
                      <span style="font-size:0.65rem">a favor de<br>{elo_fav}</span>
                    </div>
                    <div>
                      <div style="font-size:1.5rem;font-weight:800;color:{'#f85149' if a_elo>h_elo else '#8b949e'}">{a_elo:.0f}</div>
                      <div style="font-size:0.72rem;color:#8b949e">✈️ {away_team}</div>
                    </div>
                  </div>
                  <div style="font-size:0.7rem;color:#484f58;text-align:center;margin-top:0.5rem">
                    El Elo es el predictor más fuerte del modelo (55% de su decisión)
                  </div>
                </div>
                """, unsafe_allow_html=True)

                                # ── Badge de modelo + xG (si élite) ───────────────────────────
                model_used = full_pred.get("model_used", "base")
                home_xg = full_pred.get("home_xg")
                away_xg = full_pred.get("away_xg")

                if model_used.startswith("élite") and home_xg and away_xg and home_xg.get("has_data") and away_xg.get("has_data"):
                    st.markdown(f"""
                    <div style="text-align:center;margin:0.3rem 0">
                      <span style="background:#1a2e1a;border:1px solid #21d060;color:#21d060;
                                   font-size:0.7rem;font-weight:600;padding:0.2rem 0.7rem;border-radius:20px">
                        ⭐ Modelo de élite activado · usa xG real de StatsBomb
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-title">xG histórico (StatsBomb)</div>', unsafe_allow_html=True)
                    col_hx, col_ax = st.columns(2)
                    for col, team, xg, side in [
                        (col_hx, home_team, home_xg, "🏠"),
                        (col_ax, away_team, away_xg, "✈️"),
                    ]:
                        with col:
                            st.markdown(f"""
                            <div class="pred-card">
                              <div style="font-weight:700;font-size:0.85rem;margin-bottom:0.4rem">{side} {team}</div>
                              <div style="font-family:monospace;font-size:0.82rem;color:#8b949e">
                                xG ataque: <strong style="color:#21d060">{xg['xg_for']:.2f}</strong><br>
                                xG defensa: <strong style="color:#f85149">{xg['xg_against']:.2f}</strong><br>
                                xG neto: <strong style="color:#58a6ff">{xg['xg_diff']:+.2f}</strong>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                                # ── Tiempo extra y penales ────────────────────────────────────
                etp = full_pred.get("et_penalties")

                if is_ko and etp:
                    st.markdown('<div class="section-title">Si hay empate en 90\'</div>', unsafe_allow_html=True)

                    prob_et  = etp["prob_goes_to_et"]
                    prob_pen = etp["prob_goes_to_penalties"]
                    et_data  = etp["extra_time"]
                    pen_data = etp["penalties"]

                    # Card de probabilidades del camino
                    st.markdown(f"""
                    <div class="pred-card et-card">
                      <div style="font-size:0.75rem;color:#d29922;font-weight:600">⏱ TIEMPO EXTRA</div>
                      <div style="margin-top:0.5rem;display:flex;justify-content:space-around;text-align:center">
                        <div>
                          <div style="font-size:1.3rem;font-weight:700;color:#21d060">{et_data['prob_home_et']:.0%}</div>
                          <div style="font-size:0.7rem;color:#8b949e">{home_team}<br>gana ET</div>
                        </div>
                        <div>
                          <div style="font-size:1.3rem;font-weight:700;color:#d29922">{et_data['prob_penalties']:.0%}</div>
                          <div style="font-size:0.7rem;color:#8b949e">Va a<br>penales</div>
                        </div>
                        <div>
                          <div style="font-size:1.3rem;font-weight:700;color:#f85149">{et_data['prob_away_et']:.0%}</div>
                          <div style="font-size:0.7rem;color:#8b949e">{away_team}<br>gana ET</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Card de penales
                    h_pen_pct = pen_data["prob_home_wins"]
                    a_pen_pct = pen_data["prob_away_wins"]
                    h_hist = pen_data["home_historical"]
                    a_hist = pen_data["away_historical"]

                    st.markdown(f"""
                    <div class="pred-card pen-card">
                      <div style="font-size:0.75rem;color:#58a6ff;font-weight:600">🎯 TANDA DE PENALES</div>
                      <div style="margin-top:0.5rem;display:flex;justify-content:space-around;text-align:center">
                        <div>
                          <div style="font-size:1.5rem;font-weight:800;color:#21d060">{h_pen_pct:.0%}</div>
                          <div style="font-size:0.7rem;color:#8b949e">{home_team}</div>
                          <div style="font-size:0.65rem;color:#484f58">
                            {h_hist['wins']}V/{h_hist['total']}P hist.
                          </div>
                        </div>
                        <div style="font-size:1.2rem;color:#484f58;padding-top:0.5rem">vs</div>
                        <div>
                          <div style="font-size:1.5rem;font-weight:800;color:#f85149">{a_pen_pct:.0%}</div>
                          <div style="font-size:0.7rem;color:#8b949e">{away_team}</div>
                          <div style="font-size:0.65rem;color:#484f58">
                            {a_hist['wins']}V/{a_hist['total']}P hist.
                          </div>
                        </div>
                      </div>
                      <div style="font-size:0.72rem;color:#484f58;text-align:center;margin-top:0.5rem">
                        Basado en {len(shootouts_df)} tandas históricas internacionales · Monte Carlo 50k sim.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Simulación de tanda ejemplo
                    with st.expander("🎲 Ver simulación de tanda (ejemplo)"):
                        example = pen_data["example_shootout"]
                        h_kicks = example["home_kicks"]
                        a_kicks = example["away_kicks"]
                        sd_kicks = example["sudden_death"]
                        winner  = example["winner"]

                        col_h, col_a = st.columns(2)
                        with col_h:
                            st.markdown(f"**🏠 {home_team}**")
                            for k in h_kicks:
                                icon = "🟢" if k["scored"] else "🔴"
                                st.markdown(f'<div class="kick-row"><span class="kick-dot">{icon}</span> Tiro {k["kick"]} · {k["cumulative"]} gol(es)</div>', unsafe_allow_html=True)
                        with col_a:
                            st.markdown(f"**✈️ {away_team}**")
                            for k in a_kicks:
                                icon = "🟢" if k["scored"] else "🔴"
                                st.markdown(f'<div class="kick-row"><span class="kick-dot">{icon}</span> Tiro {k["kick"]} · {k["cumulative"]} gol(es)</div>', unsafe_allow_html=True)

                        if sd_kicks:
                            st.markdown("**⚡ Muerte súbita**")
                            for sd in sd_kicks:
                                h_icon = "🟢" if sd["home_scored"] else "🔴"
                                a_icon = "🟢" if sd["away_scored"] else "🔴"
                                st.markdown(f"Ronda {sd['round']}: {h_icon} {home_team} · {a_icon} {away_team}")

                        st.markdown(f"""
                        <div style="text-align:center;font-size:1rem;font-weight:700;
                                    color:#21d060;margin-top:0.8rem;padding:0.6rem;
                                    background:#0d2818;border-radius:8px">
                          🏆 Gana la tanda: {winner}<br>
                          <span style="font-size:0.8rem;color:#8b949e">
                            {example['home_goals']} – {example['away_goals']}
                          </span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probabilidad final de clasificación
                    st.markdown('<div class="section-title">Probabilidad de clasificar (90\' + ET + P)</div>', unsafe_allow_html=True)

                    p_h_class = etp["prob_home_classifies"]
                    p_a_class = etp["prob_away_classifies"]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=[f"🏠 {home_team}", f"✈️ {away_team}"],
                        y=[p_h_class * 100, p_a_class * 100],
                        marker_color=["#21d060", "#f85149"],
                        text=[f"{p_h_class:.1%}", f"{p_a_class:.1%}"],
                        textposition="outside",
                        textfont=dict(color="#e6edf3", size=16, family="Inter"),
                    ))
                    fig2.update_layout(
                        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                        font_color="#e6edf3", showlegend=False,
                        margin=dict(t=20, b=10, l=10, r=10), height=200,
                        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
                        xaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

                elif is_ko and not etp:
                    st.info("Partido eliminatorio detectado, pero no hay empate probable en 90' para simular ET/penales.")

                # ── Forma reciente ────────────────────────────────────────────
                st.markdown('<div class="section-title">Forma reciente (últimos 5)</div>', unsafe_allow_html=True)
                col_hf, col_af = st.columns(2)
                for col, team, form, side in [
                    (col_hf, home_team, full_pred["home_form"], "🏠"),
                    (col_af, away_team, full_pred["away_form"], "✈️"),
                ]:
                    with col:
                        pts = form.get("pts", np.nan)
                        gf  = form.get("gf", np.nan)
                        ga  = form.get("ga", np.nan)
                        pts_s = f"{pts:.1f}" if not np.isnan(pts) else "—"
                        gf_s  = f"{gf:.1f}"  if not np.isnan(gf)  else "—"
                        ga_s  = f"{ga:.1f}"  if not np.isnan(ga)  else "—"
                        st.markdown(f"""
                        <div class="pred-card">
                          <div style="font-weight:700;font-size:0.85rem;margin-bottom:0.4rem">{side} {team}</div>
                          <div style="font-family:monospace;font-size:0.82rem;color:#8b949e">
                            Pts: <strong style="color:#21d060">{pts_s}</strong><br>
                            GF:  <strong style="color:#e6edf3">{gf_s}</strong><br>
                            GA:  <strong style="color:#f85149">{ga_s}</strong>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB MUNDIAL 2026
# ═══════════════════════════════════════════════════════════════════════════════
with tab_wc:
    # Cargar datos del Mundial
    wc_matches = get_wc2026_matches(raw_df)
    if "result" not in wc_matches.columns:
        wc_matches["result"] = wc_matches.apply(
            lambda r: "H" if r["home_score"] > r["away_score"]
            else ("D" if r["home_score"] == r["away_score"] else "A"), axis=1
        )
    standings = compute_group_standings(wc_matches)
    qualified  = get_qualified_teams(standings)
    remaining  = get_remaining_matches(wc_matches)

    # Header
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a2744,#0d1117);border:1px solid #21d060;
                border-radius:12px;padding:1rem;text-align:center;margin-bottom:1rem">
      <div style="font-size:1.3rem;font-weight:800;color:#21d060">🌎 Copa Mundial 2026</div>
      <div style="font-size:0.78rem;color:#8b949e">48 selecciones · 12 grupos · Datos en tiempo real</div>
    </div>
    """, unsafe_allow_html=True)

    # Sub-tabs: Grupos | Predicciones | Partidos pendientes
    sub_grupos, sub_pred, sub_pend = st.tabs(["📋 Grupos", "🎯 Predecir partido", "⏳ Pendientes"])

    # ── Sub-tab: Grupos ───────────────────────────────────────────────────────
    with sub_grupos:
        group_sel = st.selectbox("Seleccionar grupo", list(GROUPS.keys()),
                                  format_func=lambda g: f"Grupo {g}")
        table = standings[group_sel]
        q_info = qualified[group_sel]

        st.markdown(f'<div class="section-title">Grupo {group_sel}</div>', unsafe_allow_html=True)

        # Tabla de posiciones
        for pos, (_, row) in enumerate(table.iterrows(), 1):
            team = row["team"]
            pts  = int(row["Pts"])
            pj   = int(row["PJ"])
            gf   = int(row["GF"])
            ga   = int(row["GA"])
            dg   = int(row["DG"])

            # Determinar estado de clasificación
            is_1st = team == q_info["1st"]
            is_2nd = team == q_info["2nd"]
            conf   = q_info["confirmed"]

            if conf:
                if is_1st or is_2nd:
                    dot = "🟢"; status = "Clasificado"
                else:
                    dot = "🔴"; status = "Eliminado"
            else:
                if pos <= 2:
                    dot = "🟡"; status = "Clasificando"
                else:
                    dot = "⚪"; status = ""

            pos_color = "#21d060" if pos==1 else ("#58a6ff" if pos==2 else "#8b949e")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.5rem;padding:0.45rem 0.6rem;
                        background:#161b22;border-radius:8px;margin:0.2rem 0;
                        border-left:3px solid {pos_color}">
              <span style="font-weight:800;color:{pos_color};width:1rem">{pos}</span>
              <span style="flex:1;font-weight:600">{wc_display(team)}</span>
              <span style="font-family:monospace;font-size:0.82rem;color:#8b949e">{pj}PJ</span>
              <span style="font-family:monospace;font-size:0.82rem;color:#e6edf3;
                           font-weight:700;background:#21262d;padding:0.1rem 0.4rem;border-radius:4px">{pts}pts</span>
              <span style="font-family:monospace;font-size:0.78rem;color:#8b949e">{gf}:{ga}({dg:+d})</span>
              <span>{dot}</span>
            </div>
            """, unsafe_allow_html=True)

        if not conf:
            st.markdown('<div style="font-size:0.72rem;color:#484f58;margin-top:0.5rem">🟡 Clasificación provisional · Jornada 3 pendiente</div>', unsafe_allow_html=True)

        # Partidos jugados en este grupo
        group_teams = set(GROUPS[group_sel])
        gm = wc_matches[
            (wc_matches["home_team"].isin(group_teams)) &
            (wc_matches["away_team"].isin(group_teams))
        ].sort_values("date")

        if not gm.empty:
            st.markdown('<div class="section-title">Resultados</div>', unsafe_allow_html=True)
            for _, r in gm.iterrows():
                hs, as_ = int(r["home_score"]), int(r["away_score"])
                res_icon = "🟢" if r["result"]=="H" else ("🟡" if r["result"]=="D" else "🔴")
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:0.3rem 0;border-bottom:1px solid #21262d;font-size:0.85rem">
                  <span style="flex:1;text-align:right">{wc_display(r['home_team'])}</span>
                  <span style="font-family:monospace;font-weight:700;padding:0 0.8rem;
                               color:#e6edf3">{hs} {res_icon} {as_}</span>
                  <span style="flex:1">{wc_display(r['away_team'])}</span>
                  <span style="color:#484f58;font-size:0.7rem">{r['date'].strftime('%d/%m')}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── Sub-tab: Predecir partido del Mundial ─────────────────────────────────
    with sub_pred:
        st.markdown('<div class="section-title">Predecir partido del Mundial</div>', unsafe_allow_html=True)

        # Todos los equipos del Mundial
        all_wc_teams = sorted(set(t for teams in GROUPS.values() for t in teams))

        wc_home = st.selectbox("🏠 Local / primero", all_wc_teams,
                                format_func=wc_display, key="wc_home",
                                index=all_wc_teams.index("Argentina") if "Argentina" in all_wc_teams else 0)
        wc_away = st.selectbox("✈️ Visitante / segundo", all_wc_teams,
                                format_func=wc_display, key="wc_away",
                                index=all_wc_teams.index("France") if "France" in all_wc_teams else 1)

        wc_phase = st.selectbox("📍 Fase", [
            "Grupo - Jornada 3",
            "Dieciseisavos de Final",
            "Cuartos de Final",
            "Semifinal",
            "Tercer Puesto",
            "Final",
        ], index=1)

        wc_neutral = wc_phase not in ["Grupo - Jornada 3"]

        # Mostrar estado actual en el Mundial
        col_hi, col_ai = st.columns(2)
        for col, team in [(col_hi, wc_home), (col_ai, wc_away)]:
            with col:
                # Buscar en qué grupo está y su posición actual
                team_group = next((g for g, ts in GROUPS.items() if team in ts), None)
                if team_group:
                    tbl = standings[team_group]
                    team_row = tbl[tbl["team"]==team]
                    if not team_row.empty:
                        tr = team_row.iloc[0]
                        pos_num = tbl.index[tbl["team"]==team][0]
                        st.markdown(f"""
                        <div class="pred-card" style="padding:0.6rem">
                          <div style="font-size:0.7rem;color:#8b949e">Grupo {team_group} · #{pos_num}</div>
                          <div style="font-weight:700">{wc_display(team)}</div>
                          <div style="font-family:monospace;font-size:0.75rem;color:#8b949e">
                            {int(tr['Pts'])}pts · {int(tr['GF'])}:{int(tr['GA'])} · Elo {elo_map.get(team,1500):.0f}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        if wc_home != wc_away:
            if st.button("⚡ Predecir", key="wc_predict"):
                with st.spinner("Calculando..."):
                    from src.knockout_logic import full_match_prediction
                    is_ko = wc_phase not in ["Grupo - Jornada 3"]
                    ko_override = True if is_ko else False

                    base = predict_match(wc_home, wc_away, features,
                                         "FIFA World Cup", wc_neutral,
                                         clf, reg, feature_cols,
                                         clf_elite, elite_cols,
                                         statsbomb_summary, reg_h, reg_a)
                    full = full_match_prediction(base, "FIFA World Cup",
                                                  wc_phase, ko_override if is_ko else False, penalty_stats)

                result_labels = {"H": f"Gana {wc_display(wc_home)}", "D": "Empate", "A": f"Gana {wc_display(wc_away)}"}
                result_colors = {"H": "#21d060", "D": "#d29922", "A": "#f85149"}
                res = full["predicted_result"]

                gh_s = f"{full['predicted_home_goals']:.1f}" if full.get('predicted_home_goals') else "—"
                ga_s = f"{full['predicted_away_goals']:.1f}" if full.get('predicted_away_goals') else "—"

                st.markdown(f"""
                <div class="pred-card highlight">
                  <div style="font-size:0.75rem;color:#8b949e">{wc_phase.upper()}</div>
                  <div style="font-size:1.6rem;font-weight:800;color:{result_colors[res]}">{result_labels[res]}</div>
                  <div style="font-size:1.2rem;font-family:monospace;margin:0.3rem 0">
                    {wc_display(wc_home)} <strong>{gh_s} – {ga_s}</strong> {wc_display(wc_away)}
                  </div>
                  <div style="display:flex;justify-content:space-around;margin-top:0.5rem">
                    <div style="text-align:center">
                      <div style="font-size:1.2rem;font-weight:700;color:#21d060">{full['prob_home']:.0%}</div>
                      <div style="font-size:0.7rem;color:#8b949e">{wc_display(wc_home)}</div>
                    </div>
                    <div style="text-align:center">
                      <div style="font-size:1.2rem;font-weight:700;color:#d29922">{full['prob_draw']:.0%}</div>
                      <div style="font-size:0.7rem;color:#8b949e">Empate</div>
                    </div>
                    <div style="text-align:center">
                      <div style="font-size:1.2rem;font-weight:700;color:#f85149">{full['prob_away']:.0%}</div>
                      <div style="font-size:0.7rem;color:#8b949e">{wc_display(wc_away)}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if is_ko and full.get("et_penalties"):
                    etp = full["et_penalties"]
                    ph = etp["prob_home_classifies"]
                    pa = etp["prob_away_classifies"]
                    pen = etp["penalties"]
                    st.markdown(f"""
                    <div class="pred-card pen-card">
                      <div style="font-size:0.75rem;color:#58a6ff;font-weight:600">PROBABILIDAD DE CLASIFICAR</div>
                      <div style="display:flex;justify-content:space-around;margin-top:0.5rem;text-align:center">
                        <div>
                          <div style="font-size:1.4rem;font-weight:800;color:#21d060">{ph:.0%}</div>
                          <div style="font-size:0.7rem;color:#8b949e">{wc_display(wc_home)}</div>
                        </div>
                        <div>
                          <div style="font-size:1.4rem;font-weight:800;color:#f85149">{pa:.0%}</div>
                          <div style="font-size:0.7rem;color:#8b949e">{wc_display(wc_away)}</div>
                        </div>
                      </div>
                      <div style="font-size:0.72rem;color:#484f58;text-align:center;margin-top:0.4rem">
                        Penales si empate: {wc_display(wc_home)} {pen['prob_home_wins']:.0%} vs {wc_display(wc_away)} {pen['prob_away_wins']:.0%}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Sub-tab: Partidos pendientes ──────────────────────────────────────────
    with sub_pend:
        st.markdown('<div class="section-title">Jornada 3 — Partidos pendientes</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.8rem">25–27 junio 2026 · Todos simultáneos por grupo</div>', unsafe_allow_html=True)

        # Estado del dataset
        patched = [(h,a,hs,as_) for d,h,a,hs,as_ in PATCHES]
        if patched:
            st.markdown(f"""
            <div style="background:#1a2a1a;border:1px solid #21d060;border-radius:8px;
                        padding:0.6rem 0.8rem;margin-bottom:0.8rem;font-size:0.78rem">
              <strong style="color:#21d060">📌 {len(patched)} resultado(s) añadido(s) manualmente</strong>
              <div style="color:#8b949e;margin-top:0.2rem">
                El dataset de martj42 tarda 24-48h en actualizarse.
                Estos resultados ya están integrados en el modelo.
              </div>
            </div>
            """, unsafe_allow_html=True)
            for h,a,hs,as_ in patched:
                res_icon = "🟢" if hs>as_ else ("🟡" if hs==as_ else "🔴")
                st.markdown(f'<div style="font-size:0.85rem;padding:0.2rem 0">{res_icon} <strong>{wc_display(h)}</strong> {hs}–{as_} <strong>{wc_display(a)}</strong></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        if not remaining:
            st.success("✅ Todos los partidos de grupos han sido jugados.")
        else:
            # Agrupar por grupo
            for group_key, teams in GROUPS.items():
                group_rem = [m for m in remaining
                             if m["home"] in teams and m["away"] in teams]
                if group_rem:
                    st.markdown(f'<div style="font-weight:700;color:#21d060;margin-top:0.8rem">Grupo {group_key}</div>', unsafe_allow_html=True)
                    for m in group_rem:
                        h_elo = elo_map.get(m["home"], 1500)
                        a_elo = elo_map.get(m["away"], 1500)
                        fav = wc_display(m["home"]) if h_elo >= a_elo else wc_display(m["away"])
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;
                                    padding:0.4rem 0.6rem;background:#161b22;border-radius:8px;margin:0.2rem 0">
                          <span style="font-weight:600">{wc_display(m['home'])} <span style="color:#484f58">vs</span> {wc_display(m['away'])}</span>
                          <span style="font-size:0.72rem;color:#8b949e">Fav: <strong style="color:#d29922">{fav}</strong></span>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1.2rem">Equipos que necesitan resultado</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.5rem">Selecciones en posición de clasificar o ser eliminadas</div>', unsafe_allow_html=True)

        critical = []
        for g, info in qualified.items():
            if not info["confirmed"]:
                tbl = info["table"]
                for pos in range(len(tbl)):
                    row = tbl.iloc[pos]
                    if row["PJ"] < 3:
                        critical.append({
                            "Grupo": g, "Equipo": wc_display(row["team"]),
                            "Pos": pos+1, "Pts": int(row["Pts"])
                        })

        if critical:
            cdf = pd.DataFrame(critical).sort_values(["Grupo","Pos"])
            for _, r in cdf.iterrows():
                color = "#21d060" if r["Pos"]<=2 else "#f85149"
                st.markdown(f'<div style="display:flex;gap:0.5rem;padding:0.25rem 0"><span style="color:{color}">{"🟢" if r["Pos"]<=2 else "🔴"}</span><span>Grupo {r["Grupo"]} #{r["Pos"]}</span><strong>{r["Equipo"]}</strong><span style="color:#8b949e">{r["Pts"]}pts</span></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB SIMULADOR DEL MUNDIAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a2744,#0d1117);border:1px solid #21d060;
                border-radius:12px;padding:1rem;text-align:center;margin-bottom:1rem">
      <div style="font-size:1.2rem;font-weight:800;color:#21d060">🎲 Simulador Monte Carlo</div>
      <div style="font-size:0.78rem;color:#8b949e">3,000 Mundiales simulados · Actualizado con estado actual de grupos</div>
    </div>
    """, unsafe_allow_html=True)

    # Cargar o generar simulación
    @st.cache_data(show_spinner="🎲 Simulando 3,000 Mundiales...", ttl=3600)
    def get_simulation():
        from src.wc2026 import get_wc2026_matches, compute_group_standings, get_remaining_matches, GROUPS
        from src.model import predict_match
        from src.wc2026_simulator import run_simulation

        raw_wc = get_wc2026_matches(raw_df)
        raw_wc_r = raw_wc.copy()
        raw_wc_r["result"] = raw_wc_r.apply(
            lambda r: "H" if r["home_score"]>r["away_score"] else ("D" if r["home_score"]==r["away_score"] else "A"), axis=1)
        standings_s = compute_group_standings(raw_wc_r)
        remaining = get_remaining_matches(raw_wc_r)

        j3_probs = {}
        for m in remaining:
            h, a = m["home"], m["away"]
            p = predict_match(h, a, features, "FIFA World Cup", True,
                             clf, reg, feature_cols, clf_elite, elite_cols,
                             statsbomb_summary, reg_h, reg_a)
            j3_probs[(h,a)] = (p["prob_home"], p["prob_draw"], p["prob_away"])

        pts = {}; gf = {}; ga = {}
        for g, table in standings_s.items():
            for _, row in table.iterrows():
                t = row["team"]; pts[t]=int(row["Pts"]); gf[t]=int(row["GF"]); ga[t]=int(row["GA"])

        return run_simulation(GROUPS, pts, gf, ga, j3_probs, elo_map, n_sim=3000, seed=42)

    sim_results = get_simulation()
    top_teams = get_top_contenders(sim_results, 48)

    # Filtro
    sim_filter = st.radio("Mostrar", ["Top 16", "Top 32", "Todos"], horizontal=True)
    n_show = {"Top 16": 16, "Top 32": 32, "Todos": 48}[sim_filter]
    show_teams = [t for t in top_teams if t["champion"] > 0][:n_show]

    st.markdown('<div class="section-title">Probabilidad de avanzar por ronda</div>', unsafe_allow_html=True)

    for i, t in enumerate(show_teams, 1):
        team = t["team"]
        champ = t["champion"]
        final = t["final"]
        sf = t["sf"]
        qf = t["qf"]
        r16 = t["r16"]
        elo_val = elo_map.get(team, 1500)

        # Color según probabilidad de campeón
        if champ >= 0.15: bar_color = "#21d060"
        elif champ >= 0.05: bar_color = "#58a6ff"
        elif champ >= 0.01: bar_color = "#d29922"
        else: bar_color = "#484f58"

        st.markdown(f"""
        <div style="background:#161b22;border-radius:8px;padding:0.6rem 0.8rem;margin:0.2rem 0;
                    border-left:3px solid {bar_color}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
              <span style="font-weight:700;font-size:0.9rem">{i}. {team}</span>
              <span style="color:#484f58;font-size:0.7rem;margin-left:0.5rem">Elo {elo_val:.0f}</span>
            </div>
            <span style="font-size:1.1rem;font-weight:800;color:{bar_color}">🏆 {champ:.1%}</span>
          </div>
          <div style="display:flex;gap:1rem;margin-top:0.3rem;font-size:0.75rem;color:#8b949e;font-family:monospace">
            <span>Final: <strong style="color:#e6edf3">{final:.1%}</strong></span>
            <span>SF: <strong style="color:#e6edf3">{sf:.1%}</strong></span>
            <span>QF: <strong style="color:#e6edf3">{qf:.1%}</strong></span>
            <span>R16: <strong style="color:#e6edf3">{r16:.1%}</strong></span>
          </div>
          <div style="background:#21262d;border-radius:3px;height:4px;margin-top:0.4rem">
            <div style="background:{bar_color};width:{min(100,champ*300)}%;height:100%;border-radius:3px;max-width:100%"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.72rem;color:#484f58;margin-top:0.8rem;text-align:center">3,000 simulaciones Monte Carlo · Incluye jornada 3 pendiente · Bracket según cruces FIFA 2026</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB ANÁLISIS DE VALOR EN CUOTAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_value:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#0d1117);border:1px solid #58a6ff;
                border-radius:12px;padding:1rem;text-align:center;margin-bottom:1rem">
      <div style="font-size:1.2rem;font-weight:800;color:#58a6ff">💰 Análisis de Valor</div>
      <div style="font-size:0.78rem;color:#8b949e">Compara la probabilidad del modelo vs las cuotas del mercado</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pred-card" style="font-size:0.8rem;color:#8b949e;margin-bottom:1rem">
      <strong style="color:#e6edf3">¿Qué es el valor?</strong><br>
      Cuando el modelo dice 50% y la cuota implica 35%, hay un <strong style="color:#21d060">+15% de edge</strong>.
      Significa que la cuota está mejor pagada de lo que debería. Los apostadores profesionales
      buscan este tipo de situaciones sistemáticamente.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Analizar un partido</div>', unsafe_allow_html=True)

    v_home = st.selectbox("🏠 Local", team_list, key="v_home",
                          index=team_list.index("Argentina") if "Argentina" in team_list else 0)
    v_away = st.selectbox("✈️ Visitante", team_list, key="v_away",
                          index=team_list.index("France") if "France" in team_list else 1)
    v_neutral = st.toggle("Campo neutral", value=True, key="v_neutral")

    st.markdown('<div class="section-title">Cuotas del mercado (opcional)</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.75rem;color:#8b949e;margin-bottom:0.5rem">Ingresa las cuotas decimales de tu casa de apuestas (ej: 2.50 = paga 2.5x la apuesta)</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        odds_h = st.number_input(f"Cuota {v_home[:8]}", min_value=1.01, max_value=50.0, value=2.00, step=0.05, format="%.2f")
    with col2:
        odds_d = st.number_input("Cuota Empate", min_value=1.01, max_value=50.0, value=3.20, step=0.05, format="%.2f")
    with col3:
        odds_a = st.number_input(f"Cuota {v_away[:8]}", min_value=1.01, max_value=50.0, value=3.80, step=0.05, format="%.2f")

    if v_home != v_away:
        if st.button("🔍 Analizar valor", key="v_analyze"):
            with st.spinner("Calculando..."):
                from src.knockout_logic import full_match_prediction
                base_v = predict_match(v_home, v_away, features, "FIFA World Cup",
                                       v_neutral, clf, reg, feature_cols, clf_elite,
                                       elite_cols, statsbomb_summary, reg_h, reg_a)
                # Blend con cuotas si las hay en caché
                match_odds_v = get_match_odds(v_home, v_away, odds_cache)
                if match_odds_v:
                    from src.odds_features import blend_elo_with_odds
                    elo_p = {"prob_home": base_v["prob_home"], "prob_draw": base_v["prob_draw"], "prob_away": base_v["prob_away"]}
                    blended_v = blend_elo_with_odds(elo_p, match_odds_v)
                    ph, pd, pa = blended_v["prob_home"], blended_v["prob_draw"], blended_v["prob_away"]
                else:
                    ph, pd, pa = base_v["prob_home"], base_v["prob_draw"], base_v["prob_away"]

                analysis = analyze_match_value(ph, pd, pa, odds_h, odds_d, odds_a)

            # Mostrar resultado
            colors_map = {"home": ("#21d060", v_home), "draw": ("#d29922", "Empate"), "away": ("#f85149", v_away)}
            for side, (color, label) in colors_map.items():
                v = analysis[side]
                bg = "#0d2818" if v["edge"] > 0.02 else "#161b22"
                border = color if v["edge"] > 0.02 else "#30363d"
                ev_str = f"{v['ev']:+.2f}"
                edge_str = f"{v['edge']:+.1%}"

                st.markdown(f"""
                <div style="background:{bg};border:1px solid {border};border-radius:10px;
                            padding:0.8rem;margin:0.4rem 0">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <span style="font-weight:700;color:{color}">{label}</span>
                      <span style="font-size:0.75rem;color:#8b949e;margin-left:0.5rem">{v['recommendation']}</span>
                    </div>
                    <span style="font-family:monospace;font-size:1rem;font-weight:700;
                                 color:{'#21d060' if v['edge']>0.02 else '#8b949e'}">
                      Cuota {v['market_odds']:.2f}
                    </span>
                  </div>
                  <div style="display:flex;gap:1.5rem;margin-top:0.4rem;font-size:0.78rem;font-family:monospace;color:#8b949e">
                    <span>Modelo: <strong style="color:{color}">{v['model_prob']:.0%}</strong></span>
                    <span>Mercado: <strong style="color:#e6edf3">{v['implied_prob']:.0%}</strong></span>
                    <span>Edge: <strong style="color:{'#21d060' if v['edge']>0 else '#f85149'}">{edge_str}</strong></span>
                    <span>EV: <strong style="color:{'#21d060' if v['ev']>0 else '#f85149'}">{ev_str}</strong></span>
                  </div>
                  {"<div style='font-size:0.72rem;color:#21d060;margin-top:0.3rem'>Kelly stake recomendado: " + f"{v['kelly_stake']:.1%}" + " del bankroll</div>" if v['kelly_stake']>0.005 else ""}
                </div>
                """, unsafe_allow_html=True)

            # Advertencia legal
            st.markdown("""
            <div style="font-size:0.7rem;color:#484f58;margin-top:0.8rem;padding:0.5rem;
                        background:#0d1117;border-radius:6px;text-align:center">
              ⚠️ Este análisis es informativo. Las apuestas conllevan riesgo de pérdida.
              El edge del modelo es una estimación estadística, no una garantía.
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB RANKING ELO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rank:
    st.markdown('<div class="section-title">Ranking Elo mundial de selecciones</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.8rem">Calculado desde 32,000 partidos internacionales. Mayor Elo = mayor fuerza.</div>', unsafe_allow_html=True)

    elo_ranked = sorted(elo_map.items(), key=lambda x: -x[1])

    # Filtro de búsqueda
    search = st.text_input("🔍 Buscar selección", "")
    if search:
        elo_ranked = [(t,e) for t,e in elo_ranked if search.lower() in t.lower()]

    # Top 30 o resultados de búsqueda
    show = elo_ranked[:30] if not search else elo_ranked[:15]

    for i, (team, elo) in enumerate(show):
        rank = [t for t,_ in sorted(elo_map.items(), key=lambda x:-x[1])].index(team) + 1
        # Color por tier
        if elo >= 2000:
            bar_color = "#21d060"
        elif elo >= 1800:
            bar_color = "#58a6ff"
        elif elo >= 1600:
            bar_color = "#d29922"
        else:
            bar_color = "#8b949e"
        # Barra proporcional (Elo 800-2200 -> 0-100%)
        pct = max(0, min(100, (elo - 800) / 14))
        medal = "🥇" if rank==1 else ("🥈" if rank==2 else ("🥉" if rank==3 else f"{rank}"))
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.6rem;padding:0.35rem 0;border-bottom:1px solid #21262d">
          <div style="width:2rem;text-align:center;font-weight:700;color:#8b949e;font-size:0.85rem">{medal}</div>
          <div style="flex:1">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem">
              <span style="font-weight:600">{team}</span>
              <span style="font-family:monospace;color:{bar_color};font-weight:700">{elo:.0f}</span>
            </div>
            <div style="background:#21262d;border-radius:4px;height:5px;margin-top:0.2rem">
              <div style="background:{bar_color};width:{pct}%;height:100%;border-radius:4px"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if not search:
        st.markdown('<div style="font-size:0.72rem;color:#484f58;margin-top:0.8rem">Mostrando top 30 · usa el buscador para ver cualquier selección</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown('<div class="section-title">Explorar equipo</div>', unsafe_allow_html=True)
    sel_team = st.selectbox("Selección", team_list, key="stats_team",
                            index=team_list.index("Brazil") if "Brazil" in team_list else 0)
    year_range = st.slider("Rango de años", 2000, int(raw_df["date"].dt.year.max()), (2010, 2024))

    team_mask = (
        ((raw_df["home_team"] == sel_team) | (raw_df["away_team"] == sel_team)) &
        (raw_df["date"].dt.year >= year_range[0]) &
        (raw_df["date"].dt.year <= year_range[1])
    )
    team_matches = raw_df[team_mask].copy()

    if team_matches.empty:
        st.info("Sin datos para este equipo en el rango seleccionado.")
    else:
        def match_result_for_team(row, team):
            if row["home_team"] == team:
                gf, ga = row["home_score"], row["away_score"]
            else:
                gf, ga = row["away_score"], row["home_score"]
            res = "W" if gf > ga else ("D" if gf == ga else "L")
            return res, gf, ga

        stats_rows = team_matches.apply(lambda r: pd.Series(match_result_for_team(r, sel_team), index=["res","gf","ga"]), axis=1)
        team_matches = team_matches.join(stats_rows)

        total = len(team_matches)
        wins  = (team_matches["res"] == "W").sum()
        avg_gf = team_matches["gf"].mean()
        avg_ga = team_matches["ga"].mean()

        # Stats de penales
        pen_stat = penalty_stats.get(sel_team, {"total": 0, "wins": 0, "win_rate": 0.5})
        pen_label = f"{pen_stat['wins']}/{pen_stat['total']}" if pen_stat["total"] > 0 else "—"

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [(c1, f"{wins/total:.0%}", "victorias"), (c2, f"{avg_gf:.1f}", "GF/p"), (c3, f"{avg_ga:.1f}", "GA/p"), (c4, pen_label, "penales V/T")]:
            col.markdown(f'<div class="mini-metric"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        team_matches["year"] = team_matches["date"].dt.year
        by_year = team_matches.groupby(["year", "res"]).size().unstack(fill_value=0).reset_index()

        fig_stats = go.Figure()
        for res, color_s, label_s in [("W","#21d060","Victoria"),("D","#d29922","Empate"),("L","#f85149","Derrota")]:
            if res in by_year.columns:
                fig_stats.add_trace(go.Bar(name=label_s, x=by_year["year"], y=by_year[res], marker_color=color_s))
        fig_stats.update_layout(
            barmode="stack", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font_color="#e6edf3", legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, b=40, l=10, r=10), height=280,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#21262d"),
        )
        st.plotly_chart(fig_stats, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: H2H
# ═══════════════════════════════════════════════════════════════════════════════
with tab_h2h:
    st.markdown('<div class="section-title">Enfrentamientos directos</div>', unsafe_allow_html=True)
    h2h_home = st.selectbox("Equipo A", team_list, key="h2h_home",
                             index=team_list.index("Brazil") if "Brazil" in team_list else 0)
    h2h_away = st.selectbox("Equipo B", team_list, key="h2h_away",
                             index=team_list.index("Argentina") if "Argentina" in team_list else 1)

    if h2h_home != h2h_away:
        mask_h2h = (
            ((raw_df["home_team"] == h2h_home) & (raw_df["away_team"] == h2h_away)) |
            ((raw_df["home_team"] == h2h_away) & (raw_df["away_team"] == h2h_home))
        )
        h2h_df = raw_df[mask_h2h].sort_values("date", ascending=False).head(20)

        if h2h_df.empty:
            st.info("Sin enfrentamientos registrados.")
        else:
            a_wins = ((h2h_df["home_team"] == h2h_home) & (h2h_df["result"] == "H")).sum() + \
                     ((h2h_df["away_team"] == h2h_home) & (h2h_df["result"] == "A")).sum()
            b_wins = len(h2h_df) - a_wins - (h2h_df["result"] == "D").sum()
            draws_h2h = (h2h_df["result"] == "D").sum()

            fig_h2h = go.Figure(go.Pie(
                labels=[h2h_home, "Empate", h2h_away],
                values=[a_wins, draws_h2h, b_wins],
                hole=0.6,
                marker_colors=["#21d060", "#d29922", "#f85149"],
                textinfo="label+percent",
                textfont=dict(color="#e6edf3", size=12),
            ))
            fig_h2h.update_layout(
                paper_bgcolor="#161b22", font_color="#e6edf3", showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10), height=220,
                annotations=[dict(text=f"{len(h2h_df)}<br>partidos", x=0.5, y=0.5,
                                  font_size=16, showarrow=False, font_color="#e6edf3")],
            )
            st.plotly_chart(fig_h2h, use_container_width=True, config={"displayModeBar": False})

            # Historial de penales entre estos equipos
            sh_mask = (
                ((shootouts_df["home_team"] == h2h_home) & (shootouts_df["away_team"] == h2h_away)) |
                ((shootouts_df["home_team"] == h2h_away) & (shootouts_df["away_team"] == h2h_home))
            )
            sh_between = shootouts_df[sh_mask]
            if not sh_between.empty:
                st.markdown('<div class="section-title">Penales históricos entre estos equipos</div>', unsafe_allow_html=True)
                for _, r in sh_between.iterrows():
                    w_icon = "🏆" if r["winner"] == h2h_home else "🥈"
                    st.markdown(f"<div class='h2h-row'><span style='color:#8b949e'>{r['date'].strftime('%Y-%m-%d')}</span><span>{w_icon} {r['winner']}</span><span style='color:#8b949e'>{r.get('tournament','')}</span></div>", unsafe_allow_html=True)

            st.markdown('<div class="section-title">Últimos encuentros</div>', unsafe_allow_html=True)
            for _, row in h2h_df.iterrows():
                if row["home_team"] == h2h_home:
                    score_txt = f"{int(row['home_score'])} – {int(row['away_score'])}"
                    win_side  = "home" if row["result"] == "H" else ("draw" if row["result"] == "D" else "away")
                else:
                    score_txt = f"{int(row['away_score'])} – {int(row['home_score'])}"
                    win_side  = "home" if row["result"] == "A" else ("draw" if row["result"] == "D" else "away")

                dot = "🟢" if win_side == "home" else ("🟡" if win_side == "draw" else "🔴")
                st.markdown(
                    f"<div class='h2h-row'>"
                    f"<span style='color:#8b949e;font-size:0.8rem'>{row['date'].strftime('%Y-%m-%d')}</span>"
                    f"<span style='font-weight:600'>{score_txt}</span>"
                    f"<span>{dot}</span>"
                    f"<span style='color:#8b949e;font-size:0.75rem'>{row['tournament'][:22]}</span>"
                    f"</div>", unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_info:
    # ── Panel de estado del sistema ───────────────────────────────────────────
    st.markdown('<div class="section-title">Estado del sistema</div>', unsafe_allow_html=True)

    # Estado de la API key de cuotas
    api_key = get_api_key()
    n_odds = len(odds_cache)

    if api_key:
        odds_status_icon = "✅"
        odds_status_color = "#21d060"
        odds_status_txt = f"API key configurada · {n_odds} partidos con cuotas en caché"
        odds_detail = "Las predicciones fusionan modelo (45%) + mercado (55%)"
    else:
        odds_status_icon = "⚠️"
        odds_status_color = "#d29922"
        odds_status_txt = "Sin API key · Cuotas desactivadas"
        odds_detail = "Registrarse gratis en the-odds-api.com y añadir ODDS_API_KEY en Streamlit Secrets"

    # Estado del dataset
    last_match = raw_df["date"].max().strftime("%d/%m/%Y")
    n_wc = len(raw_df[raw_df["tournament"]=="FIFA World Cup"])

    st.markdown(f"""
    <div class="pred-card" style="padding:1rem">
      <div style="display:flex;flex-direction:column;gap:0.6rem">

        <div style="display:flex;align-items:center;gap:0.6rem">
          <span style="font-size:1.1rem">✅</span>
          <div style="flex:1">
            <div style="font-weight:600;font-size:0.85rem">Dataset martj42</div>
            <div style="font-size:0.75rem;color:#8b949e">Último partido: {last_match} · {len(raw_df):,} partidos totales</div>
          </div>
          <span style="font-size:0.7rem;color:#21d060;font-weight:600">ACTIVO</span>
        </div>

        <div style="display:flex;align-items:center;gap:0.6rem">
          <span style="font-size:1.1rem">✅</span>
          <div style="flex:1">
            <div style="font-weight:600;font-size:0.85rem">Modelo XGBoost + Elo</div>
            <div style="font-size:0.75rem;color:#8b949e">326 selecciones · precisión 60% · se recalcula cada 24h</div>
          </div>
          <span style="font-size:0.7rem;color:#21d060;font-weight:600">ACTIVO</span>
        </div>

        <div style="display:flex;align-items:center;gap:0.6rem">
          <span style="font-size:1.1rem">✅</span>
          <div style="flex:1">
            <div style="font-weight:600;font-size:0.85rem">Mundial 2026</div>
            <div style="font-size:0.75rem;color:#8b949e">{n_wc} partidos · {len(PATCHES)} parche(s) manual(es) activo(s)</div>
          </div>
          <span style="font-size:0.7rem;color:#21d060;font-weight:600">ACTIVO</span>
        </div>

        <div style="display:flex;align-items:center;gap:0.6rem">
          <span style="font-size:1.1rem">{odds_status_icon}</span>
          <div style="flex:1">
            <div style="font-weight:600;font-size:0.85rem">Cuotas de apuestas (The Odds API)</div>
            <div style="font-size:0.75rem;color:#8b949e">{odds_status_txt}</div>
            <div style="font-size:0.7rem;color:#484f58;margin-top:0.1rem">{odds_detail}</div>
          </div>
          <span style="font-size:0.7rem;color:{odds_status_color};font-weight:600">{"ACTIVO" if api_key else "INACTIVO"}</span>
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)

    # Instrucciones para activar cuotas si no hay API key
    if not api_key:
        with st.expander("💰 Cómo activar las cuotas (mejora +2-4% precisión)"):
            st.markdown("""
            **Paso 1** — Regístrate gratis en [the-odds-api.com](https://the-odds-api.com)
            - Sin tarjeta de crédito
            - 500 requests/mes gratis (suficiente para todo el Mundial 2026)

            **Paso 2** — Copia tu API key desde el dashboard

            **Paso 3** — En Streamlit Cloud:
            - Tu app → **Settings** (esquina inferior derecha)
            - → **Secrets**
            - → Añade esta línea:
            ```
            ODDS_API_KEY = "pega_tu_key_aquí"
            ```
            - → **Save**

            La app se recarga automáticamente y las cuotas aparecen activas en este panel.
            """)

    st.markdown('<div class="section-title">Fuentes de datos</div>', unsafe_allow_html=True)
    for icon, name, desc in [
        ("✅", "martj42/international_results", f"{len(raw_df):,} partidos · 1872–2026"),
        ("✅", "martj42/shootouts", f"{len(shootouts_df)} tandas de penales históricas"),
        ("⚡", "Rating Elo propio", "Calculado de 32,000 partidos · predictor #1"),
        ("📊", "StatsBomb Open Data", "xG real · solo visualización"),
        ("✅", "Feature engineering", "Forma, H2H, contexto, ET/penales"),
        ("⚠️", "Rankings FIFA (Kaggle)", "Añadir data/raw/fifa_ranking.csv para activar"),
    ]:
        st.markdown(f'<div class="pred-card" style="margin:0.3rem 0;padding:0.8rem"><span>{icon}</span> <strong style="margin-left:0.4rem">{name}</strong><div style="color:#8b949e;font-size:0.8rem;margin-top:0.2rem;margin-left:1.6rem">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Precisión del modelo</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pred-card">
      <div style="font-size:0.8rem;color:#8b949e;line-height:1.8">
        <div style="margin-bottom:0.5rem;color:#e6edf3;font-weight:600">Evaluado en partidos 2023–2025 (nunca vistos en entrenamiento)</div>
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #21262d;padding:0.3rem 0">
          <span>Total (H/D/A)</span><strong style="color:#21d060">60.0%</strong>
        </div>
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #21262d;padding:0.3rem 0">
          <span>Eliminatorias</span><strong style="color:#21d060">~60%</strong>
        </div>
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #21262d;padding:0.3rem 0">
          <span>Amistosos</span><strong style="color:#d29922">51.1%</strong>
        </div>
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #21262d;padding:0.3rem 0">
          <span>Mundiales / EURO / Copa América</span><strong style="color:#21d060">56.9%</strong>
        </div>
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid #21262d;padding:0.3rem 0">
          <span>Baseline naive (siempre predecir local)</span><span style="color:#484f58">46.7%</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.3rem 0">
          <span>Goles totales — Error medio (MAE)</span><strong style="color:#21d060">±1.38 goles</strong>
        </div>
      </div>
    </div>
    <div class="pred-card" style="border-color:#d29922;background:#1a1500;margin-top:0.4rem">
      <div style="font-size:0.78rem;color:#8b949e">
        <strong style="color:#d29922">⚠️ Contexto importante</strong><br>
        El predictor más potente es el <strong style="color:#21d060">rating Elo</strong>, calculado
        desde 32,000 partidos históricos (la misma metodología que adoptó FIFA en 2018). Aporta el
        55% de la decisión del modelo y cubre el 100% de las selecciones, sin data leakage.
        <br><br>Tras una auditoría, se descartó el xG de StatsBomb como feature predictiva: el Elo ya
        captura la fuerza del equipo con mejor cobertura. El xG se conserva solo para visualización.
        <br><br>El motor es <strong style="color:#21d060">XGBoost</strong> (gradient boosting), validado
        walk-forward: supera a RandomForest en los 6 de 6 años probados (promedio 58.9% vs 56.8%).
        <br><br>La varianza inherente del fútbol pone un techo real: la investigación académica seria
        sitúa la predicción de resultado 1X2 en 52–56% para ligas. Este modelo supera ese rango en
        selecciones gracias al Elo. Nadie predice fútbol al 70%; desconfía de quien lo prometa.
      </div>
    </div>
    """, unsafe_allow_html=True)
    if train_metrics:
        st.markdown(f'<div style="color:#484f58;font-size:0.75rem;margin-top:0.3rem">Entrenado con {train_metrics["n_train"]:,} partidos · Test: {train_metrics["n_test"]:,} partidos · Monte Carlo penales: 50,000 sim.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Deploy gratuito</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pred-card" style="font-size:0.85rem;color:#8b949e">
      <strong style="color:#e6edf3">Streamlit Community Cloud</strong><br>
      1. Sube a GitHub (repo público)<br>
      2. share.streamlit.io → conectar repo<br>
      3. URL pública · funciona desde el celular ✅
    </div>
    """, unsafe_allow_html=True)
