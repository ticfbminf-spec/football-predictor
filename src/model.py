"""
model.py
Modelo predictivo para:
  1. Resultado del partido (H / D / A) — clasificación multiclase (XGBoost)
  2. Goles totales — regresión (XGBoost)
XGBoost validado walk-forward: supera a RandomForest en 6/6 años (58.9% vs 56.8%).
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
import xgboost as xgb
import pickle
from pathlib import Path

MODELS_PATH = Path(__file__).parent.parent / "data" / "processed"

# Codificación de resultado para XGBoost (requiere enteros)
RESULT_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_RESULT = {0: "H", 1: "D", 2: "A"}

FEATURE_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "form_pts_diff",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_n",
    "competition_weight", "home_advantage",
    "year", "month",
    "home_elo", "away_elo", "elo_diff",
    "elo_diff_abs", "elo_similar", "form_similar", "h2h_balance",
]
RANKING_COLS = ["home_fifa_rank", "away_fifa_rank", "rank_diff"]
STATSBOMB_COLS = [
    "home_xg_for", "home_xg_against", "home_xg_diff",
    "away_xg_for", "away_xg_against", "away_xg_diff",
    "xg_diff_matchup",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Retorna las columnas de features disponibles en el dataset."""
    available = FEATURE_COLS.copy()
    for col in RANKING_COLS:
        if col in df.columns and df[col].notna().sum() > 100:
            available.append(col)
    # NOTA: StatsBomb xG se descartó como feature predictiva tras evaluación
    # (el Elo ya captura la fuerza del equipo con mejor cobertura y sin leakage).
    # El xG se conserva solo para visualización en la app.
    return [c for c in available if c in df.columns]


def prepare_xy(df: pd.DataFrame, mode: str = "base"):
    """
    Prepara X (features) e y (targets) para entrenamiento.

    mode="base":  usa solo features disponibles para todos los partidos
                  (sin xG de StatsBomb). Maximiza el nº de partidos.
    mode="elite": añade las features xG de StatsBomb, conservando solo
                  partidos donde ambos equipos tienen datos de élite.
    """
    base_cols = [c for c in FEATURE_COLS if c in df.columns]
    for col in RANKING_COLS:
        if col in df.columns and df[col].notna().sum() > 100:
            base_cols.append(col)

    if mode == "elite":
        elite_cols = [c for c in STATSBOMB_COLS if c in df.columns]
        feature_cols = base_cols + elite_cols
        # Conservar partidos con datos xG en ambos equipos
        df_clean = df[feature_cols + ["result", "total_goals"]].dropna(
            subset=base_cols + ["home_xg_diff", "away_xg_diff"]
        )
    else:
        feature_cols = base_cols
        df_clean = df[feature_cols + ["result", "total_goals"]].dropna(subset=feature_cols)

    X = df_clean[feature_cols]
    y_result = df_clean["result"]
    y_goals = df_clean["total_goals"]

    return X, y_result, y_goals, feature_cols


def train(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Modelo de producción: ensemble XGBoost + Poisson (w=0.90/0.10).
    - XGBoost: clasifica resultado H/D/A
    - Poisson: deriva probabilidades desde goles esperados (local y visita por separado)
    - Ensemble: 90% XGBoost + 10% Poisson → mejor calibración, especialmente en alta competencia
    Validado walk-forward 6 años: 58.9% promedio vs 56.8% RandomForest puro.
    """
    from scipy.stats import poisson as poisson_dist
    has_statsbomb = False

    X, y_result, y_goals, feature_cols = prepare_xy(df, mode="base")
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_result_train, y_result_test = y_result.iloc[:split_idx], y_result.iloc[split_idx:]
    y_goals_train, y_goals_test = y_goals.iloc[:split_idx], y_goals.iloc[split_idx:]

    y_result_train_enc = y_result_train.map(RESULT_TO_INT)
    y_result_test_enc  = y_result_test.map(RESULT_TO_INT)

    # Goles local y visita separados (para Poisson)
    home_goals_train = df.loc[X_train.index, "home_score"] if "home_score" in df.columns else y_goals_train / 2
    away_goals_train = df.loc[X_train.index, "away_score"] if "away_score" in df.columns else y_goals_train / 2
    home_goals_test  = df.loc[X_test.index,  "home_score"] if "home_score" in df.columns else y_goals_test / 2
    away_goals_test  = df.loc[X_test.index,  "away_score"] if "away_score" in df.columns else y_goals_test / 2

    xgb_params = dict(n_estimators=500, max_depth=4, learning_rate=0.03,
                      subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                      reg_alpha=0.5, reg_lambda=2, random_state=42, n_jobs=-1,
                      eval_metric="mlogloss")

    # ── Clasificador XGBoost (resultado) ───────────────────────────────────────
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, **xgb_params)
    clf.fit(X_train, y_result_train_enc)

    # ── Regresores XGBoost (goles por separado para Poisson) ─────────────────
    reg_params = dict(n_estimators=400, max_depth=4, learning_rate=0.03,
                      subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5,
                      reg_lambda=2, objective="reg:squarederror", random_state=42, n_jobs=-1)
    reg_h = xgb.XGBRegressor(**reg_params)
    reg_a = xgb.XGBRegressor(**reg_params)
    reg_h.fit(X_train, home_goals_train)
    reg_a.fit(X_train, away_goals_train)

    # Regresor de goles totales (para mostrar en UI)
    reg = xgb.XGBRegressor(**reg_params)
    reg.fit(X_train, y_goals_train)

    # ── Evaluar ensemble ───────────────────────────────────────────────────────
    ENSEMBLE_W = 0.90  # peso de XGBoost vs Poisson
    MAX_G = 8

    def poisson_probs(lh, la):
        p = np.array([[poisson_dist.pmf(i, lh) * poisson_dist.pmf(j, la)
                       for j in range(MAX_G)] for i in range(MAX_G)])
        return np.tril(p, -1).sum(), np.diag(p).sum(), np.triu(p, 1).sum()

    proba_xgb = clf.predict_proba(X_test)
    lam_h = np.clip(reg_h.predict(X_test), 0.1, 8)
    lam_a = np.clip(reg_a.predict(X_test), 0.1, 8)
    proba_poi = np.array([poisson_probs(h, a) for h, a in zip(lam_h, lam_a)])

    proba_ens = ENSEMBLE_W * proba_xgb + (1 - ENSEMBLE_W) * proba_poi
    pred_ens  = pd.Series(np.argmax(proba_ens, axis=1), index=X_test.index).map(INT_TO_RESULT)
    result_acc = accuracy_score(y_result_test, pred_ens)
    goals_mae  = mean_absolute_error(y_goals_test, reg.predict(X_test))
    importance = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # ── Guardar ────────────────────────────────────────────────────────────────
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    for name, obj in [("clf_result", clf), ("reg_goals", reg),
                      ("reg_home_goals", reg_h), ("reg_away_goals", reg_a),
                      ("feature_cols", feature_cols)]:
        with open(MODELS_PATH / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    return {
        "result_accuracy": result_acc,
        "goals_mae": goals_mae,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": importance,
        "clf": clf, "reg": reg, "reg_h": reg_h, "reg_a": reg_a,
        "feature_cols": feature_cols,
        "clf_elite": None, "elite_cols": None, "elite_accuracy": None,
        "has_statsbomb": has_statsbomb,
        "ensemble_w": ENSEMBLE_W,
    }


def load_models():
    """Carga modelos entrenados desde disco."""
    try:
        with open(MODELS_PATH / "clf_result.pkl", "rb") as f:
            clf = pickle.load(f)
        with open(MODELS_PATH / "reg_goals.pkl", "rb") as f:
            reg = pickle.load(f)
        with open(MODELS_PATH / "feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)
    except FileNotFoundError:
        return None, None, None, None, None, None, None

    reg_h = reg_a = None
    try:
        with open(MODELS_PATH / "reg_home_goals.pkl", "rb") as f:
            reg_h = pickle.load(f)
        with open(MODELS_PATH / "reg_away_goals.pkl", "rb") as f:
            reg_a = pickle.load(f)
    except FileNotFoundError:
        pass

    return clf, reg, feature_cols, None, None, reg_h, reg_a


def predict_match(
    home_team: str,
    away_team: str,
    df: pd.DataFrame,
    competition: str = "FIFA World Cup",
    neutral: bool = False,
    clf=None,
    reg=None,
    feature_cols: list = None,
    clf_elite=None,
    elite_cols: list = None,
    statsbomb_summary=None,
    reg_h=None,
    reg_a=None,
) -> dict:
    """
    Predice resultado y goles usando ensemble XGBoost + Poisson.
    El Poisson deriva probabilidades desde goles esperados por separado,
    mejorando la calibración en partidos parejos.
    """
    if clf is None or reg is None:
        return {"error": "Modelos no entrenados. Ejecuta el entrenamiento primero."}

    # Calcular forma reciente de cada equipo
    def team_recent_form(team, n=5):
        home_m = df[df["home_team"] == team][["date", "home_score", "away_score", "result"]].copy()
        home_m["gf"] = home_m["home_score"]
        home_m["ga"] = home_m["away_score"]
        home_m["pts"] = home_m["result"].map({"H": 3, "D": 1, "A": 0})

        away_m = df[df["away_team"] == team][["date", "home_score", "away_score", "result"]].copy()
        away_m["gf"] = away_m["away_score"]
        away_m["ga"] = away_m["home_score"]
        away_m["pts"] = away_m["result"].map({"A": 3, "D": 1, "H": 0})

        all_m = pd.concat([home_m, away_m]).sort_values("date").tail(n)
        if all_m.empty:
            return {"pts": np.nan, "gf": np.nan, "ga": np.nan}

        return {
            "pts": all_m["pts"].mean(),
            "gf": all_m["gf"].mean(),
            "ga": all_m["ga"].mean(),
        }

    def h2h_stats(t1, t2, n=10):
        mask = (
            ((df["home_team"] == t1) & (df["away_team"] == t2)) |
            ((df["home_team"] == t2) & (df["away_team"] == t1))
        )
        past = df[mask].tail(n)
        if past.empty:
            return {"hw": 0, "aw": 0, "d": 0, "n": 0}
        hw = ((past["home_team"] == t1) & (past["result"] == "H")).sum() + \
             ((past["away_team"] == t1) & (past["result"] == "A")).sum()
        aw = ((past["home_team"] == t2) & (past["result"] == "H")).sum() + \
             ((past["away_team"] == t2) & (past["result"] == "A")).sum()
        return {"hw": hw, "aw": aw, "d": (past["result"] == "D").sum(), "n": len(past)}

    competition_weight = {
        "FIFA World Cup": 5, "Copa América": 4, "UEFA Euro": 4,
        "Africa Cup of Nations": 4, "FIFA World Cup qualification": 3,
        "UEFA Euro qualification": 2, "UEFA Nations League": 2, "Friendly": 1,
    }

    hf = team_recent_form(home_team)
    af = team_recent_form(away_team)
    h2h = h2h_stats(home_team, away_team)
    cw = competition_weight.get(competition, 2)

    # Stats avanzadas de StatsBomb (xG) si disponibles
    home_sb = away_sb = None
    if statsbomb_summary is not None:
        from src.statsbomb_features import get_team_advanced_stats
        home_sb = get_team_advanced_stats(statsbomb_summary, home_team)
        away_sb = get_team_advanced_stats(statsbomb_summary, away_team)

    # Rating Elo actual de cada equipo
    from src.elo_ratings import load_current_elo, HOME_ADVANTAGE
    elo_map = load_current_elo()
    home_elo = elo_map.get(home_team, 1500)
    away_elo = elo_map.get(away_team, 1500)

    feature_map = {
        "home_form_pts": hf["pts"],
        "home_form_gf": hf["gf"],
        "home_form_ga": hf["ga"],
        "away_form_pts": af["pts"],
        "away_form_gf": af["gf"],
        "away_form_ga": af["ga"],
        "form_pts_diff": (hf["pts"] - af["pts"]) if not (np.isnan(hf["pts"]) or np.isnan(af["pts"])) else np.nan,
        "h2h_home_wins": h2h["hw"],
        "h2h_away_wins": h2h["aw"],
        "h2h_draws": h2h["d"],
        "h2h_n": h2h["n"],
        "competition_weight": cw,
        "home_advantage": int(not neutral),
        "year": pd.Timestamp.now().year,
        "month": pd.Timestamp.now().month,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_fifa_rank": np.nan,
        "away_fifa_rank": np.nan,
        "rank_diff": np.nan,
    }

    # Añadir features xG al mapa
    if home_sb and away_sb:
        feature_map.update({
            "home_xg_for": home_sb["xg_for"],
            "home_xg_against": home_sb["xg_against"],
            "home_xg_diff": home_sb["xg_diff"],
            "away_xg_for": away_sb["xg_for"],
            "away_xg_against": away_sb["xg_against"],
            "away_xg_diff": away_sb["xg_diff"],
            "xg_diff_matchup": (home_sb["xg_diff"] - away_sb["xg_diff"])
                if not (np.isnan(home_sb["xg_diff"]) or np.isnan(away_sb["xg_diff"])) else np.nan,
        })

    # ── Routing: usar modelo élite si ambos equipos tienen datos xG ────────────
    use_elite = (
        clf_elite is not None and elite_cols is not None and
        home_sb is not None and away_sb is not None and
        home_sb["has_data"] and away_sb["has_data"]
    )

    if use_elite:
        active_clf = clf_elite
        active_cols = elite_cols
        model_used = "élite (con xG)"
    else:
        active_clf = clf
        active_cols = feature_cols
        model_used = "base"

    X = pd.DataFrame([{col: feature_map.get(col, np.nan) for col in active_cols}])

    # ── XGBoost probabilidades ────────────────────────────────────────────────
    proba_xgb = active_clf.predict_proba(X)[0]

    # ── Poisson probabilidades (si hay regresores de goles separados) ─────────
    ENSEMBLE_W = 0.90
    MAX_G = 8
    if reg_h is not None and reg_a is not None:
        from scipy.stats import poisson as poisson_dist
        lam_h = float(np.clip(reg_h.predict(X)[0], 0.1, 8))
        lam_a = float(np.clip(reg_a.predict(X)[0], 0.1, 8))
        poi_grid = np.array([
            [poisson_dist.pmf(i, lam_h) * poisson_dist.pmf(j, lam_a)
             for j in range(MAX_G)] for i in range(MAX_G)
        ])
        p_poi = np.array([
            np.tril(poi_grid, -1).sum(),
            np.diag(poi_grid).sum(),
            np.triu(poi_grid, 1).sum(),
        ])
        proba_final = ENSEMBLE_W * proba_xgb + (1 - ENSEMBLE_W) * p_poi
    else:
        proba_final = proba_xgb
        lam_h = lam_a = None

    # ── Resultado ─────────────────────────────────────────────────────────────
    classes = [INT_TO_RESULT.get(c, c) for c in active_clf.classes_]
    result_proba = dict(zip(classes, proba_final))
    predicted_result = classes[int(np.argmax(proba_final))]

    predicted_goals = float(reg.predict(
        pd.DataFrame([{col: feature_map.get(col, np.nan) for col in feature_cols}])
    )[0])
    predicted_goals = max(0.0, predicted_goals)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_result": predicted_result,
        "prob_home": result_proba.get("H", 0),
        "prob_draw": result_proba.get("D", 0),
        "prob_away": result_proba.get("A", 0),
        "predicted_total_goals": round(predicted_goals, 1),
        "predicted_home_goals": round(lam_h, 1) if lam_h else None,
        "predicted_away_goals": round(lam_a, 1) if lam_a else None,
        "home_form": hf,
        "away_form": af,
        "h2h": h2h,
        "model_used": model_used,
        "home_xg": home_sb,
        "away_xg": away_sb,
    }

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_result": predicted_result,
        "prob_home": result_proba.get("H", 0),
        "prob_draw": result_proba.get("D", 0),
        "prob_away": result_proba.get("A", 0),
        "predicted_total_goals": round(predicted_goals, 1),
        "home_form": hf,
        "away_form": af,
        "h2h": h2h,
        "model_used": model_used,
        "home_xg": home_sb,
        "away_xg": away_sb,
    }
