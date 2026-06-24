"""
wc2026_patches.py
Parches de resultados del Mundial 2026 que aún no están en el dataset de martj42.
Se fusionan con el dataset principal para que el modelo tenga datos al día.

Actualizar manualmente con cada jornada hasta que martj42 los incluya.
Una vez que el dataset de martj42 se actualiza (24-48h), estos parches se
ignoran automáticamente (no duplican si ya existen).

Última actualización: 23 junio 2026
"""
import pandas as pd

# Resultados confirmados que aún no están en el dataset de martj42
# Formato: (fecha, local, visitante, goles_local, goles_visita)
PATCHES = [
    # ── Jornada 2 Grupo K y L (23 junio 2026) ────────────────────────────────
    ("2026-06-23", "Portugal",  "Uzbekistan", 5, 0),   # Doblete Ronaldo (histórico)
    ("2026-06-23", "England",   "Ghana",      0, 0),   # Empate en Boston
    # Panamá vs Croacia y Colombia vs RD Congo (noche del 23 junio — en curso)
    # Añadir cuando terminen:
    # ("2026-06-23", "Panama",   "Croatia",    ?, ?),
    # ("2026-06-23", "Colombia", "DR Congo",   ?, ?),
]


def apply_patches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade los parches al dataset si el partido aún no existe.
    No duplica si martj42 ya lo tiene.
    """
    existing = set(zip(
        df['home_team'].tolist(),
        df['away_team'].tolist(),
        df['date'].dt.strftime('%Y-%m-%d').tolist()
    ))

    new_rows = []
    for date_str, home, away, hs, as_ in PATCHES:
        if (home, away, date_str) not in existing:
            result = 'H' if hs > as_ else ('D' if hs == as_ else 'A')
            new_rows.append({
                'date': pd.Timestamp(date_str),
                'home_team': home,
                'away_team': away,
                'home_score': float(hs),
                'away_score': float(as_),
                'tournament': 'FIFA World Cup',
                'city': '',
                'country': '',
                'neutral': True,
                'result': result,
                'total_goals': float(hs + as_),
                'competition_type': 'Mundial',
                'is_competitive': True,
            })

    if new_rows:
        patch_df = pd.DataFrame(new_rows)
        # Asegurar que tiene las mismas columnas que el df original
        for col in df.columns:
            if col not in patch_df.columns:
                patch_df[col] = None
        df = pd.concat([df, patch_df[df.columns]], ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        print(f"   📌 {len(new_rows)} parche(s) aplicado(s) al dataset")

    return df


def add_patch(date_str: str, home: str, away: str, home_score: int, away_score: int):
    """
    Añade un parche al archivo. Llamar cuando termina un partido.
    Uso: add_patch("2026-06-23", "Panama", "Croatia", 1, 2)
    """
    import ast, os
    path = __file__
    content = open(path).read()
    new_line = f'    ("{date_str}", "{home}", "{away}", {home_score}, {away_score}),'
    # Insertar antes del cierre del listado
    content = content.replace(
        '    # Añadir cuando terminen:',
        f'{new_line}\n    # Añadir cuando terminen:'
    )
    open(path, 'w').write(content)
    print(f"✅ Parche añadido: {home} {home_score}-{away_score} {away}")
