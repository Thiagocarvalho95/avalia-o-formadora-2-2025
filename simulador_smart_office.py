"""
Simulador de dados de sensores para o projeto Smart Office.
Gera um CSV com 7 dias de dados (granularidade de 15 minutos)
para 3 tipos de sensores (temperatura [°C], luminosidade [lux],
ocupação [0/1]) em 3 salas.

Requisitos atendidos (conforme enunciado):
- pandas, numpy, datetime
- 3 tipos de sensores
- 7 dias, a cada 15 minutos
- variação lógica (dia/noite, horário comercial, picos ocasionais no fim de semana)
- colunas: timestamp, sensor_id, valor

Uso:
    python simulador_smart_office.py
Saída:
    smart_office_data.csv (no mesmo diretório do script)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random

def generate_time_index(start: datetime, days: int = 7, freq="15T"):
    end = start + timedelta(days=days)
    return pd.date_range(start=start, end=end, freq=freq, inclusive="left")

def is_work_hour(ts: pd.Timestamp):
    # horário comercial: 08:00–18:00, seg–sex
    return (ts.weekday() < 5) and (8 <= ts.hour < 18)

def is_night(ts: pd.Timestamp):
    return (ts.hour >= 22) or (ts.hour < 6)

def simulate_for_room(room_id: int, time_index: pd.DatetimeIndex, seed_base: int = 42):
    rng = np.random.default_rng(seed_base + room_id)

    # --- Temperatura base: padrão diário suave (mais frio à noite)
    # faixa ~ 20–25 °C em horário comercial; cai à noite.
    temp = []
    for ts in time_index:
        # componente diário (seno), pico por volta de 15:00
        # normaliza hora do dia em radianos
        rad = ((ts.hour * 60 + ts.minute) / (24*60)) * 2 * np.pi
        daily = 2.5 * np.sin(rad - np.pi/2) + 22.5  # ~20 a ~25
        # mais quente se horário comercial (pessoas + equipamentos)
        if is_work_hour(ts):
            daily += 0.6 + rng.normal(0, 0.2)
        # mais frio à noite
        if is_night(ts):
            daily -= 1.0 + rng.normal(0, 0.2)
        # ruído geral
        val = daily + rng.normal(0, 0.3)
        temp.append(max(17.0, min(28.0, val)))  # clamp suave

    # --- Luminosidade: 0 à noite; em horário comercial mantém ~400–600 lux
    lux = []
    for ts in time_index:
        if is_night(ts):
            base = 0.0
        elif is_work_hour(ts):
            base = 500 + rng.normal(0, 60)  # iluminação de escritório
        else:
            # fora do horário comercial de dia -> baixo (luzes geralmente apagadas)
            base = max(0, 50 + rng.normal(0, 30))
        lux.append(max(0.0, float(base)))

    # --- Ocupação: 1 com alta probabilidade no horário comercial (seg–sex), 0 caso contrário
    occ = []
    for ts in time_index:
        p = 0.0
        if ts.weekday() < 5:  # seg–sex
            if 8 <= ts.hour < 18:
                p = 0.85  # ocupado com alta probabilidade
            elif 7 <= ts.hour < 8 or 18 <= ts.hour < 19:
                p = 0.35  # transição
            else:
                p = 0.05  # fora do horário
        else:  # fim de semana
            p = 0.03  # quase sempre vazio

        # Picos ocasionais e "inesperados" no fim de semana (para a ata)
        if ts.weekday() >= 5 and 10 <= ts.hour < 14:
            if rng.random() < 0.06:
                p = 0.7

        occ.append(int(rng.random() < p))

    # Monta DataFrame long para a sala
    df_temp = pd.DataFrame({
        "timestamp": time_index,
        "sensor_id": [f"temp_{room_id}"] * len(time_index),
        "valor": np.round(temp, 2),
    })
    df_lux = pd.DataFrame({
        "timestamp": time_index,
        "sensor_id": [f"lux_{room_id}"] * len(time_index),
        "valor": np.round(lux, 1),
    })
    df_occ = pd.DataFrame({
        "timestamp": time_index,
        "sensor_id": [f"occ_{room_id}"] * len(time_index),
        "valor": occ,
    })
    return pd.concat([df_temp, df_lux, df_occ], ignore_index=True)

def main(output_csv: str = "smart_office_data.csv", start: datetime | None = None):
    if start is None:
        # ancorar no início da semana atual (segunda-feira, 00:00)
        today = datetime.now()
        monday = today - timedelta(days=today.weekday())
        start = monday.replace(hour=0, minute=0, second=0, microsecond=0)

    time_index = generate_time_index(start, days=7, freq="15T")

    # 3 salas
    frames = []
    for room in [1, 2, 3]:
        frames.append(simulate_for_room(room, time_index, seed_base=2025))

    df = pd.concat(frames, ignore_index=True)

    # ordena e salva
    df = df.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Gerado: {output_csv} com {len(df):,} linhas.")

if __name__ == "__main__":
    main()
