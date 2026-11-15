"""
Backend do painel JIKUFIRE: harmoniza VIIRS (375 m) e MODIS (1 km) do NASA FIRMS.

Execução:
    pip install flask requests pandas numpy scipy
    export FIRMS_MAP_KEY="a_sua_chave"      # Windows (PowerShell): $env:FIRMS_MAP_KEY="a_sua_chave"
    python app.py

Estrutura esperada:
    app.py
    templates/index.html
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template
from scipy.spatial import cKDTree

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
MAP_KEY = "09b0fb12f88011b420547f31cae7b5e3" # os.environ.get("FIRMS_MAP_KEY", "").strip()
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov"
AREA = "-180,-90,180,90"   # mundo inteiro
DIAS = 1                   # janela de 24 h (o FIRMS aceita 1 a 5)

# Cada fonte indica a coluna de temperatura de brilho (Kelvin) e o instrumento.
FONTES = {
    "VIIRS_NOAA21_NRT": {"instrumento": "VIIRS", "col_temp": "bright_ti4"},
    "VIIRS_NOAA20_NRT": {"instrumento": "VIIRS", "col_temp": "bright_ti4"},
    "VIIRS_SNPP_NRT":   {"instrumento": "VIIRS", "col_temp": "bright_ti4"},
    "MODIS_NRT":        {"instrumento": "MODIS", "col_temp": "brightness"},
}

NOMES_SATELITE = {"N21": "NOAA-21", "N20": "NOAA-20", "N": "Suomi NPP",
                  "Terra": "Terra", "Aqua": "Aqua"}

RAIO_FUSAO_KM = 0.6        # ~resolução do MODIS (1 km): deteções mais próximas que isto são o mesmo foco
CRITICO_TEMP_C = 120       # °C (na prática só o MODIS chega aqui, ver nota abaixo)
CRITICO_FRP_MW = 100       # MW: critério comparável entre sensores
FRP_REF_MW = 100           # FRP que corresponde à intensidade máxima (1.0)
MAX_PONTOS_MAPA = 15000
CACHE_TTL = 600            # segundos

# Nota: o canal I4 do VIIRS satura perto de 367 K (~94 °C), enquanto o canal 21
# do MODIS satura perto de 500 K. As temperaturas dos dois sensores não são
# diretamente comparáveis, por isso a classificação usa também o FRP.


class ErroFIRMS(Exception):
    def __init__(self, mensagem, status=502):
        super().__init__(mensagem)
        self.status = status


def seguro(msg):
    """Nunca devolver a chave ao browser (as mensagens do requests incluem o URL)."""
    texto = str(msg)
    return texto.replace(MAP_KEY, "***") if MAP_KEY else texto


# ----------------------------------------------------------------------------
# Recolha e normalização por fonte
# ----------------------------------------------------------------------------
def normalizar_confianca(serie):
    """VIIRS: l/n/h (texto). MODIS: 0-100 (numérico). Tudo para low/nominal/high."""
    texto = serie.astype(str).str.lower().str.strip()
    mapa = {"l": "low", "low": "low", "n": "nominal", "nominal": "nominal",
            "h": "high", "high": "high"}
    out = texto.map(mapa)
    num = pd.to_numeric(serie, errors="coerce")
    # Limiares usados pela própria NASA para o MODIS: <30 baixa, 30-79 nominal, >=80 alta
    faixas = pd.cut(num, bins=[-1, 29, 79, 100], labels=["low", "nominal", "high"]).astype(object)
    return out.where(num.isna(), faixas)


def normalizar(df, fonte):
    cfg = FONTES[fonte]
    obrigatorias = ["latitude", "longitude", cfg["col_temp"], "confidence", "acq_date", "acq_time"]
    em_falta = [c for c in obrigatorias if c not in df.columns]
    if em_falta:
        raise ValueError(f"colunas em falta: {', '.join(em_falta)}")

    out = pd.DataFrame(index=df.index)
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["temp_celsius"] = pd.to_numeric(df[cfg["col_temp"]], errors="coerce") - 273.15
    out["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0) if "frp" in df.columns else 0.0
    out["confianca"] = normalizar_confianca(df["confidence"])
    out["acq_date"] = df["acq_date"].astype(str)

    hhmm = pd.to_numeric(df["acq_time"], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(4)
    out["acq_hora"] = hhmm.str[:2] + ":" + hhmm.str[2:]

    sat = df["satellite"].astype(str) if "satellite" in df.columns else pd.Series(fonte, index=df.index)
    out["satelite"] = sat.map(NOMES_SATELITE).fillna(sat)
    out["instrumento"] = cfg["instrumento"]
    out["prioridade"] = 0 if cfg["instrumento"] == "VIIRS" else 1  # VIIRS tem melhor resolução

    return out.dropna(subset=["latitude", "longitude", "temp_celsius"])


def buscar_fonte(fonte):
    url = f"{FIRMS_BASE}/api/area/csv/{MAP_KEY}/{fonte}/{AREA}/{DIAS}"
    resposta = requests.get(url, timeout=60)
    if resposta.status_code != 200:
        raise ValueError(f"HTTP {resposta.status_code}")

    texto = resposta.text
    if not texto.lstrip().lower().startswith("latitude"):
        raise ValueError(seguro(texto[:150].strip()) or "resposta vazia")

    return normalizar(pd.read_csv(StringIO(texto)), fonte)


# ----------------------------------------------------------------------------
# Harmonização
# ----------------------------------------------------------------------------
def deduplicar(df):
    """
    Funde deteções do mesmo foco (várias passagens e vários satélites) por raio:
    percorre os pontos por ordem de prioridade (VIIRS primeiro e, dentro do mesmo
    instrumento, maior FRP primeiro); cada ponto ainda livre torna-se "líder" e
    absorve os vizinhos a menos de RAIO_FUSAO_KM. Ao contrário de uma grelha fixa,
    dois pontos quase coincidentes nunca ficam separados por caírem em células
    vizinhas. Guarda quantos satélites distintos viram o foco.
    """
    df = df.sort_values(["prioridade", "frp", "temp_celsius"],
                        ascending=[True, False, False]).reset_index(drop=True)
    n = len(df)

    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    xy = np.column_stack([lat * 110.574, lon * 111.320 * np.cos(np.radians(lat))])  # km

    pares = cKDTree(xy).query_pairs(RAIO_FUSAO_KM, output_type="ndarray")  # i < j
    if len(pares):
        pares = pares[np.argsort(pares[:, 0], kind="stable")]
        inicio = np.searchsorted(pares[:, 0], np.arange(n + 1))

    cluster = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if cluster[i] != -1:          # 
            continue
        cluster[i] = i
        if len(pares):
            viz = pares[inicio[i]:inicio[i + 1], 1]
            cluster[viz[cluster[viz] == -1]] = i

    df["n_satelites"] = df.groupby(cluster)["satelite"].transform("nunique")
    return df[cluster == np.arange(n)]


def amostrar_mapa(df):
    """Mantém sempre os focos mais intensos e amostra o resto até ao limite."""
    if len(df) <= MAX_PONTOS_MAPA:
        return df
    fixos = df.sort_values("frp", ascending=False).head(MAX_PONTOS_MAPA // 4)
    resto = df.drop(fixos.index)
    resto = resto.sample(n=MAX_PONTOS_MAPA - len(fixos), random_state=42)
    return pd.concat([fixos, resto])


COLUNAS_SAIDA = ["latitude", "longitude", "temp_celsius", "frp", "intensidade", "critico",
                 "acq_date", "acq_hora", "satelite", "instrumento", "n_satelites"]


def serializar(df):
    df = df[COLUNAS_SAIDA].copy()
    df["latitude"] = df["latitude"].round(4)
    df["longitude"] = df["longitude"].round(4)
    df["temp_celsius"] = df["temp_celsius"].round(1)
    df["frp"] = df["frp"].round(1)
    df["intensidade"] = df["intensidade"].round(2)
    return json.loads(df.to_json(orient="records"))  # evita tipos numpy no jsonify


def construir_dados():
    if not MAP_KEY:
        raise ErroFIRMS("Chave FIRMS em falta: defina a variável de ambiente FIRMS_MAP_KEY.", 500)

    tabelas, contagens, falhas = [], {}, {}
    with ThreadPoolExecutor(max_workers=len(FONTES)) as ex:
        futuros = {ex.submit(buscar_fonte, f): f for f in FONTES}
        for fut in as_completed(futuros):
            fonte = futuros[fut]
            try:
                tabela = fut.result()
                tabelas.append(tabela)
                contagens[fonte] = int(len(tabela))
            except Exception as e:
                falhas[fonte] = seguro(e)

    if not tabelas:
        raise ErroFIRMS("Falha em todas as fontes da NASA: " + "; ".join(f"{k}: {v}" for k, v in falhas.items()))

    bruto = pd.concat(tabelas, ignore_index=True)
    filtrado = bruto[bruto["confianca"].isin(["nominal", "high"])]
    if filtrado.empty:
        raise ErroFIRMS("A NASA devolveu 0 fogos ativos nesta região.", 404)

    df = deduplicar(filtrado).reset_index(drop=True)

    # Intensidade 0-1 baseada no FRP (comparável entre VIIRS e MODIS)
    df["intensidade"] = np.clip(np.log1p(df["frp"]) / np.log1p(FRP_REF_MW), 0.1, 1.0)
    df["critico"] = (df["temp_celsius"] > CRITICO_TEMP_C) | (df["frp"] >= CRITICO_FRP_MW)

    top10 = df.sort_values("temp_celsius", ascending=False).head(10)

    return {
        "total": int(len(df)),
        "temp_media": round(float(df["temp_celsius"].mean()), 1),
        "temp_max": round(float(df["temp_celsius"].max()), 1),
        "criticos": int(df["critico"].sum()),
        "pontos": serializar(amostrar_mapa(df)),
        "top10": serializar(top10),
        "harmonizacao": {
            "deteccoes_brutas": int(len(bruto)),
            "apos_filtro_confianca": int(len(filtrado)),
            "focos_unicos": int(len(df)),
            "por_instrumento": {k: int(v) for k, v in df["instrumento"].value_counts().items()},
            "confirmados_por_2_ou_mais_satelites": int((df["n_satelites"] >= 2).sum()),
            "linhas_por_fonte": contagens,
            "fontes_com_falha": falhas,
            "raio_fusao_km": RAIO_FUSAO_KM,
        },
    }


# ----------------------------------------------------------------------------
# Rotas
# ----------------------------------------------------------------------------
_cache = {"dados": None, "ts": 0.0}
_lock = threading.Lock()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/fires")
def get_fires():
    with _lock:  # evita vários pedidos simultâneos à NASA
        if _cache["dados"] is not None and time.time() - _cache["ts"] < CACHE_TTL:
            return jsonify(_cache["dados"])

        try:
            dados = construir_dados()
        except ErroFIRMS as e:
            if _cache["dados"] is not None:  # serve a última versão boa
                return jsonify({**_cache["dados"], "desatualizado": True, "aviso": str(e)})
            return jsonify({"erro": str(e)}), e.status
        except Exception as e:
            return jsonify({"erro": f"Erro interno: {seguro(e)}"}), 500

        _cache.update(dados=dados, ts=time.time())
        return jsonify(dados)


if __name__ == "__main__":
    app.run(debug=True, port=5000)