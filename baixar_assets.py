"""
Baixa os pesos ajustados e o video de demonstracao da Release do GitHub.
========================================================================

Arquivos grandes nao sao versionados no git. Eles ficam anexados a Release
do repositorio e este script os coloca nas pastas esperadas pelos demais.

    pesos/visdrone_y8m_50ep_best.pt   YOLOv8m ajustado por 50 epocas no VisDrone
    videos/demo_nadir_30s.mp4         30 s de video nadir (1080p) para testar a contagem

------------------------------------------------------------------
COMO USAR:

    python baixar_assets.py              # baixa tudo
    python baixar_assets.py --so-pesos   # so os pesos
------------------------------------------------------------------
"""

import argparse
import os
import urllib.request

REPO = "pedreirajr/yolo_visdrone"
TAG = "v1.0"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

ASSETS = {
    "pesos": [("visdrone_y8m_50ep_best.pt", "pesos")],
    "videos": [("demo_nadir_30s.mp4", "videos")],
}


def baixar(nome, pasta):
    destino = os.path.join(pasta, nome)
    if os.path.exists(destino):
        print(f"Ja existe, pulando  {destino}")
        return
    os.makedirs(pasta, exist_ok=True)
    url = f"{BASE}/{nome}"
    print(f"Baixando {url}")

    def progresso(blocos, tam_bloco, total):
        if total > 0:
            pct = min(100, blocos * tam_bloco * 100 / total)
            print(f"\r  {pct:5.1f}% de {total / 1e6:.1f} MB", end="", flush=True)

    tmp = destino + ".part"
    urllib.request.urlretrieve(url, tmp, progresso)
    os.replace(tmp, destino)          # so aparece com o nome final se completou
    print(f"\n  salvo em {destino}")


def main():
    ap = argparse.ArgumentParser(description="Baixa pesos e video de demonstracao")
    ap.add_argument("--so-pesos", action="store_true", help="Baixa apenas os pesos")
    args = ap.parse_args()

    grupos = ["pesos"] if args.so_pesos else ["pesos", "videos"]
    for g in grupos:
        for nome, pasta in ASSETS[g]:
            baixar(nome, pasta)


if __name__ == "__main__":
    main()
