"""
Grafico da evolucao do ajuste fino ao longo das epocas.
=======================================================

Le o results.csv que o Ultralytics grava em cada pasta de treino
(runs/detect/<nome>/results.csv) e desenha dois paineis lado a lado.

    esquerda  mAP50 e mAP50-95 no conjunto de validacao do VisDrone
    direita   perda de localizacao (box loss) no treino e na validacao

As duas curvas de perda caindo juntas indicam que o modelo esta aprendendo sem
decorar o conjunto de treino. Se a perda de validacao subisse enquanto a de
treino continua caindo, haveria sobreajuste (overfitting).

------------------------------------------------------------------
COMO USAR:

    # resultado publicado neste repositorio
    python plotar_treino.py --run resultados/treino_visdrone_y8m_50ep

    # um treino feito por voce
    python plotar_treino.py --run runs/detect/visdrone_y8m_50ep
------------------------------------------------------------------
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")           # grava em arquivo sem abrir janela
import matplotlib.pyplot as plt

# Paleta categorica (azul e laranja), validada para daltonismo
AZUL, LARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"


def ler_resultados(caminho):
    """Devolve um dict {coluna: lista de valores} a partir do results.csv."""
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    # O Ultralytics as vezes grava espacos no nome das colunas
    linhas = [{k.strip(): v for k, v in l.items()} for l in linhas]
    return {k: [float(l[k]) for l in linhas] for k in linhas[0]}


def estilizar(ax, titulo, ylabel):
    ax.set_title(titulo, loc="left", fontsize=11, color=TEXTO)
    ax.set_xlabel("Época", color=TEXTO_2)
    ax.set_ylabel(ylabel, color=TEXTO_2)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)


def curva(ax, x, y, cor, rotulo):
    """Linha de 2 px com o valor final rotulado na ponta."""
    ax.plot(x, y, color=cor, linewidth=2, label=rotulo)
    ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=TEXTO)


def main():
    ap = argparse.ArgumentParser(description="Grafico das metricas de treino do YOLO")
    ap.add_argument("--run", required=True,
                    help="Pasta do treino (contendo results.csv)")
    ap.add_argument("--saida", default=None,
                    help="PNG de saida (padrao <run>/curvas_treino.png)")
    args = ap.parse_args()

    csv_path = os.path.join(args.run, "results.csv")
    if not os.path.exists(csv_path):
        print(f"Nao encontrei {csv_path}")
        return
    r = ler_resultados(csv_path)
    ep = r["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")

    curva(ax1, ep, r["metrics/mAP50(B)"], AZUL, "mAP50")
    curva(ax1, ep, r["metrics/mAP50-95(B)"], LARANJA, "mAP50-95")
    estilizar(ax1, "Precisão média na validação", "mAP")

    curva(ax2, ep, r["train/box_loss"], AZUL, "treino")
    curva(ax2, ep, r["val/box_loss"], LARANJA, "validação")
    estilizar(ax2, "Perda de localização (box loss)", "perda")

    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXTO_2)
        ax.set_xlim(ep[0], ep[-1] * 1.08)    # espaco para o rotulo final

    fig.tight_layout()
    saida = args.saida or os.path.join(args.run, "curvas_treino.png")
    fig.savefig(saida, dpi=150)
    print(f"Grafico salvo em {saida}")

    # Resumo numerico no terminal
    i_best = max(range(len(ep)), key=lambda i: r["metrics/mAP50-95(B)"][i])
    print(f"\nEpocas treinadas: {int(ep[-1])}")
    print(f"Melhor mAP50-95: {r['metrics/mAP50-95(B)'][i_best]:.4f} (epoca {int(ep[i_best])})")
    print(f"mAP50 nessa epoca: {r['metrics/mAP50(B)'][i_best]:.4f}")
    if "time" in r:
        print(f"Tempo total de treino: {r['time'][-1] / 3600:.1f} h")


if __name__ == "__main__":
    main()
