"""
Seletor interativo de linhas de contagem.
=========================================

Abre um frame do video numa janela e deixa voce clicar as linhas com o mouse.
Ao sair, imprime (e salva) os argumentos --linha prontos para colar no
contagem_trafego.py / contagem_trafego_nadir.py.

Foi feito para DESENHAR VARIAS linhas de uma vez, ao contrario do --draw-line
do script principal, que faz uma so.

------------------------------------------------------------------
COMO USAR:

    python escolher_linhas.py --video videos/demo_nadir_30s.mp4

    # escolher um frame com transito (por tempo em segundos ou por numero):
    python escolher_linhas.py --video "...MP4" --tempo 300
    python escolher_linhas.py --video "...MP4" --frame 9000

CONTROLES (dentro da janela):
    clique esquerdo   -> marca um ponto, e a cada 2 pontos, fecha uma linha
    u                 -> desfaz a ULTIMA linha inteira
    z                 -> desfaz o ultimo PONTO (quando ha um ponto pendente)
    r                 -> apaga tudo e recomeca
    s                 -> salva e sai (imprime os --linha e grava arquivo .txt)
    ESC / q           -> sai sem salvar

O nome de cada linha e automatico (A, B, C, ...). Renomeie depois se quiser,
editando o texto gerado.
------------------------------------------------------------------
"""

import argparse
import os
import string

import cv2


def main():
    ap = argparse.ArgumentParser(description="Seletor interativo de linhas de contagem")
    ap.add_argument("--video", required=True, help="Caminho do video")
    ap.add_argument("--frame", type=int, default=0, help="Numero do frame a usar de fundo")
    ap.add_argument("--tempo", type=float, default=None,
                    help="Alternativa a --frame, instante em segundos (usa o fps do video)")
    ap.add_argument("--saida", default=None,
                    help="Arquivo .txt onde gravar os --linha (padrao linhas_<video>.txt)")
    ap.add_argument("--max-larg", type=int, default=1600,
                    help="Largura maxima da janela. O frame e reduzido para caber na tela "
                         "(as coordenadas impressas SAO sempre as do video original)")
    args = ap.parse_args()

    # ---- carrega o frame de fundo ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Nao consegui abrir o video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    idx = int(args.tempo * fps) if args.tempo is not None else args.frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Nao consegui ler o frame {idx}.")
        return
    H, W = frame.shape[:2]

    # ---- fator de escala para a janela caber na tela ----
    # Desenhamos numa versao reduzida, mas guardamos e imprimimos SEMPRE em
    # coordenadas do video original (divisao/multiplicacao pela escala).
    escala = min(1.0, args.max_larg / W)

    def para_orig(px, py):
        return (int(round(px / escala)), int(round(py / escala)))

    def para_tela(p):
        return (int(round(p[0] * escala)), int(round(p[1] * escala)))

    linhas = []            # lista de (nome, (x1,y1), (x2,y2)) em coords ORIGINAIS
    pendente = []          # 0 ou 1 ponto ja clicado da linha atual (coords orig)
    mouse = [0, 0]         # posicao atual do mouse (coords orig), para o crosshair

    CORES = [(0, 0, 255), (255, 0, 255), (0, 200, 0), (255, 160, 0),
             (0, 255, 255), (255, 0, 0), (128, 0, 255), (0, 128, 255)]

    def proximo_nome():
        # A, B, ... Z, depois A1, B1, ...
        i = len(linhas)
        return string.ascii_uppercase[i] if i < 26 else f"L{i}"

    def clique(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse[0], mouse[1] = para_orig(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            ponto = para_orig(x, y)
            pendente.append(ponto)
            if len(pendente) == 2:
                linhas.append((proximo_nome(), pendente[0], pendente[1]))
                pendente.clear()

    janela = "Escolher linhas  |  clique 2 pontos/linha  |  u=desfaz  r=reset  s=salvar  ESC=sair"
    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(janela, clique)

    while True:
        disp = cv2.resize(frame, None, fx=escala, fy=escala) if escala != 1.0 else frame.copy()

        # linhas ja fechadas
        for i, (nome, a, b) in enumerate(linhas):
            cor = CORES[i % len(CORES)]
            pa, pb = para_tela(a), para_tela(b)
            cv2.line(disp, pa, pb, cor, 2)
            cv2.circle(disp, pa, 5, cor, -1)
            cv2.circle(disp, pb, 5, cor, -1)
            mx, my = (pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2
            cv2.putText(disp, nome, (mx - 8, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

        # ponto pendente (primeiro clique de uma linha ainda nao fechada)
        if pendente:
            pp = para_tela(pendente[0])
            cv2.circle(disp, pp, 5, (255, 255, 255), -1)
            cv2.line(disp, pp, para_tela(tuple(mouse)), (255, 255, 255), 1)

        # crosshair + leitura das coordenadas na posicao do mouse
        mt = para_tela(tuple(mouse))
        cv2.line(disp, (mt[0], 0), (mt[0], disp.shape[0]), (60, 60, 60), 1)
        cv2.line(disp, (0, mt[1]), (disp.shape[1], mt[1]), (60, 60, 60), 1)
        cv2.putText(disp, f"({mouse[0]}, {mouse[1]})", (mt[0] + 8, mt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # rodape de ajuda
        cv2.putText(disp, f"{len(linhas)} linha(s)  |  u=desfaz  z=ponto  r=reset  s=salvar  ESC=sair",
                    (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(janela, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):          # sair sem salvar
            linhas_para_salvar = None
            break
        elif k == ord('u') and linhas:   # desfaz ultima linha
            linhas.pop()
        elif k == ord('z') and pendente: # desfaz ponto pendente
            pendente.clear()
        elif k == ord('r'):              # reset total
            linhas.clear()
            pendente.clear()
        elif k == ord('s'):              # salvar e sair
            linhas_para_salvar = linhas
            break

    cv2.destroyAllWindows()

    if not linhas_para_salvar:
        print("Saiu sem salvar (ou nenhuma linha marcada).")
        return

    # ---- monta os argumentos --linha e grava ----
    partes = [f"--linha {nome} {a[0]} {a[1]} {b[0]} {b[1]}"
              for nome, a, b in linhas_para_salvar]
    bloco = " ".join(partes)

    saida = args.saida or ("linhas_" + os.path.splitext(os.path.basename(args.video))[0] + ".txt")
    with open(saida, "w", encoding="utf-8") as f:
        f.write("# Linhas escolhidas em " + args.video + f" (frame {idx})\n")
        for nome, a, b in linhas_para_salvar:
            f.write(f"# {nome}: {a} -> {b}\n")
        f.write("\n" + bloco + "\n")

    print(f"\n{len(linhas_para_salvar)} linha(s) salvas em: {saida}\n")
    print("Cole isto na contagem:")
    print("  " + bloco + "\n")


if __name__ == "__main__":
    main()
