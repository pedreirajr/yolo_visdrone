"""
Contagem de trafego em video NADIR (vista de cima) com pesos VisDrone.
======================================================================

O QUE MUDA EM RELACAO AO contagem_trafego.py
--------------------------------------------
A logica de contagem e identica (deteccao, rastreamento e cruzamento de linha,
com multiplas linhas nomeadas). Este script reaproveita essa logica importando
as pecas do contagem_trafego.py (a classe LinhaContagem e a geometria), para
nao duplicar codigo.

A diferenca esta em duas coisas:

1. O MODELO. Aqui o padrao e usar pesos ajustados no VisDrone (imagem aerea),
   e nao o yolov8m.pt do COCO. No video nadir deste projeto, o modelo COCO nao
   produziu nenhuma deteccao na pista principal. O VisDrone foi feito com fotos
   de drone e reconhece o veiculo visto de cima.

2. AS CLASSES. O VisDrone usa indices e nomes diferentes dos do COCO.
       COCO      2=car          3=motorcycle  5=bus  7=truck   1=bicycle
       VisDrone  3=car  4=van   9=motor       8=bus  5=truck   2=bicycle
   Com indices fixos (como faz o contagem_trafego.py), trocar de modelo
   quebraria tudo. Por isso o mapeamento e feito pelo NOME. O script le os
   nomes que o proprio modelo declara e os traduz para as nossas categorias,
   o que funciona com pesos VisDrone e COCO sem alterar o codigo.

------------------------------------------------------------------
COMO USAR:

    # Pre-requisito: os pesos ajustados. Baixe os publicados na Release
    # (python baixar_assets.py) ou gere os seus com treinar_visdrone.py.

    # Escolher as linhas (varias de uma vez)
    python escolher_linhas.py --video videos/demo_nadir_30s.mp4

    # Rodar a contagem no video de demonstracao
    python contagem_trafego_nadir.py --video videos/demo_nadir_30s.mp4 ^
        --linha A 570 142 707 163 --linha B 701 690 767 540 ^
        --linha C 1002 306 1234 341 --linha D 1258 120 1493 82 ^
        --saida saidas/demo_nadir
------------------------------------------------------------------
"""

import argparse
import csv
import os
import sys

import cv2
from ultralytics import YOLO

# Reaproveita as pecas do script principal (mesmo diretorio).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contagem_trafego import LinhaContagem, montar_linhas, desenhar_linha_interativa


# ---------------------------------------------------------------------------
# MAPEAMENTO DE CLASSES POR NOME
# ---------------------------------------------------------------------------
# Traduz os nomes que o modelo declara para as nossas categorias de trafego.
# Cobrimos tanto os nomes do VisDrone quanto os do COCO, para o script funcionar
# com qualquer um dos dois sem alteracao. O que nao estiver aqui (pedestrian,
# people, tricycle, awning-tricycle) e ignorado -- nao e veiculo de interesse.
NOME_PARA_CATEGORIA = {
    # VisDrone
    "bicycle": "bicicleta",
    "car": "carro",
    "van": "carro",          # vans contam como carro; separe aqui se quiser
    "motor": "moto",
    "truck": "caminhao",
    "bus": "onibus",
    # COCO (nomes que diferem)
    "motorcycle": "moto",
}


def resolver_classes(modelo):
    """Constroi, a partir dos nomes do modelo, o dicionario {indice: categoria}.

    modelo.names e algo como {0: 'pedestrian', 3: 'car', 9: 'motor', ...}.
    Devolvemos so os indices que correspondem a veiculos de interesse, ja
    traduzidos para as nossas categorias. Esse dicionario cumpre o mesmo papel
    que o CLASSES_VEICULO fixo do script original, mas montado dinamicamente.
    """
    classes = {}
    for idx, nome in modelo.names.items():
        cat = NOME_PARA_CATEGORIA.get(str(nome).lower())
        if cat:
            classes[idx] = cat
    return classes


def main():
    # -----------------------------------------------------------------------
    # 1) ARGUMENTOS (espelham o contagem_trafego.py, com outro padrao de modelo)
    # -----------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Contagem de trafego em video nadir (VisDrone)")
    ap.add_argument("--video", required=True, help="Caminho do video de entrada")
    ap.add_argument("--modelo", default="pesos/visdrone_y8m_50ep_best.pt",
                    help="Pesos ajustados no VisDrone (baixe com baixar_assets.py ou gere "
                         "com treinar_visdrone.py). Pode-se passar yolov8m.pt para "
                         "comparar com o COCO.")
    ap.add_argument("--linha", action="append", nargs=5,
                    metavar=("NOME", "X1", "Y1", "X2", "Y2"),
                    help="Linha de contagem nomeada; repita para varias.")
    ap.add_argument("--line", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="Modo simples: uma unica linha sem nome (x1 y1 x2 y2)")
    ap.add_argument("--draw-line", action="store_true",
                    help="Abre o 1o frame para desenhar a linha com o mouse e sai")
    ap.add_argument("--conf", type=float, default=0.25, help="Confianca minima da deteccao")
    ap.add_argument("--imgsz", type=int, default=1280, help="Resolucao de inferencia")
    ap.add_argument("--saida", default="resultado_nadir",
                    help="Prefixo dos arquivos de saida")
    ap.add_argument("--sem-video", action="store_true", help="Nao gravar o video anotado")
    ap.add_argument("--video-ate-s", type=float, default=0,
                    help="Grava o video anotado so ate N segundos (0 = inteiro)")
    ap.add_argument("--processar-ate-s", type=float, default=0,
                    help="PARA a contagem apos N segundos de video (0 = processa "
                         "o video inteiro). Util para testes rapidos: ao contrario "
                         "de --video-ate-s, este interrompe o processamento em si.")
    ap.add_argument("--device", default=None, help="0 (GPU), cpu. Padrao: automatico")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # 2) VALIDACOES
    # -----------------------------------------------------------------------
    if not os.path.exists(args.video):
        print(f"Video nao encontrado: {args.video}")
        return

    if args.draw_line:
        desenhar_linha_interativa(args.video)
        return

    # O modelo VisDrone e um arquivo local. Se nao existir, orientamos o usuario.
    # Nomes simples como yolov8m.pt sao baixados pelo proprio Ultralytics.
    if args.modelo.endswith(".pt") and ("/" in args.modelo or "\\" in args.modelo) \
            and not os.path.exists(args.modelo):
        print(f"Pesos nao encontrados: {args.modelo}")
        print("Baixe os pesos publicados com  python baixar_assets.py")
        print("ou gere os seus com            python treinar_visdrone.py")
        print("Ou passe --modelo yolov8m.pt para comparar com o COCO.")
        return

    linhas = montar_linhas(args)
    if linhas is None:
        return
    if not linhas:
        print("Defina ao menos uma linha com --linha NOME X1 Y1 X2 Y2.")
        return

    # -----------------------------------------------------------------------
    # 3) MODELO E RESOLUCAO DAS CLASSES POR NOME
    # -----------------------------------------------------------------------
    modelo = YOLO(args.modelo)
    classes_veiculo = resolver_classes(modelo)
    if not classes_veiculo:
        print("O modelo carregado nao declara nenhuma classe de veiculo conhecida.")
        print("Nomes do modelo:", modelo.names)
        return

    # Ordem estavel das categorias para relatorios (sem repetir).
    categorias = []
    for cat in classes_veiculo.values():
        if cat not in categorias:
            categorias.append(cat)

    print(f"Modelo: {args.modelo}")
    print("Classes de veiculo reconhecidas:",
          {idx: cat for idx, cat in classes_veiculo.items()})

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Linhas de contagem ({len(linhas)}):")
    for l in linhas:
        print(f"  {l.nome}: {l.a} -> {l.b}")

    # -----------------------------------------------------------------------
    # 4) VIDEO ANOTADO DE SAIDA (opcional)
    # -----------------------------------------------------------------------
    # Cria a pasta de saida, se o prefixo incluir uma (ex.: saidas/demo)
    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    writer = None
    if not args.sem_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(f"{args.saida}_anotado.mp4", fourcc, fps, (largura, altura))
    limite_video = int(args.video_ate_s * fps) if args.video_ate_s > 0 else 0
    limite_proc = int(args.processar_ate_s * fps) if args.processar_ate_s > 0 else 0
    CORES_LINHA = [(0, 0, 255), (255, 0, 255), (0, 200, 0), (255, 160, 0),
                   (0, 255, 255), (255, 0, 0)]

    eventos = []   # (frame, tempo_s, id, classe, sentido, linha)

    # -----------------------------------------------------------------------
    # 5) DETECCAO + TRACKING
    # -----------------------------------------------------------------------
    frame_idx = 0
    resultados = modelo.track(
        source=args.video,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(classes_veiculo.keys()),   # filtro vem do mapeamento por nome
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # 6) LACO PRINCIPAL (igual ao contagem_trafego.py, muda so a origem das classes)
    # -----------------------------------------------------------------------
    for r in resultados:
        frame_idx += 1
        tempo_s = frame_idx / fps

        # Teste rapido, interrompe o processamento apos --processar-ate-s.
        if limite_proc and frame_idx > limite_proc:
            break

        gravar = writer is not None and (limite_video == 0 or frame_idx <= limite_video)
        frame = r.orig_img.copy() if gravar else None

        if frame_idx % 300 == 0:
            pct = (100.0 * frame_idx / total_frames) if total_frames else 0
            resumo = "  ".join(f"{l.nome}={l.total()}" for l in linhas)
            print(f"  frame {frame_idx}/{total_frames} ({pct:.1f}%)  {resumo}", flush=True)

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()

            for tid, cls_id, box in zip(ids, clss, xyxy):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                centro = (cx, cy)
                nome = classes_veiculo.get(cls_id, str(cls_id))

                for l in linhas:
                    l.processar(tid, nome, centro, frame_idx, tempo_s, eventos)

                if frame is not None:
                    contado = any(tid in l.ja_contado for l in linhas)
                    cor = (0, 255, 0) if contado else (255, 180, 0)
                    cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])), cor, 2)
                    cv2.putText(frame, f"{nome} #{tid}", (int(box[0]), int(box[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        if frame is not None:
            y0 = 30
            for i, l in enumerate(linhas):
                cor = CORES_LINHA[i % len(CORES_LINHA)]
                cv2.line(frame, l.a, l.b, cor, 2)
                mx, my = (l.a[0] + l.b[0]) // 2, (l.a[1] + l.b[1]) // 2
                cv2.putText(frame, l.nome, (mx - 8, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
                cv2.putText(frame, f"{l.nome}: {l.total()}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
                y0 += 28
            writer.write(frame)

    if writer is not None:
        writer.release()

    # -----------------------------------------------------------------------
    # 7) EXPORTACAO
    # -----------------------------------------------------------------------
    with open(f"{args.saida}_eventos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "tempo_s", "id", "classe", "sentido", "linha"])
        w.writerows(eventos)

    with open(f"{args.saida}_resumo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["linha", "classe", "A_para_B", "B_para_A", "total"])
        for l in linhas:
            total_linha = 0
            for cat in categorias:
                t = l.totais[cat]
                tot = t["A_para_B"] + t["B_para_A"]
                total_linha += tot
                w.writerow([l.nome, cat, t["A_para_B"], t["B_para_A"], tot])
            w.writerow([l.nome, "TOTAL", "", "", total_linha])

    # -----------------------------------------------------------------------
    # 8) RELATORIO NO TERMINAL
    # -----------------------------------------------------------------------
    print("\n=== Contagem concluida ===")
    for l in linhas:
        print(f"\nLinha {l.nome}  ({l.a} -> {l.b})   total={l.total()}")
        for cat in categorias:
            t = l.totais[cat]
            print(f"  {cat:10s}: A->B={t['A_para_B']:4d}  B->A={t['B_para_A']:4d}  "
                  f"total={t['A_para_B']+t['B_para_A']:4d}")
    print(f"\nArquivos gerados:")
    print(f"  {args.saida}_eventos.csv")
    print(f"  {args.saida}_resumo.csv")
    if writer is not None:
        print(f"  {args.saida}_anotado.mp4")


if __name__ == "__main__":
    main()
