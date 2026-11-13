"""
Contagem classificada de trafego a partir de video de drone usando YOLO.
==========================================================================

O QUE ESTE PROGRAMA FAZ
-----------------------
Transforma um video aereo em numeros de trafego, isto e, quantos veiculos
passaram por uma ou mais secoes da via, de que tipo eram (bicicleta, carro,
moto, onibus, caminhao), em que sentido seguiam e em que instante cruzaram.
Este script usa os pesos COCO e e indicado para captacao OBLIQUA. Para
captacao vertical (nadir), use o contagem_trafego_nadir.py.

O PIPELINE, EM QUATRO ETAPAS
----------------------------
    video .MP4
       -> 1. DETECCAO   (YOLO)       "onde estao os veiculos neste frame?"
       -> 2. TRACKING   (ByteTrack)  "este carro e o mesmo do frame anterior?"
       -> 3. CONTAGEM   (geometria)  "ele acabou de cruzar alguma linha? em que sentido?"
       -> 4. EXPORTACAO (CSV + MP4)  planilhas e video anotado para auditoria

A IDEIA CENTRAL
---------------
Contar deteccoes nao funciona, porque a rede neural nao tem memoria e
reencontra os mesmos carros a cada frame. Um carro visivel por 5 segundos a
30 fps geraria 150 contagens.

Por isso o rastreamento (tracking) e obrigatorio. Cada veiculo recebe um ID
que o acompanha ao longo do video, e a contagem so acontece no instante em que
o centro daquele ID atravessa uma linha imaginaria desenhada pelo operador
sobre a pista. Como o ID e estavel, o mesmo carro e contado UMA unica vez POR
LINHA, ainda que apareca em centenas de frames. Cada linha e independente, e um
veiculo que cruze duas linhas diferentes e contado uma vez em cada, o que e o
desejado para medir aproximacoes distintas de um mesmo trevo.

ATENCAO AO ANGULO DA CAMERA (licao aprendida em campo)
------------------------------------------------------
Com os pesos COCO, a camera do drone deve ser INCLINADA, e nao apontada direto
para baixo.

O COCO e feito de fotografias tiradas do nivel do solo, onde o carro aparece de
frente, de tras ou de perfil. Numa visada vertical (nadir) o veiculo vira um
retangulo de teto, some a silhueta que a rede aprendeu e a deteccao despenca.
Em teste real neste projeto, o mesmo cruzamento rendeu ZERO deteccoes na pista
principal em visada vertical, contra 73 a 81 deteccoes por frame em visada
obliqua. Baixar --conf nao resolve, pois o problema e de reconhecimento e nao
de limiar. A solucao adotada foi ajustar a rede no VisDrone (treinar_visdrone.py)
e contar com o contagem_trafego_nadir.py.

Objetos pequenos (MOTO e BICICLETA) sofrem uma versao mais branda do mesmo
problema. Ocupam poucos pixels vistos do alto e sao mal detectados mesmo em
visada obliqua, entao os totais dessas duas classes tendem a ser subestimados
e devem ser lidos com ressalva.

------------------------------------------------------------------
COMO USAR (no terminal do VS Code):

    # 1) Instale as dependencias (de preferencia num ambiente virtual)
    pip install -r requirements.txt

    # 2) Descubra as coordenadas das linhas clicando sobre um frame
    python escolher_linhas.py --video "SEU_VIDEO.MP4"

    # 3) Rode a contagem com UMA ou VARIAS linhas nomeadas
    python contagem_trafego.py --video "SEU_VIDEO.MP4" ^
        --linha A 650 378 790 432 ^
        --linha B 1130 461 1300 449 ^
        --imgsz 1920 --conf 0.25 --video-ate-s 60 ^
        --saida saidas/trevo

    # (--draw-line abre o 1o frame para marcar uma unica linha, e
    #  --line X1 Y1 X2 Y2 continua aceito para o modo de uma linha so.)
------------------------------------------------------------------
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2                      # leitura do video, desenho das anotacoes, gravacao
import numpy as np              # aritmetica vetorial da geometria da linha
from ultralytics import YOLO    # modelo de deteccao + tracker ByteTrack integrado


# ---------------------------------------------------------------------------
# CLASSES DE INTERESSE
# ---------------------------------------------------------------------------
# O modelo foi treinado no COCO, que reconhece 80 categorias de objetos. Aqui
# ficam so as que importam para trafego, referenciadas pelo indice que elas tem
# naquele dataset. Esse dicionario e usado em dois lugares:
#   - as CHAVES viram o filtro `classes=` da inferencia, de modo que pessoas,
#     semaforos e demais objetos sequer sao considerados (mais rapido e menos
#     ruido na contagem)
#   - os VALORES sao os nomes que aparecem nos relatorios e no video anotado.
#
# NOTA sobre bicicleta (1) e moto (3). Sao objetos pequenos e mal detectados em
# imagem aerea. Ficam aqui porque foram pedidos, mas seus totais sao pouco
# confiaveis (ver o aviso no cabecalho).
CLASSES_VEICULO = {
    1: "bicicleta",
    2: "carro",
    3: "moto",
    5: "onibus",
    7: "caminhao",
}


# ===========================================================================
# GEOMETRIA DA CONTAGEM
# ===========================================================================

def lado_da_linha(ponto, a, b):
    """Retorna o sinal (>0, <0) que indica de qual lado da linha AB o ponto esta.

    Calcula o produto vetorial entre o vetor da linha (A->B) e o vetor que vai
    de A ate o ponto. O sinal do resultado diz o lado (positivo de um lado,
    negativo do outro, zero exatamente em cima da linha).

    Guardando esse sinal a cada frame, uma INVERSAO entre um frame e o seguinte
    indica que o veiculo passou de um lado para o outro. O sentido do fluxo sai
    de brinde, pois se o sinal anterior era negativo, o veiculo foi de A para B.
    """
    return np.sign((b[0] - a[0]) * (ponto[1] - a[1]) - (b[1] - a[1]) * (ponto[0] - a[0]))


def segmentos_cruzam(p1, p2, a, b):
    """Diz se o deslocamento do veiculo (p1->p2) cruza o SEGMENTO ab.

    Por que isso e necessario? A troca de sinal em lado_da_linha() sozinha
    descreve uma reta INFINITA, nao o trecho que o operador desenhou. Um veiculo
    trafegando em outra pista, longe da linha mas alinhado com o seu
    prolongamento, tambem inverte o sinal e seria contado por engano. Num trevo
    com varias pistas paralelas o erro nao e marginal e infla a contagem de forma
    sistematica.

    A correcao e exigir uma intersecao real entre DOIS segmentos:
        - o passo dado pelo veiculo   (p1 -> p2), parametrizado por t
        - a linha de contagem         (a  -> b),  parametrizada por u
    Resolvendo o sistema, o cruzamento so vale quando ambos os parametros caem
    no intervalo [0, 1] -- isto e, dentro do passo do veiculo E dentro do trecho
    efetivamente desenhado.
    """
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]      # vetor do passo do veiculo
    sx, sy = b[0] - a[0], b[1] - a[1]          # vetor da linha de contagem
    denom = rx * sy - ry * sx                  # produto vetorial entre os dois
    if denom == 0:
        return False                           # paralelos: nunca se cruzam
    qpx, qpy = a[0] - p1[0], a[1] - p1[1]      # vetor de p1 ate a
    t = (qpx * sy - qpy * sx) / denom          # onde no passo do veiculo
    u = (qpx * ry - qpy * rx) / denom          # onde no segmento da linha
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


# ===========================================================================
# UMA LINHA DE CONTAGEM E SEU ESTADO
# ===========================================================================

class LinhaContagem:
    """Uma linha de contagem, com todo o estado necessario para contar por ela.

    Cada linha carrega o SEU proprio estado por ID de rastro. E o que torna as
    linhas independentes, pois o mesmo veiculo pode ser contado uma vez em cada linha
    que cruzar, sem uma interferir na outra.
    """

    def __init__(self, nome, a, b):
        self.nome = nome
        self.a = a                    # ponto A (x, y)
        self.b = b                    # ponto B (x, y)
        self.ultimo_lado = {}         # id -> sinal do lado no frame anterior
        self.ultimo_centro = {}       # id -> centro no frame anterior
        self.ja_contado = set()       # ids ja contados NESTA linha
        self.totais = defaultdict(lambda: {"A_para_B": 0, "B_para_A": 0})

    def total(self):
        """Soma de todos os cruzamentos desta linha, em todas as classes/sentidos."""
        return sum(t["A_para_B"] + t["B_para_A"] for t in self.totais.values())

    def processar(self, tid, nome_classe, centro, frame_idx, tempo_s, eventos):
        """Atualiza esta linha com um veiculo detectado no frame atual.

        Se o veiculo acabou de cruzar a linha, contabiliza e registra o evento.
        Devolve True se contou agora (util para colorir a caixa no video).
        """
        lado = lado_da_linha(centro, self.a, self.b)     # lado agora
        anterior = self.ultimo_lado.get(tid)             # lado no frame anterior
        contou = False

        # ---- TESTE DE CRUZAMENTO ----
        # Cinco condicoes, todas necessarias:
        #   1. ja vimos este ID nesta linha      (anterior is not None)
        #   2. nao esta exatamente sobre a linha (lado != 0, anterior != 0)
        #   3. o lado inverteu                   (lado != anterior)
        #   4. este ID ainda nao foi contado AQUI (evita dupla contagem na linha)
        #   5. o cruzamento caiu DENTRO do segmento desenhado
        # A condicao 5 e avaliada por ultimo de proposito, ja que o Python curto-circuita
        # o `and`, entao a funcao mais cara so roda nos poucos casos que passaram
        # por todo o resto.
        if anterior is not None and lado != 0 and anterior != 0 \
                and lado != anterior and tid not in self.ja_contado \
                and segmentos_cruzam(self.ultimo_centro[tid], centro, self.a, self.b):
            # O sinal ANTERIOR define o sentido. Vindo do lado negativo, o veiculo
            # foi de A para B.
            sentido = "A_para_B" if anterior < 0 else "B_para_A"
            self.totais[nome_classe][sentido] += 1
            self.ja_contado.add(tid)
            eventos.append((frame_idx, round(tempo_s, 2), tid, nome_classe, sentido, self.nome))
            contou = True

        # Atualiza o estado para o proximo frame. Posicoes exatamente sobre a
        # linha (lado == 0) sao ignoradas, e manter o ultimo lado nao-nulo evita
        # que uma parada em cima da linha apague a referencia.
        if lado != 0:
            self.ultimo_lado[tid] = lado
            self.ultimo_centro[tid] = centro

        return contou


# ===========================================================================
# APOIO: ESCOLHA INTERATIVA DA LINHA
# ===========================================================================

def desenhar_linha_interativa(video_path):
    """Abre o primeiro frame e deixa voce clicar 2 pontos para definir a linha.

    Nao conta nada, serve so para DESCOBRIR as coordenadas, que sao impressas no
    terminal prontas para colar em --linha na execucao seguinte. Rode uma vez por
    aproximacao, ja que as coordenadas sao em pixels e dependem do enquadramento.
    """
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()          # so o 1o frame basta para posicionar a linha
    cap.release()
    if not ok:
        print("Nao consegui ler o video.")
        return
    pontos = []

    def clique(event, x, y, flags, param):
        """Callback do mouse, guarda ate 2 cliques com o botao esquerdo."""
        if event == cv2.EVENT_LBUTTONDOWN and len(pontos) < 2:
            pontos.append((x, y))

    cv2.namedWindow("Clique 2 pontos para a linha (ESC para sair)")
    cv2.setMouseCallback("Clique 2 pontos para a linha (ESC para sair)", clique)

    # Laco de exibicao, redesenha o frame a cada iteracao mostrando o que ja foi
    # clicado (bolinhas nos pontos e, com 2 pontos, a linha entre eles).
    while True:
        vis = frame.copy()                      # copia para nao sujar o original
        for p in pontos:
            cv2.circle(vis, p, 6, (0, 0, 255), -1)
        if len(pontos) == 2:
            cv2.line(vis, pontos[0], pontos[1], (0, 0, 255), 2)
        cv2.imshow("Clique 2 pontos para a linha (ESC para sair)", vis)
        # Sai por ESC (codigo 27) ou assim que o 2o ponto for marcado.
        if cv2.waitKey(20) == 27 or len(pontos) == 2:
            if len(pontos) == 2:
                cv2.waitKey(600)                # pausa para voce ver a linha
            break
    cv2.destroyAllWindows()

    if len(pontos) == 2:
        (x1, y1), (x2, y2) = pontos
        print("\nLinha escolhida. Rode novamente com (troque NOME por um rotulo):")
        print(f'  --linha NOME {x1} {y1} {x2} {y2}\n')


# ===========================================================================
# CONSTRUCAO DAS LINHAS A PARTIR DOS ARGUMENTOS
# ===========================================================================

def montar_linhas(args):
    """Traduz --linha (nomeadas, varias) e --line (simples, uma) em objetos.

    Devolve a lista de LinhaContagem, ou None se algo estiver invalido.
    """
    linhas = []

    # --linha NOME X1 Y1 X2 Y2, repetivel. argparse entrega cada uma como 5
    # strings (nome + 4 numeros), e convertemos os quatro ultimos para int.
    if args.linha:
        for item in args.linha:
            nome = item[0]
            try:
                x1, y1, x2, y2 = (int(v) for v in item[1:])
            except ValueError:
                print(f"Coordenadas invalidas em --linha {item}")
                return None
            linhas.append(LinhaContagem(nome, (x1, y1), (x2, y2)))

    # --line X1 Y1 X2 Y2 e o modo simples de uma linha so, mantido por compatibilidade.
    if args.line:
        linhas.append(LinhaContagem("L1", (args.line[0], args.line[1]),
                                    (args.line[2], args.line[3])))

    return linhas


# ===========================================================================
# PROGRAMA PRINCIPAL
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    # 1) ARGUMENTOS DA LINHA DE COMANDO
    # -----------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Contagem classificada de trafego com YOLO")
    ap.add_argument("--video", required=True, help="Caminho do video de entrada")
    ap.add_argument("--modelo", default="yolov8m.pt",
                    help="Pesos YOLO (ex.: yolov8n.pt=rapido, yolov8m.pt=equilibrio, "
                         "yolov8x.pt=mais preciso). Baixa automaticamente na 1a vez.")
    ap.add_argument("--linha", action="append", nargs=5,
                    metavar=("NOME", "X1", "Y1", "X2", "Y2"),
                    help="Linha de contagem nomeada. Repita o argumento para varias. "
                         "Ex.: --linha A 650 378 790 432")
    ap.add_argument("--line", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="Modo simples, uma unica linha sem nome (x1 y1 x2 y2)")
    ap.add_argument("--draw-line", action="store_true",
                    help="Abre o 1o frame para desenhar a linha com o mouse e sai")
    ap.add_argument("--conf", type=float, default=0.25, help="Confianca minima da deteccao")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Resolucao de inferencia (maior = melhor para objetos pequenos de drone)")
    ap.add_argument("--saida", default="resultado",
                    help="Prefixo dos arquivos de saida (CSV e video anotado)")
    ap.add_argument("--sem-video", action="store_true",
                    help="Nao gravar o video anotado (mais rapido)")
    ap.add_argument("--video-ate-s", type=float, default=0,
                    help="Grava o video anotado so ate N segundos (0 = video inteiro). "
                         "A contagem continua no video todo. Serve para conferir "
                         "visualmente as linhas sem gerar arquivo gigante.")
    ap.add_argument("--device", default=None,
                    help="Dispositivo de inferencia (0 = GPU, cpu). Padrao automatico")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # 2) VALIDACOES E MODOS DE SAIDA ANTECIPADA
    # -----------------------------------------------------------------------
    if not os.path.exists(args.video):
        print(f"Video nao encontrado: {args.video}")
        return

    # Modo auxiliar, so descobre as coordenadas de uma linha e encerra.
    if args.draw_line:
        desenhar_linha_interativa(args.video)
        return

    linhas = montar_linhas(args)
    if linhas is None:
        return
    if not linhas:
        print("Defina ao menos uma linha com --linha NOME X1 Y1 X2 Y2 "
              "(ou use --draw-line para descobrir as coordenadas).")
        return

    # -----------------------------------------------------------------------
    # 3) MODELO E METADADOS DO VIDEO
    # -----------------------------------------------------------------------
    # Na primeira execucao o Ultralytics baixa os pesos sozinho. Se houver GPU
    # com PyTorch compilado para CUDA, ele a utiliza automaticamente.
    modelo = YOLO(args.modelo)

    # Abrimos o video so para ler os metadados (fps, dimensoes, total de frames)
    # e fechamos em seguida, porque quem le o video de verdade e o modelo.track().
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30      # o "or 30" cobre fps invalido/zero
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # so para o progresso
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # codec do arquivo de saida
        writer = cv2.VideoWriter(f"{args.saida}_anotado.mp4", fourcc, fps, (largura, altura))

    # Ate que frame gravar o video anotado (0 = ate o fim). Gravar 15 minutos em
    # 1080p e lento e gera arquivo enorme, quando em geral basta conferir o
    # primeiro minuto para validar as linhas e as classes.
    limite_video = int(args.video_ate_s * fps) if args.video_ate_s > 0 else 0

    # Cores fixas para desenhar cada linha no video (ciclo, se houver mais linhas).
    CORES_LINHA = [(0, 0, 255), (255, 0, 255), (0, 200, 0), (255, 160, 0),
                   (0, 255, 255), (255, 0, 0)]

    # eventos e o acumulado global. Cada cruzamento vira uma linha do CSV, agora com
    # a coluna extra "linha" identificando por onde o veiculo passou.
    eventos = []              # (frame, tempo_s, id, classe, sentido, linha)

    # -----------------------------------------------------------------------
    # 5) DETECCAO + TRACKING
    # -----------------------------------------------------------------------
    # modelo.track() percorre o video inteiro aplicando YOLO e ByteTrack juntos.
    #   stream=True   -> devolve frame a frame em vez de acumular todos os
    #                    resultados na memoria (essencial em videos longos)
    #   persist=True  -> mantem o estado dos rastros entre frames
    #   classes=      -> filtra so os veiculos ainda dentro da inferencia
    frame_idx = 0
    resultados = modelo.track(
        source=args.video,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(CLASSES_VEICULO.keys()),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # 6) LACO PRINCIPAL: UM FRAME POR VEZ
    # -----------------------------------------------------------------------
    for r in resultados:
        frame_idx += 1
        tempo_s = frame_idx / fps          # instante do frame, para o CSV

        # Copiamos o frame apenas se ele for mesmo ser gravado, pois a copia custa
        # tempo e memoria, e na maior parte do video ela e desnecessaria.
        gravar = writer is not None and (limite_video == 0 or frame_idx <= limite_video)
        frame = r.orig_img.copy() if gravar else None

        # Progresso a cada 300 frames (~10 s de video), para acompanhar rodadas longas.
        if frame_idx % 300 == 0:
            pct = (100.0 * frame_idx / total_frames) if total_frames else 0
            resumo = "  ".join(f"{l.nome}={l.total()}" for l in linhas)
            print(f"  frame {frame_idx}/{total_frames} ({pct:.1f}%)  {resumo}", flush=True)

        # r.boxes.id vem None quando o tracker ainda nao firmou nenhum rastro
        # neste frame (por exemplo, logo no inicio do video).
        if r.boxes is not None and r.boxes.id is not None:
            # Trazemos os tensores da GPU para a CPU uma unica vez por frame.
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()

            # ---- por veiculo detectado neste frame ----
            for tid, cls_id, box in zip(ids, clss, xyxy):
                # O centro da caixa e o ponto que "cruza" a linha.
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                centro = (cx, cy)
                nome = CLASSES_VEICULO.get(cls_id, str(cls_id))

                # Testa o cruzamento em TODAS as linhas (cada uma e independente).
                for l in linhas:
                    l.processar(tid, nome, centro, frame_idx, tempo_s, eventos)

                # ---- anotacao visual do veiculo ----
                # Verde = ja contado em ALGUMA linha, azul-claro = ainda nao.
                if frame is not None:
                    contado = any(tid in l.ja_contado for l in linhas)
                    cor = (0, 255, 0) if contado else (255, 180, 0)
                    cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])), cor, 2)
                    cv2.putText(frame, f"{nome} #{tid}", (int(box[0]), int(box[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        # ---- painel fixo do frame (as linhas e o placar por linha) ----
        if frame is not None:
            y0 = 30                                  # altura da 1a linha do placar
            for i, l in enumerate(linhas):
                cor = CORES_LINHA[i % len(CORES_LINHA)]
                # a linha em si, com o nome no meio
                cv2.line(frame, l.a, l.b, cor, 2)
                mx, my = (l.a[0] + l.b[0]) // 2, (l.a[1] + l.b[1]) // 2
                cv2.putText(frame, l.nome, (mx - 8, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
                # o total desta linha no placar (canto superior esquerdo)
                cv2.putText(frame, f"{l.nome}: {l.total()}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
                y0 += 28
            writer.write(frame)

    # Fecha o arquivo de video. Sem isso o MP4 pode ficar corrompido.
    if writer is not None:
        writer.release()

    # -----------------------------------------------------------------------
    # 7) EXPORTACAO
    # -----------------------------------------------------------------------
    # 7.1) CSV de eventos, uma linha por cruzamento. Dado bruto, com a coluna
    #      "linha" indicando por qual aproximacao o veiculo passou.
    with open(f"{args.saida}_eventos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "tempo_s", "id", "classe", "sentido", "linha"])
        w.writerows(eventos)

    # 7.2) CSV resumo, com, para cada linha, os totais por classe e sentido, mais uma
    #      linha TOTAL da aproximacao. E o formato direto para o relatorio.
    with open(f"{args.saida}_resumo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["linha", "classe", "A_para_B", "B_para_A", "total"])
        for l in linhas:
            total_linha = 0
            for nome in CLASSES_VEICULO.values():
                t = l.totais[nome]
                tot = t["A_para_B"] + t["B_para_A"]
                total_linha += tot
                w.writerow([l.nome, nome, t["A_para_B"], t["B_para_A"], tot])
            w.writerow([l.nome, "TOTAL", "", "", total_linha])

    # -----------------------------------------------------------------------
    # 8) RELATORIO NO TERMINAL
    # -----------------------------------------------------------------------
    print("\n=== Contagem concluida ===")
    for l in linhas:
        print(f"\nLinha {l.nome}  ({l.a} -> {l.b})   total={l.total()}")
        for nome in CLASSES_VEICULO.values():
            t = l.totais[nome]
            print(f"  {nome:10s}: A->B={t['A_para_B']:4d}  B->A={t['B_para_A']:4d}  "
                  f"total={t['A_para_B']+t['B_para_A']:4d}")
    print(f"\nArquivos gerados:")
    print(f"  {args.saida}_eventos.csv  (1 linha por cruzamento, com tempo e linha)")
    print(f"  {args.saida}_resumo.csv   (totais por linha, classe e sentido)")
    if writer is not None:
        print(f"  {args.saida}_anotado.mp4  (video com boxes e linhas)")


if __name__ == "__main__":
    main()
