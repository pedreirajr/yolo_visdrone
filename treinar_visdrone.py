"""
Ajuste fino (fine-tuning) do YOLOv8 no dataset VisDrone.
========================================================

POR QUE ESTE SCRIPT EXISTE
--------------------------
Os pesos padrao do YOLO (yolov8m.pt) foram treinados no COCO, um conjunto de
fotografias tiradas do nivel do solo. Em video de drone, sobretudo em visada
vertical (nadir), os veiculos aparecem de cima e pequenos, e a rede deixa de
reconhece-los.

O VisDrone e um dataset de imagens capturadas por drones em varias cidades,
altitudes e angulos. Ajustar a rede nele ensina o modelo a reconhecer o
veiculo como ele aparece nessa geometria. Este script automatiza o treino, e o
resultado e um best.pt que o contagem_trafego_nadir.py usa no lugar do
yolov8m.pt.

O QUE E O AJUSTE FINO
---------------------
Em vez de treinar do zero, partimos dos pesos COCO, que ja aprenderam a
enxergar bordas, texturas e formas gerais, e reajustamos essa base para o
dominio aereo. O custo e muito menor do que treinar uma rede nova e o
resultado costuma ser melhor quando ha poucos dados.

CLASSES DO VISDRONE (diferentes das do COCO)
--------------------------------------------
    0 pedestrian  1 people   2 bicycle  3 car   4 van
    5 truck       6 tricycle 7 awning-tricycle  8 bus   9 motor

O modelo treinado tera esses indices e nomes. O contagem_trafego_nadir.py
resolve as classes pelo NOME, e nao por indice fixo, entao o mapeamento se
ajusta sozinho.

GPU E MEMORIA
-------------
O treino consome bastante memoria de video. Os pesos publicados neste
repositorio foram gerados numa RTX 4070 de 8 GB com os valores padrao abaixo
(imgsz 768, batch 4, 50 epocas), em cerca de 14 horas. Nao rode o treino ao
mesmo tempo que uma contagem, porque as duas tarefas disputam a mesma memoria
e uma delas pode estourar (out of memory).

Na primeira execucao o Ultralytics baixa o VisDrone (cerca de 2 GB) para a
pasta `datasets/VisDrone` e converte as anotacoes para o formato YOLO.

------------------------------------------------------------------
COMO USAR:

    # 1) Teste de tempo com uma epoca, para dimensionar o treino completo
    python treinar_visdrone.py --epochs 1 --nome teste_1ep

    # 2) Ajuste completo (reproduz os pesos publicados na Release)
    python treinar_visdrone.py

    # 3) Grafico da evolucao das metricas ao longo das epocas
    python plotar_treino.py --run runs/detect/visdrone_y8m_50ep

    # 4) Contagem com os pesos gerados
    python contagem_trafego_nadir.py --video "...nadir.MP4" ^
        --modelo runs/detect/visdrone_y8m_50ep/weights/best.pt --linha ...
------------------------------------------------------------------
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Ajuste fino do YOLO no VisDrone")
    ap.add_argument("--base", default="yolov8m.pt",
                    help="Pesos de partida do ajuste (padrao yolov8m.pt, baixado "
                         "automaticamente na primeira vez)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Numero de epocas. Comece com 1 para medir o tempo.")
    ap.add_argument("--imgsz", type=int, default=768,
                    help="Resolucao de treino. O VisDrone tem objetos pequenos, entao "
                         "resolucoes maiores ajudam, ao custo de mais memoria. Com 8 GB, "
                         "768 com batch 4 coube sem problemas.")
    ap.add_argument("--batch", type=int, default=4,
                    help="Tamanho do lote FIXO. Evite -1 (automatico) com 8 GB, pois o "
                         "teste do AutoBatch em resolucoes altas pode estourar a memoria.")
    ap.add_argument("--device", default=0,
                    help="Dispositivo (0 = primeira GPU, cpu = processador, inviavel para treino)")
    ap.add_argument("--nome", default="visdrone_y8m_50ep",
                    help="Nome da pasta de saida em runs/detect/")
    args = ap.parse_args()

    # Carrega os pesos de partida. O ajuste continua a partir deles.
    modelo = YOLO(args.base)

    # model.train() cuida de todas as etapas (download do VisDrone na primeira
    # vez via VisDrone.yaml, que acompanha o Ultralytics, conversao das
    # anotacoes, treino, validacao a cada epoca e gravacao dos pesos).
    #   patience=15  interrompe o treino se a metrica parar de melhorar
    #   plots=True   gera os graficos de treino para auditoria
    modelo.train(
        data="VisDrone.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.nome,
        patience=15,
        plots=True,
    )

    # Se runs/detect/<nome> ja existir, o Ultralytics cria <nome>2, <nome>3 etc.
    # O caminho real fica em trainer.save_dir, entao usamos ele em vez de
    # remontar o caminho na mao.
    save_dir = getattr(modelo.trainer, "save_dir", f"runs/detect/{args.nome}")
    print("\n=== Treino concluido ===")
    print(f"Pesos finais em {save_dir}/weights/best.pt")
    print("Aponte a contagem para esse arquivo com --modelo.")


if __name__ == "__main__":
    main()
