"""
Compara dois modelos no conjunto de validacao do VisDrone, nas mesmas condicoes.
================================================================================

Roda model.val() nos dois pesos sobre o VisDrone val (548 imagens rotuladas) e
imprime uma tabela lado a lado com mAP50, mAP50-95, precisao e recall, global e
por classe. Como o gabarito e os parametros sao os mesmos, a comparacao e justa.

Usos tipicos:
    - medir o ganho de treinar mais epocas (ex.: 1 epoca contra 50 epocas)
    - comparar os pesos deste repositorio com outros pesos VisDrone publicados
      por terceiros (verifique a licenca antes de usa-los)

Os dois modelos precisam ter sido treinados com as classes do VisDrone. Pesos
COCO (yolov8m.pt) tem outras classes e nao podem ser avaliados neste conjunto.

------------------------------------------------------------------
COMO USAR:

    python comparar_modelos.py ^
        --a runs/detect/teste_1ep/weights/best.pt   --nome-a 1_epoca ^
        --b pesos/visdrone_y8m_50ep_best.pt         --nome-b 50_epocas
------------------------------------------------------------------
"""

import argparse

from ultralytics import YOLO

# Classes VisDrone que interessam ao trafego (nome no dataset -> rotulo pt)
INTERESSE = {"bicycle": "bicicleta", "car": "carro", "van": "van",
             "truck": "caminhao", "bus": "onibus", "motor": "moto"}


def avaliar(pesos, imgsz):
    """Roda val() e devolve dicts de metricas, global e por classe (por nome)."""
    modelo = YOLO(pesos)
    r = modelo.val(data="VisDrone.yaml", imgsz=imgsz, verbose=False)
    box = r.box
    glob = {"mAP50": box.map50, "mAP50-95": box.map, "P": box.mp, "R": box.mr}
    porclasse = {}
    for i, ci in enumerate(box.ap_class_index):
        nome = r.names[ci]
        porclasse[nome] = {
            "mAP50": box.ap50[i], "mAP50-95": box.ap[i],
            "P": box.p[i], "R": box.r[i],
        }
    return glob, porclasse


def linha(rotulo, ma, mb, chave, nome_a, nome_b):
    va = ma.get(chave)
    vb = mb.get(chave)
    if va is None or vb is None:
        return f"  {rotulo:12s}      -         -"
    dif = vb - va
    marca = f"  ({nome_a})" if va > vb else (f"  ({nome_b})" if vb > va else "  (empate)")
    return f"  {rotulo:12s} {va:6.3f}   {vb:6.3f}   dif={dif:+.3f}{marca}"


def main():
    ap = argparse.ArgumentParser(description="Compara dois modelos no VisDrone val")
    ap.add_argument("--a", required=True, help="Pesos do modelo A")
    ap.add_argument("--b", required=True, help="Pesos do modelo B")
    ap.add_argument("--nome-a", default="A")
    ap.add_argument("--nome-b", default="B")
    ap.add_argument("--imgsz", type=int, default=768,
                    help="Mesma resolucao para os dois (padrao 768, a do treino)")
    args = ap.parse_args()

    print(f"\nAvaliando A = {args.nome_a}  ({args.a}) ...")
    ga, ca = avaliar(args.a, args.imgsz)
    print(f"Avaliando B = {args.nome_b}  ({args.b}) ...")
    gb, cb = avaliar(args.b, args.imgsz)

    print("\n" + "=" * 60)
    print(f"COMPARACAO   (imgsz={args.imgsz}, VisDrone val)")
    print(f"colunas   {args.nome_a}   vs   {args.nome_b}   (entre parenteses, o melhor)")
    print("=" * 60)

    print("\n--- GLOBAL (todas as 10 classes) ---")
    for chave in ("mAP50", "mAP50-95", "P", "R"):
        print(linha(chave, ga, gb, chave, args.nome_a, args.nome_b))

    print("\n--- mAP50 por classe de interesse ---")
    for nome_ds, rot in INTERESSE.items():
        print(linha(rot, ca.get(nome_ds, {}), cb.get(nome_ds, {}), "mAP50",
                    args.nome_a, args.nome_b))

    print("\n--- recall por classe de interesse (quem PERDE menos veiculos) ---")
    for nome_ds, rot in INTERESSE.items():
        print(linha(rot, ca.get(nome_ds, {}), cb.get(nome_ds, {}), "R",
                    args.nome_a, args.nome_b))


if __name__ == "__main__":
    main()
