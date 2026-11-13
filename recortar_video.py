"""
Recorta um trecho curto de um video de drone (e opcionalmente o reduz).
=======================================================================

Os videos originais do drone tem varios GB (10 a 15 minutos em 1080p), o que
torna inviavel distribui-los junto com o codigo. Este script extrai um trecho
de poucos segundos, que serve para testar a contagem sem precisar do arquivo
completo. Foi com ele que se gerou o video de demonstracao publicado na
Release do repositorio.

O recorte e feito com o ffmpeg que acompanha o pacote `imageio-ffmpeg`
(instalado pelo requirements.txt), gerando um MP4 em H.264 bem menor que o
original. Se o pacote nao estiver disponivel, o script recorre ao OpenCV, que
grava em MPEG-4 Part 2 (arquivo maior, mas igualmente legivel pelo YOLO).

ATENCAO AS COORDENADAS DAS LINHAS
---------------------------------
As linhas de contagem sao definidas em pixels. Se o video for reduzido com
--largura, as coordenadas escolhidas no video original deixam de valer e
precisam ser multiplicadas pelo mesmo fator de escala (o script imprime esse
fator ao final). Sem --largura, a resolucao original e mantida e as mesmas
linhas continuam validas.

------------------------------------------------------------------
COMO USAR:

    # 20 segundos a partir de 5 min (300 s), mantendo 1080p
    python recortar_video.py --video "E:\\DRONE SEMOB\\DJI_20260722094104_0178_D.MP4" ^
        --inicio 300 --duracao 20 --saida videos/demo_nadir_20s.mp4

    # mesmo trecho reduzido para 1280 px de largura
    python recortar_video.py --video "...MP4" --inicio 300 --duracao 20 ^
        --largura 1280 --saida videos/demo_nadir_20s_720p.mp4
------------------------------------------------------------------
"""

import argparse
import os
import subprocess

import cv2


def recortar_ffmpeg(args, ffmpeg):
    """Recorta com ffmpeg (H.264). Devolve True se deu certo."""
    cmd = [ffmpeg, "-y", "-ss", str(args.inicio), "-i", args.video,
           "-t", str(args.duracao), "-an",              # -an descarta o audio
           "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium",
           "-pix_fmt", "yuv420p"]
    if args.largura:
        cmd += ["-vf", f"scale={args.largura}:-2"]      # -2 mantem a proporcao (altura par)
    cmd.append(args.saida)
    print("Executando ffmpeg...")
    return subprocess.run(cmd).returncode == 0


def recortar_opencv(args):
    """Alternativa sem ffmpeg, frame a frame com o OpenCV."""
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    larg = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.largura:
        alt = int(round(alt * args.largura / larg / 2)) * 2
        larg = args.largura
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.inicio * fps))
    writer = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (larg, alt))
    for _ in range(int(args.duracao * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if args.largura:
            frame = cv2.resize(frame, (larg, alt), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser(description="Recorta um trecho curto de um video")
    ap.add_argument("--video", required=True, help="Video de entrada")
    ap.add_argument("--inicio", type=float, default=0, help="Instante inicial, em segundos")
    ap.add_argument("--duracao", type=float, default=20, help="Duracao do trecho, em segundos")
    ap.add_argument("--largura", type=int, default=0,
                    help="Nova largura em pixels (0 = manter a resolucao original)")
    ap.add_argument("--crf", type=int, default=23,
                    help="Qualidade do H.264 (menor = melhor e maior). 18 a 28 e o usual.")
    ap.add_argument("--saida", required=True, help="Arquivo MP4 de saida")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"Video nao encontrado: {args.video}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)

    try:
        import imageio_ffmpeg
        ok = recortar_ffmpeg(args, imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        print("imageio-ffmpeg nao instalado, usando o OpenCV (arquivo maior).")
        ok = recortar_opencv(args)

    if not ok:
        print("Falha no recorte.")
        return

    cap = cv2.VideoCapture(args.video)
    larg_orig = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    tam = os.path.getsize(args.saida) / 1e6
    print(f"\nTrecho salvo em {args.saida} ({tam:.1f} MB)")
    if args.largura:
        print(f"Fator de escala das coordenadas das linhas = {args.largura / larg_orig:.4f}")


if __name__ == "__main__":
    main()
