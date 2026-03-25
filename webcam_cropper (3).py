#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  WEBCAM CROPPER com Grounded SAM 2                              ║
║  Detecta e recorta EXATAMENTE a região da webcam numa imagem.   ║
║  Usa Grounding DINO (HF) + SAM 2 para segmentação precisa.     ║
╚══════════════════════════════════════════════════════════════════╝

Estratégia:
  1. Grounding DINO detecta o OBJETO "webcam/camera overlay" (não a pessoa!)
  2. SAM 2 gera máscara pixel-perfect desse objeto
  3. Usamos a bounding box DA MÁSCARA (não do detector) para recortar
  4. Adicionamos padding de segurança para não perder nenhum pixel da webcam
  5. Se múltiplas detecções, escolhemos a de maior confiança OU unimos todas

Prompts inteligentes:
  - Usamos VÁRIOS prompts descritivos do objeto "webcam" em si
  - O ponto final "." é OBRIGATÓRIO para o Grounding DINO
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path

# ─── SAM 2 ───
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─── Grounding DINO (HuggingFace) ───
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# ══════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES PADRÃO
# ══════════════════════════════════════════════════════════════════

# SAM 2.1 Large — melhor qualidade de máscara
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Grounding DINO via HuggingFace (download automático)
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"

# Prompts para detectar a WEBCAM (não a pessoa!)
# Cada prompt termina com "." como exigido pelo Grounding DINO
WEBCAM_PROMPTS = [
    "streamer webcam overlay.",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════

def load_models(
    sam2_checkpoint: str = SAM2_CHECKPOINT,
    sam2_config: str = SAM2_CONFIG,
    grounding_model_id: str = GROUNDING_MODEL,
    device: str = DEVICE,
):
    """Carrega SAM 2 e Grounding DINO."""
    print(f"[INFO] Carregando SAM 2 de: {sam2_checkpoint}")
    print(f"[INFO] Device: {device}")

    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print(f"[INFO] Carregando Grounding DINO de: {grounding_model_id}")
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounding_model_id
    ).to(device)

    return sam2_predictor, processor, grounding_model


def detect_webcam_boxes(
    image: Image.Image,
    processor,
    grounding_model,
    prompts: list,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    device: str = DEVICE,
):
    """
    Tenta múltiplos prompts para encontrar a webcam.
    Retorna TODAS as boxes encontradas com scores e labels.
    """
    all_boxes = []
    all_scores = []
    all_labels = []

    for prompt in prompts:
        # Garante lowercase + termina com "."
        prompt = prompt.lower().strip()
        if not prompt.endswith("."):
            prompt += "."

        inputs = processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]],  # (H, W)
        )

        if len(results) > 0 and len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"].cpu().numpy()
            scores = results[0]["scores"].cpu().numpy()
            labels = results[0]["labels"]

            # Filtra manualmente pelos thresholds (compatível com versões novas do transformers)
            mask = scores >= box_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = [l for l, m in zip(labels, mask) if m]

            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                all_boxes.append(box)
                all_scores.append(score)
                all_labels.append(f"{prompt} [{label}]")
                print(
                    f"  [DETECT] prompt='{prompt}' | "
                    f"score={score:.3f} | "
                    f"box=[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
                )

    if len(all_boxes) == 0:
        return None, None, None

    return (
        np.array(all_boxes),
        np.array(all_scores),
        all_labels,
    )


def filter_and_select_best_box(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list,
    image_size: tuple,
    min_area_ratio: float = 0.005,   # Mín 0.5% da imagem
    max_area_ratio: float = 0.50,    # Máx 50% da imagem
    min_aspect: float = 0.3,         # Mín aspect ratio (w/h)
    max_aspect: float = 3.5,         # Máx aspect ratio (w/h)
):
    """
    Filtra detecções por tamanho e proporção razoáveis para uma webcam.
    Retorna a melhor box (maior score entre as válidas).
    """
    img_w, img_h = image_size
    img_area = img_w * img_h

    valid_indices = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        area = w * h
        area_ratio = area / img_area
        aspect = w / max(h, 1)

        # Filtros de sanidade
        if area_ratio < min_area_ratio:
            print(f"  [FILTER] Box {i} descartada: muito pequena ({area_ratio:.4f})")
            continue
        if area_ratio > max_area_ratio:
            print(f"  [FILTER] Box {i} descartada: muito grande ({area_ratio:.4f})")
            continue
        if aspect < min_aspect or aspect > max_aspect:
            print(f"  [FILTER] Box {i} descartada: aspect ratio inválido ({aspect:.2f})")
            continue

        valid_indices.append(i)

    if len(valid_indices) == 0:
        print("  [WARN] Nenhuma detecção passou nos filtros!")
        # Fallback: retorna a de maior score sem filtro
        best_idx = np.argmax(scores)
        return boxes[best_idx], scores[best_idx], labels[best_idx]

    # Seleciona a de maior score entre as válidas
    valid_scores = scores[valid_indices]
    best_valid_idx = valid_indices[np.argmax(valid_scores)]

    return boxes[best_valid_idx], scores[best_valid_idx], labels[best_valid_idx]


def segment_with_sam2(
    sam2_predictor: SAM2ImagePredictor,
    image_np: np.ndarray,
    box: np.ndarray,
):
    """
    Usa SAM 2 para gerar máscara precisa a partir da box do Grounding DINO.
    Retorna a máscara binária.
    """
    sam2_predictor.set_image(image_np)

    # SAM 2 espera boxes no formato [x1, y1, x2, y2]
    input_box = box.reshape(1, 4)

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True,  # Gera 3 máscaras, pegamos a melhor
    )

    # Seleciona a máscara com maior score
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]

    print(f"  [SAM2] Máscara gerada | score={scores[best_mask_idx]:.4f}")
    print(f"  [SAM2] Máscaras disponíveis: {len(masks)}, scores: {scores}")

    return best_mask


def get_mask_bbox(mask: np.ndarray, padding: int = 0, image_shape: tuple = None):
    """
    Calcula a bounding box da máscara binária.
    Adiciona padding de segurança.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # Adiciona padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    if image_shape is not None:
        x2 = min(image_shape[1] - 1, x2 + padding)
        y2 = min(image_shape[0] - 1, y2 + padding)
    else:
        x2 = x2 + padding
        y2 = y2 + padding

    return (x1, y1, x2, y2)


def crop_webcam(
    image_path: str,
    output_dir: str = "./output",
    sam2_checkpoint: str = SAM2_CHECKPOINT,
    sam2_config: str = SAM2_CONFIG,
    grounding_model_id: str = GROUNDING_MODEL,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    padding: int = 5,
    save_debug: bool = True,
    custom_prompts: list = None,
):
    """
    Pipeline principal:
      1. Carrega modelos
      2. Detecta webcam com Grounding DINO (múltiplos prompts)
      3. Filtra e seleciona melhor detecção
      4. Segmenta com SAM 2
      5. Recorta usando bounding box DA MÁSCARA (não do detector)
      6. Salva resultado
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Carrega modelos ──
    sam2_predictor, processor, grounding_model = load_models(
        sam2_checkpoint, sam2_config, grounding_model_id
    )

    # ── Carrega imagem ──
    print(f"\n[INFO] Processando: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    img_w, img_h = image_pil.size
    print(f"[INFO] Tamanho da imagem: {img_w}x{img_h}")

    # ── Detecta webcam ──
    prompts = custom_prompts if custom_prompts else WEBCAM_PROMPTS
    print(f"\n[STEP 1] Detectando webcam com {len(prompts)} prompts...")

    boxes, scores, labels = detect_webcam_boxes(
        image_pil,
        processor,
        grounding_model,
        prompts=prompts,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if boxes is None:
        print("\n[ERRO] ❌ Nenhuma webcam detectada na imagem!")
        print("[DICA] Tente:")
        print("  - Reduzir box_threshold (ex: 0.15)")
        print("  - Reduzir text_threshold (ex: 0.10)")
        print("  - Adicionar prompts customizados com --prompts")
        return None

    print(f"\n[STEP 2] {len(boxes)} detecções encontradas, filtrando...")

    # ── Filtra e seleciona melhor ──
    best_box, best_score, best_label = filter_and_select_best_box(
        boxes, scores, labels, (img_w, img_h)
    )

    print(
        f"\n[BEST] ✅ Melhor detecção: score={best_score:.3f} | "
        f"label='{best_label}' | "
        f"box=[{best_box[0]:.0f}, {best_box[1]:.0f}, {best_box[2]:.0f}, {best_box[3]:.0f}]"
    )

    # ── Segmenta com SAM 2 ──
    print(f"\n[STEP 3] Segmentando com SAM 2...")
    mask = segment_with_sam2(sam2_predictor, image_np, best_box)

    # ── Calcula bbox da MÁSCARA (não do detector!) ──
    # Isso é crucial: a máscara do SAM 2 é mais precisa que a box do DINO
    mask_bbox = get_mask_bbox(mask, padding=padding, image_shape=image_np.shape)

    if mask_bbox is None:
        print("[ERRO] ❌ Máscara vazia! Usando box do detector como fallback.")
        x1, y1, x2, y2 = best_box.astype(int)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)
        mask_bbox = (x1, y1, x2, y2)

    x1, y1, x2, y2 = mask_bbox
    print(
        f"\n[STEP 4] Bounding box da máscara (com padding={padding}px): "
        f"[{x1}, {y1}, {x2}, {y2}] → {x2 - x1}x{y2 - y1}px"
    )

    # ── Recorta ──
    cropped = image_np[y1:y2 + 1, x1:x2 + 1]

    # ── Salva resultados ──
    basename = Path(image_path).stem
    crop_path = os.path.join(output_dir, f"{basename}_webcam_crop.png")
    cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    print(f"\n[SAVE] ✅ Webcam recortada salva em: {crop_path}")

    if save_debug:
        # 1. Imagem com box do Grounding DINO
        debug_img = image_np.copy()
        bx1, by1, bx2, by2 = best_box.astype(int)
        cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
        cv2.putText(
            debug_img,
            f"DINO: {best_score:.2f}",
            (bx1, by1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        dino_path = os.path.join(output_dir, f"{basename}_dino_detection.png")
        cv2.imwrite(dino_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

        # 2. Máscara do SAM 2
        mask_vis = image_np.copy()
        mask_overlay = np.zeros_like(image_np)
        mask_overlay[mask.astype(bool)] = [0, 120, 255]  # Azul semitransparente
        mask_vis = cv2.addWeighted(mask_vis, 0.7, mask_overlay, 0.3, 0)
        # Desenha bbox da máscara
        cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            mask_vis,
            f"MASK BBOX (crop area)",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
        sam_path = os.path.join(output_dir, f"{basename}_sam2_mask.png")
        cv2.imwrite(sam_path, cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

        # 3. Máscara binária
        mask_binary_path = os.path.join(output_dir, f"{basename}_mask_binary.png")
        cv2.imwrite(mask_binary_path, (mask.astype(np.uint8) * 255))

        print(f"[SAVE] Debug: {dino_path}")
        print(f"[SAVE] Debug: {sam_path}")
        print(f"[SAVE] Debug: {mask_binary_path}")

    print(f"\n{'═' * 60}")
    print(f"  RESULTADO: Webcam recortada com sucesso!")
    print(f"  Tamanho: {x2 - x1}x{y2 - y1} pixels")
    print(f"  Arquivo: {crop_path}")
    print(f"{'═' * 60}\n")

    return crop_path


# ══════════════════════════════════════════════════════════════════
#  PROCESSAMENTO EM LOTE (múltiplas imagens)
# ══════════════════════════════════════════════════════════════════

def crop_webcam_batch(
    input_dir: str,
    output_dir: str = "./output",
    **kwargs,
):
    """Processa todas as imagens de um diretório."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_files = sorted([
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in extensions
    ])

    print(f"[BATCH] Encontradas {len(image_files)} imagens em {input_dir}")

    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n{'━' * 60}")
        print(f"  [{i}/{len(image_files)}] {img_path.name}")
        print(f"{'━' * 60}")
        result = crop_webcam(str(img_path), output_dir=output_dir, **kwargs)
        results.append((img_path.name, result))

    # Resumo
    print(f"\n{'═' * 60}")
    print(f"  RESUMO DO LOTE")
    print(f"{'═' * 60}")
    success = sum(1 for _, r in results if r is not None)
    print(f"  ✅ Sucesso: {success}/{len(results)}")
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"    {status} {name}")
    print(f"{'═' * 60}\n")

    return results


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Recorta EXATAMENTE a webcam de uma imagem usando Grounded SAM 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Imagem única
  python webcam_cropper.py -i screenshot.png

  # Com thresholds ajustados (mais sensível)
  python webcam_cropper.py -i screenshot.png --box-threshold 0.15 --text-threshold 0.10

  # Diretório inteiro
  python webcam_cropper.py -d ./screenshots/ -o ./recortes/

  # Com prompts customizados
  python webcam_cropper.py -i img.png --prompts "facecam." "small video window."

  # Com mais padding de segurança
  python webcam_cropper.py -i img.png --padding 15
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--image", type=str, help="Caminho da imagem de entrada"
    )
    input_group.add_argument(
        "-d", "--directory", type=str, help="Diretório com múltiplas imagens"
    )

    parser.add_argument(
        "-o", "--output", type=str, default="./output",
        help="Diretório de saída (default: ./output)"
    )
    parser.add_argument(
        "--box-threshold", type=float, default=0.25,
        help="Threshold do Grounding DINO para boxes (default: 0.25)"
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.20,
        help="Threshold do Grounding DINO para texto (default: 0.20)"
    )
    parser.add_argument(
        "--padding", type=int, default=5,
        help="Padding extra em pixels ao redor da máscara (default: 5)"
    )
    parser.add_argument(
        "--prompts", nargs="+", type=str, default=None,
        help="Prompts customizados para detectar a webcam"
    )
    parser.add_argument(
        "--sam2-checkpoint", type=str, default=SAM2_CHECKPOINT,
        help=f"Checkpoint do SAM 2 (default: {SAM2_CHECKPOINT})"
    )
    parser.add_argument(
        "--sam2-config", type=str, default=SAM2_CONFIG,
        help=f"Config do SAM 2 (default: {SAM2_CONFIG})"
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Não salvar imagens de debug"
    )

    args = parser.parse_args()

    common_kwargs = dict(
        output_dir=args.output,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        padding=args.padding,
        save_debug=not args.no_debug,
        custom_prompts=args.prompts,
    )

    if args.image:
        crop_webcam(args.image, **common_kwargs)
    else:
        crop_webcam_batch(args.directory, **common_kwargs)


if __name__ == "__main__":
    main()