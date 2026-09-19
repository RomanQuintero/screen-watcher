import argparse
import time

import dxcam
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
TARGET_SIZE = (1024, 576)
MAX_NEW_TOKENS = 32
NUM_FRAMES = 5
PROMPT = (
    "Describe en español qué está ocurriendo en esta pantalla. "
    "Sé conciso: máximo 3 frases. "
    "Prioriza la actividad del usuario y las aplicaciones visibles. "
    "No enumeres detalles del código salvo que sean relevantes."
)


print(f"Cargando {MODEL_ID}…")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)


def create_camera():
    """Crea una instancia de DXcam para capturar la pantalla."""
    return dxcam.create()


def capture_frame(camera):
    """Captura un frame RGB de la pantalla."""
    frame = camera.grab()
    if frame is None:
        print("No se pudo capturar el frame")
        return None

    print(f"Frame capturado - Tamaño: {frame.shape}")
    return frame


def prepare_image(frame, target_size=TARGET_SIZE):
    """Convierte un frame RGB de DXcam a PIL y limita su resolución."""
    image = Image.fromarray(frame)  # DXcam entrega RGB por defecto.
    original_size = image.size
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    print(f"  Imagen reducida: {original_size} -> {image.size}")
    return image


def query_vlm(frame, question=PROMPT, target_size=TARGET_SIZE, max_new_tokens=MAX_NEW_TOKENS):
    """Consulta Qwen3-VL directamente con un frame capturado por DXcam."""
    image = prepare_image(frame, target_size)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decodificar solo los tokens generados, no el prompt de entrada.
    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
    )[0]


def synchronize_cuda():
    """Sincroniza CUDA cuando está disponible para medir inferencia con precisión."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(num_frames=NUM_FRAMES, question=PROMPT):
    """Mide la latencia del VLM sobre varios frames consecutivos."""
    camera = create_camera()
    latencies = []

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK VLM - {num_frames} frames")
    print(f"Modelo: {MODEL_ID}")
    print(f"Resolución máxima: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"{'=' * 60}")

    total_start = time.perf_counter()

    for i in range(num_frames):
        print(f"\n[{i + 1}/{num_frames}] Capturando y analizando frame...")
        frame = capture_frame(camera)
        if frame is None:
            continue

        synchronize_cuda()
        start = time.perf_counter()
        response = query_vlm(frame, question)
        synchronize_cuda()
        latency = time.perf_counter() - start
        latencies.append(latency)

        print(f"  Tiempo: {latency:.2f} s")
        print(f"  Respuesta: {response}")

    total_time = time.perf_counter() - total_start
    processed = len(latencies)

    if not processed:
        print("\nNo se pudieron procesar frames")
        return {}

    avg_latency = sum(latencies) / processed
    metrics = {
        "frames_processed": processed,
        "total_time_seconds": total_time,
        "avg_latency_seconds": avg_latency,
        "min_latency_seconds": min(latencies),
        "max_latency_seconds": max(latencies),
        "fps": processed / total_time,
        "latencies": latencies.copy(),
    }

    print(f"\n{'=' * 60}")
    print("RESULTADOS")
    print(f"{'=' * 60}")
    print(f"Frames procesados: {processed}")
    print(f"Tiempo total: {total_time:.2f} s")
    print(f"Latencia promedio: {metrics['avg_latency_seconds']:.2f} s/frame")
    print(f"FPS: {metrics['fps']:.3f}")
    print(f"Latencia mínima: {metrics['min_latency_seconds']:.2f} s")
    print(f"Latencia máxima: {metrics['max_latency_seconds']:.2f} s")

    return metrics


def single_test(question=PROMPT):
    """Captura una pantalla y muestra una única inferencia del VLM."""
    camera = create_camera()
    frame = capture_frame(camera)
    if frame is None:
        return

    synchronize_cuda()
    start = time.perf_counter()
    response = query_vlm(frame, question)
    synchronize_cuda()

    print(f"\nRespuesta del VLM ({time.perf_counter() - start:.2f} s):")
    print(response)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prueba y benchmark del VLM usado por Screen Watcher."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Analiza varios frames y muestra métricas de rendimiento.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=NUM_FRAMES,
        help=f"Número de frames del benchmark (por defecto: {NUM_FRAMES}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frames < 1:
        raise SystemExit("--frames debe ser mayor que 0")

    if args.benchmark:
        benchmark(num_frames=args.frames)
    else:
        single_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark detenido")
