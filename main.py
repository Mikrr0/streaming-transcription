
import os
import subprocess
import logging
from collections import deque

import numpy as np
from dotenv import load_dotenv

from transcriber import cargar_whisper, transcribir_segmento


# CONFIGURACIÓN

load_dotenv()

STREAM_URL = os.getenv(
    "STREAM_URL",
    "rtsp://127.0.0.1:8554/sala1"
)

SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2

# Leer audio cada 100 ms.
CHUNK_DURATION_MS = 100

# Terminar el segmento tras 1,2 segundos de silencio.
SILENCE_LIMIT_CHUNKS = 12

# Umbral inicial de actividad. Ajustar según el audio real.
VOLUME_THRESHOLD = 0.015

# Duración máxima del segmento.
MAX_SEGMENT_SECONDS = 15

# Guardar 500 ms anteriores al inicio de la voz.
PRE_ROLL_CHUNKS = 5

# No enviar segmentos extremadamente cortos.
MIN_SEGMENT_SECONDS = 0.5

CHUNK_SIZE = int(
    SAMPLE_RATE
    * CHANNELS
    * BYTES_PER_SAMPLE
    * CHUNK_DURATION_MS
    / 1000
)

MAX_SEGMENT_SAMPLES = SAMPLE_RATE * MAX_SEGMENT_SECONDS
MIN_SEGMENT_SAMPLES = int(SAMPLE_RATE * MIN_SEGMENT_SECONDS)


# REGISTRO DE MENSAJES

logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# CARGA DEL MODELO

whisper_model = cargar_whisper()

if whisper_model is None:
    raise RuntimeError(
        "No se pudo cargar Whisper. Revisa los errores anteriores."
    )


# INICIO DE FFMPEG

cmd = [
    "ffmpeg",
    "-nostdin",
    "-loglevel", "error",
    "-i", STREAM_URL,
    "-f", "s16le",
    "-acodec", "pcm_s16le",
    "-ac", str(CHANNELS),
    "-ar", str(SAMPLE_RATE),
    "-"
]

process = None


# ESTADO DE LA CAPTURA
pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)

audio_buffer = []
segment_start_time = None

silence_chunks = 0
is_speaking = False
audio_position = 0.0
total_buffer_samples = 0


# TRANSCRIPCIÓN

def transcribir_buffer():
    """Transcribe el audio acumulado y muestra tiempos globales."""

    global audio_buffer, segment_start_time

    if not audio_buffer or segment_start_time is None:
        return

    full_audio = np.concatenate(audio_buffer)

    if full_audio.size < MIN_SEGMENT_SAMPLES:
        return

    # Evitar procesar audio prácticamente silencioso.
    nivel_audio = float(np.max(np.abs(full_audio)))

    if nivel_audio < VOLUME_THRESHOLD:
        return

    segmentos = transcribir_segmento(
        whisper_model,
        full_audio
    )

    # No imprimir nada si Whisper no encontró voz.
    if not segmentos:
        return

    lineas = []

    for segmento in segmentos:
        texto = segmento.get("text", "").strip()

        if not texto:
            continue

        inicio = segment_start_time + segmento["start"]
        fin = segment_start_time + segmento["end"]

        lineas.append(
            f"[{inicio:06.2f} - {fin:06.2f}] {texto}"
        )


    if lineas:
        print("\n--- TRANSCRIPCIÓN ---")

        for linea in lineas:
            print(linea)

        print("-" * 22)




try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )

    if process.stdout is None:
        raise RuntimeError(
            "No se pudo abrir la salida de audio de FFmpeg."
        )

    logger.warning("Captura de audio iniciada.")

    while True:
        raw_audio = process.stdout.read(CHUNK_SIZE)


        if not raw_audio:
            if process.poll() is not None:
                logger.error("FFmpeg terminó el proceso.")
                break

            continue


        audio_data = (
            np.frombuffer(raw_audio, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )

        if audio_data.size == 0:
            continue

        chunk_start_time = audio_position
        audio_position += audio_data.size / SAMPLE_RATE


        max_amp = float(np.max(np.abs(audio_data)))
        hay_actividad = max_amp >= VOLUME_THRESHOLD


        if not is_speaking:

            if hay_actividad:

                audio_previo = list(pre_roll)

                audio_buffer = [
                    audio for audio, _ in audio_previo
                ]

                audio_buffer.append(audio_data)

                if audio_previo:
                    segment_start_time = audio_previo[0][1]
                else:
                    segment_start_time = chunk_start_time

                total_buffer_samples = sum(
                    chunk.size for chunk in audio_buffer
                )

                is_speaking = True
                silence_chunks = 0
                pre_roll.clear()

            else:

                pre_roll.append(
                    (audio_data, chunk_start_time)
                )

            continue

        audio_buffer.append(audio_data)
        total_buffer_samples += audio_data.size

        if hay_actividad:
            silence_chunks = 0
        else:
            silence_chunks += 1

        llego_a_silencio = (
            silence_chunks >= SILENCE_LIMIT_CHUNKS
        )

        llego_al_maximo = (
            total_buffer_samples >= MAX_SEGMENT_SAMPLES
        )


        if llego_a_silencio or llego_al_maximo:

            transcribir_buffer()

            audio_buffer = []
            segment_start_time = None

            total_buffer_samples = 0
            silence_chunks = 0
            is_speaking = False

            pre_roll.clear()

except KeyboardInterrupt:
    logger.warning("Deteniendo la captura...")

except Exception:
    logger.exception("Error durante la captura de audio.")

finally:

    if audio_buffer and segment_start_time is not None:
        try:
            transcribir_buffer()
        except Exception:
            logger.exception(
                "No se pudo transcribir el último segmento."
            )

    if process is not None:

        if process.stdout is not None:
            process.stdout.close()

        if process.poll() is None:
            process.terminate()

        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    logger.warning("Captura finalizada.")
