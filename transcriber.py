import os
import sys
import logging
from typing import Optional

# 1. CONFIGURACIÓN GENERAL
MODEL_NAME = "large-v3-turbo"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# Idiomas disponibles:
# "es" = español
# "en" = inglés
# "ja" = japonés
IDIOMA = "es"

IDIOMAS_COMPATIBLES = {"es", "en", "ja"}

INITIAL_PROMPT: Optional[str] = None

# Parámetros de reconocimiento
BEAM_SIZE = 5
BEST_OF = 5
CONDITION_ON_PREVIOUS_TEXT = False

# Filtros de calidad
MAX_NO_SPEECH_PROB = 0.80
MIN_AVG_LOGPROB = -1.0
MAX_COMPRESSION_RATIO = 2.4

# Si es False, las transcripciones de baja confianza se conservan y se marcan en el registro.
DESCARTAR_BAJA_CONFIANZA = False

# Si es True, se descartan segmentos con una probabilidad alta de ausencia de voz.
FILTRAR_AUSENCIA_DE_VOZ = True

# Si es True, se descartan segmentos con una relación de compresión excesiva.
FILTRAR_REPETICIONES = True


# 2. REGISTRO DE MENSAJES

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# 3. CONFIGURACIÓN DE DLL DE CUDA PARA WINDOWS

_DLL_HANDLES = []


def registrar_dll_cuda():
    """
    Registra directorios de DLL usados por CUDA y CTranslate2
    en Windows.

    Mantiene referencias a los handles durante la ejecución.
    No copia ni modifica archivos de las bibliotecas instaladas.
    """
    if os.name != "nt":
        return

    site_packages = os.path.join(
        sys.prefix,
        "Lib",
        "site-packages"
    )

    rutas_cuda = [
        os.path.join(
            site_packages,
            "nvidia",
            "cublas",
            "bin"
        ),
        os.path.join(
            site_packages,
            "nvidia",
            "cudnn",
            "bin"
        ),
        os.path.join(
            site_packages,
            "torch",
            "lib"
        ),
        os.path.join(
            site_packages,
            "ctranslate2"
        ),
    ]

    for ruta in rutas_cuda:
        if not os.path.isdir(ruta):
            continue

        try:
            handle = os.add_dll_directory(ruta)
            _DLL_HANDLES.append(handle)
            logger.info("DLL registradas: %s", ruta)

        except OSError as error:
            logger.warning(
                "No se pudo registrar %s: %s",
                ruta,
                error
            )


# Registrar DLL antes de importar Faster-Whisper.
registrar_dll_cuda()

from faster_whisper import WhisperModel


# 4. VALIDAR CONFIGURACIÓN
def validar_configuracion():
    """Comprueba que la configuración sea coherente."""

    if IDIOMA is not None and IDIOMA not in IDIOMAS_COMPATIBLES:
        raise ValueError(
            f"Idioma no compatible con esta configuración: {IDIOMA}. "
            f"Utiliza uno de {sorted(IDIOMAS_COMPATIBLES)} o None."
        )

    if DEVICE not in {"cuda", "cpu"}:
        raise ValueError(
            "DEVICE debe ser 'cuda' o 'cpu'."
        )

    if BEAM_SIZE < 1 or BEST_OF < 1:
        raise ValueError(
            "BEAM_SIZE y BEST_OF deben ser mayores que cero."
        )


# 5. CARGAR MODELO
def cargar_whisper():
    """
    Carga Whisper y devuelve el modelo preparado para transcribir.

    Los errores de carga se registran y se vuelven a lanzar
    para que el programa principal pueda gestionarlos.
    """
    validar_configuracion()

    logger.info("Cargando Whisper %s...", MODEL_NAME)
    logger.info("Dispositivo: %s", DEVICE)
    logger.info("Precisión: %s", COMPUTE_TYPE)

    if IDIOMA is None:
        logger.info("Idioma: detección automática")
    else:
        logger.info("Idioma seleccionado: %s", IDIOMA)

    try:
        model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )

        logger.info("Whisper cargado correctamente.")
        return model

    except Exception:
        logger.exception("No se pudo cargar Whisper.")
        raise



# 6. TRANSCRIBIR UN FRAGMENTO
def transcribir_segmento(model, audio):
    """
    Transcribe un fragmento de audio.

    Parámetros
    ----------
    model:
        Instancia de WhisperModel obtenida con cargar_whisper().

    audio:
        Audio aceptado por Faster-Whisper. En este proyecto,
        se espera un arreglo de audio mono a 16 kHz.

    Retorno
    -------
    list[dict]:
        Lista de segmentos compatibles con main.py.

        Cada segmento incluye:
        start, end, text, no_speech_prob, avg_logprob,
        compression_ratio y palabras.
    """

    if model is None:
        raise ValueError(
            "El modelo no está inicializado."
        )

    if audio is None:
        logger.warning("Se recibió un fragmento de audio vacío.")
        return []

    try:
        if len(audio) == 0:
            logger.warning("Se recibió un fragmento de audio vacío.")
            return []

    except TypeError as error:
        raise TypeError(
            "El audio debe tener una longitud definida y ser "
            "compatible con Faster-Whisper."
        ) from error

    opciones = {
        "language": IDIOMA,
        "task": "transcribe",
        "beam_size": BEAM_SIZE,
        "best_of": BEST_OF,
        "without_timestamps": False,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 500
        },
        "no_speech_threshold": 0.6,
        "compression_ratio_threshold": 2.2,
        "log_prob_threshold": -0.7,
        "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
    }

    # Solo enviar contexto si se ha definido.
    if INITIAL_PROMPT:
        opciones["initial_prompt"] = INITIAL_PROMPT

    resultados = []

    try:
        segments, info = model.transcribe(
            audio,
            **opciones
        )

        # Faster-Whisper puede detectar idiomas distintos de los tres que usamos habitualmente.
        idioma_detectado = getattr(info, "language", None)
        probabilidad_idioma = getattr(
            info,
            "language_probability",
            None
        )

        if idioma_detectado:
            if probabilidad_idioma is not None:
                logger.info(
                    "Idioma detectado: %s (probabilidad: %.2f)",
                    idioma_detectado,
                    probabilidad_idioma
                )
            else:
                logger.info(
                    "Idioma detectado: %s",
                    idioma_detectado
                )

            if idioma_detectado not in IDIOMAS_COMPATIBLES:
                logger.warning(
                    "El idioma detectado (%s) no está entre "
                    "los idiomas habituales: %s.",
                    idioma_detectado,
                    sorted(IDIOMAS_COMPATIBLES)
                )

        for segment in segments:
            texto = segment.text.strip()

            if not texto:
                continue

            # Filtro 1: posible ausencia de voz
            if (
                FILTRAR_AUSENCIA_DE_VOZ
                and segment.no_speech_prob is not None
                and segment.no_speech_prob >= MAX_NO_SPEECH_PROB
            ):
                logger.debug(
                    "Segmento descartado por ausencia de voz: %s",
                    texto
                )
                continue

            # Filtro 2: confianza baja
            # Por defecto, avisar sin eliminar el resultado.
            baja_confianza = (
                segment.avg_logprob is not None
                and segment.avg_logprob < MIN_AVG_LOGPROB
            )

            if baja_confianza:
                logger.warning(
                    "Transcripción de baja confianza: %s",
                    texto
                )

                if DESCARTAR_BAJA_CONFIANZA:
                    continue

            # Filtro 3: posible repetición
            if (
                FILTRAR_REPETICIONES
                and segment.compression_ratio is not None
                and segment.compression_ratio > MAX_COMPRESSION_RATIO
            ):
                logger.warning(
                    "Posible repetición o alucinación: %s",
                    texto
                )
                continue

            # Tiempos por palabra
            palabras = []

            if segment.words is not None:
                for word in segment.words:
                    palabras.append({
                        "palabra": word.word,
                        "inicio": word.start,
                        "fin": word.end,
                        "probabilidad": word.probability,
                    })

            resultado = {
                "start": segment.start,
                "end": segment.end,
                "text": texto,
                "no_speech_prob": segment.no_speech_prob,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "palabras": palabras,
            }

            resultados.append(resultado)

            logger.info(
                "[TRANSCRIPCIÓN] %.2fs - %.2fs: %s",
                resultado["start"],
                resultado["end"],
                resultado["text"]
            )

    except Exception:
        logger.exception(
            "Error durante la transcripción del fragmento."
        )
        raise

    return resultados