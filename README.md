# Pipeline de Transcripción de Audio con Faster-Whisper

Documentación técnica de la arquitectura desarrollada para la captura, procesamiento y transcripción automática de audio desde un flujo de red RTSP, utilizando aceleración por GPU (CUDA).

## Arquitectura y componentes principales

El proyecto utiliza una arquitectura modular que separa la captura y orquestación del audio del motor de transcripción.

### 1. Orquestador y control de flujo (`main.py`)

- **Ingesta de audio mediante FFmpeg:** se conecta de forma continua a un flujo de audio RTSP y extrae las muestras en formato PCM sin comprimir (`s16le`), mono y a 16 kHz.
- **Detección de actividad por amplitud:** evalúa el nivel de amplitud de bloques de audio de 100 ms para identificar actividad y períodos de silencio. Este mecanismo constituye una detección básica de actividad, no un sistema VAD especializado.
- **Gestión de segmentos:** acumula fragmentos de audio en un búfer y procesa el segmento cuando se alcanza el límite de silencio o la duración máxima configurada.
- **Control temporal:** mantiene la posición del audio recibido para asociar marcas de tiempo a los fragmentos transcritos.
- **Coordinación del procesamiento:** administra la captura, segmentación y envío de audio al módulo de transcripción.

### 2. Módulo de transcripción (`transcriber.py`)

- **Motor de reconocimiento:** utiliza Faster-Whisper con el modelo `large-v3-turbo`, ejecutado sobre CUDA con precisión `float16` para aprovechar la aceleración de una GPU NVIDIA compatible.
- **Detección de idioma:** permite configurar un idioma específico o utilizar la detección automática del modelo.
- **Parámetros de inferencia:** configura parámetros como `beam_size`, umbrales de confianza y detección de ausencia de voz para ajustar el comportamiento de la transcripción.
- **Filtrado de resultados:** aplica criterios de calidad para reducir transcripciones asociadas a silencios, repeticiones o resultados poco fiables, según los filtros habilitados en la implementación.
- **Marcas de tiempo:** obtiene los tiempos de inicio y finalización de los segmentos reconocidos y permite relacionarlos con la posición correspondiente dentro del flujo de audio.

## Flujo de procesamiento

```text
Fuente de audio
      |
      v
   MediaMTX
      |
      v
Flujo RTSP
      |
      v
    FFmpeg
      |
      v
Captura de audio PCM
      |
      v
Detección de actividad
y segmentación
      |
      v
    main.py
      |
      v
 transcriber.py
      |
      v
 Faster-Whisper
  (CUDA / float16)
      |
      v
Transcripción con
marcas de tiempo
```

## Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Desarrollo y coordinación del sistema |
| Faster-Whisper | Reconocimiento automático del habla |
| Whisper `large-v3-turbo` | Modelo de transcripción |
| CUDA | Aceleración de inferencia mediante GPU |
| FFmpeg | Captura y conversión del flujo de audio |
| MediaMTX | Distribución del audio mediante RTSP |
| NumPy | Manipulación de muestras de audio |
| python-dotenv | Gestión de variables de configuración |

## Estructura del proyecto

```text
proyecto/
├── main.py
├── transcriber.py
├── .env
├── .gitignore
└── requirements.txt
```

- **`main.py`:** captura el flujo RTSP, gestiona los búferes, detecta actividad por amplitud y coordina el procesamiento de los segmentos.
- **`transcriber.py`:** carga el modelo Faster-Whisper, configura los parámetros de inferencia y devuelve los resultados de la transcripción.
- **`.env`:** contiene variables de configuración, como la dirección del flujo RTSP.
- **`.gitignore`:** especifica los archivos y directorios excluidos del control de versiones.
- **`requirements.txt`:** registra las dependencias de Python utilizadas por el proyecto.

## Estado del desarrollo

El proyecto se encuentra en etapa de desarrollo y pruebas. El procesamiento continuo del flujo de audio está implementado como base para una transcripción con baja latencia, pero su precisión y estabilidad deben seguir evaluándose.

Las principales áreas de mejora incluyen:

- Precisión del reconocimiento en conversaciones espontáneas.
- Segmentación de voz y silencios.
- Reducción de transcripciones incorrectas o alucinaciones del modelo.
- Detección de idioma en fragmentos cortos.
- Estabilidad durante sesiones prolongadas.
- Evaluación de latencia y consumo de recursos de hardware.

## Consideraciones

Las transcripciones son resultados automáticos y pueden contener errores. El sistema debe evaluarse con grabaciones de referencia y no debe considerarse una fuente infalible de información.

En aplicaciones que procesen conversaciones sensibles, es necesario aplicar medidas adecuadas de confidencialidad, seguridad y control de acceso.
