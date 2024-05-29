import logging
import wave
from pathlib import Path
from typing import Annotated

import bentoml
from bentoml.validators import ContentType

import ChatTTS

logger = logging.getLogger("bentoml")


def save_wav_file(wav, filename="output_audio.wav"):
    # Convert numpy array to bytes and write to WAV file
    wav_bytes = (wav * 32768).astype("int16").tobytes()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(wav_bytes)


@bentoml.service(resources={"gpu": 1})
class ChatTTSService:
    def __init__(self) -> None:
        chat = ChatTTS.Chat()
        logger.info("Initializing ChatTTS...")
        chat.load_models()
        logger.info("Models loaded successfully.")
        self.chat = chat

    @bentoml.api
    def infer(
        self, text_input: str, ctx: bentoml.Context
    ) -> Annotated[Path, ContentType("audio/wav")]:
        wavs = self.chat.infer([text_input], use_decoder=True)
        logger.info("Inference completed. Audio generation successful.")
        output_path = Path(ctx.temp_dir, "output_audio.wav")
        save_wav_file(wavs[0], str(output_path))
        return output_path
