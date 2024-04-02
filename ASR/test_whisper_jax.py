from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline in bfloat16
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.float16, batch_size=16)

# JIT compile the forward call - slow, but we only do once
#task="translate"
# text = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
text = pipeline("audio.mp3",  task="transcribe")
# transcribe and return timestamps
# outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
# text = outputs["text"]  # transcription
# chunks = outputs["chunks"]  # transcription + timestamps