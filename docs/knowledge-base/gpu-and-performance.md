# GPU and performance

BytesTTS runs every voice through ONNX Runtime: on the CPU by default, or on an NVIDIA GPU when the CUDA build of
ONNX Runtime is installed and working.

## Using an NVIDIA GPU

1. Install NVIDIA's **CUDA 13 runtime** and **cuDNN 9**. BytesTTS does not bundle NVIDIA's libraries.
2. Install BytesTTS with the `gpu` extra, then make sure the GPU build of ONNX Runtime is the one on disk:

   ```
   pip install "bytestts[gpu] @ git+https://github.com/jared0565/bytesTTS"
   pip install --force-reinstall --no-deps onnxruntime-gpu
   ```

   `onnxruntime` (a core dependency) and `onnxruntime-gpu` install the same `onnxruntime` package files, and
   whichever was installed last wins. Since the install order is up to pip, reinstalling `onnxruntime-gpu` on its own
   makes sure the GPU build is the one on disk.
3. Load the voice with `device="auto"` (the default) or `device="cuda"`.

### Checking the GPU

```python
import onnxruntime as ort
print(ort.get_available_providers())   # should include "CUDAExecutionProvider"
```

Being **listed** is not the same as **working**: ONNX Runtime lists CUDA whenever the GPU build is installed, even
when it cannot load CUDA or cuDNN. BytesTTS therefore checks which provider the session really uses:

- `device="auto"`: if CUDA is listed but not working, it logs a warning and runs on the CPU.
- `device="cuda"`: it raises `NotSupported` with the same advice instead of falling back.

When CUDA is listed but not working, the CUDA 13 or cuDNN 9 libraries are missing or not found. On Windows their
`bin` folders must be on `PATH`; on Linux, on `LD_LIBRARY_PATH` or the system library path.

## CPU threads

```python
tts = BytesTTS.load_path("voices/en-ljspeech", threads=2)
```

```
bytestts say "..." --voice-path voices/en-ljspeech --threads 2 -o out.wav
```

- `threads=None` (the default) lets ONNX Runtime pick, usually one thread per physical core. It is the fastest on a
  machine that does nothing else.
- `threads=N` caps synthesis at N threads, runs operators one at a time, and stops idle threads from spinning. Use
  it on a shared server, beside other work, or on a laptop that runs hot. Synthesis is slower but the rest of the
  machine stays responsive.
- On a machine that becomes unstable under full CPU load, keep `threads` low (1 or 2).

## Speed and latency

- **Load once.** Loading reads, verifies and prepares the model. Keep the `BytesTTS` object and reuse it.
- **Stream for low latency.** `tts.stream(text)` yields each sentence when it is ready, so playback can start after
  the first sentence instead of the whole text.
- **Long texts are fine.** Sentence splitting takes time in proportion to the text's length, and sentences longer
  than 500 tokens are split near their middle.

## Memory

Each ONNX Runtime call handles one chunk of at most 500 tokens (a provisional limit that may change), so the
working memory per call stays bounded whatever the length of the text. A 500-token chunk peaked at about 500 MiB
of RAM at one thread in our measurements. `synthesize` keeps the whole result in memory; for very long texts, use
`stream` and write or play each sentence as it arrives.
