# Hörfaul

Drop voice message files (WhatsApp `.opus`, `.m4a`, or any common audio format) and get a plain-text transcript. Everything runs in the browser — no backend, no cloud, no API keys required.

Transcription is performed by [OpenAI Whisper](https://openai.com/research/whisper) compiled to WebAssembly via [Transformers.js](https://huggingface.co/docs/transformers.js). Audio never leaves the device.

**Browser support:** Chrome, Firefox, Edge, Safari 16+  
**First use:** downloads the model (~39 MB, cached for future sessions)

## Usage

Open `index.html` via a local HTTP server (required for the service worker):

```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

Or deploy as static files to GitHub Pages — the included service worker handles the cross-origin isolation headers that Transformers.js requires.
