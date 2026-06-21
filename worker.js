import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

let transcriber = null;

self.addEventListener('message', async ({ data: msg }) => {
  switch (msg.type) {
    case 'load': {
      try {
        transcriber = await pipeline(
          'automatic-speech-recognition',
          msg.modelId,
          {
            dtype: msg.dtype,
            progress_callback: (info) => self.postMessage({ type: 'progress', data: info }),
          }
        );
        self.postMessage({ type: 'ready' });
      } catch (err) {
        self.postMessage({ type: 'model-error', data: err.message });
      }
      break;
    }
    case 'transcribe': {
      try {
        const result = await transcriber(msg.url, {
          language: msg.language,
          chunk_length_s: 20,
          stride_length_s: 3,
          chunk_callback: (chunk) => {
            self.postMessage({ type: 'chunk', data: chunk.text });
          },
        });
        self.postMessage({ type: 'done', data: result.text });
      } catch (err) {
        self.postMessage({ type: 'transcribe-error', data: err.message });
      }
      break;
    }
  }
});
