import { pipeline, WhisperTextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

let transcriber = null;

self.onmessage = async ({ data }) => {
  if (data.type === 'load') {
    try {
      transcriber = await pipeline(
        'automatic-speech-recognition',
        data.modelId,
        {
          dtype: data.dtype,
          progress_callback: (info) => self.postMessage({ type: 'model-progress', info }),
        }
      );
      self.postMessage({ type: 'model-ready' });
    } catch (err) {
      self.postMessage({ type: 'model-error', message: err.message });
    }

  } else if (data.type === 'transcribe') {
    try {
      const streamer = new WhisperTextStreamer(transcriber.tokenizer, {
        skip_prompt: true,
        decode_kwargs: { skip_special_tokens: true },
        callback_function: (text) => self.postMessage({ type: 'token', text }),
      });
      const result = await transcriber(data.url, {
        language: data.lang,
        chunk_length_s: 30,
        stride_length_s: 5,
        streamer,
      });
      const text = (result.text ?? '').trim() || '(no speech detected)';
      self.postMessage({ type: 'result', text });
    } catch (err) {
      self.postMessage({ type: 'transcribe-error', message: err.message });
    }
  }
};
