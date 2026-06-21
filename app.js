import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

// ── Model config ─────────────────────────────────────────────────────────────
const MODELS = {
  german: { label: 'German fine-tune', id: 'onnx-community/whisper-large-v3-turbo-german-ONNX', dtype: { encoder_model: 'q8', decoder_model_merged: 'q4' } },
  tiny:   { label: 'Whisper Tiny',     id: 'onnx-community/whisper-tiny',                        dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } },
  base:   { label: 'Whisper Base',     id: 'onnx-community/whisper-base',                        dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } },
  small:  { label: 'Whisper Small',    id: 'onnx-community/whisper-small',                       dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } },
};

// ── DOM refs ────────────────────────────────────────────────────────────────
const modelSection  = document.getElementById('model-section');
const modelPick     = document.getElementById('model-pick');
const modelProgress = document.getElementById('model-progress');
const modelSelectEl = document.getElementById('model-select');
const btnLoadModel  = document.getElementById('btn-load-model');
const modelLabel    = document.getElementById('model-label');
const modelDetail   = document.getElementById('model-detail');
const progressFill  = document.getElementById('progress-fill');
const compatSection = document.getElementById('compat-section');
const compatMessage = document.getElementById('compat-message');
const dropZone      = document.getElementById('drop-zone');
const dropHint      = document.getElementById('drop-hint');
const fileInput     = document.getElementById('file-input');
const queueEl       = document.getElementById('queue');
const toolbar       = document.getElementById('toolbar');
const btnCopyAll    = document.getElementById('btn-copy-all');
const btnClearAll   = document.getElementById('btn-clear-all');
const langSelect    = document.getElementById('lang-select');
const toastEl       = document.getElementById('toast');

// ── State ────────────────────────────────────────────────────────────────────
const STORAGE_KEY = 'hoerfaul-v1';
let transcriber = null;
let processing  = false;
const pending   = [];
const cards     = new Map();  // fileKey → <article> element
let pendingSharedFile = null; // file received via Web Share Target, queued until model ready

// ── Restore saved transcripts from previous session ──────────────────────────
let savedTranscripts = loadSaved();
if (savedTranscripts.length > 0) {
  savedTranscripts.forEach(({ id, name, text }) => addCard(id, name, text));
  toolbar.hidden = false;
}

// ── Web Share Target: retrieve file stashed by the service worker ─────────────
if (location.search.includes('shared=1')) {
  history.replaceState({}, '', location.pathname);
  (async () => {
    try {
      const cache = await caches.open('hoerfaul-share');
      const response = await cache.match('shared-file');
      if (!response) return;
      await cache.delete('shared-file');
      const blob = await response.blob();
      const name = decodeURIComponent(response.headers.get('X-File-Name') || 'shared-audio');
      pendingSharedFile = new File([blob], name, { type: blob.type });
      checkWasmSupport();  // auto-load when a file arrives via share target
    } catch (err) {
      console.error('Failed to retrieve shared file:', err);
    }
  })();
}

// ── Model initialization ─────────────────────────────────────────────────────
btnLoadModel.addEventListener('click', checkWasmSupport);

async function checkWasmSupport() {
  try {
    // Quick WASM availability check
    await WebAssembly.compile(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
  } catch {
    showCompat('Your browser does not support WebAssembly, which is required for on-device transcription. Try Chrome, Firefox, or Safari 15+.');
    return;
  }
  initModel();
}

async function initModel() {
  const model = MODELS[modelSelectEl.value] ?? MODELS.german;
  modelPick.hidden = true;
  modelProgress.hidden = false;
  modelLabel.textContent = `Loading ${model.label}…`;

  try {
    transcriber = await pipeline(
      'automatic-speech-recognition',
      model.id,
      { dtype: model.dtype, progress_callback: onModelProgress }
    );

    modelSection.hidden = true;
    enableDropZone();
  } catch (err) {
    modelLabel.textContent = 'Failed to load model.';
    modelDetail.textContent = err.message;
    console.error(err);
  }
}

function onModelProgress(info) {
  if (info.status === 'progress') {
    const pct = info.total ? Math.round((info.loaded / info.total) * 100) : 0;
    progressFill.style.width = pct + '%';
    modelDetail.textContent = `${info.file ?? ''}  ${pct}%`;
  } else if (info.status === 'done') {
    modelDetail.textContent = '';
  } else if (info.status === 'ready') {
    progressFill.style.width = '100%';
  }
}

function enableDropZone() {
  dropZone.classList.add('ready');
  dropZone.setAttribute('aria-disabled', 'false');
  dropHint.textContent = 'tap to choose a file · or drag & drop';
  if (pendingSharedFile) {
    handleFiles([pendingSharedFile]);
    pendingSharedFile = null;
  }
}

function showCompat(msg) {
  compatSection.hidden = false;
  compatMessage.textContent = msg;
}

// ── File input wiring ────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => {
  if (transcriber) fileInput.click();
});

dropZone.addEventListener('keydown', e => {
  if ((e.key === 'Enter' || e.key === ' ') && transcriber) fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFiles([...fileInput.files]);
  fileInput.value = '';
});

dropZone.addEventListener('dragover', e => {
  if (!transcriber) return;
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (!transcriber) return;
  const file = [...e.dataTransfer.files].find(isAudio);
  if (file) handleFiles([file]);
});

function isAudio(file) {
  return file.type.startsWith('audio/') || /\.(opus|m4a|ogg|webm|mp3|wav|flac|aac|oga)$/i.test(file.name);
}

// ── File handling ────────────────────────────────────────────────────────────
function handleFiles(files) {
  const file = files[0];
  if (!file) return;

  const id = fileKey(file);
  if (cards.has(id)) {
    showToast('Already in list');
    return;
  }

  pending.push({ file, id });
  addCard(id, file.name, null);
  toolbar.hidden = false;
  drainQueue();
}

function fileKey(file) {
  return `${file.name}::${file.size}`;
}

async function drainQueue() {
  if (processing) return;
  processing = true;
  while (pending.length > 0) {
    const { file, id } = pending.shift();
    await transcribeFile(file, id);
  }
  processing = false;
}

async function transcribeFile(file, id) {
  const entry = cards.get(id);
  if (!entry) return;

  setCardBody(entry.body, 'working');

  let url;
  try {
    url = URL.createObjectURL(file);
    const lang = langSelect.value || null;  // empty string = auto-detect
    const result = await transcriber(url, { language: lang });
    const text = (result.text ?? '').trim() || '(no speech detected)';
    setCardBody(entry.body, 'done', text);
    if (cards.has(id)) {  // skip if cleared during transcription
      persistTranscript(id, file.name, text);
    }
  } catch (err) {
    setCardBody(entry.body, 'error', err.message);
    console.error(err);
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
}

// ── Card rendering ───────────────────────────────────────────────────────────
function addCard(id, name, text) {
  const card = document.createElement('article');
  card.className = 'card';

  const header = document.createElement('div');
  header.className = 'card-header';

  const nameEl = document.createElement('span');
  nameEl.className = 'card-name';
  nameEl.title = name;
  nameEl.textContent = name;

  const copyBtn = document.createElement('button');
  copyBtn.className = 'btn-icon';
  copyBtn.title = 'Copy transcript';
  copyBtn.setAttribute('aria-label', 'Copy transcript');
  copyBtn.innerHTML = `<svg viewBox="0 0 24 24" width="15" height="15" fill="none"
    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2"/>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg>`;
  copyBtn.addEventListener('click', () => {
    const t = body.querySelector('.transcript');
    if (t) copyText(t.textContent);
  });

  header.appendChild(nameEl);
  header.appendChild(copyBtn);

  const body = document.createElement('div');
  body.className = 'card-body';

  card.appendChild(header);
  card.appendChild(body);

  cards.set(id, { card, body });
  queueEl.appendChild(card);

  setCardBody(body, text !== null ? 'done' : 'queued', text ?? undefined);
}

function setCardBody(body, state, detail) {
  const p = document.createElement('p');
  switch (state) {
    case 'queued':
    case 'working': {
      p.className = 'status-text';
      const spinner = document.createElement('span');
      spinner.className = 'spinner';
      p.append(spinner, state === 'queued' ? ' Waiting…' : ' Transcribing…');
      break;
    }
    case 'done':
      p.className = 'transcript';
      p.textContent = detail;
      break;
    case 'error':
      p.className = 'status-text error';
      p.textContent = `Error: ${detail}`;
      break;
  }
  body.replaceChildren(p);
}

// ── Toolbar actions ──────────────────────────────────────────────────────────
btnCopyAll.addEventListener('click', () => {
  const parts = [...queueEl.querySelectorAll('.transcript')]
    .map(el => el.textContent)
    .filter(Boolean);
  if (parts.length) copyText(parts.join('\n\n---\n\n'));
});

btnClearAll.addEventListener('click', () => {
  pending.length = 0;
  savedTranscripts = [];
  queueEl.innerHTML = '';
  cards.clear();
  toolbar.hidden = true;
  localStorage.removeItem(STORAGE_KEY);
});

// ── Clipboard ────────────────────────────────────────────────────────────────
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast('Copied');
  } catch {
    showToast('Copy failed');
  }
}

// ── Toast ────────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 2000);
}

// ── Persistence ──────────────────────────────────────────────────────────────
function loadSaved() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]');
    if (!Array.isArray(raw)) return [];
    return raw.filter(e =>
      e !== null &&
      typeof e === 'object' &&
      typeof e.id === 'string' &&
      typeof e.name === 'string' &&
      typeof e.text === 'string'
    );
  } catch { return []; }
}

function persistTranscript(id, name, text) {
  try {
    const idx = savedTranscripts.findIndex(t => t.id === id);
    if (idx >= 0) {
      savedTranscripts[idx] = { id, name, text };
    } else {
      savedTranscripts.push({ id, name, text });
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(savedTranscripts));
  } catch {
    // localStorage unavailable (private mode quota, etc.) — silently skip
  }
}

// ── Future: summarize with LLM API ───────────────────────────────────────────
//
// To enable:
// 1. Uncomment and fill in an API key (or add a UI input that stores it in localStorage).
// 2. Unhide the #btn-summarize element in index.html.
//
// import Anthropic from 'https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk/+esm';
//
// async function summarize(text) {
//   const key = localStorage.getItem('api-key') ?? '';
//   const client = new Anthropic({ apiKey: key, dangerouslyAllowBrowser: true });
//   const msg = await client.messages.create({
//     model: 'claude-sonnet-4-5',
//     max_tokens: 1024,
//     messages: [{ role: 'user', content: `Summarize this voice message:\n\n${text}` }],
//   });
//   return msg.content[0].text;
// }
