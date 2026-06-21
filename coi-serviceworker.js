/* coi-serviceworker v0.1.7 - MIT License - https://github.com/gzuidhof/coi-serviceworker */
/* Injects Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers so that
   SharedArrayBuffer (required by Transformers.js WASM) is available on static hosts. */

(() => {
  const RELOAD_KEY = "coi-reload";

  if (typeof window === "undefined") {
    // ── Service worker context ──────────────────────────────────────────────
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));

    self.addEventListener("fetch", e => {
      const req = e.request;
      if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;

      e.respondWith(
        fetch(req)
          .then(res => {
            if (res.status === 0) return res;

            const headers = new Headers(res.headers);
            headers.set("Cross-Origin-Opener-Policy", "same-origin");
            headers.set("Cross-Origin-Embedder-Policy", "require-corp");
            headers.set("Cross-Origin-Resource-Policy", "cross-origin");

            return new Response(res.body, {
              status: res.status,
              statusText: res.statusText,
              headers,
            });
          })
          .catch(() => Response.error())
      );
    });
  } else {
    // ── Page context: register the service worker then reload ───────────────
    if (window.crossOriginIsolated) return;

    if (!window.isSecureContext) {
      console.warn("coi-serviceworker: requires a secure context (HTTPS or localhost).");
      return;
    }

    if (!("serviceWorker" in navigator)) {
      console.warn("coi-serviceworker: service workers not supported in this browser.");
      return;
    }

    const alreadyReloaded = sessionStorage.getItem(RELOAD_KEY);

    navigator.serviceWorker
      .register(document.currentScript.src)
      .then(reg => {
        if (alreadyReloaded) {
          sessionStorage.removeItem(RELOAD_KEY);
          return;
        }
        sessionStorage.setItem(RELOAD_KEY, "1");

        const sw = reg.installing ?? reg.waiting ?? reg.active;
        if (!sw) return;

        if (sw.state === "activated") {
          location.reload();
        } else {
          sw.addEventListener("statechange", () => {
            if (sw.state === "activated") location.reload();
          });
        }
      })
      .catch(err => console.error("coi-serviceworker: registration failed:", err));
  }
})();
