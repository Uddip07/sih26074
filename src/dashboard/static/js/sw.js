/* Offline support (F33): cache the app shell and geometry; serve the last forecast/bulletin when offline. */
const SHELL = "agromet-shell-v15";
const DATA = "agromet-data-v15";
const SHELL_URLS = ["/", "/static/css/style.css?v=15", "/static/js/dashboard.js?v=15", "/manifest.webmanifest",
  "/static/img/icon.svg", "/api/geo/panchayats", "/api/geo/blocks"];

self.addEventListener("install", (e) => {
  // cache: "reload" bypasses the HTTP cache so a new worker never pre-caches stale files
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_URLS.map((u) => new Request(u, { cache: "reload" }))))
    .then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => ![SHELL, DATA].includes(k)).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  const isData = url.pathname.startsWith("/api/");
  // network-first for API data (fresh when online, last copy when offline); cache-first for the shell
  if (isData) {
    e.respondWith(fetch(e.request).then((r) => {
      if (r.ok && !url.pathname.startsWith("/api/nowcast")) { const cp = r.clone(); caches.open(DATA).then((c) => c.put(e.request, cp)); }
      return r;
    }).catch(() => caches.match(e.request).then((m) => m || new Response(JSON.stringify({ detail: "offline" }),
      { status: 503, headers: { "Content-Type": "application/json" } }))));
  } else {
    // stale-while-revalidate: instant from cache, refreshed in the background so updates ship
    e.respondWith(caches.open(SHELL).then((c) => c.match(e.request).then((m) => {
      const net = fetch(e.request).then((r) => { if (r.ok) c.put(e.request, r.clone()); return r; }).catch(() => m);
      return m || net;
    })));
  }
});
