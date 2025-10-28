// Service Worker for Astra - Optimized for fast loading and low network
const CACHE_VERSION = 'astra-v1';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const IMAGE_CACHE = `${CACHE_VERSION}-images`;

// Cache limits
const CACHE_LIMITS = {
	images: 50,
	dynamic: 100
};

// Resources to cache immediately on install
const STATIC_ASSETS = [
	'/',
	'/manifest.json',
	'/static/favicon.png',
	'/static/custom.css'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
	console.log('[SW] Installing service worker...');
	event.waitUntil(
		caches.open(STATIC_CACHE).then((cache) => {
			console.log('[SW] Caching static assets');
			return cache.addAll(STATIC_ASSETS);
		})
	);
	self.skipWaiting();
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
	console.log('[SW] Activating service worker...');
	event.waitUntil(
		caches.keys().then((cacheNames) => {
			return Promise.all(
				cacheNames
					.filter((cacheName) => {
						return cacheName.startsWith('astra-') && cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE && cacheName !== IMAGE_CACHE;
					})
					.map((cacheName) => {
						console.log('[SW] Deleting old cache:', cacheName);
						return caches.delete(cacheName);
					})
			);
		})
	);
	self.clients.claim();
});

// Fetch event - serve from cache with network fallback
self.addEventListener('fetch', (event) => {
	const { request } = event;
	const url = new URL(request.url);

	// Skip non-GET requests
	if (request.method !== 'GET') return;

	// Skip API calls and WebSocket
	if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) {
		return;
	}

	// Handle images with cache-first strategy
	if (request.destination === 'image') {
		event.respondWith(
			caches.match(request).then((response) => {
				if (response) return response;

				return fetch(request).then((response) => {
					// Clone response before caching
					const responseToCache = response.clone();
					
					caches.open(IMAGE_CACHE).then((cache) => {
						cache.put(request, responseToCache);
						limitCacheSize(IMAGE_CACHE, CACHE_LIMITS.images);
					});

					return response;
				});
			})
		);
		return;
	}

	// Handle static assets (JS, CSS, fonts)
	if (
		request.destination === 'script' ||
		request.destination === 'style' ||
		request.destination === 'font' ||
		url.pathname.includes('/_app/')
	) {
		event.respondWith(
			caches.match(request).then((response) => {
				return response || fetch(request).then((response) => {
					const responseToCache = response.clone();
					caches.open(STATIC_CACHE).then((cache) => {
						cache.put(request, responseToCache);
					});
					return response;
				});
			})
		);
		return;
	}

	// Handle other requests with network-first, fallback to cache
	event.respondWith(
		fetch(request)
			.then((response) => {
				// Don't cache non-successful responses
				if (!response || response.status !== 200 || response.type === 'error') {
					return response;
				}

				const responseToCache = response.clone();
				caches.open(DYNAMIC_CACHE).then((cache) => {
					cache.put(request, responseToCache);
					limitCacheSize(DYNAMIC_CACHE, CACHE_LIMITS.dynamic);
				});

				return response;
			})
			.catch(() => {
				// Network failed, try cache
				return caches.match(request);
			})
	);
});

// Helper function to limit cache size
function limitCacheSize(cacheName, maxItems) {
	caches.open(cacheName).then((cache) => {
		cache.keys().then((keys) => {
			if (keys.length > maxItems) {
				// Delete oldest entries
				cache.delete(keys[0]).then(() => {
					limitCacheSize(cacheName, maxItems);
				});
			}
		});
	});
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
	console.log('[SW] Background sync:', event.tag);
	if (event.tag === 'sync-chats') {
		event.waitUntil(syncChats());
	}
});

async function syncChats() {
	// Implement chat sync logic here
	console.log('[SW] Syncing chats...');
}
