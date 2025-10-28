import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
	plugins: [
		sveltekit(),
		viteStaticCopy({
			targets: [
				{
					src: 'node_modules/onnxruntime-web/dist/*.jsep.*',

					dest: 'wasm'
				}
			]
		})
	],
	define: {
		APP_VERSION: JSON.stringify(process.env.npm_package_version),
		APP_BUILD_HASH: JSON.stringify(process.env.APP_BUILD_HASH || 'dev-build')
	},
	build: {
		sourcemap: process.env.ENV === 'prod' ? false : true,
		minify: 'esbuild',
		chunkSizeWarningLimit: 1000,
		target: 'esnext',
		cssCodeSplit: true,
		rollupOptions: {
			external: ['y-protocols/awareness'],
			output: {
				manualChunks: (id) => {
					// Vendor chunks for better caching
					if (id.includes('node_modules')) {
						if (id.includes('svelte')) return 'svelte-vendor';
						if (id.includes('socket.io')) return 'socket-vendor';
						if (id.includes('openai') || id.includes('anthropic')) return 'ai-vendor';
						if (id.includes('marked') || id.includes('highlight')) return 'markdown-vendor';
						return 'vendor';
					}
				},
				chunkFileNames: 'chunks/[name]-[hash].js',
				assetFileNames: 'assets/[name]-[hash][extname]'
			}
		}
	},
	worker: {
		format: 'es'
	},
	server: {
		fs: {
			strict: false
		}
	},
	esbuild: {
		pure: process.env.ENV === 'dev' ? [] : ['console.log', 'console.debug', 'console.error'],
		legalComments: 'none',
		treeShaking: true
	},
	optimizeDeps: {
		include: ['svelte', 'socket.io-client'],
		exclude: ['onnxruntime-web']
	}
});
