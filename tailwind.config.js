import typography from '@tailwindcss/typography';
import containerQueries from '@tailwindcss/container-queries';

/** @type {import('tailwindcss').Config} */
export default {
	darkMode: 'class',
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {
			colors: {
				gray: {
					50: 'var(--color-gray-50, #fafafa)',
					100: 'var(--color-gray-100, #f4f4f4)',
					200: 'var(--color-gray-200, #e5e5e5)',
					300: 'var(--color-gray-300, #d4d4d4)',
					400: 'var(--color-gray-400, #a3a3a3)',
					500: 'var(--color-gray-500, #737373)',
					600: 'var(--color-gray-600, #525252)',
					700: 'var(--color-gray-700, #404040)',
					800: 'var(--color-gray-800, #2a2a2a)',
					850: 'var(--color-gray-850, #1f1f1f)',
					900: 'var(--color-gray-900, #171717)',
					950: 'var(--color-gray-950, #0a0a0a)'
				},
				primary: {
					50: '#f0f9ff',
					100: '#e0f2fe',
					200: '#bae6fd',
					300: '#7dd3fc',
					400: '#38bdf8',
					500: '#0ea5e9',
					600: '#0284c7',
					700: '#0369a1',
					800: '#075985',
					900: '#0c4a6e'
				},
				accent: {
					50: '#fdf4ff',
					100: '#fae8ff',
					200: '#f5d0fe',
					300: '#f0abfc',
					400: '#e879f9',
					500: '#d946ef',
					600: '#c026d3',
					700: '#a21caf',
					800: '#86198f',
					900: '#701a75'
				}
			},
			typography: {
				DEFAULT: {
					css: {
						pre: false,
						code: false,
						'pre code': false,
						'code::before': false,
						'code::after': false
					}
				}
			},
			padding: {
				'safe-bottom': 'env(safe-area-inset-bottom)'
			},
			transitionProperty: {
				width: 'width'
			}
		}
	},
	plugins: [typography, containerQueries]
};
