<script lang="ts">
	import DOMPurify from 'dompurify';
	import { marked } from 'marked';

	import { getAdminDetails } from '$lib/apis/auths';
	import { onMount, tick, getContext } from 'svelte';
	import { config } from '$lib/stores';

	const i18n = getContext('i18n');

	let adminDetails = null;

	onMount(async () => {
		adminDetails = await getAdminDetails(localStorage.token).catch((err) => {
			console.error(err);
			return null;
		});
	});
</script>

<div class="fixed w-full h-full flex z-999">
	<div
		class="absolute w-full h-full backdrop-blur-lg bg-white/10 dark:bg-gray-900/50 flex justify-center items-center p-4"
	>
		<div class="w-full max-w-md">
			<div class="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8">
				<div
					class="text-center dark:text-white text-2xl font-medium"
					style="white-space: pre-wrap;"
				>
					{#if ($config?.ui?.pending_user_overlay_title ?? '').trim() !== ''}
						{$config.ui.pending_user_overlay_title}
					{:else}
						{$i18n.t('Account Activation Pending')}<br />
						{$i18n.t('Contact Admin for Astra Access')}
					{/if}
				</div>

				<div
					class="mt-4 text-center text-sm dark:text-gray-300 text-gray-600"
					style="white-space: pre-wrap;"
				>
					{#if ($config?.ui?.pending_user_overlay_content ?? '').trim() !== ''}
						{@html marked.parse(
							DOMPurify.sanitize(
								($config?.ui?.pending_user_overlay_content ?? '').replace(/\n/g, '<br>')
							)
						)}
					{:else}
						{$i18n.t('Your account status is currently pending activation.')}{'\n'}{$i18n.t(
							'To access the Astra, please reach out to the administrator.'
						)}
					{/if}
				</div>

				{#if adminDetails}
					<div class="mt-6 text-sm font-medium text-center dark:text-gray-200 text-gray-700">
						<div>{$i18n.t('Admin')}: {adminDetails.name} ({adminDetails.email})</div>
					</div>
				{/if}

				<div class="mt-6 flex flex-col items-center space-y-3">
					<button
						class="w-full px-5 py-2.5 rounded-full bg-gray-900 dark:bg-white hover:bg-gray-800 dark:hover:bg-gray-100 text-white dark:text-gray-900 transition font-medium text-sm shadow-lg"
						on:click={async () => {
							location.href = '/';
						}}
					>
						{$i18n.t('Check Again')}
					</button>

					<button
						class="text-xs text-center text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 underline transition"
						on:click={async () => {
							localStorage.removeItem('token');
							location.href = '/auth';
						}}>{$i18n.t('Sign Out')}</button
					>
				</div>
			</div>
		</div>
	</div>
</div>
