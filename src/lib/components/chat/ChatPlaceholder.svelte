<script lang="ts">
	import { WEBUI_BASE_URL } from '$lib/constants';
	import { marked } from 'marked';

	import { config, user, models as _models, temporaryChatEnabled, mobile } from '$lib/stores';
	import { onMount, getContext } from 'svelte';

	import { blur, fade } from 'svelte/transition';

	import Suggestions from './Suggestions.svelte';
	import { sanitizeResponseContent } from '$lib/utils';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import EyeSlash from '$lib/components/icons/EyeSlash.svelte';

	const i18n = getContext('i18n');

	export let modelIds = [];
	export let models = [];
	export let atSelectedModel;

	export let onSelect = (e) => {};

	let mounted = false;
	let selectedModelIdx = 0;

	// ChatGPT-like random greeting titles
	const greetingTitles = [
		'What are you working on?',
		"What's on your mind today?",
		'Hey, {{name}}. Ready to dive in?',
		'Where should we begin?',
		'How can I help, {{name}}?',
		'Ready when you are.',
		'Good to see you, {{name}}.',
		'What would you like to create?',
		'Let\'s get started.',
		'What can I help with?'
	];

	let randomGreeting = '';

	$: if (modelIds.length > 0) {
		selectedModelIdx = models.length - 1;
	}

	$: models = modelIds.map((id) => $_models.find((m) => m.id === id));

	onMount(() => {
		mounted = true;
		// Select a random greeting on mount
		const randomIndex = Math.floor(Math.random() * greetingTitles.length);
		const greeting = greetingTitles[randomIndex];
		// Replace {{name}} with actual user name
		randomGreeting = greeting.replace('{{name}}', $user?.name || 'there');
	});
</script>

{#key mounted}
	<div class="{$mobile ? 'h-full flex flex-col' : 'm-auto'} w-full max-w-3xl px-4 md:px-6 lg:px-8">
		<!-- Spacer to push content down on mobile -->
		{#if $mobile}
			<div class="flex-1"></div>
		{/if}
		
		<div class="flex flex-col items-center justify-center {$mobile ? '' : 'min-h-[40vh] md:min-h-[50vh]'}">
			<!-- Model Icons -->
			<div class="flex justify-center mb-6" in:fade={{ duration: 200 }}>
				<div class="flex {$mobile ? '' : '-space-x-3'}">
					{#each models as model, modelIdx}
						<button
							on:click={() => {
								selectedModelIdx = modelIdx;
							}}
							class="transition-transform hover:scale-110 hover:z-10"
						>
							<Tooltip
								content={marked.parse(
									sanitizeResponseContent(
										models[selectedModelIdx]?.info?.meta?.description ?? ''
									).replaceAll('\n', '<br>')
								)}
								placement="top"
							>
								<img
									crossorigin="anonymous"
									src={model?.info?.meta?.profile_image_url ??
										($i18n.language === 'dg-DG'
											? `${WEBUI_BASE_URL}/doge.png`
											: `${WEBUI_BASE_URL}/static/favicon.png`)}
									class="{$mobile ? 'size-16' : 'size-12 md:size-14'} rounded-full border-2 border-white dark:border-gray-900 shadow-lg"
									alt="logo"
									draggable="false"
								/>
							</Tooltip>
						</button>
					{/each}
				</div>
			</div>

			<!-- Temporary Chat Indicator -->
			{#if $temporaryChatEnabled}
				<Tooltip
					content={$i18n.t("This chat won't appear in history and your messages will not be saved.")}
					className="mb-4"
					placement="top"
				>
					<div class="flex items-center gap-2 text-gray-500 text-sm md:text-base">
						<EyeSlash strokeWidth="2.5" className="size-4 md:size-5" />
						<span>{$i18n.t('Temporary Chat')}</span>
					</div>
				</Tooltip>
			{/if}

			<!-- Greeting Title -->
			<div class="text-center mb-8 md:mb-12">
				<div
					class="text-2xl md:text-3xl lg:text-4xl text-gray-800 dark:text-gray-100 font-semibold mb-3"
					in:fade={{ duration: 200 }}
				>
					{randomGreeting}
				</div>

				<!-- Model Description (Hidden on Mobile for Cleaner Look) -->
				{#if models[selectedModelIdx]?.info?.meta?.description ?? null}
					<div in:fade={{ duration: 200, delay: 100 }}>
						<div
							class="hidden md:block text-sm lg:text-base text-gray-500 dark:text-gray-400 max-w-2xl mx-auto"
						>
							{@html marked.parse(
								sanitizeResponseContent(
									models[selectedModelIdx]?.info?.meta?.description
								).replaceAll('\n', '<br>')
							)}
						</div>
						{#if models[selectedModelIdx]?.info?.meta?.user}
							<div class="hidden md:block mt-2 text-xs text-gray-400 dark:text-gray-500">
								By
								{#if models[selectedModelIdx]?.info?.meta?.user.community}
									<a
										href="https://openwebui.com/m/{models[selectedModelIdx]?.info?.meta?.user
											.username}"
										class="hover:underline"
										>{models[selectedModelIdx]?.info?.meta?.user.name
											? models[selectedModelIdx]?.info?.meta?.user.name
											: `@${models[selectedModelIdx]?.info?.meta?.user.username}`}</a
									>
								{:else}
									{models[selectedModelIdx]?.info?.meta?.user.name}
								{/if}
							</div>
						{/if}
					</div>
				{/if}
			</div>

			<!-- Suggestions -->
			<div class="w-full max-w-2xl" in:fade={{ duration: 200, delay: 200 }}>
				<Suggestions
					className="grid grid-cols-1 md:grid-cols-2 gap-2"
					suggestionPrompts={atSelectedModel?.info?.meta?.suggestion_prompts ??
						models[selectedModelIdx]?.info?.meta?.suggestion_prompts ??
						$config?.default_prompt_suggestions ??
						[]}
					{onSelect}
				/>
			</div>
		</div>
	</div>
{/key}
