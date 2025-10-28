<script lang="ts">
	import { marked } from 'marked';

	import { getContext, tick } from 'svelte';
	import dayjs from '$lib/dayjs';

	import { mobile, settings, user } from '$lib/stores';
	import { WEBUI_BASE_URL } from '$lib/constants';

	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { copyToClipboard, sanitizeResponseContent } from '$lib/utils';
	import ArrowUpTray from '$lib/components/icons/ArrowUpTray.svelte';
	import Check from '$lib/components/icons/Check.svelte';
	import ModelItemMenu from './ModelItemMenu.svelte';
	import EllipsisHorizontal from '$lib/components/icons/EllipsisHorizontal.svelte';
	import { toast } from 'svelte-sonner';
	import Tag from '$lib/components/icons/Tag.svelte';
	import Label from '$lib/components/icons/Label.svelte';

	const i18n = getContext('i18n');

	export let selectedModelIdx: number = -1;
	export let item: any = {};
	export let index: number = -1;
	export let value: string = '';

	export let unloadModelHandler: (modelValue: string) => void = () => {};
	export let pinModelHandler: (modelId: string) => void = () => {};

	export let onClick: () => void = () => {};

	const copyLinkHandler = async (model) => {
		const baseUrl = window.location.origin;
		const res = await copyToClipboard(`${baseUrl}/?model=${encodeURIComponent(model.id)}`);

		if (res) {
			toast.success($i18n.t('Copied link to clipboard'));
		} else {
			toast.error($i18n.t('Failed to copy link'));
		}
	};

	let showMenu = false;
</script>

<button
	aria-roledescription="model-item"
	aria-label={item.label}
	class="flex group/item w-full text-left font-medium select-none items-center rounded-lg py-3 px-3 mb-1 text-sm text-gray-700 dark:text-gray-100 outline-hidden transition-all duration-75 hover:bg-gray-100 dark:hover:bg-gray-850 cursor-pointer border border-transparent hover:border-gray-200 dark:hover:border-gray-700 {index ===
	selectedModelIdx
		? 'bg-gray-100 dark:bg-gray-850 border-gray-200 dark:border-gray-700'
		: ''}"
	data-arrow-selected={index === selectedModelIdx}
	data-value={item.value}
	on:click={() => {
		onClick();
	}}
>
	<div class="flex items-start gap-3 flex-1 min-w-0">
		<!-- Model Icon -->
		<div class="flex-shrink-0 mt-0.5">
			<img
				src={item.model?.info?.meta?.profile_image_url ??
					`${WEBUI_BASE_URL}/static/favicon.png`}
				alt="Model"
				class="rounded-full size-6"
			/>
		</div>

		<!-- Model Info -->
		<div class="flex flex-col gap-1 flex-1 min-w-0">
			<div class="font-semibold text-sm text-gray-900 dark:text-gray-100 line-clamp-1">
				{item.label}
			</div>
			
			{#if item.model?.info?.meta?.description}
				<div class="text-xs text-gray-500 dark:text-gray-400 line-clamp-2">
					{item.model?.info?.meta?.description}
				</div>
			{/if}
		</div>

		<!-- Check Mark -->
		{#if index === selectedModelIdx}
			<div class="ml-auto flex-shrink-0">
				<Check className="size-4 text-gray-900 dark:text-gray-100" />
			</div>
		{/if}
	</div>
</button>
