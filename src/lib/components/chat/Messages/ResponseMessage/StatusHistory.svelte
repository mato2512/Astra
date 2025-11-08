<script>
	import { getContext } from 'svelte';
	const i18n = getContext('i18n');

	import StatusItem from './StatusHistory/StatusItem.svelte';
	export let statusHistory = [];
	export let expand = false;

	let showHistory = true;

	$: if (expand) {
		showHistory = true;
	} else {
		showHistory = false;
	}

	let history = [];
	let status = null;

	$: if (history && history.length > 0) {
		status = history.at(-1);
	}

	$: if (JSON.stringify(statusHistory) !== JSON.stringify(history)) {
		history = statusHistory;
	}
</script>

{#if history && history.length > 0 && history.filter(h => !h.hidden && h.description !== 'No search query generated' && h.description !== 'Generating search query' && h.action !== 'sources_retrieved' && h.action !== 'web_search' && h.description !== 'Searching the web').length > 0}
	{#if status?.hidden !== true}
		<div class="text-sm flex flex-col w-full">
			{#if showHistory}
				<div class="flex flex-row">
					{#if history.length > 1}
						<div class="w-full">
							{#each history as status, idx}
								{#if idx !== history.length - 1}
									<div class="flex items-stretch gap-2 mb-1">
										<StatusItem {status} done={true} />
									</div>
								{/if}
							{/each}
						</div>
					{/if}
				</div>
			{/if}

			<button
				class="w-full"
				on:click={() => {
					showHistory = !showHistory;
				}}
			>
				<div class="flex items-start gap-2">
					<StatusItem {status} />
				</div>
			</button>
		</div>
	{/if}
{/if}
