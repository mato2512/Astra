# Astra Customization Implementation Plan

## Completed ✅

1. ✅ **About Section** - Version check removed
2. ✅ **Translation Link** - "Help us translate" link removed  
3. ✅ **Advanced Parameters** - Already wrapped in admin check
4. ✅ **README.md** - All OpenWebUI references replaced with Astra
5. ✅ **Branding** - All logos, names updated to Astra/Prasad Navale

## Pending Changes 🚧

### 1. Hide Settings Tabs for Regular Users
**File:** `src/lib/components/chat/SettingsModal.svelte`
**Action:** Wrap these tabs in `{#if $user?.role === 'admin'}`:
- Personalization tab
- Interface tab
- Audio tab  
- Data Controls tab

### 2. Hide Active Users Display  
**File:** Search for "Active Users" or user list display
**Action:** Hide from regular users

### 3. Hide Keyboard Shortcuts
**File:** Search for keyboard shortcut button/modal
**Action:** Hide from regular users

### 4. Hide Top Corner Controls
**File:** Layout/navigation files
**Action:** Hide certain controls from regular users

### 5. Restrict Model Addition
**File:** Model management components
**Action:** Users can only SELECT models, not ADD new ones

### 6. Custom Update Notifications
**Current:** Checks GitHub releases from open-webui/open-webui  
**Needed:** Check YOUR custom endpoint for updates
**Files to modify:**
- `backend/open_webui/main.py` or version check API
- `src/lib/apis/index.ts` - getVersionUpdates function
- Point to YOUR server/API for version checks

## Implementation Notes

### For Admin-Only Features:
Use this pattern:
```svelte
{#if $user?.role === 'admin'}
  <!-- Admin-only content -->
{/if}
```

### For Custom Update Endpoint:
Change version check URL from:
```
https://api.github.com/repos/open-webui/open-webui/releases/latest
```
To YOUR custom endpoint:
```
https://astra.ngts.tech/api/version/latest
```

### Chat Landing Page:
Default chat interface remains accessible to all users - NO changes needed.

## Next Steps

1. Search and modify SettingsModal.svelte to hide tabs
2. Find and hide user list/active users component
3. Find and hide keyboard shortcuts button
4. Modify model management to disable "Add Model" for users
5. Create custom version check API endpoint
6. Update frontend to call YOUR version API

## Your Control

You can control update notifications by:
1. Creating an API endpoint at `https://astra.ngts.tech/api/updates`
2. Return JSON: `{"version": "1.0.5", "message": "New version available!", "url": "..."}`
3. Update frontend to poll YOUR endpoint instead of GitHub

This way, YOU control when users see update notifications, not OpenWebUI.
