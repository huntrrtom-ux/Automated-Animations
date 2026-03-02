# Implementation Plan — Hunter Motions UI & Data Tweaks

## Phase 1: Backend Changes (`app.py`)

### 1.1 Logo Resolution — Resize to 800×800 on Upload
- In `save_channel()` (line ~203) and `update_channel()` (line ~248): after writing logo bytes to `logo.png`, open with PIL, resize to 800×800 using `Image.LANCZOS`, save back as PNG.
- PIL is already imported (line 18: `from PIL import Image`).
- Helper function `_resize_logo(path, size=800)` to avoid duplication.

### 1.2 One-Time Startup Migration (Formats, Copies, Renames)
- New function `migrate_v53_channel_updates()` gated by a flag file `/data/channels/_migration_v53.done`.
- Runs at startup after `migrate_presets_to_channels()`.
- Steps:
  1. **Switch all channels to flash format**: Iterate all channels, set `config['format'] = copy.deepcopy(BASE_FORMATS['flash'])`.
  2. **Copy + rename Pet Psychology**: Find channel named "Pet Psychology - Flash", create 2 copies (duplicate folder contents), rename original to "Psychology of Cats", copies to "Whisker Theory" and "Psychology of Dogs". Keep same tags, images, all other config.
  3. **Rename channels**:
     - "Pastel - Pregnancy Advice - Flash" → "Mama Knowledge"
     - "Watercolour - Pregnancy Advice - Flash" → "Pregnancy with Grace"
     - "Fatherhood Advice - Flash" → "Coach Dan"
     - "Sally - Housing Market - Flash" → "Sally Saves"
     - "Betty - Housing Market - Flash" → "Financial Betty"
     - "Howard - Housing Market - Flash" → "Housing Howard"
  4. Write flag file to prevent re-running.

### 1.3 Alphabetical Tag Ordering
- Modify `get_all_channels()` (line ~148): after collecting all channels, sort by first tag alphabetically. Channels with no tags go to end.
- This replaces the current registry-based ordering.

### 1.4 Add `project_title` to Generation History
- In `log_generation()` calls (lines ~2209 and ~2234): add `'project_title': project_title` field.
- In `recent_generations()` API (line ~2847): add `'project_title'` to `safe_fields`.

### 1.5 Active Sessions Tracking + API
- Add module-level dict: `active_sessions = {}` — maps `session_id` → `{ 'channel_name': ..., 'started_at': ... }`.
- At start of `process_voiceover()`: add entry to `active_sessions`.
- At completion/error: remove entry from `active_sessions`.
- New endpoint `GET /api/active-generations` returns `{ 'count': len(active_sessions) }`.

---

## Phase 2: Admin Page (`templates/admin.html`)

### 2.1 Channel Card Size 1.5×
- `.admin-channel-grid`: change `minmax(200px, 1fr)` → `minmax(290px, 1fr)` (fits 3 in 960px).
- `.admin-ch-logo-wrap`, `.admin-ch-logo`, `.admin-ch-logo-ph`: change 56px → 84px.
- `.admin-ch-name`: increase font-size from 0.88rem → 1rem.

### 2.2 Logo Upload Hint
- Next to the logo file input in the edit panel (line ~131): add text "Recommended: 800×800px".

---

## Phase 3: Main Page (`templates/index.html`)

### 3.1 Remove Beta Stamp & Version Number
- Delete `<div class="version-badge">Beta v52</div>` (line 14).
- In footer (line ~241): change `<button class="link-btn changelog-link" id="open-changelog">What's New · v52</button>` to just `What's New`.
- Remove `.version-badge` CSS from `style.css`.

### 3.2 Ongoing Generations Pill
- Add a small pill element in the header area: `<div id="active-gen-pill" class="active-gen-pill">...</div>`.
- Shows a green circle + "Ongoing Generations: N" when count ≥ 1, red circle + "Ongoing Generations: 0" when count = 0.
- Frontend polls `/api/active-generations` every ~5 seconds (or listens via SocketIO).

---

## Phase 4: Frontend JS (`static/js/app.js`)

### 4.1 Remove Subject Tick & Format from Channel Tiles
- In `renderChannelGrid()` (line ~260): remove the `.tile-meta` div that shows format icon + base name and "✓ Subject" / "✗ No Subject".

### 4.2 Add Format + Tags to Audio Page Info
- In `setupUploadSummary()` (line ~306):
  - Keep Channel and Format rows.
  - For Format: detect if "tailored" by comparing channel's format against BASE_FORMATS[base]. If any field differs, prefix with "Tailored" (e.g., "Tailored Flash").
  - Add a Tags row showing the channel's tags as colored pills.
  - Add Subject row.

### 4.3 Recent Generations — Title + Channel Pill
- In `loadRecentGenerations()` (line ~173):
  - Show first 10 chars of `project_title` + "…" at the start.
  - Move channel name into a pill-shaped element.
  - Date stays after the pill.
- Add CSS for `.recent-channel-pill` styling.

### 4.4 Active Generations Pill Logic
- On page load and every ~5s: fetch `/api/active-generations`, update the pill count and indicator color.

---

## Phase 5: CSS (`static/css/style.css`)

- Remove `.version-badge` styles.
- Add `.active-gen-pill` styles (small pill, flexbox, green/red dot indicator).
- Add `.recent-channel-pill` styles (rounded pill with background).
- Adjust `.tile-meta` removal if needed.

---

## Phase 6: Deferred — Concurrent Generation Bugs (Item 16)
- Wait for user to provide Railway logs and previous diagnosis.
- Likely involves race conditions in shared state during concurrent `process_voiceover()` calls.
- Will investigate after all other changes are deployed.

---

## Implementation Order
1. Backend: migration function, logo resize, tag ordering, project_title in history, active sessions API
2. Admin HTML: card sizes, logo hint
3. Main HTML: remove beta, add ongoing pill
4. JS: channel tiles cleanup, audio page info, recent gen redesign, active gen polling
5. CSS: supporting styles
6. Test, commit, push
