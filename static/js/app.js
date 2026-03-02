// HUNTER MOTIONS — Wizard Frontend v52
document.addEventListener('DOMContentLoaded', () => {

    // ===================== UTILITIES =====================
    function showToast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icons = { success: '\u2713', error: '\u2715', info: 'i' };
        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || 'i'}</div>
            <div class="toast-msg">${message}</div>
            <button class="toast-close" onclick="this.parentElement.classList.add('removing');setTimeout(()=>this.parentElement.remove(),300)">\u2715</button>
        `;
        container.appendChild(toast);
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.add('removing');
                setTimeout(() => toast.remove(), 300);
            }
        }, duration);
    }

    function fmtTime(s) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; }
    function escHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
    function formatSize(bytes) {
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
    function formatDuration(seconds) {
        if (!seconds) return '—';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${String(s).padStart(2, '0')}`;
    }

    // ===================== STATE =====================
    const WIZARD_STEPS = ['home', 'channel', 'upload', 'generate'];
    let currentStep = 'home';
    let selectedChannelId = null;
    let selectedChannelData = null;
    let channelCache = [];
    let selectedFile = null;
    let currentSessionId = null;
    let isGenerating = false;
    let generationStartTime = null;

    // ===================== THEME =====================
    const themeToggle = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    themeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    });

    // ===================== WEBSOCKET =====================
    const socket = io();
    socket.on('connect', () => console.log('Connected'));
    socket.on('progress', (data) => {
        if (data.session_id !== currentSessionId) return;
        handleProgress(data);
    });

    // ===================== WIZARD NAVIGATION =====================
    const panels = {
        home: document.getElementById('step-home'),
        channel: document.getElementById('step-channel'),
        upload: document.getElementById('step-upload'),
        generate: document.getElementById('step-generate')
    };
    const wizardNav = document.getElementById('wizard-nav');
    const btnBack = document.getElementById('btn-back');
    const btnNext = document.getElementById('btn-next');

    function goToStep(step) {
        currentStep = step;
        // Hide all panels
        Object.values(panels).forEach(p => p.classList.remove('active'));
        panels[step].classList.add('active');

        // Update breadcrumb
        const stepIdx = WIZARD_STEPS.indexOf(step);
        document.querySelectorAll('.wiz-step').forEach((el, i) => {
            el.classList.toggle('active', i === stepIdx);
            el.classList.toggle('completed', i < stepIdx);
        });

        // Show/hide nav buttons
        if (step === 'home' || step === 'generate') {
            wizardNav.classList.add('hidden');
        } else {
            wizardNav.classList.remove('hidden');
            btnBack.style.visibility = 'visible';
            updateNextButton();
        }

        // Update channel context bar
        const ctxBar = document.getElementById('channel-context');
        if (selectedChannelData && step !== 'home') {
            ctxBar.classList.remove('hidden');
            document.getElementById('ctx-name').textContent = selectedChannelData.name;
            const formatBase = selectedChannelData.format ? selectedChannelData.format.base : 'pulse';
            document.getElementById('ctx-format').textContent = formatBase.charAt(0).toUpperCase() + formatBase.slice(1);
            const logoImg = document.getElementById('ctx-logo');
            if (selectedChannelData.has_logo) {
                logoImg.src = `/api/channels/${selectedChannelData.id}/logo.png`;
                logoImg.style.display = 'block';
            } else {
                logoImg.style.display = 'none';
            }
        } else {
            ctxBar.classList.add('hidden');
        }

        // Step-specific setup
        if (step === 'channel') loadChannels();
        if (step === 'upload') setupUploadSummary();
        if (step === 'generate') setupConfirmCard();
    }

    function updateNextButton() {
        if (currentStep === 'channel') {
            btnNext.disabled = !selectedChannelId;
        } else if (currentStep === 'upload') {
            btnNext.disabled = !selectedFile;
        } else {
            btnNext.disabled = true;
        }
    }

    btnBack.addEventListener('click', () => {
        const idx = WIZARD_STEPS.indexOf(currentStep);
        if (idx > 0) goToStep(WIZARD_STEPS[idx - 1]);
    });

    btnNext.addEventListener('click', () => {
        const idx = WIZARD_STEPS.indexOf(currentStep);
        if (idx < WIZARD_STEPS.length - 1) goToStep(WIZARD_STEPS[idx + 1]);
    });

    // ===================== STEP 1: HOME =====================
    document.getElementById('new-video-btn').addEventListener('click', () => {
        // Reset state for new video
        selectedChannelId = null;
        selectedChannelData = null;
        selectedFile = null;
        currentSessionId = null;
        isGenerating = false;
        generationStartTime = null;
        document.getElementById('file-input').value = '';
        document.getElementById('file-selected').classList.add('hidden');
        document.getElementById('project-title').value = '';
        document.title = 'Hunter Motions';
        goToStep('channel');
    });

    // Load recent generations
    async function loadRecentGenerations() {
        try {
            const resp = await fetch('/api/recent-generations');
            const items = await resp.json();
            const list = document.getElementById('recent-list');
            const empty = document.getElementById('recent-empty');
            if (!items || items.length === 0) {
                empty.style.display = 'block';
                return;
            }
            empty.style.display = 'none';
            list.innerHTML = '';
            items.forEach(item => {
                const el = document.createElement('div');
                el.className = 'recent-item';
                el.dataset.session = item.session_id;
                const date = item.timestamp ? new Date(item.timestamp).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : '\u2014';
                const channelName = item.channel_name || item.preset_name || 'Unknown';
                const statusClass = item.status === 'complete' ? 'complete' : (item.status === 'error' ? 'error' : 'pending');
                const title = item.project_title || '';
                const titleDisplay = title.length > 10 ? escHtml(title.substring(0, 10)) + '\u2026' : escHtml(title);

                // Time ago calculation
                let agoText = '';
                if (item.timestamp) {
                    const diffMs = Date.now() - new Date(item.timestamp).getTime();
                    const diffMins = Math.floor(diffMs / 60000);
                    const diffHrs = Math.floor(diffMins / 60);
                    const diffDays = Math.floor(diffHrs / 24);
                    if (diffMins < 1) agoText = 'just now';
                    else if (diffMins < 60) agoText = `${diffMins}min ago`;
                    else if (diffHrs < 24) agoText = `${diffHrs}hr${diffHrs > 1 ? 's' : ''} ago`;
                    else agoText = `${diffDays}d ago`;
                }

                el.innerHTML = `
                    <span class="recent-channel-name">${escHtml(channelName)}</span>
                    ${titleDisplay ? `<span class="recent-title-snippet">${titleDisplay}</span>` : ''}
                    <span class="recent-right">
                        ${agoText ? `<span class="recent-ago">${agoText}</span>` : ''}
                        <span class="recent-date">${date}</span>
                        <span class="recent-status ${statusClass}">&bull;</span>
                    </span>
                `;
                el.addEventListener('click', () => reopenSession(item.session_id));
                list.appendChild(el);
            });
        } catch (err) {
            console.error('Load recent generations error:', err);
        }
    }

    async function reopenSession(sessionId) {
        try {
            const resp = await fetch(`/api/session/${sessionId}/state`);
            if (!resp.ok) { showToast('Session not found', 'error'); return; }
            const state = await resp.json();
            currentSessionId = sessionId;
            // Go to generate step and show results
            goToStep('generate');
            showResult({
                video_url: state.video_url || `/download/${sessionId}/${state.output_filename}`,
                scenes: state.scenes || [],
                session_id: sessionId
            });
        } catch (err) {
            showToast('Failed to load session', 'error');
        }
    }

    // ===================== STEP 2: CHANNEL SELECT =====================
    async function loadChannels() {
        try {
            const resp = await fetch('/api/channels');
            channelCache = await resp.json();
            renderChannelGrid(channelCache);
        } catch (err) {
            console.error('Load channels error:', err);
            showToast('Failed to load channels', 'error');
        }
    }

    function renderChannelGrid(channels) {
        const grid = document.getElementById('channel-grid');
        grid.innerHTML = '';
        if (channels.length === 0) {
            grid.innerHTML = '<p class="empty-state">No channels configured. Add channels in Admin.</p>';
            return;
        }
        channels.forEach(ch => {
            const tile = document.createElement('div');
            tile.className = 'channel-tile';
            if (ch.id === selectedChannelId) tile.classList.add('selected');
            tile.dataset.id = ch.id;

            const hasLogo = ch.has_logo;

            let logoHtml;
            if (hasLogo) {
                logoHtml = `<img class="tile-logo" src="/api/channels/${ch.id}/logo.png" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                            <div class="tile-logo-placeholder" style="display:none">\uD83D\uDD27</div>`;
            } else {
                logoHtml = `<div class="tile-logo-placeholder">\uD83D\uDD27</div>
                            <div class="tile-warning-tape"></div>`;
            }

            let tileTagsHtml = '';
            if (ch.tags && ch.tags.length) {
                tileTagsHtml = `<div class="tile-tags">${ch.tags.map(t => {
                    const color = (ch.tag_colors || {})[t] || 'var(--accent)';
                    return `<span class="tile-tag" style="background:${color}20;color:${color};border-color:${color}40">${escHtml(t)}</span>`;
                }).join('')}</div>`;
            }

            tile.innerHTML = `
                <div class="tile-select-check">\u2713</div>
                <div class="tile-logo-wrap">${logoHtml}</div>
                <div class="tile-name">${escHtml(ch.name)}</div>
                ${tileTagsHtml}
            `;

            tile.addEventListener('click', () => {
                document.querySelectorAll('.channel-tile.selected').forEach(t => t.classList.remove('selected'));
                tile.classList.add('selected');
                selectedChannelId = ch.id;
                selectedChannelData = ch;
                updateNextButton();
            });

            grid.appendChild(tile);
        });
    }

    // Channel search filter
    document.getElementById('channel-search').addEventListener('input', (e) => {
        const q = e.target.value.toLowerCase().trim();
        if (!q) {
            renderChannelGrid(channelCache);
            return;
        }
        const filtered = channelCache.filter(ch => {
            return ch.name.toLowerCase().includes(q) ||
                   (ch.tags || []).some(t => t.toLowerCase().includes(q)) ||
                   (ch.format && ch.format.base && ch.format.base.toLowerCase().includes(q));
        });
        renderChannelGrid(filtered);
    });

    // ===================== STEP 3: UPLOAD =====================
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const fileSelected = document.getElementById('file-selected');
    const fileNameEl = document.getElementById('file-name');
    const fileSizeEl = document.getElementById('file-size');

    function setupUploadSummary() {
        const summary = document.getElementById('upload-summary');
        if (!selectedChannelData) {
            summary.innerHTML = '<p class="text-muted">No channel selected</p>';
            return;
        }
        const ch = selectedChannelData;
        const formatBase = ch.format ? ch.format.base : 'flash';
        const formatLabel = ch.format ? (ch.format.label || formatBase) : formatBase;
        const tailoredPrefix = ch.format_tailored ? 'Tailored ' : '';

        let tagsHtml = '';
        if (ch.tags && ch.tags.length) {
            tagsHtml = ch.tags.map(t => {
                const color = (ch.tag_colors || {})[t] || 'var(--accent)';
                return `<span class="summary-tag-pill" style="background:${color}20;color:${color};border:1px solid ${color}40">${escHtml(t)}</span>`;
            }).join(' ');
        }

        summary.innerHTML = `
            <div class="summary-row">
                <span class="summary-label">Channel</span>
                <span class="summary-value">${escHtml(ch.name)}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Format</span>
                <span class="summary-value">${escHtml(tailoredPrefix + formatLabel)}</span>
            </div>
            ${tagsHtml ? `<div class="summary-row">
                <span class="summary-label">Tags</span>
                <span class="summary-value">${tagsHtml}</span>
            </div>` : ''}
            <div class="summary-row">
                <span class="summary-label">Subject</span>
                <span class="summary-value"${ch.has_subject ? ' style="color:var(--success)"' : ''}>${ch.has_subject ? '\u2713 Yes' : '\u2717 No'}</span>
            </div>
        `;
    }

    browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => { if (e.target.files.length) selectFile(e.target.files[0]); });
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
    });

    function selectFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'webm', 'mp4'].includes(ext)) {
            showToast('Invalid file type \u2014 use MP3, WAV, OGG, FLAC, M4A, AAC, WebM, or MP4', 'error');
            return;
        }
        selectedFile = file;
        fileNameEl.textContent = file.name;
        fileSizeEl.textContent = formatSize(file.size);
        fileSelected.classList.remove('hidden');
        updateNextButton();
    }

    document.getElementById('file-remove').addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        fileSelected.classList.add('hidden');
        updateNextButton();
    });

    // ===================== STEP 4: CONFIRM & GENERATE =====================
    function setupConfirmCard() {
        const confirmCard = document.getElementById('confirm-card');
        const processingCard = document.getElementById('processing-card');
        const resultCard = document.getElementById('result-card');
        confirmCard.classList.remove('hidden');
        processingCard.classList.add('hidden');
        resultCard.classList.add('hidden');

        if (selectedChannelData) {
            document.getElementById('confirm-channel').textContent = selectedChannelData.name;
            const formatBase = selectedChannelData.format ? selectedChannelData.format.base : 'pulse';
            const formatIcons = { pulse: '\u26A1 Pulse', flash: '\uD83C\uDF93 Flash', deep: '\uD83D\uDCDA Deep' };
            document.getElementById('confirm-format').textContent = formatIcons[formatBase] || formatBase;
        }
        if (selectedFile) {
            document.getElementById('confirm-audio').textContent = `${selectedFile.name} (${formatSize(selectedFile.size)})`;
        }
        const titleVal = document.getElementById('project-title').value.trim() || 'Untitled';
        document.getElementById('confirm-title').textContent = titleVal.replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }

    document.getElementById('confirm-back').addEventListener('click', () => goToStep('upload'));

    document.getElementById('confirm-start').addEventListener('click', async () => {
        if (!selectedFile || !selectedChannelId) return;

        const confirmCard = document.getElementById('confirm-card');
        const processingCard = document.getElementById('processing-card');
        confirmCard.classList.add('hidden');
        processingCard.classList.remove('hidden');

        const projectTitle = document.getElementById('project-title').value.trim().toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_-]/g, '') || '';
        const formData = new FormData();
        formData.append('audio', selectedFile);
        formData.append('channel_id', selectedChannelId);
        formData.append('project_title', projectTitle);

        try {
            const resp = await fetch('/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            if (resp.ok) {
                currentSessionId = data.session_id;
                isGenerating = true;
                generationStartTime = Date.now();
                // Show meta
                const displayTitle = (document.getElementById('project-title').value.trim() || selectedFile.name)
                    .replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                document.getElementById('processing-meta').textContent = `${displayTitle} \u00B7 ${selectedChannelData ? selectedChannelData.name : ''}`;
                document.title = `(0%) ${displayTitle} \u2014 Hunter Motions`;
                // Reset progress UI
                document.querySelectorAll('.step').forEach(s => s.classList.remove('active', 'complete'));
                document.getElementById('progress-bar').style.width = '0%';
                document.getElementById('progress-text').textContent = '0%';
                document.getElementById('status-message').style.color = '';
            } else {
                showToast(data.error || 'Upload failed', 'error');
                confirmCard.classList.remove('hidden');
                processingCard.classList.add('hidden');
            }
        } catch {
            showToast('Upload failed \u2014 check your connection', 'error');
            confirmCard.classList.remove('hidden');
            processingCard.classList.add('hidden');
        }
    });

    // ===================== PROGRESS HANDLER =====================
    function handleProgress(data) {
        const { step, progress, message } = data;
        const statusMessage = document.getElementById('status-message');
        statusMessage.textContent = message;

        if (progress >= 0) {
            document.getElementById('progress-bar').style.width = `${progress}%`;
            document.getElementById('progress-text').textContent = `${progress}%`;

            // Update spinner arc
            const arc = document.querySelector('.spinner-arc');
            if (arc) {
                const circumference = 2 * Math.PI * 26; // r=26
                const offset = circumference - (circumference * progress / 100);
                arc.style.strokeDashoffset = offset;
            }

            // ETA
            const etaEl = document.getElementById('eta-value');
            if (etaEl && progress >= 35 && generationStartTime) {
                const elapsed = (Date.now() - generationStartTime) / 1000;
                const rate = progress / elapsed;
                const remaining = (100 - progress) / rate;
                if (remaining < 60) {
                    etaEl.textContent = `${Math.ceil(remaining)}s`;
                } else if (remaining < 3600) {
                    etaEl.textContent = `${Math.floor(remaining / 60)}m ${Math.floor(remaining % 60)}s`;
                } else {
                    etaEl.textContent = `${Math.floor(remaining / 3600)}h ${Math.floor((remaining % 3600) / 60)}m`;
                }
            } else if (etaEl) {
                etaEl.textContent = 'Calculating...';
            }

            // Tab title
            const metaText = document.getElementById('processing-meta').textContent;
            const projLabel = metaText ? metaText.split(' \u00B7 ')[0] : '';
            document.title = projLabel ? `(${progress}%) ${projLabel} \u2014 Hunter Motions` : `(${progress}%) Hunter Motions`;
        }

        // Step indicators
        const stepOrder = ['transcription', 'scene_detection', 'generation', 'compositing'];
        const idx = stepOrder.indexOf(step);
        stepOrder.forEach((s, i) => {
            const el = document.getElementById(`step-${s}`);
            if (!el) return;
            if (i < idx) { el.classList.remove('active'); el.classList.add('complete'); }
            else if (i === idx) {
                el.classList.add('active'); el.classList.remove('complete');
                if (progress >= 100 && step !== 'complete') { el.classList.remove('active'); el.classList.add('complete'); }
            } else { el.classList.remove('active', 'complete'); }
        });

        if (step === 'complete' && data.data) {
            isGenerating = false;
            document.title = 'Completed - Hunter Motions';
            stepOrder.forEach(s => {
                const el = document.getElementById(`step-${s}`);
                if (el) { el.classList.remove('active'); el.classList.add('complete'); }
            });
            fireConfetti();
            showToast('Video generated successfully!', 'success', 6000);
            setTimeout(() => showResult(data.data), 600);
        }

        if (step === 'error') {
            isGenerating = false;
            document.title = 'Hunter Motions';
            statusMessage.style.color = 'var(--error)';
        }
    }

    // ===================== RESULTS =====================
    function showResult(data) {
        document.getElementById('confirm-card').classList.add('hidden');
        document.getElementById('processing-card').classList.add('hidden');
        const resultCard = document.getElementById('result-card');
        resultCard.classList.remove('hidden');

        const downloadBtn = document.getElementById('download-btn');
        downloadBtn.href = data.video_url;
        currentSessionId = data.session_id || currentSessionId;

        const videoPreview = document.getElementById('video-preview');
        videoPreview.src = data.video_url;

        window._currentScenes = data.scenes || [];
        window._selectedScenes = new Set();

        const sceneTimeline = document.getElementById('scene-timeline');
        sceneTimeline.innerHTML = '';
        if (data.scenes && data.scenes.length) {
            data.scenes.forEach(scene => {
                const isVideo = scene.is_video;
                const item = document.createElement('div');
                item.className = 'scene-item scene-item-interactive';
                item.dataset.sceneNum = scene.scene_number;
                item.dataset.startTime = scene.start_time;
                item.innerHTML = `
                    <label class="scene-checkbox-col" onclick="event.stopPropagation()">
                        <input type="checkbox" class="scene-checkbox" data-scene="${scene.scene_number}" />
                    </label>
                    <div class="scene-thumb-col">
                        <img class="scene-thumb" src="/scene-image/${currentSessionId}/${scene.scene_number}"
                             alt="Scene ${scene.scene_number}" onerror="this.style.display='none'" />
                        <div class="scene-badge ${isVideo ? 'video' : 'image'}">${scene.scene_number}</div>
                    </div>
                    <div class="scene-info-col">
                        <div class="scene-time">${fmtTime(scene.start_time)} \u2014 ${fmtTime(scene.end_time)}</div>
                        <div class="scene-desc">${escHtml(scene.visual_description).substring(0, 120)}${scene.visual_description.length > 120 ? '...' : ''}</div>
                    </div>
                    <div class="scene-actions-col">
                        <span class="scene-type-tag ${isVideo ? 'video' : 'image'}">${isVideo ? 'Veo' : 'Imagen'}</span>
                        <button class="btn-regen" title="Regenerate this scene" data-scene="${scene.scene_number}">
                            <svg viewBox="0 0 20 20" fill="none" width="14" height="14"><path d="M3 10a7 7 0 0113.07-3.5M17 10a7 7 0 01-13.07 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M16 3v4h-4M4 17v-4h4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
                        </button>
                    </div>
                `;
                item.addEventListener('click', (e) => {
                    if (e.target.closest('.btn-regen') || e.target.closest('.scene-checkbox-col')) return;
                    videoPreview.currentTime = scene.start_time;
                    videoPreview.play();
                    document.querySelectorAll('.scene-item-interactive').forEach(s => s.classList.remove('scene-active'));
                    item.classList.add('scene-active');
                });
                sceneTimeline.appendChild(item);
            });

            videoPreview.addEventListener('timeupdate', () => {
                const t = videoPreview.currentTime;
                document.querySelectorAll('.scene-item-interactive').forEach(item => {
                    const sceneData = window._currentScenes.find(s => s.scene_number == item.dataset.sceneNum);
                    if (sceneData && t >= sceneData.start_time && t < sceneData.end_time) {
                        item.classList.add('scene-active');
                    } else {
                        item.classList.remove('scene-active');
                    }
                });
            });
        }
        updateBatchBar();
    }

    // ===================== BATCH REGENERATION =====================
    function updateBatchBar() {
        const checked = document.querySelectorAll('.scene-checkbox:checked');
        const bar = document.getElementById('batch-regen-bar');
        const count = document.getElementById('batch-count');
        window._selectedScenes = new Set([...checked].map(c => parseInt(c.dataset.scene)));
        if (checked.length > 0) {
            bar.classList.remove('hidden');
            count.textContent = `${checked.length} scene${checked.length > 1 ? 's' : ''} selected`;
        } else {
            bar.classList.add('hidden');
        }
    }

    document.addEventListener('change', (e) => {
        if (e.target.classList.contains('scene-checkbox')) {
            const item = e.target.closest('.scene-item-interactive');
            if (e.target.checked) item.classList.add('scene-selected');
            else item.classList.remove('scene-selected');
            updateBatchBar();
        }
    });

    document.getElementById('batch-clear').addEventListener('click', () => {
        document.querySelectorAll('.scene-checkbox:checked').forEach(cb => {
            cb.checked = false;
            cb.closest('.scene-item-interactive').classList.remove('scene-selected');
        });
        updateBatchBar();
    });

    document.getElementById('batch-regen-btn').addEventListener('click', async () => {
        const sceneNums = [...window._selectedScenes].sort((a, b) => a - b);
        if (sceneNums.length === 0) return;

        const btn = document.getElementById('batch-regen-btn');
        const btnText = btn.querySelector('.btn-text');
        btn.disabled = true;
        btnText.textContent = `Regenerating ${sceneNums.length} scenes...`;

        try {
            const resp = await fetch('/api/regenerate-batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, scene_numbers: sceneNums })
            });
            const result = await resp.json();
            if (result.success) {
                window._currentScenes = result.scenes;
                const videoPreview = document.getElementById('video-preview');
                const currentTime = videoPreview.currentTime;
                videoPreview.src = result.video_url + '?t=' + Date.now();
                videoPreview.currentTime = currentTime;
                document.getElementById('download-btn').href = result.video_url;
                const successful = result.results.filter(r => r.success);
                successful.forEach(r => {
                    const thumb = document.querySelector(`.scene-item-interactive[data-scene-num="${r.scene_number}"] .scene-thumb`);
                    if (thumb) thumb.src = `/scene-image/${currentSessionId}/${r.scene_number}?t=` + Date.now();
                });
                showBatchComplete(successful.length, sceneNums.length);
            } else {
                showToast('Error: ' + (result.error || 'Batch regeneration failed'), 'error');
            }
        } catch {
            showToast('Network error \u2014 please try again', 'error');
        }

        btn.disabled = false;
        btnText.textContent = 'Regenerate Selected';
        document.querySelectorAll('.scene-checkbox:checked').forEach(cb => {
            cb.checked = false;
            cb.closest('.scene-item-interactive').classList.remove('scene-selected');
        });
        updateBatchBar();
    });

    // ===================== SINGLE SCENE REGENERATE =====================
    document.addEventListener('click', (e) => {
        const regenBtn = e.target.closest('.btn-regen');
        if (!regenBtn) return;
        const sceneNum = parseInt(regenBtn.dataset.scene);
        const scene = window._currentScenes.find(s => s.scene_number === sceneNum);
        if (!scene) return;

        document.getElementById('regen-scene-num').textContent = sceneNum;
        document.getElementById('regen-time').textContent = `${fmtTime(scene.start_time)} \u2014 ${fmtTime(scene.end_time)}`;
        document.getElementById('regen-prompt').value = scene.visual_description;
        const img = document.getElementById('regen-current-img');
        img.src = `/scene-image/${currentSessionId}/${sceneNum}`;
        img.style.display = 'block';
        document.getElementById('regen-status').classList.add('hidden');
        document.getElementById('regen-submit').disabled = false;
        document.getElementById('regen-submit').querySelector('.btn-text').textContent = 'Regenerate';
        document.getElementById('regen-overlay').classList.remove('hidden');
    });

    document.getElementById('regen-close').addEventListener('click', () => document.getElementById('regen-overlay').classList.add('hidden'));
    document.getElementById('regen-cancel').addEventListener('click', () => document.getElementById('regen-overlay').classList.add('hidden'));

    document.getElementById('regen-submit').addEventListener('click', async () => {
        const sceneNum = parseInt(document.getElementById('regen-scene-num').textContent);
        const customPrompt = document.getElementById('regen-prompt').value.trim();
        const submitBtn = document.getElementById('regen-submit');
        const status = document.getElementById('regen-status');

        submitBtn.disabled = true;
        submitBtn.querySelector('.btn-text').textContent = 'Regenerating...';
        status.textContent = 'Generating new image...';
        status.classList.remove('hidden');

        try {
            const resp = await fetch('/api/regenerate-scene', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, scene_number: sceneNum, custom_prompt: customPrompt })
            });
            const result = await resp.json();
            if (result.success) {
                status.textContent = 'Scene regenerated! Reloading...';
                window._currentScenes = result.scenes;
                const videoPreview = document.getElementById('video-preview');
                const currentTime = videoPreview.currentTime;
                videoPreview.src = result.video_url + '?t=' + Date.now();
                videoPreview.currentTime = currentTime;
                document.getElementById('download-btn').href = result.video_url;
                const thumb = document.querySelector(`.scene-item-interactive[data-scene-num="${sceneNum}"] .scene-thumb`);
                if (thumb) thumb.src = result.image_url + '?t=' + Date.now();
                const descEl = document.querySelector(`.scene-item-interactive[data-scene-num="${sceneNum}"] .scene-desc`);
                if (descEl) descEl.textContent = customPrompt.substring(0, 120) + (customPrompt.length > 120 ? '...' : '');
                document.getElementById('regen-current-img').src = result.image_url + '?t=' + Date.now();
                showToast('Scene regenerated successfully!', 'success');
                setTimeout(() => document.getElementById('regen-overlay').classList.add('hidden'), 1500);
            } else {
                status.textContent = 'Error: ' + (result.error || 'Unknown error');
                submitBtn.disabled = false;
                submitBtn.querySelector('.btn-text').textContent = 'Retry';
            }
        } catch {
            status.textContent = 'Network error \u2014 please try again';
            submitBtn.disabled = false;
            submitBtn.querySelector('.btn-text').textContent = 'Retry';
        }
    });

    // ===================== NEW PROJECT =====================
    document.getElementById('new-btn').addEventListener('click', () => {
        selectedFile = null;
        currentSessionId = null;
        isGenerating = false;
        generationStartTime = null;
        selectedChannelId = null;
        selectedChannelData = null;
        document.title = 'Hunter Motions';
        document.getElementById('file-input').value = '';
        document.getElementById('file-selected').classList.add('hidden');
        goToStep('home');
    });

    // ===================== CONFETTI =====================
    function fireConfetti() {
        const canvas = document.getElementById('confetti-canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const particles = [];
        const colors = ['#5B9BD5', '#4CAF7D', '#A0C4E8', '#FFD700', '#FF6B6B', '#C084FC'];
        for (let i = 0; i < 120; i++) {
            particles.push({
                x: canvas.width / 2 + (Math.random() - 0.5) * 200,
                y: canvas.height / 2,
                vx: (Math.random() - 0.5) * 12,
                vy: Math.random() * -14 - 4,
                size: Math.random() * 6 + 3,
                color: colors[Math.floor(Math.random() * colors.length)],
                rotation: Math.random() * 360,
                rotSpeed: (Math.random() - 0.5) * 10,
                life: 1
            });
        }
        let frame = 0;
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            let alive = false;
            particles.forEach(p => {
                if (p.life <= 0) return;
                alive = true;
                p.x += p.vx;
                p.vy += 0.25;
                p.y += p.vy;
                p.rotation += p.rotSpeed;
                p.life -= 0.008;
                ctx.save();
                ctx.translate(p.x, p.y);
                ctx.rotate(p.rotation * Math.PI / 180);
                ctx.globalAlpha = Math.max(0, p.life);
                ctx.fillStyle = p.color;
                ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
                ctx.restore();
            });
            frame++;
            if (alive && frame < 200) requestAnimationFrame(animate);
            else ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        animate();
    }

    // ===================== MISC =====================
    window.addEventListener('beforeunload', (e) => {
        if (isGenerating) {
            e.preventDefault();
            e.returnValue = 'A video is being generated. Are you sure you want to leave?';
        }
    });

    // Changelog
    document.getElementById('open-changelog').addEventListener('click', () => document.getElementById('changelog-overlay').classList.remove('hidden'));
    document.getElementById('changelog-close').addEventListener('click', () => document.getElementById('changelog-overlay').classList.add('hidden'));
    document.getElementById('changelog-dismiss').addEventListener('click', () => document.getElementById('changelog-overlay').classList.add('hidden'));

    // Batch complete popup
    document.getElementById('batch-complete-continue').addEventListener('click', () => {
        document.getElementById('batch-complete-overlay').classList.add('hidden');
    });

    function showBatchComplete(completed, total) {
        document.getElementById('batch-complete-msg').textContent = `${completed} of ${total} scene${total > 1 ? 's' : ''} regenerated successfully`;
        document.getElementById('batch-complete-overlay').classList.remove('hidden');
    }

    // ===================== ACTIVE GENERATIONS POLL =====================
    async function pollActiveGenerations() {
        try {
            const resp = await fetch('/api/active-generations');
            const data = await resp.json();
            const count = data.count || 0;
            const dot = document.getElementById('active-gen-dot');
            const text = document.getElementById('active-gen-text');
            text.textContent = `Ongoing Generations: ${count}`;
            if (count > 0) {
                dot.className = 'active-gen-dot live';
            } else {
                dot.className = 'active-gen-dot idle';
            }
        } catch (e) { /* ignore poll errors */ }
    }
    pollActiveGenerations();
    setInterval(pollActiveGenerations, 5000);

    // ===================== INIT =====================
    loadRecentGenerations();
    goToStep('home');
});
