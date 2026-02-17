// HUNTER MOTIONS — Frontend
document.addEventListener('DOMContentLoaded', () => {

    // Toast notifications
    function showToast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icons = { success: '✓', error: '✕', info: 'i' };
        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || 'i'}</div>
            <div class="toast-msg">${message}</div>
            <button class="toast-close" onclick="this.parentElement.classList.add('removing');setTimeout(()=>this.parentElement.remove(),300)">✕</button>
        `;
        container.appendChild(toast);
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.add('removing');
                setTimeout(() => toast.remove(), 300);
            }
        }, duration);
    }
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const fileSelected = document.getElementById('file-selected');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const generateBtn = document.getElementById('generate-btn');
    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultSection = document.getElementById('result-section');
    const statusMessage = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const sceneTimeline = document.getElementById('scene-timeline');
    const downloadBtn = document.getElementById('download-btn');
    const newBtn = document.getElementById('new-btn');
    const animateToggle = document.getElementById('animate-toggle');

    const presetNameInput = document.getElementById('preset-name');
    const styleTextInput = document.getElementById('style-text');
    const styleInput = document.getElementById('style-input');
    const subjectInput = document.getElementById('subject-input');
    const styleUploadBox = document.getElementById('style-upload-box');
    const subjectUploadBox = document.getElementById('subject-upload-box');
    const stylePlaceholder = document.getElementById('style-placeholder');
    const subjectPlaceholder = document.getElementById('subject-placeholder');
    const stylePreview = document.getElementById('style-preview');
    const subjectPreview = document.getElementById('subject-preview');
    const savePresetBtn = document.getElementById('save-preset-btn');
    const presetGallery = document.getElementById('preset-gallery');

    const presetBarEmpty = document.getElementById('preset-bar-empty');
    const presetBarActive = document.getElementById('preset-bar-active');
    const presetBarThumb = document.getElementById('preset-bar-thumb');
    const presetBarName = document.getElementById('preset-bar-name');
    const goToSettings = document.getElementById('go-to-settings');
    const changePreset = document.getElementById('change-preset');

    let selectedFile = null;
    let currentSessionId = null;
    let activePresetId = null;
    let styleFile = null;
    let subjectFile = null;

    // Theme
    const themeToggle = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);

    themeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    });

    // WebSocket
    const socket = io();
    socket.on('connect', () => console.log('Connected'));
    socket.on('progress', (data) => {
        if (data.session_id !== currentSessionId) return;
        handleProgress(data);
    });

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
            if (btn.dataset.tab === 'settings') loadPresets();
        });
    });

    goToSettings.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.querySelector('[data-tab="settings"]').classList.add('active');
        document.getElementById('tab-settings').classList.add('active');
        loadPresets();
    });
    changePreset.addEventListener('click', () => goToSettings.click());

    // File
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
        if (!['mp3','wav','ogg','flac','m4a','aac','webm','mp4'].includes(ext)) { showToast('Invalid file type — use MP3, WAV, OGG, FLAC, M4A, AAC, WebM, or MP4', 'error'); return; }
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatSize(file.size);
        fileSelected.classList.remove('hidden');
    }

    function formatSize(bytes) {
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // Generate
    generateBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        if (!activePresetId) { showToast('Select a preset in Settings first', 'error'); return; }
        generateBtn.disabled = true;
        generateBtn.querySelector('.btn-text').textContent = 'Uploading...';
        const formData = new FormData();
        formData.append('audio', selectedFile);
        formData.append('preset_id', activePresetId);
        formData.append('animate_intro', animateToggle.checked ? 'true' : 'false');
        try {
            const resp = await fetch('/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            if (resp.ok) { currentSessionId = data.session_id; showProcessing(); }
            else { showToast(data.error || 'Upload failed', 'error'); resetBtn(); }
        } catch { showToast('Upload failed — check your connection', 'error'); resetBtn(); }
    });

    function resetBtn() {
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').textContent = 'Generate';
    }

    function showProcessing() {
        uploadSection.classList.add('hidden');
        processingSection.classList.remove('hidden');
        resultSection.classList.add('hidden');
        document.querySelectorAll('.step').forEach(s => s.classList.remove('active', 'complete'));
        progressBar.style.width = '0%';
        progressText.textContent = '0%';
        statusMessage.style.color = '';
    }

    function handleProgress(data) {
        const { step, progress, message } = data;
        statusMessage.textContent = message;
        
        // Only update progress if >= 0 (-1 means keep current)
        if (progress >= 0) {
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${progress}%`;

            // Update spinner ring and percentage
            const spinnerPct = document.getElementById('spinner-pct');
            const spinnerFill = document.querySelector('.spinner-fill');
            if (spinnerPct) spinnerPct.textContent = `${progress}%`;
            if (spinnerFill) {
                const circumference = 125.6; // 2 * PI * 20
                spinnerFill.style.strokeDashoffset = circumference - (circumference * progress / 100);
            }
        }

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
            stepOrder.forEach(s => {
                const el = document.getElementById(`step-${s}`);
                if (el) { el.classList.remove('active'); el.classList.add('complete'); }
            });
            setTimeout(() => showResult(data.data), 600);
        }

        if (step === 'error') {
            statusMessage.style.color = 'var(--error)';
        }
    }

    function showResult(data) {
        processingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        downloadBtn.href = data.video_url;
        sceneTimeline.innerHTML = '';
        if (data.scenes && data.scenes.length) {
            data.scenes.forEach(scene => {
                const isVideo = scene.is_video;
                const item = document.createElement('div');
                item.className = 'scene-item';
                item.innerHTML = `
                    <div class="scene-badge ${isVideo ? 'video' : 'image'}">${scene.scene_number}</div>
                    <div style="flex:1;min-width:0">
                        <div class="scene-time">${fmtTime(scene.start_time)} — ${fmtTime(scene.end_time)}</div>
                        <div class="scene-desc">${escHtml(scene.visual_description)}</div>
                    </div>
                    <span class="scene-type-tag ${isVideo ? 'video' : 'image'}">${isVideo ? 'Veo' : 'Imagen'}</span>
                `;
                sceneTimeline.appendChild(item);
            });
        }
    }

    function fmtTime(s) { return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`; }
    function escHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

    newBtn.addEventListener('click', () => {
        selectedFile = null; currentSessionId = null;
        fileInput.value = ''; fileSelected.classList.add('hidden');
        resetBtn(); statusMessage.style.color = '';
        resultSection.classList.add('hidden'); processingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    });

    // Settings
    styleUploadBox.addEventListener('click', () => styleInput.click());
    subjectUploadBox.addEventListener('click', () => subjectInput.click());

    styleInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            styleFile = e.target.files[0];
            stylePreview.src = URL.createObjectURL(styleFile);
            stylePreview.classList.remove('hidden');
            stylePlaceholder.classList.add('hidden');
        }
    });
    subjectInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            subjectFile = e.target.files[0];
            subjectPreview.src = URL.createObjectURL(subjectFile);
            subjectPreview.classList.remove('hidden');
            subjectPlaceholder.classList.add('hidden');
        }
    });

    savePresetBtn.addEventListener('click', async () => {
        const name = presetNameInput.value.trim() || 'Untitled';
        const styleText = styleTextInput.value.trim();
        if (!styleFile && !styleText) { showToast('Provide a style image, text description, or both', 'error'); return; }
        savePresetBtn.disabled = true;
        savePresetBtn.querySelector('span').textContent = 'Saving...';
        const formData = new FormData();
        formData.append('name', name);
        formData.append('style_text', styleText);
        if (styleFile) formData.append('style', styleFile);
        if (subjectFile) formData.append('subject', subjectFile);
        try {
            const resp = await fetch('/api/presets', { method: 'POST', body: formData });
            const data = await resp.json();
            if (resp.ok) {
                setActivePreset(data.id, name);
                showToast(`Preset "${name}" saved and activated`, 'success');
                presetNameInput.value = ''; styleTextInput.value = '';
                styleFile = null; subjectFile = null;
                stylePreview.classList.add('hidden'); stylePlaceholder.classList.remove('hidden');
                subjectPreview.classList.add('hidden'); subjectPlaceholder.classList.remove('hidden');
                styleInput.value = ''; subjectInput.value = '';
                loadPresets();
            } else { showToast(data.error || 'Save failed', 'error'); }
        } catch { showToast('Failed to save preset', 'error'); }
        savePresetBtn.disabled = false;
        savePresetBtn.querySelector('span').textContent = 'Save Preset';
    });

    async function loadPresets() {
        try {
            const resp = await fetch('/api/presets');
            const presets = await resp.json();
            presetGallery.innerHTML = '';
            if (!presets.length) {
                presetGallery.innerHTML = '<p class="empty-state">No presets saved yet</p>';
                return;
            }
            presets.forEach(p => {
                const item = document.createElement('div');
                item.className = 'preset-item';
                const isActive = p.id === activePresetId;
                item.innerHTML = `
                    <div class="preset-thumbs">
                        <img class="preset-thumb" src="/api/presets/${p.id}/style.png" alt="Style">
                        ${p.has_subject ? `<img class="preset-thumb" src="/api/presets/${p.id}/subject.png" alt="Subject">` : `<div class="preset-thumb-empty">—</div>`}
                    </div>
                    <div class="preset-info">
                        <div class="preset-info-name">${escHtml(p.name)}${isActive ? ' ✓' : ''}</div>
                        <div class="preset-info-meta">${p.has_subject ? 'Style + Subject' : 'Style only'}${p.style_text ? ' · Text' : ''}</div>
                    </div>
                    <div class="preset-actions">
                        <button class="preset-use-btn" data-id="${p.id}" data-name="${escHtml(p.name)}">${isActive ? 'Active' : 'Use'}</button>
                        <button class="preset-delete-btn" data-id="${p.id}">✕</button>
                    </div>
                `;
                presetGallery.appendChild(item);
            });
            presetGallery.querySelectorAll('.preset-use-btn').forEach(btn => {
                btn.addEventListener('click', () => { setActivePreset(btn.dataset.id, btn.dataset.name); loadPresets(); });
            });
            presetGallery.querySelectorAll('.preset-delete-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    if (!confirm('Delete this preset?')) return;
                    await fetch(`/api/presets/${btn.dataset.id}`, { method: 'DELETE' });
                    if (activePresetId === btn.dataset.id) { activePresetId = null; updatePresetBar(); }
                    loadPresets();
                });
            });
        } catch (err) { console.error('Load presets error:', err); }
    }

    function setActivePreset(id, name) {
        activePresetId = id;
        localStorage.setItem('activePresetId', id);
        localStorage.setItem('activePresetName', name);
        updatePresetBar();
    }

    function updatePresetBar() {
        if (activePresetId) {
            presetBarEmpty.classList.add('hidden');
            presetBarActive.classList.remove('hidden');
            presetBarThumb.src = `/api/presets/${activePresetId}/style.png`;
            presetBarName.textContent = localStorage.getItem('activePresetName') || 'Preset';
        } else {
            presetBarEmpty.classList.remove('hidden');
            presetBarActive.classList.add('hidden');
        }
    }

    // Export presets
    document.getElementById('export-presets-btn').addEventListener('click', async () => {
        try {
            const resp = await fetch('/api/presets/export');
            if (!resp.ok) { showToast('Export failed', 'error'); return; }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `hunter-motions-presets-${new Date().toISOString().slice(0,10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
            showToast('Presets exported', 'success');
        } catch { showToast('Export failed', 'error'); }
    });

    // Import presets
    const importFileInput = document.getElementById('import-file-input');
    document.getElementById('import-presets-btn').addEventListener('click', () => importFileInput.click());
    importFileInput.addEventListener('change', async (e) => {
        if (!e.target.files.length) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);
        try {
            const resp = await fetch('/api/presets/import', { method: 'POST', body: formData });
            const data = await resp.json();
            if (resp.ok) {
                showToast(`Imported ${data.count} preset(s)`, 'success');
                loadPresets();
            } else {
                showToast(data.error || 'Import failed', 'error');
            }
        } catch { showToast('Import failed', 'error'); }
        importFileInput.value = '';
    });

    // Init
    const savedPresetId = localStorage.getItem('activePresetId');
    if (savedPresetId) { activePresetId = savedPresetId; updatePresetBar(); }
});
