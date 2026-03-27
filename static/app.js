/**
 * PDF Comparison Tool - Frontend Application
 * Vanilla JS with i18n, SSE progress, and API calls.
 */

// ============================================================
//  i18n Translations
// ============================================================
const I18N = {
    vi: {
        appTitle: 'Công cụ so sánh PDF',
        appDesc: 'So sánh tự động PDF Brief và Produit Fini',
        // Step 1
        step1Title: 'Chọn PDF A (Brief)',
        optUpload: 'Upload file mới',
        optExisting: 'Chọn từ danh sách',
        uploadHint: 'Kéo thả file PDF hoặc nhấn để chọn',
        uploadHintSub: 'Chỉ chấp nhận file .pdf',
        noProcessed: 'Chưa có PDF nào đã xử lý.',
        selectPdf: 'Chọn PDF đã xử lý:',
        btnStep1: 'Chạy Step 1 (Detect Bbox)',
        btnStep23: 'Chạy Step 2 & 3 (Detect ID + OCR)',
        pdfAReady: 'PDF A đã sẵn sàng',
        bboxCount: '{0} bbox',
        idCount: '{0} ID',
        alreadyProcessed: 'PDF này đã có dữ liệu bbox + ID.',
        changeBtn: 'Thay đổi',
        deleteBtn: 'Xóa dữ liệu',
        deleteConfirm: 'Bạn có chắc muốn xóa toàn bộ dữ liệu của PDF này?',
        deleteWarning: 'Hành động này không thể hoàn tác!',
        confirmBtn: 'Xác nhận xóa',
        cancelBtn: 'Hủy',
        // Step 2
        step2Title: 'Chọn PDF B (Produit Fini)',
        uploadBHint: 'Kéo thả file PDF B (có thể nhiều file)',
        addMoreFiles: 'Thêm file',
        // Step 3
        step3Title: 'Chạy so sánh',
        aiModelLabel: 'Chọn AI Model:',
        btnRunPipeline: 'Chạy',
        // Progress
        processing: 'Đang xử lý...',
        completed: 'Hoàn thành',
        // Results
        resultsTitle: 'Kết quả',
        metricBbox: 'Bbox',
        metricMatched: 'Matched',
        metricCompared: 'So sánh',
        metricHighlight: 'Highlight',
        downloadBtn: 'Tải PDF',
        previewBtn: 'Xem PDF',
        success: 'Thành công',
        failed: 'Thất bại',
        noMatch: 'Không trùng ID',
        noIframeSupport: 'Trình duyệt không hỗ trợ xem PDF trực tiếp. Bạn có thể',
        // Messages
        step1Done: 'Step 1 hoàn thành: {0} bbox',
        step23Done: 'Step 2&3 hoàn thành: {0} ID, {1} renamed',
        uploadFirst: 'Vui lòng chọn file trước',
        modelNotFound: 'Không tìm thấy model YOLO',
        selectedFiles: 'Đã chọn {0} file:',
    },
    en: {
        appTitle: 'PDF Comparison Tool',
        appDesc: 'Automatically compare Brief vs. Produit Fini PDFs',
        step1Title: 'Select PDF A (Brief)',
        optUpload: 'Upload new file',
        optExisting: 'Choose from list',
        uploadHint: 'Drag & drop PDF or click to select',
        uploadHintSub: 'Only .pdf files accepted',
        noProcessed: 'No processed PDFs yet.',
        selectPdf: 'Select a processed PDF:',
        btnStep1: 'Run Step 1 (Detect Bbox)',
        btnStep23: 'Run Step 2 & 3 (Detect ID + OCR)',
        pdfAReady: 'PDF A is ready',
        bboxCount: '{0} bbox',
        idCount: '{0} IDs',
        alreadyProcessed: 'This PDF already has bbox + ID data.',
        changeBtn: 'Change',
        deleteBtn: 'Delete data',
        deleteConfirm: 'Are you sure you want to delete all data for this PDF?',
        deleteWarning: 'This action cannot be undone!',
        confirmBtn: 'Confirm delete',
        cancelBtn: 'Cancel',
        step2Title: 'Select PDF B (Produit Fini)',
        uploadBHint: 'Drag & drop PDF B files (multiple allowed)',
        addMoreFiles: 'Add file',
        step3Title: 'Run Comparison',
        aiModelLabel: 'Choose AI Model:',
        btnRunPipeline: 'Run',
        processing: 'Processing...',
        completed: 'Completed',
        resultsTitle: 'Results',
        metricBbox: 'Bbox',
        metricMatched: 'Matched',
        metricCompared: 'Compared',
        metricHighlight: 'Highlights',
        downloadBtn: 'Download PDF',
        previewBtn: 'View PDF',
        success: 'Success',
        failed: 'Failed',
        noMatch: 'No ID match',
        noIframeSupport: 'Browser does not support inline PDFs. You can',
        step1Done: 'Step 1 done: {0} bbox',
        step23Done: 'Step 2&3 done: {0} IDs, {1} renamed',
        uploadFirst: 'Please select a file first',
        modelNotFound: 'YOLO model not found',
        selectedFiles: 'Selected {0} file(s):',
    },
    fr: {
        appTitle: 'Outil de Comparaison PDF',
        appDesc: 'Comparer automatiquement les PDF Brief et Produit Fini',
        step1Title: 'Sélectionner PDF A (Brief)',
        optUpload: 'Importer un nouveau fichier',
        optExisting: 'Choisir dans la liste',
        uploadHint: 'Glisser-déposer le PDF ou cliquer pour sélectionner',
        uploadHintSub: 'Fichiers .pdf uniquement',
        noProcessed: 'Aucun PDF traité.',
        selectPdf: 'Sélectionner un PDF traité :',
        btnStep1: 'Lancer Étape 1 (Detect Bbox)',
        btnStep23: 'Lancer Étapes 2 & 3 (Detect ID + OCR)',
        pdfAReady: 'PDF A est prêt',
        bboxCount: '{0} bbox',
        idCount: '{0} ID',
        alreadyProcessed: 'Ce PDF possède déjà des données bbox + ID.',
        changeBtn: 'Changer',
        deleteBtn: 'Supprimer les données',
        deleteConfirm: 'Êtes-vous sûr de vouloir supprimer toutes les données de ce PDF ?',
        deleteWarning: 'Cette action est irréversible !',
        confirmBtn: 'Confirmer la suppression',
        cancelBtn: 'Annuler',
        step2Title: 'Sélectionner PDF B (Produit Fini)',
        uploadBHint: 'Glisser-déposer les PDF B (multiples autorisés)',
        addMoreFiles: 'Ajouter fichier',
        step3Title: 'Lancer la Comparaison',
        aiModelLabel: 'Choisir le modèle IA :',
        btnRunPipeline: 'Lancer',
        processing: 'Traitement en cours...',
        completed: 'Terminé',
        resultsTitle: 'Résultats',
        metricBbox: 'Bbox',
        metricMatched: 'Correspondances',
        metricCompared: 'Comparaisons',
        metricHighlight: 'Surlignés',
        downloadBtn: 'Télécharger PDF',
        previewBtn: 'Voir PDF',
        success: 'Succès',
        failed: 'Échoué',
        noMatch: 'Pas de correspondance',
        step1Done: 'Étape 1 terminée : {0} bbox',
        step23Done: 'Étapes 2&3 terminées : {0} ID, {1} renommés',
        uploadFirst: "Veuillez d'abord sélectionner un fichier",
        modelNotFound: 'Modèle YOLO introuvable',
        selectedFiles: '{0} fichier(s) sélectionné(s) :',
    },
};

// ============================================================
//  App State
// ============================================================
const state = {
    lang: localStorage.getItem('pdf_tool_lang') || 'vi',
    sessionId: null,
    pdfAName: null,
    pdfADisplayName: null,
    pdfAStoredAtDisplay: null,
    pdfAFile: null, // Track selected file A
    pdfAReady: false,
    pdfBFiles: [],
    appendPdfB: false, // true khi bấm "Thêm file"
    running: false,
};

// ============================================================
//  Helpers
// ============================================================
function t(key, ...args) {
    let text = (I18N[state.lang] || I18N.vi)[key] || key;
    args.forEach((arg, i) => { text = text.replace(`{${i}}`, arg); });
    return text;
}
function tShared(key) {
    const i = window.I18N_SHARED && I18N_SHARED[state.lang] ? I18N_SHARED[state.lang] : {};
    return i[key] || key;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function showAlert(container, type, msg) {
    const icons = { info: 'info', success: 'check-circle', warning: 'alert-triangle', error: 'x-circle' };
    const el = document.createElement('div');
    el.className = `alert alert-${type}`;
    el.innerHTML = `<i data-lucide="${icons[type] || 'info'}"></i><span>${msg}</span>`;
    container.prepend(el);
    if (window.lucide) lucide.createIcons();
    setTimeout(() => el.remove(), 8000);
}

async function api(method, url, data = null, isForm = true) {
    const opts = { method };
    if (data) {
        if (isForm && !(data instanceof FormData)) {
            const fd = new FormData();
            Object.entries(data).forEach(([k, v]) => fd.append(k, v));
            opts.body = fd;
        } else {
            opts.body = data;
        }
    }
    const res = await fetch(`/api/${url}`, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API error');
    }
    return res.json();
}

// ============================================================
//  Render Functions
// ============================================================
function renderApp() {
    const app = $('#app');
    if (!app) {
        document.body.innerHTML = '<p style="padding:20px;color:red">Lỗi: không tìm thấy #app</p>';
        return;
    }
    app.innerHTML = `
        <div class="app-header">
            <h1><i data-lucide="layers" class="icon-lg"></i> ${t('appTitle')}</h1>
            <p id="header-desc">${t('appDesc')}</p>
            <div class="mode-switch-tabs">
                <button type="button" class="mode-tab" data-mode="1">Mode 1</button>
                <button type="button" class="mode-tab" data-mode="2">Mode 2</button>
                <button type="button" class="mode-tab" data-mode="3">Mode 3</button>
            </div>
            <div class="header-controls">
                <button class="theme-toggle" id="theme-toggle" title="Đổi sáng/tối">
                    <span class="theme-icon theme-sun">○</span>
                    <span class="theme-icon theme-moon">☾</span>
                </button>
                <div class="lang-switcher">
                    <button class="lang-btn ${state.lang === 'vi' ? 'active' : ''}" data-lang="vi">VN</button>
                    <button class="lang-btn ${state.lang === 'en' ? 'active' : ''}" data-lang="en">EN</button>
                    <button class="lang-btn ${state.lang === 'fr' ? 'active' : ''}" data-lang="fr">FR</button>
                </div>
            </div>
        </div>

        <div id="mode1-panel" class="mode-panel">
        <div class="main-layout">
            <!-- Top Left: PDF A -->
            <div class="card" id="card-step1">
                <div class="card-title">
                    <span class="step-badge">1</span>
                    <i data-lucide="file-text"></i>
                    <span>${t('step1Title')}</span>
                </div>
                <div class="radio-group" id="pdf-a-mode">
                    <div class="radio-option">
                        <input type="radio" name="pdfAMode" id="modeUpload" value="upload" checked>
                        <label for="modeUpload"><i data-lucide="upload-cloud"></i> ${t('optUpload')}</label>
                    </div>
                    <div class="radio-option">
                        <input type="radio" name="pdfAMode" id="modeExisting" value="existing">
                        <label for="modeExisting"><i data-lucide="folder-open"></i> ${t('optExisting')}</label>
                    </div>
                </div>

                <div id="upload-a-section" style="margin-top:18px;">
                    <div class="upload-zone" id="dropzone-a">
                        <div class="upload-icon"><i data-lucide="file-up"></i></div>
                        <div class="upload-title">${t('uploadHint')}</div>
                        <div class="upload-hint">${t('uploadHintSub')}</div>
                        <input type="file" accept=".pdf" id="file-a-input">
                    </div>
                    <div id="file-a-info"></div>
                </div>

                <div id="existing-a-section" class="hidden" style="margin-top:18px;">
                    <div id="processed-list"></div>
                </div>

                <div id="pdf-a-status" style="margin-top:14px;"></div>
                <div id="pdf-a-actions" style="margin-top:14px;"></div>
            </div>

            <!-- Top Right: PDF B & Run -->
            <div id="card-step2-wrapper">
                <div class="card" id="card-step2">
                    <div class="card-title">
                        <span class="step-badge">2</span>
                        <i data-lucide="files"></i>
                        <span>${t('step2Title')}</span>
                    </div>
                    <div class="upload-zone" id="dropzone-b">
                        <div class="upload-icon"><i data-lucide="file-plus-2"></i></div>
                        <div class="upload-title">${t('uploadBHint')}</div>
                        <div class="upload-hint">${t('uploadHintSub')}</div>
                        <input type="file" accept=".pdf" multiple id="file-b-input">
                    </div>
                    <div id="file-b-list" class="file-list"></div>
                </div>

                <div class="card" id="card-step3" style="margin-top:20px;">
                    <div class="card-title">
                        <span class="step-badge">3</span>
                        <i data-lucide="cog"></i>
                        <span>${t('step3Title')}</span>
                    </div>
                    <div class="form-group">
                        <label class="form-label"><i data-lucide="cpu"></i> ${t('aiModelLabel')}</label>
                        <div class="radio-group">
                            <div class="radio-option">
                                <input type="radio" name="aiModel" id="aiOpenAI" value="OpenAI GPT" checked>
                                <label for="aiOpenAI">GPT</label>
                            </div>
                            <div class="radio-option hidden" id="aiGeminiOption">
                                <input type="radio" name="aiModel" id="aiGemini" value="Google Gemini" disabled>
                                <label for="aiGemini">Gemini</label>
                            </div>
                        </div>
                    </div>
                    <button class="btn btn-primary btn-full" id="btn-run-pipeline" disabled>
                        <i data-lucide="play"></i> ${t('btnRunPipeline')}
                    </button>
                </div>
            </div>

            <!-- Bottom: Progress & Results -->
            <div id="result-section-wrapper">
                <div class="card ${!state.running && !state.sessionId ? 'hidden' : ''}" id="card-progress">
                    <div class="card-title">
                        <i data-lucide="refresh-cw" class="spinner-icon"></i>
                        <span id="progress-title">${t('processing')}</span>
                    </div>
                    <div id="pipeline-progress">
                        <div class="progress-wrapper">
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" id="progress-fill" style="width:0%"></div>
                            </div>
                            <div class="progress-text">
                                <span id="progress-label">${t('processing')}</span>
                                <span id="progress-pct">0%</span>
                            </div>
                        </div>
                        <div class="log-container" id="progress-log"></div>
                    </div>
                </div>

                <!-- Results -->
                <div class="card hidden" id="card-results">
                    <div class="card-title">
                        <i data-lucide="bar-chart-3"></i>
                        <span>${t('resultsTitle')}</span>
                    </div>
                    <div id="results-container"></div>
                </div>
            </div>
        </div>
        </div>

        <!-- Mode 2 Panel -->
        <div id="mode2-panel" class="mode-panel hidden">
            <div class="main-layout">
                <div class="card">
                    <div class="card-title">
                        <span class="step-badge">1</span>
                        <i data-lucide="file-text"></i>
                        <span id="mode2-label-ref">${tShared('labelRefPdf')}</span>
                    </div>
                    <div class="upload-zone" id="mode2-dropzone-ref">
                        <div class="upload-icon"><i data-lucide="file-up"></i></div>
                        <div class="upload-title">${t('uploadHint')}</div>
                        <div class="upload-hint">${t('uploadHintSub')}</div>
                        <input type="file" accept=".pdf" id="mode2-refPdf">
                    </div>
                    <div id="mode2-ref-selected" class="file-selected-row hidden"></div>
                    <div class="mode2-preview-wrap"><iframe id="mode2-previewRef" title="Preview Reference" src="about:blank"></iframe></div>
                </div>
                <div class="card">
                    <div class="card-title">
                        <span class="step-badge">2</span>
                        <i data-lucide="files"></i>
                        <span id="mode2-label-final">${tShared('labelFinalPdf')}</span>
                    </div>
                    <div class="upload-zone" id="mode2-dropzone-final">
                        <div class="upload-icon"><i data-lucide="file-up"></i></div>
                        <div class="upload-title">${t('uploadHint')}</div>
                        <div class="upload-hint">${t('uploadHintSub')}</div>
                        <input type="file" accept=".pdf" id="mode2-finalPdf">
                    </div>
                    <div id="mode2-final-selected" class="file-selected-row hidden"></div>
                    <div class="mode2-preview-wrap"><iframe id="mode2-previewFinal" title="Preview Final" src="about:blank"></iframe></div>
                    <button type="button" id="mode2-btnCompare" class="btn btn-primary btn-full" style="margin-top:16px" disabled>
                        <i data-lucide="play"></i> ${tShared('btnCompare')}
                    </button>
                </div>
            </div>
            <div id="mode2-status" class="card" hidden style="margin-top:24px"><span id="mode2-statusText"></span></div>
            <div id="mode2-error" class="card" hidden style="margin-top:24px;background:rgba(255,107,107,0.12);border-color:rgba(255,107,107,0.35)"><p id="mode2-errorText"></p></div>
            <div id="mode2-result" class="card" hidden style="margin-top:24px">
                <div class="card-title"><i data-lucide="bar-chart-3"></i> <span id="mode2-results-title">${tShared('resultsTitle')}</span></div>
                <div class="mode2-summary" id="mode2-summary" style="margin-bottom:12px"></div>
                <div class="mode2-links" style="gap:12px;margin-bottom:16px">
                    <a id="mode2-linkView" href="#" target="_blank" rel="noopener" class="btn btn-secondary btn-sm">Mở tab mới</a>
                    <a id="mode2-linkDownload" href="#" download class="btn btn-primary btn-sm"><i data-lucide="download"></i> ${t('downloadBtn')}</a>
                </div>
                <div class="mode2-pdf-view"><iframe id="mode2-pdfViewer" title="PDF kết quả" src="about:blank"></iframe></div>
            </div>
        </div>

        <!-- Mode 3 Panel -->
        <div id="mode3-panel" class="mode-panel hidden">
            <div class="mode3-layout">
                <div class="mode3-left">
                    <div class="card" style="max-width:500px">
                        <div class="card-title">
                            <span class="step-badge">1</span>
                            <i data-lucide="file-text"></i>
                            <span id="mode3-label-produit">${tShared('labelProduitFini')}</span>
                        </div>
                        <div class="upload-zone" id="mode3-dropzone-final">
                            <div class="upload-icon"><i data-lucide="file-up"></i></div>
                            <div class="upload-title">${t('uploadHint')}</div>
                            <div class="upload-hint">${t('uploadHintSub')}</div>
                            <input type="file" accept=".pdf" id="mode3-finalPdf">
                        </div>
                        <div id="mode3-final-selected" class="file-selected-row hidden"></div>
                        <div class="mode2-preview-wrap"><iframe id="mode3-previewFinal" src="about:blank"></iframe></div>
                        <button id="mode3-btnCheck" class="btn btn-primary btn-full" style="margin-top:16px" disabled>
                            <i data-lucide="play"></i> ${tShared('btnCheck')}
                        </button>
                    </div>
                </div>
                <div class="mode3-right">
                    <div id="mode3-status" class="card" hidden><span id="mode3-statusText"></span></div>
                    <div id="mode3-error" class="card" hidden style="background:rgba(255,107,107,0.12);border-color:rgba(255,107,107,0.35)"><p id="mode3-errorText"></p></div>
                    <div id="mode3-result" class="card" hidden>
                        <div class="card-title"><i data-lucide="bar-chart-3"></i> <span id="mode3-results-title">${tShared('resultsTitle3')}</span></div>
                        <div id="mode3-hardSummary" class="mode3-summary-badges"></div>
                        <div id="mode3-zonesInfo" class="mode3-zones-info"></div>

                        <div id="mode3-debugPanel" class="mode3-debug-panel">
                            <details open>
                                <summary class="mode3-debug-title" id="mode3-dbg-s1">${tShared('debugOverlay')}</summary>
                                <div id="mode3-debugOverlays" class="mode3-debug-images"></div>
                            </details>
                            <details>
                                <summary class="mode3-debug-title" id="mode3-dbg-s2">${tShared('debugYoloCrops')}</summary>
                                <div id="mode3-debugYoloCrops" class="mode3-debug-crops"></div>
                            </details>
                            <details>
                                <summary class="mode3-debug-title" id="mode3-dbg-s3">${tShared('debugSubzones')}</summary>
                                <div id="mode3-debugCrops" class="mode3-debug-crops"></div>
                            </details>
                            <details>
                                <summary class="mode3-debug-title" id="mode3-dbg-s4">${tShared('debugGptCrops')}</summary>
                                <div id="mode3-debugGptCrops" class="mode3-debug-crops"></div>
                            </details>
                        </div>

                        <div id="mode3-hardDetails" class="mode3-rule-table-wrap"></div>
                        <div id="mode3-softSummary" class="mode3-soft-section"></div>
                        <div class="mode2-links" style="gap:12px;margin:16px 0">
                            <a id="mode3-linkView" href="#" target="_blank" rel="noopener" class="btn btn-secondary btn-sm">Mở PDF</a>
                            <a id="mode3-linkDownload" href="#" download class="btn btn-primary btn-sm"><i data-lucide="download"></i> ${t('downloadBtn')}</a>
                            <a id="mode3-linkReport" href="#" target="_blank" rel="noopener" class="btn btn-secondary btn-sm">Report JSON</a>
                        </div>
                        <div class="mode2-pdf-view"><iframe id="mode3-pdfViewer" src="about:blank"></iframe></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Dialog -->
        <div id="dialog-overlay" class="dialog-overlay hidden"></div>
    `;

    bindEvents();
    if (window.lucide) lucide.createIcons();
}

// Shared Produit Fini file (sync between Mode 2 & 3)
let sharedProduitFiniFile = null;

function setFileToInput(inputEl, file) {
    if (!inputEl || !file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    inputEl.files = dt.files;
}

function clearProduitFiniEverywhere() {
    sharedProduitFiniFile = null;
    state.pdfBFiles = [];
    const fbInput = $('#file-b-input');
    if (fbInput) fbInput.value = '';
    const m2Final = $('#mode2-finalPdf');
    const m3Final = $('#mode3-finalPdf');
    if (m2Final) { m2Final.value = ''; m2Final.dispatchEvent(new Event('change')); }
    if (m3Final) { m3Final.value = ''; m3Final.dispatchEvent(new Event('change')); }
    renderPdfBList();
    const m2DropFinal = $('#mode2-dropzone-final'), m2SelFinal = $('#mode2-final-selected'), m2Prev = $('#mode2-previewFinal');
    if (m2DropFinal) m2DropFinal.classList.remove('hidden');
    if (m2SelFinal) { m2SelFinal.classList.add('hidden'); m2SelFinal.innerHTML = ''; }
    if (m2Prev) { if (m2Prev._prevBlobUrl) URL.revokeObjectURL(m2Prev._prevBlobUrl); m2Prev._prevBlobUrl = null; m2Prev.src = 'about:blank'; m2Prev.style.display = 'none'; }
    const m3Drop = $('#mode3-dropzone-final'), m3Sel = $('#mode3-final-selected'), m3Prev = $('#mode3-previewFinal');
    if (m3Drop) m3Drop.classList.remove('hidden');
    if (m3Sel) { m3Sel.classList.add('hidden'); m3Sel.innerHTML = ''; }
    if (m3Prev) { if (m3Prev._prevBlobUrl) URL.revokeObjectURL(m3Prev._prevBlobUrl); m3Prev._prevBlobUrl = null; m3Prev.src = 'about:blank'; m3Prev.style.display = 'none'; }
    const btnCompare = $('#mode2-btnCompare'), btnCheck = $('#mode3-btnCheck'), btnRun = $('#btn-run-pipeline');
    if (btnCompare) btnCompare.disabled = true;
    if (btnCheck) btnCheck.disabled = true;
    if (btnRun) btnRun.disabled = true;
}

function switchMode(mode) {
    const m = String(mode);
    $$('.mode-panel').forEach(p => p.classList.add('hidden'));
    $$('.mode-tab').forEach(t => t.classList.remove('active'));
    const panel = $(`#mode${m === '1' ? '1' : m}-panel`);
    const tab = $(`.mode-tab[data-mode="${m}"]`);
    if (panel) panel.classList.remove('hidden');
    if (tab) tab.classList.add('active');
    location.hash = m === '1' ? '' : `#mode${m}`;
    if (m === '1' && sharedProduitFiniFile) {
        state.pdfBFiles = [sharedProduitFiniFile];
        renderPdfBList();
        const inp = $('#file-b-input');
        if (inp) setFileToInput(inp, sharedProduitFiniFile);
        const runBtn = $('#btn-run-pipeline');
        if (runBtn) runBtn.disabled = state.pdfBFiles.length === 0 || (!state.pdfAName && !state.pdfAFile);
    }
    if (m === '2' && sharedProduitFiniFile) {
        const inp = $('#mode2-finalPdf');
        if (inp) { setFileToInput(inp, sharedProduitFiniFile); inp.dispatchEvent(new Event('change')); }
    }
    if (m === '3' && sharedProduitFiniFile) {
        const inp = $('#mode3-finalPdf');
        if (inp) { setFileToInput(inp, sharedProduitFiniFile); inp.dispatchEvent(new Event('change')); }
    }
    const descEl = $('#header-desc');
    if (descEl) {
        if (m === '1') descEl.textContent = t('appDesc');
        else if (m === '2') descEl.textContent = tShared('mode2Desc');
        else if (m === '3') descEl.textContent = tShared('mode3Desc');
    }
}

function bindEvents() {
    // Mode tabs - SPA routing (dùng button để tránh xung đột href/hash)
    $$('.mode-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const m = tab.dataset.mode;
            if (m) switchMode(m);
        });
    });
    window.addEventListener('hashchange', () => {
        const hash = (location.hash || '').replace(/^#/, '');
        const m = hash === 'mode2' ? '2' : hash === 'mode3' ? '3' : '1';
        switchMode(m);
    });
    // Áp dụng mode theo hash khi load (hoặc mode 1 nếu không có hash)
    const hash = (location.hash || '').replace(/^#/, '');
    const initialMode = hash === 'mode2' ? '2' : hash === 'mode3' ? '3' : '1';
    switchMode(initialMode);

    // Theme toggle
    const themeToggle = $('#theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const html = document.documentElement;
            const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('pdf_tool_theme', next);
        });
    }

    // Language switcher
    $$('.lang-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            state.lang = btn.dataset.lang;
            localStorage.setItem('pdf_tool_lang', state.lang);
            renderApp();
            // Restore state
            if (state.pdfAReady) {
                const info = { bbox_count: 0, id_count: 0 }; // Placeholder, will be refreshed
                api('GET', `pdf-info/${state.pdfAName}`).then(showPdfAStatus);
            } else if (state.pdfAFile) {
                preparePdfA(state.pdfAFile);
            }
        });
    });

    // PDF A mode toggle
    $$('#pdf-a-mode input').forEach(radio => {
        radio.addEventListener('change', () => {
            const mode = radio.value;
            if (mode === 'upload') {
                $('#upload-a-section').classList.remove('hidden');
                $('#existing-a-section').classList.add('hidden');
            } else {
                $('#upload-a-section').classList.add('hidden');
                $('#existing-a-section').classList.remove('hidden');
                loadProcessedPdfs();
            }
        });
    });

    // Upload PDF A
    const fileAInput = $('#file-a-input');
    const dropzoneA = $('#dropzone-a');

    fileAInput.addEventListener('change', () => {
        if (fileAInput.files.length > 0) preparePdfA(fileAInput.files[0]);
    });

    ['dragover', 'dragenter'].forEach(ev => {
        dropzoneA.addEventListener(ev, e => { e.preventDefault(); dropzoneA.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(ev => {
        dropzoneA.addEventListener(ev, e => { e.preventDefault(); dropzoneA.classList.remove('dragover'); });
    });
    dropzoneA.addEventListener('drop', e => {
        if (e.dataTransfer.files.length > 0) preparePdfA(e.dataTransfer.files[0]);
    });
    dropzoneA.addEventListener('click', () => fileAInput.click());

    // Upload PDF B
    const fileBInput = $('#file-b-input');
    const dropzoneB = $('#dropzone-b');

    if (fileBInput) {
        fileBInput.addEventListener('change', () => {
            if (fileBInput.files.length > 0) {
                handlePdfBUpload(fileBInput.files, state.appendPdfB);
                state.appendPdfB = false;
            }
        });

        ['dragover', 'dragenter'].forEach(ev => {
            dropzoneB.addEventListener(ev, e => { e.preventDefault(); dropzoneB.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(ev => {
            dropzoneB.addEventListener(ev, e => { e.preventDefault(); dropzoneB.classList.remove('dragover'); });
        });
        dropzoneB.addEventListener('drop', e => {
            if (e.dataTransfer.files.length > 0) handlePdfBUpload(e.dataTransfer.files);
        });
        dropzoneB.addEventListener('click', () => fileBInput.click());
    }

    // Run pipeline
    const btnRun = $('#btn-run-pipeline');
    if (btnRun) {
        btnRun.addEventListener('click', runPipeline);
    }

    // Mode 2
    initMode2();
    // Mode 3
    initMode3();
}

function initMode2() {
    const refPdf = $('#mode2-refPdf');
    const finalPdf = $('#mode2-finalPdf');
    const btnCompare = $('#mode2-btnCompare');
    const dropzoneRef = $('#mode2-dropzone-ref');
    const dropzoneFinal = $('#mode2-dropzone-final');
    const refSelected = $('#mode2-ref-selected');
    const finalSelected = $('#mode2-final-selected');
    if (!refPdf || !finalPdf) return;
    const previewRef = $('#mode2-previewRef');
    const previewFinal = $('#mode2-previewFinal');
    function setPreview(iframe, file) {
        if (!iframe) return;
        if (iframe._prevBlobUrl) URL.revokeObjectURL(iframe._prevBlobUrl);
        if (file) { iframe._prevBlobUrl = URL.createObjectURL(file); iframe.src = iframe._prevBlobUrl; iframe.style.display = 'block'; }
        else { iframe._prevBlobUrl = null; iframe.src = 'about:blank'; iframe.style.display = 'none'; }
    }
    function updateMode2RefUI() {
        if (refPdf.files?.length) {
            setPreview(previewRef, refPdf.files[0]);
            dropzoneRef?.classList.add('hidden');
            refSelected?.classList.remove('hidden');
            refSelected.innerHTML = `<span class="file-name">${refPdf.files[0].name}</span><button type="button" class="btn btn-secondary btn-sm btn-clear">Xóa</button>`;
            refSelected.querySelector('.btn-clear')?.addEventListener('click', () => { refPdf.value = ''; setPreview(previewRef, null); dropzoneRef?.classList.remove('hidden'); refSelected?.classList.add('hidden'); refSelected.innerHTML = ''; enableCompare(); });
        } else { setPreview(previewRef, null); dropzoneRef?.classList.remove('hidden'); refSelected?.classList.add('hidden'); refSelected.innerHTML = ''; }
    }
    function updateMode2FinalUI() {
        if (finalPdf.files?.length) {
            sharedProduitFiniFile = finalPdf.files[0];
            setPreview(previewFinal, finalPdf.files[0]);
            dropzoneFinal?.classList.add('hidden');
            finalSelected?.classList.remove('hidden');
            finalSelected.innerHTML = `<span class="file-name">${finalPdf.files[0].name}</span><button type="button" class="btn btn-secondary btn-sm btn-clear">Xóa</button>`;
            finalSelected.querySelector('.btn-clear')?.addEventListener('click', () => { clearProduitFiniEverywhere(); enableCompare(); });
        } else { sharedProduitFiniFile = null; setPreview(previewFinal, null); dropzoneFinal?.classList.remove('hidden'); finalSelected?.classList.add('hidden'); finalSelected.innerHTML = ''; }
    }
    function enableCompare() {
        if (btnCompare) btnCompare.disabled = !(refPdf.files?.length && finalPdf.files?.length);
    }
    refPdf.addEventListener('change', () => { updateMode2RefUI(); enableCompare(); });
    finalPdf.addEventListener('change', () => { updateMode2FinalUI(); enableCompare(); });
    ['dragover','dragenter'].forEach(ev => dropzoneRef?.addEventListener(ev, e => { e.preventDefault(); dropzoneRef.classList.add('dragover'); }));
    ['dragleave','drop'].forEach(ev => dropzoneRef?.addEventListener(ev, e => { e.preventDefault(); dropzoneRef.classList.remove('dragover'); }));
    dropzoneRef?.addEventListener('drop', e => { if (e.dataTransfer.files.length) refPdf.files = e.dataTransfer.files; refPdf.dispatchEvent(new Event('change')); });
    ['dragover','dragenter'].forEach(ev => dropzoneFinal?.addEventListener(ev, e => { e.preventDefault(); dropzoneFinal.classList.add('dragover'); }));
    ['dragleave','drop'].forEach(ev => dropzoneFinal?.addEventListener(ev, e => { e.preventDefault(); dropzoneFinal.classList.remove('dragover'); }));
    dropzoneFinal?.addEventListener('drop', e => { if (e.dataTransfer.files.length) { finalPdf.files = e.dataTransfer.files; finalPdf.dispatchEvent(new Event('change')); } });
    dropzoneRef?.addEventListener('click', () => refPdf.click());
    dropzoneFinal?.addEventListener('click', () => finalPdf.click());
    updateMode2RefUI();
    updateMode2FinalUI();
    if (btnCompare) btnCompare.addEventListener('click', async () => {
        if (!refPdf.files?.length || !finalPdf.files?.length) return;
        const result = $('#mode2-result'), error = $('#mode2-error'), status = $('#mode2-status'), statusText = $('#mode2-statusText'), summary = $('#mode2-summary');
        const linkView = $('#mode2-linkView'), linkDownload = $('#mode2-linkDownload');
        result.hidden = true; error.hidden = true; status.hidden = false;
        statusText.textContent = tShared('processing');
        btnCompare.disabled = true;
        try {
            const fd = new FormData();
            fd.append('ref_pdf', refPdf.files[0]);
            fd.append('final_pdf', finalPdf.files[0]);
            const res = await fetch('/api/mode2/compare', { method: 'POST', body: fd });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || res.statusText || 'Lỗi không xác định');
            statusText.textContent = tShared('done');
            const s = data.summary || {};
            summary.innerHTML = [`Tổng annotations: ${s.total_annotations||0}`,`Đã thực hiện: ${s.implemented||0}`,`Chưa thực hiện: ${s.not_implemented||0}`,`Một phần: ${s.partial||0}`,`Không rõ: ${s.unclear||0}`].map(x=>`<div>${x}</div>`).join('');
            const base = window.location.origin;
            linkView.href = base + (data.view_url||'');
            linkDownload.href = base + (data.download_url||'');
            linkDownload.download = data.output_file || 'output_mode2_diff.pdf';
            const pdfViewer = $('#mode2-pdfViewer');
            if (pdfViewer) pdfViewer.src = linkView.href;
            result.hidden = false;
        } catch (err) {
            $('#mode2-errorText').textContent = err.message || 'Có lỗi xảy ra';
            error.hidden = false;
            statusText.textContent = 'Lỗi';
        } finally { btnCompare.disabled = false; }
    });
}

function initMode3() {
    const finalPdf = $('#mode3-finalPdf');
    const btnCheck = $('#mode3-btnCheck');
    const dropzoneFinal = $('#mode3-dropzone-final');
    const finalSelected = $('#mode3-final-selected');
    if (!finalPdf) return;
    const previewFinal = $('#mode3-previewFinal');
    function setPreview(iframe, file) {
        if (!iframe) return;
        if (iframe._prevBlobUrl) URL.revokeObjectURL(iframe._prevBlobUrl);
        if (file) { iframe._prevBlobUrl = URL.createObjectURL(file); iframe.src = iframe._prevBlobUrl; iframe.style.display = 'block'; }
        else { iframe._prevBlobUrl = null; iframe.src = 'about:blank'; iframe.style.display = 'none'; }
    }
    function updateMode3FinalUI() {
        if (finalPdf.files?.length) {
            sharedProduitFiniFile = finalPdf.files[0];
            setPreview(previewFinal, finalPdf.files[0]);
            dropzoneFinal?.classList.add('hidden');
            finalSelected?.classList.remove('hidden');
            finalSelected.innerHTML = `<span class="file-name">${finalPdf.files[0].name}</span><button type="button" class="btn btn-secondary btn-sm btn-clear">Xóa</button>`;
            finalSelected.querySelector('.btn-clear')?.addEventListener('click', () => { clearProduitFiniEverywhere(); if (btnCheck) btnCheck.disabled = true; });
        } else { sharedProduitFiniFile = null; setPreview(previewFinal, null); dropzoneFinal?.classList.remove('hidden'); finalSelected?.classList.add('hidden'); finalSelected.innerHTML = ''; if (btnCheck) btnCheck.disabled = true; }
    }
    finalPdf.addEventListener('change', () => { updateMode3FinalUI(); if (btnCheck) btnCheck.disabled = !finalPdf.files?.length; });
    ['dragover','dragenter'].forEach(ev => dropzoneFinal?.addEventListener(ev, e => { e.preventDefault(); dropzoneFinal.classList.add('dragover'); }));
    ['dragleave','drop'].forEach(ev => dropzoneFinal?.addEventListener(ev, e => { e.preventDefault(); dropzoneFinal.classList.remove('dragover'); }));
    dropzoneFinal?.addEventListener('drop', e => { if (e.dataTransfer.files.length) { finalPdf.files = e.dataTransfer.files; finalPdf.dispatchEvent(new Event('change')); } });
    dropzoneFinal?.addEventListener('click', () => finalPdf.click());
    updateMode3FinalUI();
    if (btnCheck) btnCheck.addEventListener('click', async () => {
        if (!finalPdf.files?.length) return;
        const result = $('#mode3-result'), error = $('#mode3-error'), status = $('#mode3-status'), statusText = $('#mode3-statusText');
        const hardSummary = $('#mode3-hardSummary'), hardDetails = $('#mode3-hardDetails');
        const softSummary = $('#mode3-softSummary'), zonesInfo = $('#mode3-zonesInfo');
        const linkView = $('#mode3-linkView'), linkDownload = $('#mode3-linkDownload'), linkReport = $('#mode3-linkReport');
        const pdfViewer = $('#mode3-pdfViewer');
        result.hidden = true; error.hidden = true; status.hidden = false;
        statusText.textContent = tShared('processing3');
        btnCheck.disabled = true;
        try {
            const fd = new FormData();
            fd.append('final_pdf', finalPdf.files[0]);
            fd.append('lang', state.lang || 'vi');
            const res = await fetch('/api/mode3/check', { method: 'POST', body: fd });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || res.statusText || 'Lỗi không xác định');
            statusText.textContent = tShared('done');

            const hs = data.hard_summary || {};
            const _s = (k) => tShared(k);
            const sevMap = {blocker: _s('sevBlocker'), major: _s('sevMajor'), minor: _s('sevMinor')};
            const statusMap = {pass: _s('pass'), fail: _s('fail'), unclear: _s('unclear')};

            hardSummary.innerHTML = `
                <span class="mode3-badge mode3-badge-total">${hs.total||0} ${_s('rules')}</span>
                <span class="mode3-badge mode3-badge-pass">${hs.pass||0} ${_s('pass')}</span>
                <span class="mode3-badge mode3-badge-fail">${hs.fail||0} ${_s('fail')}</span>
                <span class="mode3-badge mode3-badge-unclear">${hs.unclear||0} ${_s('unclear')}</span>
            `;

            const zd = data.zones_detected || {};
            const zdArr = Object.entries(zd);
            const dm = data.detect_method || 'unknown';
            const dmLabel = dm === 'yolo+segmentation' ? 'YOLO → Segmentation' : dm === 'yolo' ? 'YOLO only' : dm;
            zonesInfo.innerHTML = zdArr.length
                ? `<div class="mode3-zones-label">${dmLabel}: ${zdArr.map(([k,v])=>`${k}(${v})`).join(', ')}</div>`
                : `<div class="mode3-zones-label">${dmLabel}: ${_s('noZonesDetected')}</div>`;

            const hr = data.hard_results || [];
            const issuesOnly = hr.filter(r => r.status !== 'pass');
            if (hr.length) {
                const statusIcon = {pass:'✓',fail:'✗',unclear:'?'};
                const typeI18n = {color:'typeColor',dimension:'typeDimension',spacing:'typeSpacing',typography:'typeTypography',forbidden_effect:'typeForbiddenEffect',layout:'typeLayout'};
                const typeLabel = t => _s(typeI18n[t] || '') || t || '';
                let detailsHtml = `<div class="mode3-summary-note">${_s('summaryLabel')}: ${hs.total||0} ${_s('rules')} — ${hs.pass||0} ${_s('pass')}, ${hs.fail||0} ${_s('fail')}, ${hs.unclear||0} ${_s('unclear')}. ${issuesOnly.length ? _s('showOnlyIssues') : ''}</div>`;
                if (issuesOnly.length === 0) {
                    detailsHtml += `<div class="mode3-all-passed">${_s('allPassed')}</div>`;
                } else {
                    const byType = {};
                    issuesOnly.forEach(r => {
                        const t = r.type || 'other';
                        if (!byType[t]) byType[t] = [];
                        byType[t].push(r);
                    });
                    const typeOrder = ['color','dimension','spacing','typography','forbidden_effect','layout'];
                    const orderedTypes = typeOrder.filter(t => byType[t]).concat(Object.keys(byType).filter(t => !typeOrder.includes(t)));
                    let tbody = '';
                    orderedTypes.forEach(type => {
                        tbody += `<tr class="mode3-type-header"><td colspan="5"><strong>${typeLabel(type)}</strong></td></tr>`;
                        byType[type].forEach(r => {
                            const localStatus = statusMap[r.status] || r.status;
                            const localSev = sevMap[r.severity] || r.severity || '';
                            const localReason = (r.reason||'').replace(/: found$/, `: ${_s('found')}`).replace(/: not found$/, `: ${_s('notFound')}`);
                            tbody += `<tr class="mode3-row-${r.status}">
                        <td title="${r.title||''}">${r.title || r.rule_id || '?'}</td>
                        <td>${typeLabel(type)}</td>
                        <td><span class="mode3-sev mode3-sev-${r.severity||'major'}">${localSev}</span></td>
                        <td><span class="mode3-status-chip mode3-status-${r.status}">${statusIcon[r.status]||'?'} ${localStatus}</span></td>
                        <td class="mode3-reason">${localReason}</td>
                    </tr>`;
                        });
                    });
                    detailsHtml += `<table class="mode3-rule-table">
                    <thead><tr><th>${_s('colRule')}</th><th>${_s('colType')}</th><th>${_s('colSeverity')}</th><th>${_s('colStatus')}</th><th>${_s('colReason')}</th></tr></thead>
                    <tbody>${tbody}</tbody>
                </table>`;
                }
                hardDetails.innerHTML = detailsHtml;
            } else {
                hardDetails.innerHTML = `<div class="mode3-empty-msg">${_s('noHardResults')}</div>`;
            }

            const sd = data.soft_details || data.soft_summary || {};
            const softStatusLocal = statusMap[sd.status] || sd.status || 'n/a';
            let softHtml = `<div class="mode3-soft-header">${_s('softRulesLabel')}: <strong>${softStatusLocal}</strong> &mdash; ${_s('scoreLabel')}: <strong>${(sd.score??0).toFixed(2)}</strong></div>`;
            if (sd.reason) softHtml += `<div class="mode3-soft-reason">${sd.reason}</div>`;
            const issues = sd.issues || [];
            if (issues.length) softHtml += `<ul class="mode3-soft-issues">${issues.map(i=>`<li>${i}</li>`).join('')}</ul>`;
            softSummary.innerHTML = softHtml;

            const base = window.location.origin;
            const debugBase = base + (data.debug_base_url || '');
            const debug = data.debug || {};

            const overlayEl = $('#mode3-debugOverlays');
            if (overlayEl && debug.overlays && debug.overlays.length) {
                overlayEl.innerHTML = debug.overlays.map(o =>
                    `<div class="mode3-debug-img-wrap">
                        <img src="${debugBase}/${o.file}" alt="Page ${o.page}" loading="lazy">
                        <div class="mode3-debug-img-label">Page ${o.page} — ${o.zone_count} zones</div>
                    </div>`
                ).join('');
            } else if (overlayEl) {
                overlayEl.innerHTML = '<div class="mode3-debug-empty">No overlay</div>';
            }

            const yoloCropsEl = $('#mode3-debugYoloCrops');
            if (yoloCropsEl && debug.yolo_crops && debug.yolo_crops.length) {
                yoloCropsEl.innerHTML = debug.yolo_crops.map(yc =>
                    `<div class="mode3-crop-card mode3-crop-yolo">
                        <div class="mode3-crop-pair">
                            <img src="${debugBase}/${yc.crop_file}" alt="YOLO crop" loading="lazy">
                            <img src="${debugBase}/${yc.mask_file}" alt="seg mask" loading="lazy" class="mode3-crop-mask">
                        </div>
                        <div class="mode3-crop-label">
                            <span class="mode3-crop-class">YOLO bbox → ${yc.classes.length} classes</span>
                            <span class="mode3-crop-seg">${yc.classes.join(', ') || 'no class'}</span>
                        </div>
                    </div>`
                ).join('');
            } else if (yoloCropsEl) {
                yoloCropsEl.innerHTML = '<div class="mode3-debug-empty">No YOLO crops</div>';
            }

            const cropsEl = $('#mode3-debugCrops');
            if (cropsEl && debug.crops && debug.crops.length) {
                cropsEl.innerHTML = debug.crops.map(c => {
                    const maskImg = c.mask_file ? `<img src="${debugBase}/${c.mask_file}" alt="seg mask" loading="lazy" class="mode3-crop-mask">` : '';
                    const segInfo = c.seg_ratio != null ? `<span class="mode3-crop-seg">${c.class} (${(c.seg_ratio*100).toFixed(0)}%)</span>` : '';
                    return `<div class="mode3-crop-card">
                        <div class="mode3-crop-pair">
                            <img src="${debugBase}/${c.file}" alt="${c.class}" loading="lazy">
                            ${maskImg}
                        </div>
                        <div class="mode3-crop-label">
                            <span class="mode3-crop-class">${c.class}</span>
                            ${segInfo}
                            <span class="mode3-crop-dim">${c.width_mm?.toFixed(1) || '?'}×${c.height_mm?.toFixed(1) || '?'}mm</span>
                        </div>
                    </div>`;
                }).join('');
            } else if (cropsEl) {
                cropsEl.innerHTML = '<div class="mode3-debug-empty">No crops</div>';
            }

            const gptEl = $('#mode3-debugGptCrops');
            if (gptEl && debug.gpt_crops && debug.gpt_crops.length) {
                gptEl.innerHTML = debug.gpt_crops.map(g =>
                    `<div class="mode3-crop-card mode3-crop-gpt">
                        <img src="${debugBase}/${g.file}" alt="${g.rule_id}" loading="lazy">
                        <div class="mode3-crop-label">
                            <span class="mode3-crop-class">${g.rule_id}</span>
                            <span class="mode3-crop-dim">${g.rule_type}: ${g.rule_title}</span>
                        </div>
                    </div>`
                ).join('');
            } else if (gptEl) {
                gptEl.innerHTML = '<div class="mode3-debug-empty">No GPT crops</div>';
            }

            linkView.href = base + (data.view_url||'');
            linkDownload.href = base + (data.download_url||'');
            linkDownload.download = data.output_file || 'output_mode3_review.pdf';
            linkReport.href = base + (data.report_url||'');
            if (pdfViewer) pdfViewer.src = linkView.href;
            result.hidden = false;
        } catch (e) {
            $('#mode3-errorText').textContent = e.message || 'Có lỗi xảy ra';
            error.hidden = false;
            statusText.textContent = 'Lỗi';
        } finally { btnCheck.disabled = false; }
    });
}

// ============================================================
//  PDF A Handlers
// ============================================================
function preparePdfA(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showAlert($('#pdf-a-status'), 'error', t('uploadFirst'));
        return;
    }
    state.pdfAFile = file;
    state.pdfAName = file.name.replace('.pdf', '');
    state.pdfADisplayName = file.name;
    state.pdfAStoredAtDisplay = null;
    state.pdfAReady = false; // Not yet processed

    const statusEl = $('#pdf-a-status');
    const objectUrl = URL.createObjectURL(file);
    statusEl.innerHTML = `
        <div class="file-item">
            <i data-lucide="file"></i>
            <span class="file-name">${file.name}</span>
        </div>
        <div class="pdf-preview" style="height: 400px; margin-top: 10px;">
            <iframe src="${objectUrl}"></iframe>
        </div>
    `;
    if (window.lucide) lucide.createIcons();

    const actionsEl = $('#pdf-a-actions');
    actionsEl.innerHTML = `<button class="btn btn-secondary btn-sm" id="btn-change-a">${t('changeBtn')}</button>`;
    $('#btn-change-a').addEventListener('click', changePdfA);

    const runBtn = $('#btn-run-pipeline');
    if (runBtn) runBtn.disabled = state.pdfBFiles.length === 0;

    $('#pdf-a-mode').classList.add('hidden');
    $('#upload-a-section').classList.add('hidden');
}

function changePdfA() {
    state.pdfAName = null;
    state.pdfADisplayName = null;
    state.pdfAStoredAtDisplay = null;
    state.pdfAFile = null;
    state.pdfAReady = false;

    $('#pdf-a-status').innerHTML = '';
    $('#pdf-a-actions').innerHTML = '';
    $('#pdf-a-mode').classList.remove('hidden');
    $('#upload-a-section').classList.add('hidden');
    $('#existing-a-section').classList.remove('hidden');

    const fileAInput = $('#file-a-input');
    if (fileAInput) fileAInput.value = '';

    $('#btn-run-pipeline').disabled = true;
    loadProcessedPdfs();
    const modeExisting = document.getElementById('modeExisting');
    if (modeExisting) modeExisting.checked = true;
    if (window.lucide) lucide.createIcons();
}

async function handlePdfAUpload() {
    if (!state.pdfAFile) return;

    const statusEl = $('#pdf-a-status');
    statusEl.innerHTML = `<div class="alert alert-info"><span class="spinner"></span> Uploading PDF A...</div>`;

    const fd = new FormData();
    fd.append('file', state.pdfAFile);
    const res = await api('POST', 'upload-pdf-a', fd, false);

    state.sessionId = res.session_id;
    state.pdfAName = res.pdf_name;
    state.pdfADisplayName = res.display_name || `${res.pdf_name}.pdf`;
    state.pdfAStoredAtDisplay = res.stored_at_display || null;

    if (res.already_processed) {
        state.pdfAReady = true;
        const previewUrl = `/api/pdf-a/${res.pdf_name}`;
        showPdfAStatus(res, previewUrl);
        return true;
    } else {
        // Need to run step 1 and 2&3
        appendLog('PDF A uploaded, starting analysis...');
        await runStep1();
        await runStep23();
        // Since we now have results, show the status with a preview
        const previewUrl = `/api/pdf-a/${res.pdf_name}`;
        showPdfAStatus({ ...res, bbox_count: state.bboxCount, id_count: state.idCount }, previewUrl);
        return true;
    }
}

async function selectExistingPdf(name) {
    const statusEl = $('#pdf-a-status');
    statusEl.innerHTML = `<div class="alert alert-info"><span class="spinner"></span> Loading...</div>`;

    try {
        const res = await api('POST', 'select-pdf-a', { name });
        state.sessionId = res.session_id;
        state.pdfAName = res.pdf_name;
        state.pdfADisplayName = res.display_name || `${res.pdf_name}.pdf`;
        state.pdfAStoredAtDisplay = res.stored_at_display || null;
        state.pdfAReady = true;
        state.pdfAFile = null;

        // Highlight selected
        $$('.pdf-list-item').forEach(el => el.classList.remove('selected'));
        const selectedEl = document.querySelector(`.pdf-list-item[data-name="${name}"]`);
        if (selectedEl) selectedEl.classList.add('selected');

        const previewUrl = `/api/pdf-a/${res.pdf_name}`;
        showPdfAStatus(res, previewUrl);
        
        // Enable run button if PDF B files are also present
        const runBtn = $('#btn-run-pipeline');
        if (runBtn) runBtn.disabled = state.pdfBFiles.length === 0;

        // Thu gá»n khu vá»±c chá»n/existing
        $('#pdf-a-mode').classList.add('hidden');
        $('#existing-a-section').classList.add('hidden');
    } catch (err) {
        statusEl.innerHTML = '';
        showAlert(statusEl, 'error', err.message);
    }
}

function showPdfAStatus(data, previewUrl = null) {
    const statusEl = $('#pdf-a-status');
    const info = data || {};
    if (info.display_name) {
        state.pdfADisplayName = info.display_name;
    }
    if (Object.prototype.hasOwnProperty.call(info, 'stored_at_display')) {
        state.pdfAStoredAtDisplay = info.stored_at_display || null;
    }
    let finalUrl = previewUrl;
    if (!finalUrl && state.pdfAFile) {
        finalUrl = URL.createObjectURL(state.pdfAFile);
    } else if (!finalUrl && state.pdfAName) {
        finalUrl = `/api/pdf-a/${state.pdfAName}`;
    }

    const visibleName = state.pdfADisplayName || `${state.pdfAName}.pdf`;
    const savedAtLine = state.pdfAStoredAtDisplay
        ? `<div class="file-meta-muted">Lưu lúc: ${escapeHtml(state.pdfAStoredAtDisplay)}</div>`
        : '';

    statusEl.innerHTML = `
        <div class="alert alert-success">
            <i data-lucide="check"></i>
            <div>
                <strong>${t('pdfAReady')}: ${escapeHtml(visibleName)}</strong>
                ${savedAtLine}
                <div style="margin-top: 5px;">
                    <span class="status-chip chip-success">${t('bboxCount', info.bbox_count || '...')}</span>
                    <span class="status-chip chip-info">${t('idCount', info.id_count || '...')}</span>
                </div>
            </div>
        </div>
        ${finalUrl ? `<div class="pdf-preview" style="height: 400px; margin-top: 10px;"><iframe src="${finalUrl}"></iframe></div>` : ''}
    `;
    if (window.lucide) lucide.createIcons();

    // Thu gá»n toĂ n bá»™ khu vá»±c chá»n PDF A
    $('#pdf-a-mode').classList.add('hidden');
    $('#upload-a-section').classList.add('hidden');
    $('#existing-a-section').classList.add('hidden');

    const actionsEl = $('#pdf-a-actions');
    actionsEl.innerHTML = `
        <button class="btn btn-secondary btn-sm" id="btn-change-a">${t('changeBtn')}</button>
        <button class="btn btn-danger btn-sm" id="btn-delete-a">${t('deleteBtn')}</button>
    `;
    $('#btn-change-a').addEventListener('click', changePdfA);
    $('#btn-delete-a').addEventListener('click', () => showDeleteDialog());
}

function showPdfAStepButtons(data) {
    const statusEl = $('#pdf-a-status');
    const visibleName = state.pdfADisplayName || `${state.pdfAName}.pdf`;
    statusEl.innerHTML = `
        <div class="file-item">
            <i data-lucide="file"></i>
            <span class="file-name">${escapeHtml(visibleName)}</span>
        </div>
    `;
    if (window.lucide) lucide.createIcons();

    const actionsEl = $('#pdf-a-actions');
    const needStep1 = !data.already_processed && data.bbox_count === 0;
    const needStep23 = data.bbox_count > 0 && data.id_count === 0;

    if (needStep1) {
        actionsEl.innerHTML = `<button class="btn btn-primary" id="btn-step1">${t('btnStep1')}</button>`;
        $('#btn-step1').addEventListener('click', runStep1);
    } else if (needStep23) {
        actionsEl.innerHTML = `<button class="btn btn-primary" id="btn-step23">${t('btnStep23')}</button>`;
        $('#btn-step23').addEventListener('click', runStep23);
    }
}

async function runStep1() {
    const btn = $('#btn-step1');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner"></span> Processing...`;
    }

    const statusEl = $('#pdf-a-status');

    try {
        const res = await api('POST', 'run-step1', {
            session_id: state.sessionId,
            pdf_name: state.pdfAName,
        });
        showAlert(statusEl, 'success', t('step1Done', res.total_bbox));

        // Now show step23 button if it's manual mode
        const actionsEl = $('#pdf-a-actions');
        if (actionsEl && !state.running) {
            actionsEl.innerHTML = `<button class="btn btn-primary" id="btn-step23">${t('btnStep23')}</button>`;
            $('#btn-step23').addEventListener('click', runStep23);
        }
    } catch (err) {
        if (statusEl) showAlert(statusEl, 'error', err.message);
        if (btn) {
            btn.disabled = false;
            btn.textContent = t('btnStep1');
        }
    }
}

async function runStep23() {
    const btn = $('#btn-step23');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner"></span> Processing...`;
    }

    const statusEl = $('#pdf-a-status');

    try {
        const res = await api('POST', 'run-step23', {
            session_id: state.sessionId,
            pdf_name: state.pdfAName,
        });
        if (statusEl) showAlert(statusEl, 'success', t('step23Done', res.step2_saved, res.step3_renamed));

        state.pdfAReady = true;
        const info = await api('GET', `pdf-info/${state.pdfAName}`);
        showPdfAStatus(info);
    } catch (err) {
        if (statusEl) showAlert(statusEl, 'error', err.message);
        if (btn) {
            btn.disabled = false;
            btn.textContent = t('btnStep23');
        }
    }
}

async function loadProcessedPdfs() {
    const container = $('#processed-list');
    container.innerHTML = `<div class="alert alert-info"><span class="spinner"></span> Loading...</div>`;

    try {
        const res = await api('GET', 'processed-pdfs');
        if (res.pdfs.length === 0) {
            container.innerHTML = `<div class="alert alert-warning">${t('noProcessed')}</div>`;
            return;
        }

        container.innerHTML = `<div class="pdf-list">${res.pdfs.map(p => `
            <div class="pdf-list-item" data-name="${p.name}">
                <i data-lucide="file-check" class="pdf-icon"></i>
                <div class="pdf-details">
                    <div class="pdf-name">${escapeHtml(p.display_name || `${p.name}.pdf`)}</div>
                    <div class="pdf-meta">${t('bboxCount', p.bbox_count)} · ${t('idCount', p.id_count)}</div>
                    ${p.stored_at_display ? `<div class="file-meta-muted">Lưu lúc: ${escapeHtml(p.stored_at_display)}</div>` : ''}
                </div>
                <span class="status-chip ${p.has_bbox && p.has_id ? 'chip-success' : 'chip-warning'}">
                    <i data-lucide="${p.has_bbox && p.has_id ? 'check' : 'alert-circle'}"></i>
                </span>
            </div>
        `).join('')}</div>`;

        if (window.lucide) lucide.createIcons();

        $$('.pdf-list-item').forEach(item => {
            item.addEventListener('click', () => selectExistingPdf(item.dataset.name));
        });
    } catch (err) {
        container.innerHTML = `<div class="alert alert-error">${err.message}</div>`;
    }
}

// ============================================================
//  PDF B Handlers
// ============================================================
async function handlePdfBUpload(files, append = false) {
    const newFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    state.pdfBFiles = append ? [...state.pdfBFiles, ...newFiles] : newFiles;
    sharedProduitFiniFile = state.pdfBFiles[0] || null;
    renderPdfBList();
    const fbInput = $('#file-b-input');
    if (fbInput) fbInput.value = ''; // Reset để chọn lại file cũ được

    const btnRun = $('#btn-run-pipeline');
    if (btnRun) {
        btnRun.disabled = state.pdfBFiles.length === 0 || (!state.pdfAName && !state.pdfAFile);
    }
}

function renderPdfBList() {
    const container = $('#file-b-list');
    if (state.pdfBFiles.length === 0) {
        container.innerHTML = '';
        const fb = $('#file-b-input');
        if (fb) fb.value = ''; // Reset input Mode 1 để chọn lại file cũ được
        $('#dropzone-b').classList.remove('hidden');
        return;
    }
    $('#dropzone-b').classList.add('hidden');
    const addBtn = `<div style="margin-bottom:12px"><button type="button" class="btn btn-secondary btn-sm" id="btn-add-pdfb"><i data-lucide="plus"></i> ${t('addMoreFiles')}</button></div>`;
    container.innerHTML = addBtn + state.pdfBFiles.map((f, i) => {
        const objectUrl = URL.createObjectURL(f);
        return `
            <div class="file-item-wrapper" style="margin-bottom: 20px;">
                <div class="file-item">
                    <i data-lucide="file-text"></i>
                    <span class="file-name">${f.name}</span>
                    <span class="file-remove" data-idx="${i}"><i data-lucide="x"></i></span>
                </div>
                <div class="pdf-preview" style="height: 300px; margin-top: 8px;">
                    <iframe src="${objectUrl}"></iframe>
                </div>
            </div>
        `;
    }).join('');

    if (window.lucide) lucide.createIcons();

    const btnAdd = $('#btn-add-pdfb');
    if (btnAdd) btnAdd.addEventListener('click', () => { state.appendPdfB = true; $('#file-b-input')?.click(); });

    $$('.file-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.dataset.idx, 10);
            state.pdfBFiles.splice(idx, 1);
            if (state.pdfBFiles.length === 0) {
                clearProduitFiniEverywhere();
                return;
            }
            sharedProduitFiniFile = state.pdfBFiles[0];
            const m2Final = $('#mode2-finalPdf'), m3Final = $('#mode3-finalPdf');
            if (m2Final) { setFileToInput(m2Final, sharedProduitFiniFile); m2Final.dispatchEvent(new Event('change')); }
            if (m3Final) { setFileToInput(m3Final, sharedProduitFiniFile); m3Final.dispatchEvent(new Event('change')); }
            renderPdfBList();
            const runBtn = $('#btn-run-pipeline');
            if (runBtn) runBtn.disabled = state.pdfBFiles.length === 0 || (!state.pdfAName && !state.pdfAFile);
            if (window.lucide) lucide.createIcons();
        });
    });
}



// ============================================================
//  Pipeline Execution
// ============================================================
async function runPipeline() {
    if (state.pdfBFiles.length === 0 || state.running) return;
    if (!state.pdfAName && !state.pdfAFile) return;

    state.running = true;

    const btn = $('#btn-run-pipeline');
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> ${t('processing')}`;

    const progressCard = $('#card-progress');
    if (progressCard) progressCard.classList.remove('hidden');

    const progressSection = $('#pipeline-progress');
    if (progressSection) progressSection.classList.remove('hidden');
    
    $('#progress-log').innerHTML = '';
    updateProgress(0, t('processing'));

    try {
        // 1) Handle PDF A if not yet uploaded/processed
        if (!state.pdfAReady && state.pdfAFile) {
            appendLog('Starting PDF A preparation...');
            await handlePdfAUpload(); 
        }

        // 2) Upload PDF B files
        appendLog('Uploading PDF B files...');
        const uploadFd = new FormData();
        uploadFd.append('session_id', state.sessionId);
        state.pdfBFiles.forEach(f => uploadFd.append('files', f));
        await api('POST', 'upload-pdf-b', uploadFd, false);

        // 3) Start SSE listener for final steps
        const sseUrl = `/api/stream-progress/${state.sessionId}`;
        const eventSource = new EventSource(sseUrl);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                appendLog(data.detail, data.status);
                const progressLabel = data.step === 'complete' ? t('completed') : data.detail;
                updateProgress(data.progress, progressLabel);

                if (data.step === 'complete') {
                    eventSource.close();
                    showResults(data.data?.results || []);
                    state.running = false;
                    btn.disabled = false;
                    btn.textContent = t('btnRunPipeline');
                }
            } catch (e) { /* ignore parse errors */ }
        };

        eventSource.onerror = () => {
            eventSource.close();
        };

        // 4) Trigger comparison pipeline (Step 4-7)
        // Gemini đã bị ẩn/disable (chưa có nhu cầu) -> mặc định luôn dùng OpenAI GPT nếu chưa check
        const aiModel = document.querySelector('input[name="aiModel"]:checked')?.value || 'OpenAI GPT';
        appendLog(`Starting comparison using ${aiModel}...`);
        await api('POST', 'run-pipeline', {
            session_id: state.sessionId,
            pdf_a_name: state.pdfAName,
            ai_model: aiModel,
        });

    } catch (err) {
        showAlert($('#card-step3'), 'error', err.message);
        state.running = false;
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="play"></i> ${t('btnRunPipeline')}`;
        if (window.lucide) lucide.createIcons();
    }
}

function updateProgress(pct, label) {
    const fill = $('#progress-fill');
    const pctLabel = $('#progress-pct');
    const textLabel = $('#progress-label');
    const titleLabel = $('#progress-title');
    if (fill) fill.style.width = `${pct}%`;
    if (pctLabel) pctLabel.textContent = `${pct}%`;
    if (textLabel) textLabel.textContent = label;
    if (titleLabel) titleLabel.textContent = label;
}

function appendLog(msg, status) {
    const log = $('#progress-log');
    if (!log || !msg) return;
    const line = document.createElement('div');
    line.className = `log-line log-${status || 'running'}`;
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
}

// ============================================================
//  Results
// ============================================================
function showResults(results) {
    const card = $('#card-results');
    if (card) card.classList.remove('hidden');

    const progressCard = $('#card-progress');
    // We can keep progress card visible or hide it. Let's keep it for logs.

    const container = $('#results-container');
    if (!results || results.length === 0) {
        container.innerHTML = `<div class="alert alert-warning">No results</div>`;
        return;
    }

    container.innerHTML = results.map(r => {
        if (r.status === 'success') {
            const fileId = r.output_file.replace(/[^a-zA-Z0-9]/g, '_');
            return `
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">
                            <i data-lucide="file-check-2"></i> ${r.file_name}
                        </div>
                        <span class="status-chip chip-success"><i data-lucide="check-circle"></i> ${t('success')}</span>
                    </div>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${r.step4?.step1 || 0}</div>
                            <div class="metric-label">${t('metricBbox')}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${r.step5?.matched || 0}</div>
                            <div class="metric-label">${t('metricMatched')}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${r.step6?.comparisons || 0}</div>
                            <div class="metric-label">${t('metricCompared')}</div>
                        </div>
                        <div class="metric-card ${r.step7?.highlighted === 0 ? 'metric-warning' : ''}">
                            <div class="metric-value">${r.step7?.highlighted || 0}</div>
                            <div class="metric-label">${t('metricHighlight')}</div>
                        </div>
                    </div>
                    <div class="btn-group">
                        <a class="btn btn-primary btn-sm" href="/api/download/${state.sessionId}/${encodeURIComponent(r.output_file)}" download>
                            <i data-lucide="download"></i> ${t('downloadBtn')}
                        </a>
                    </div>
                </div>
                <div class="pdf-preview" id="preview-${fileId}">
                    <iframe src="/api/view/${state.sessionId}/${encodeURIComponent(r.output_file)}">
                        <p>${t('noIframeSupport')} <a href="/api/download/${state.sessionId}/${encodeURIComponent(r.output_file)}">${t('downloadBtn')}</a></p>
                    </iframe>
                </div>
            `;
        } else {
            const chipClass = r.status === 'no_match' ? 'chip-warning' : 'chip-error';
            const icon = r.status === 'no_match' ? 'alert-circle' : 'x-octagon';
            const label = r.status === 'no_match' ? t('noMatch') : (r.error || t('failed'));
            return `
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title"><i data-lucide="file-warning"></i> ${r.file_name}</div>
                        <span class="status-chip ${chipClass}"><i data-lucide="${icon}"></i> ${label}</span>
                    </div>
                </div>
            `;
        }
    }).join('');

    if (window.lucide) lucide.createIcons();

    // Preview buttons
    $$('.preview-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const previewId = btn.dataset.fileId;
            const preview = document.getElementById(previewId);
            if (preview) preview.classList.toggle('hidden');
        });
    });
}

// ============================================================
//  Delete Dialog
// ============================================================
function showDeleteDialog() {
    const overlay = $('#dialog-overlay');
    overlay.classList.remove('hidden');
    overlay.innerHTML = `
        <div class="dialog-box">
            <div class="dialog-title"><i data-lucide="alert-triangle" class="icon-danger"></i> ${t('deleteConfirm')}</div>
            <div class="dialog-body">
                <strong>${state.pdfAName}</strong><br>
                ${t('deleteWarning')}
            </div>
            <div class="dialog-actions">
                <button class="btn btn-secondary btn-sm" id="dialog-cancel">${t('cancelBtn')}</button>
                <button class="btn btn-danger btn-sm" id="dialog-confirm">${t('confirmBtn')}</button>
            </div>
        </div>
    `;
    if (window.lucide) lucide.createIcons();

    $('#dialog-cancel').addEventListener('click', () => overlay.classList.add('hidden'));
    $('#dialog-confirm').addEventListener('click', async () => {
        try {
            await api('DELETE', `delete-pdf-a/${state.pdfAName}`);
            state.pdfAReady = false;
            state.pdfAName = null;
            overlay.classList.add('hidden');
            renderApp();
        } catch (err) {
            showAlert($('#pdf-a-status'), 'error', err.message);
            overlay.classList.add('hidden');
        }
    });
}

// ============================================================
//  Init
// ============================================================
async function init() {
    try {
        renderApp();
    } catch (e) {
        document.body.innerHTML = '<p style="padding:20px;color:red">Lỗi: ' + (e.message || e) + '</p>';
        console.error(e);
        return;
    }

    // Health check
    try {
        const health = await api('GET', 'health');
        if (!health.model_found) {
            showAlert($('#card-step1'), 'error', t('modelNotFound'));
        }
    } catch (err) {
        showAlert($('#card-step1'), 'error', 'Cannot connect to server: ' + err.message);
    }
}

document.addEventListener('DOMContentLoaded', init);



