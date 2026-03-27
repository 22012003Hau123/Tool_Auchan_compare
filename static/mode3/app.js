(function () {
    const lang = () => localStorage.getItem('pdf_tool_lang') || 'vi';
    const t = (key) => (window.I18N_SHARED && I18N_SHARED[lang()] ? I18N_SHARED[lang()][key] : key) || key;

    function applyI18n() {
        const L = lang();
        const i = window.I18N_SHARED && I18N_SHARED[L] ? I18N_SHARED[L] : {};
        document.getElementById('header-title') && (document.getElementById('header-title').textContent = i.appTitle || 'Công cụ so sánh PDF');
        document.getElementById('header-desc') && (document.getElementById('header-desc').textContent = i.mode3Desc || '');
        document.getElementById('label-produit') && (document.getElementById('label-produit').textContent = i.labelProduitFini || '');
        document.getElementById('btnCheck') && (document.getElementById('btnCheck').textContent = i.btnCheck || '');
        document.getElementById('results-title') && (document.getElementById('results-title').textContent = i.resultsTitle3 || '');
        document.getElementById('view-pdf') && (document.getElementById('view-pdf').textContent = i.viewPdf || '');
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === L);
        });
    }

    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const html = document.documentElement;
            const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('pdf_tool_theme', next);
        });
    }

    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            localStorage.setItem('pdf_tool_lang', btn.dataset.lang);
            applyI18n();
        });
    });
    applyI18n();

    const finalPdf = document.getElementById('finalPdf');
    const btn = document.getElementById('btnCheck');
    const status = document.getElementById('status');
    const statusText = document.getElementById('statusText');
    const result = document.getElementById('result');
    const error = document.getElementById('error');
    const errorText = document.getElementById('errorText');

    const charteInfo = document.getElementById('charteInfo');
    const hardSummary = document.getElementById('hardSummary');
    const softSummary = document.getElementById('softSummary');
    const linkView = document.getElementById('linkView');
    const linkDownload = document.getElementById('linkDownload');
    const linkReport = document.getElementById('linkReport');
    const pdfViewer = document.getElementById('pdfViewer');

    function showPreview(input, iframeId) {
        const iframe = document.getElementById(iframeId);
        if (!iframe) return;
        if (input.files?.length) {
            if (iframe._prevBlobUrl) URL.revokeObjectURL(iframe._prevBlobUrl);
            const url = URL.createObjectURL(input.files[0]);
            iframe._prevBlobUrl = url;
            iframe.src = url;
            iframe.style.display = 'block';
        } else {
            if (iframe._prevBlobUrl) {
                URL.revokeObjectURL(iframe._prevBlobUrl);
                iframe._prevBlobUrl = null;
            }
            iframe.src = 'about:blank';
            iframe.style.display = 'none';
        }
    }

    function checkEnabled() {
        btn.disabled = !(finalPdf.files?.length);
    }

    finalPdf.addEventListener('change', () => {
        showPreview(finalPdf, 'previewFinal');
        checkEnabled();
    });

    btn.addEventListener('click', async () => {
        result.hidden = true;
        error.hidden = true;
        status.hidden = false;
        statusText.textContent = t('processing3');
        btn.disabled = true;

        const fd = new FormData();
        fd.append('final_pdf', finalPdf.files[0]);

        try {
            const res = await fetch('/api/mode3/check', { method: 'POST', body: fd });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || res.statusText || 'Lỗi không xác định');

            statusText.textContent = t('done');
            charteInfo.innerHTML = `<div>Charte source: ${data.charte_source || 'fixed-default'}</div>`;

            const hs = data.hard_summary || {};
            hardSummary.innerHTML = [
                `Hard rules total: ${hs.total || 0}`,
                `Pass: ${hs.pass || 0}`,
                `Fail: ${hs.fail || 0}`,
                `Unclear: ${hs.unclear || 0}`,
            ].map((x) => `<div>${x}</div>`).join('');

            const ss = data.soft_summary || {};
            softSummary.innerHTML = [
                `Soft status: ${ss.status || 'n/a'}`,
                `Soft score: ${ss.score ?? 0}`,
            ].map((x) => `<div>${x}</div>`).join('');

            const base = window.location.origin;
            linkView.href = base + (data.view_url || '');
            linkDownload.href = base + (data.download_url || '');
            linkDownload.download = data.output_file || 'output_mode3_review.pdf';
            linkReport.href = base + (data.report_url || '');
            pdfViewer.src = linkView.href;

            result.hidden = false;
        } catch (e) {
            errorText.textContent = e.message || 'Có lỗi xảy ra';
            error.hidden = false;
            statusText.textContent = 'Lỗi';
        } finally {
            btn.disabled = false;
        }
    });
})();
