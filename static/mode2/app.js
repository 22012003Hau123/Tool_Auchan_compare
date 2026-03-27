(function () {
    const lang = () => localStorage.getItem('pdf_tool_lang') || 'vi';
    const t = (key) => (window.I18N_SHARED && I18N_SHARED[lang()] ? I18N_SHARED[lang()][key] : key) || key;

    function applyI18n() {
        const L = lang();
        const i = window.I18N_SHARED && I18N_SHARED[L] ? I18N_SHARED[L] : {};
        document.getElementById('header-title') && (document.getElementById('header-title').textContent = i.appTitle || 'Công cụ so sánh PDF');
        document.getElementById('header-desc') && (document.getElementById('header-desc').textContent = i.mode2Desc || '');
        document.getElementById('label-ref') && (document.getElementById('label-ref').textContent = i.labelRefPdf || '');
        document.getElementById('label-final') && (document.getElementById('label-final').textContent = i.labelFinalPdf || '');
        document.getElementById('btnCompare') && (document.getElementById('btnCompare').textContent = i.btnCompare || '');
        document.getElementById('results-title') && (document.getElementById('results-title').textContent = i.resultsTitle || '');
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

    const refPdf = document.getElementById('refPdf');
    const finalPdf = document.getElementById('finalPdf');
    const btnCompare = document.getElementById('btnCompare');
    const status = document.getElementById('status');
    const statusText = document.getElementById('statusText');
    const result = document.getElementById('result');
    const summary = document.getElementById('summary');
    const linkView = document.getElementById('linkView');
    const linkDownload = document.getElementById('linkDownload');
    const error = document.getElementById('error');
    const errorText = document.getElementById('errorText');

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

    function enableCompare() {
        btnCompare.disabled = !(refPdf.files?.length && finalPdf.files?.length);
    }

    refPdf.addEventListener('change', () => {
        showPreview(refPdf, 'previewRef');
        enableCompare();
    });

    finalPdf.addEventListener('change', () => {
        showPreview(finalPdf, 'previewFinal');
        enableCompare();
    });

    btnCompare.addEventListener('click', async () => {
        if (!refPdf.files?.length || !finalPdf.files?.length) return;

        result.hidden = true;
        error.hidden = true;
        status.hidden = false;
        statusText.textContent = t('processing');
        btnCompare.disabled = true;

        const formData = new FormData();
        formData.append('ref_pdf', refPdf.files[0]);
        formData.append('final_pdf', finalPdf.files[0]);

        try {
            const res = await fetch('/api/mode2/compare', { method: 'POST', body: formData });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || res.statusText || 'Lỗi không xác định');

            statusText.textContent = t('done');
            const s = data.summary || {};
            summary.innerHTML = [
                `Tổng annotations: ${s.total_annotations || 0}`,
                `Đã thực hiện: ${s.implemented || 0}`,
                `Chưa thực hiện: ${s.not_implemented || 0}`,
                `Một phần: ${s.partial || 0}`,
                `Không rõ: ${s.unclear || 0}`,
            ].map((t) => `<div>${t}</div>`).join('');

            const base = window.location.origin;
            const viewUrl = base + (data.view_url || '');
            linkView.href = viewUrl;
            linkDownload.href = base + (data.download_url || '');
            linkDownload.download = data.output_file || 'output_mode2_diff.pdf';

            const pdfViewer = document.getElementById('pdfViewer');
            if (pdfViewer) pdfViewer.src = viewUrl;

            result.hidden = false;
        } catch (err) {
            errorText.textContent = err.message || 'Có lỗi xảy ra';
            error.hidden = false;
            statusText.textContent = 'Lỗi';
        } finally {
            btnCompare.disabled = false;
        }
    });
})();
