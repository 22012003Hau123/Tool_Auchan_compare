/**
 * Mode 2: So sánh PDF Annotations - Frontend
 * Gọi POST /api/mode2/compare, hiển thị summary + link xem/tải.
 */

(function () {
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
        statusText.textContent = 'Đang xử lý...';
        btnCompare.disabled = true;

        const formData = new FormData();
        formData.append('ref_pdf', refPdf.files[0]);
        formData.append('final_pdf', finalPdf.files[0]);

        try {
            const res = await fetch('/api/mode2/compare', {
                method: 'POST',
                body: formData,
            });

            const data = await res.json().catch(() => ({}));

            if (!res.ok) {
                throw new Error(data.detail || res.statusText || 'Lỗi không xác định');
            }

            statusText.textContent = 'Hoàn thành';

            // Summary
            const s = data.summary || {};
            summary.innerHTML = [
                `Tổng annotations: ${s.total_annotations || 0}`,
                `Đã thực hiện: ${s.implemented || 0}`,
                `Chưa thực hiện: ${s.not_implemented || 0}`,
                `Một phần: ${s.partial || 0}`,
                `Không rõ: ${s.unclear || 0}`,
            ].map((t) => `<div>${t}</div>`).join('');

            // Links
            const base = window.location.origin;
            const viewUrl = base + (data.view_url || '');
            linkView.href = viewUrl;
            linkView.target = '_blank';
            linkDownload.href = base + (data.download_url || '');
            linkDownload.download = data.output_file || 'output_mode2_diff.pdf';

            // Nhúng PDF vào iframe để xem trực tiếp
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
