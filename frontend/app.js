/* app.js – Vietnamese RAG QA frontend logic */
"use strict";

// ─── Config ──────────────────────────────────────────────────────────────────
// In Docker: nginx proxies /api/ → backend:8000
// For local dev (no Docker): point directly to backend
const API_BASE = window.location.port === "3000"
  ? "/api"          // Docker / nginx proxy
  : "http://localhost:8000"; // Local dev

// ─── Helpers ─────────────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function scrollBottom() {
  const c = document.getElementById("chatMessages");
  c.scrollTop = c.scrollHeight;
}

// ─── System status ────────────────────────────────────────────────────────────
async function checkHealth() {
  const badge = document.getElementById("systemStatus");
  try {
    const data = await apiFetch("/health");
    badge.textContent = "🟢 Online";
    badge.className = "status-badge online";
    // Populate info tooltip
    const info = await apiFetch("/info");
    badge.title = `Model: ${info.embedding_model} | LLM: ${info.llm_provider} | Chunks: ${info.collection?.count ?? 0}`;
  } catch {
    badge.textContent = "🔴 Offline";
    badge.className = "status-badge offline";
    badge.title = "Backend không khả dụng";
  }
}

// ─── Documents ───────────────────────────────────────────────────────────────
async function loadDocuments() {
  const list = document.getElementById("docList");
  const filterSel = document.getElementById("filterSource");
  try {
    const data = await apiFetch("/documents");

    // Fill doc list
    if (data.files.length === 0) {
      list.innerHTML = '<span class="placeholder">Không có file PDF/DOCX nào</span>';
    } else {
      list.innerHTML = data.files.map(f => `
        <div class="doc-item">
          📄 <span>${escapeHtml(f.name)}</span>
          <span class="size">${f.size_kb} KB</span>
        </div>`).join("");
    }

    // Fill filter dropdown
    filterSel.innerHTML = '<option value="">Tất cả tài liệu</option>' +
      data.files.map(f => `<option value="${escapeHtml(f.name)}">${escapeHtml(f.name)}</option>`).join("");
  } catch (e) {
    list.innerHTML = `<span class="placeholder">Lỗi: ${escapeHtml(e.message)}</span>`;
  }
}

// ─── Ingestion ────────────────────────────────────────────────────────────────
async function ingest(recreate = false) {
  const statusEl = document.getElementById("ingestStatus");
  const btnIngest = document.getElementById("btnIngest");
  const btnRec = document.getElementById("btnIngestRecreate");
  [btnIngest, btnRec].forEach(b => b.disabled = true);
  statusEl.innerHTML = '<span class="spinner"></span> Đang nhập tài liệu…';
  try {
    const data = await apiFetch(`/ingest?recreate=${recreate}`, { method: "POST" });
    statusEl.textContent = `✅ Đã nhập ${data.chunks_ingested} chunks từ ${data.files_processed.length} file`;
    await checkHealth();
  } catch (e) {
    statusEl.textContent = `❌ Lỗi: ${e.message}`;
  } finally {
    [btnIngest, btnRec].forEach(b => b.disabled = false);
  }
}

// ─── Chat message builders ────────────────────────────────────────────────────
function appendUserMsg(text) {
  const msgs = document.getElementById("chatMessages");
  msgs.innerHTML += `
    <div class="msg-user">
      <div class="msg-label">Câu hỏi</div>
      <div class="bubble">${escapeHtml(text)}</div>
    </div>`;
  scrollBottom();
}

function appendThinking() {
  const msgs = document.getElementById("chatMessages");
  const id = "thinking_" + Date.now();
  msgs.innerHTML += `
    <div class="msg-assistant" id="${id}">
      <div class="msg-label">Hệ thống</div>
      <div class="bubble"><span class="spinner"></span> Đang tìm kiếm và sinh câu trả lời…</div>
    </div>`;
  scrollBottom();
  return id;
}

function removeEl(id) {
  document.getElementById(id)?.remove();
}

function buildSourcesHtml(retrieved) {
  if (!retrieved || retrieved.length === 0) return "";
  const chips = retrieved.map(r => `
    <span class="source-chip">
      📄 ${escapeHtml(r.source)}
      <span class="score">${(r.score * 100).toFixed(0)}%</span>
    </span>`).join("");
  return `<div class="sources">
    <div class="sources-title">📎 Nguồn tham khảo (${retrieved.length} đoạn)</div>
    ${chips}
  </div>`;
}

function appendRagAnswer(data) {
  const msgs = document.getElementById("chatMessages");
  msgs.innerHTML += `
    <div class="msg-assistant">
      <div class="msg-label">🔍 RAG – Câu trả lời từ tài liệu</div>
      <div class="bubble markdown-body">
        ${marked.parse(data.answer)}
        ${buildSourcesHtml(data.retrieved)}
      </div>
    </div>`;
  scrollBottom();
}

function appendNoRagAnswer(data) {
  const msgs = document.getElementById("chatMessages");
  msgs.innerHTML += `
    <div class="msg-assistant">
      <div class="msg-label">🤖 LLM – Câu trả lời không có tìm kiếm</div>
      <div class="bubble markdown-body">${marked.parse(data.answer)}</div>
    </div>`;
  scrollBottom();
}

function appendCompareAnswer(data) {
  const msgs = document.getElementById("chatMessages");
  msgs.innerHTML += `
    <div class="msg-compare">
      <div class="msg-label">⚖️ So sánh RAG vs Không RAG</div>
      <div class="compare-grid">
        <div class="compare-col rag">
          <div class="col-label">✅ RAG – Có tìm kiếm tài liệu</div>
          <div class="bubble markdown-body">
            ${marked.parse(data.rag.answer)}
            ${buildSourcesHtml(data.rag.retrieved)}
          </div>
        </div>
        <div class="compare-col no-rag">
          <div class="col-label">⚠️ Không RAG – LLM thuần</div>
          <div class="bubble markdown-body">${marked.parse(data.no_rag.answer)}</div>
        </div>
      </div>
    </div>`;
  scrollBottom();
}

function appendError(msg) {
  const msgs = document.getElementById("chatMessages");
  msgs.innerHTML += `
    <div class="msg-assistant">
      <div class="msg-label">Lỗi</div>
      <div class="bubble" style="color:#fca5a5">❌ ${escapeHtml(msg)}</div>
    </div>`;
  scrollBottom();
}

// ─── Query form submit ────────────────────────────────────────────────────────
document.getElementById("queryForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("questionInput");
  const question = input.value.trim();
  if (!question) return;

  const mode = document.getElementById("queryMode").value;
  const topK = parseInt(document.getElementById("topK").value) || 5;
  const filterSource = document.getElementById("filterSource").value || null;
  const btnAsk = document.getElementById("btnAsk");

  // Remove welcome message if present
  document.querySelector(".welcome-msg")?.remove();

  input.value = "";
  input.style.height = "auto";
  btnAsk.disabled = true;

  appendUserMsg(question);
  const thinkId = appendThinking();

  try {
    let endpoint, body;
    if (mode === "rag") {
      endpoint = "/query";
      body = { question, top_k: topK, filter_source: filterSource };
    } else if (mode === "no-rag") {
      endpoint = "/query/no-rag";
      body = { question };
    } else {
      endpoint = "/compare";
      body = { question, top_k: topK };
    }

    const data = await apiFetch(endpoint, {
      method: "POST",
      body: JSON.stringify(body),
    });

    removeEl(thinkId);

    if (mode === "rag") appendRagAnswer(data);
    else if (mode === "no-rag") appendNoRagAnswer(data);
    else appendCompareAnswer(data);

  } catch (err) {
    removeEl(thinkId);
    appendError(err.message);
  } finally {
    btnAsk.disabled = false;
    input.focus();
  }
});

// Auto-resize textarea
document.getElementById("questionInput").addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 160) + "px";
});

// Ctrl+Enter to submit
document.getElementById("questionInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    document.getElementById("queryForm").requestSubmit();
  }
});

// ─── Evaluation ───────────────────────────────────────────────────────────────
document.getElementById("btnEvaluate").addEventListener("click", async () => {
  const statusEl = document.getElementById("evalStatus");
  const btn = document.getElementById("btnEvaluate");
  btn.disabled = true;
  statusEl.innerHTML = '<span class="spinner"></span>';

  try {
    const data = await apiFetch("/evaluate", { method: "POST", body: JSON.stringify({}) });
    statusEl.textContent = `✅ ${data.total} câu hỏi`;
    renderEvalModal(data);
  } catch (e) {
    statusEl.textContent = `❌ ${e.message}`;
  } finally {
    btn.disabled = false;
  }
});

function renderEvalModal(data) {
  const modal = document.getElementById("evalModal");
  const results = document.getElementById("evalResults");
  const { aggregate, per_question, total } = data;

  const metricLabel = { exact_match: "Exact Match", token_f1: "Token F1", rouge_l: "ROUGE-L", grounding: "Grounding" };

  const aggHtml = `
    <div class="eval-aggregate">
      <div class="eval-card">
        <h3>✅ RAG (có tìm kiếm)</h3>
        ${Object.entries(aggregate.rag).map(([k, v]) =>
          `<div class="metric-row"><span>${metricLabel[k] || k}</span><span class="metric-value">${(v * 100).toFixed(1)}%</span></div>`
        ).join("")}
      </div>
      <div class="eval-card">
        <h3>⚠️ Không RAG (LLM thuần)</h3>
        ${Object.entries(aggregate.no_rag).map(([k, v]) =>
          `<div class="metric-row"><span>${metricLabel[k] || k}</span><span class="metric-value">${(v * 100).toFixed(1)}%</span></div>`
        ).join("")}
      </div>
    </div>`;

  const rowsHtml = per_question.map((item, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${escapeHtml(item.question.substring(0, 80))}</td>
      <td>${(item.rag_metrics.rouge_l * 100).toFixed(1)}%</td>
      <td>${(item.rag_metrics.grounding * 100).toFixed(1)}%</td>
      <td>${(item.no_rag_metrics.rouge_l * 100).toFixed(1)}%</td>
      <td>${(item.no_rag_metrics.grounding * 100).toFixed(1)}%</td>
    </tr>`).join("");

  results.innerHTML = aggHtml + `
    <h3 style="margin:12px 0 8px;">Chi tiết từng câu (${total} câu)</h3>
    <table class="eval-table">
      <thead><tr>
        <th>#</th><th>Câu hỏi</th>
        <th>RAG ROUGE-L</th><th>RAG Grounding</th>
        <th>No-RAG ROUGE-L</th><th>No-RAG Grounding</th>
      </tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>`;

  modal.classList.remove("hidden");
}

document.getElementById("evalModalClose").addEventListener("click", () => {
  document.getElementById("evalModal").classList.add("hidden");
});
document.getElementById("evalModal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) e.currentTarget.classList.add("hidden");
});

// ─── Sidebar button handlers ──────────────────────────────────────────────────
document.getElementById("btnRefreshDocs").addEventListener("click", loadDocuments);
document.getElementById("btnIngest").addEventListener("click", () => ingest(false));
document.getElementById("btnIngestRecreate").addEventListener("click", () => {
  if (confirm("Xóa toàn bộ dữ liệu vector và nhập lại? Hành động này không thể hoàn tác.")) {
    ingest(true);
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
(async function init() {
  await checkHealth();
  await loadDocuments();
})();
