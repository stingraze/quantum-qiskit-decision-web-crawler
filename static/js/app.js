/* =========================================================
   QuantumCrawler UI – app.js
   ========================================================= */

'use strict';

/* --------------------------------------------------------
   Utility helpers
   -------------------------------------------------------- */
function $(id) { return document.getElementById(id); }

function elems(sel, ctx) { return Array.from((ctx || document).querySelectorAll(sel)); }

function fmtDuration(seconds) {
  if (seconds == null || isNaN(seconds)) return '—';
  if (seconds < 60) return seconds.toFixed(1) + 's';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function fmtDatetime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  } catch {
    return iso;
  }
}

function calcDuration(startIso, endIso) {
  if (!startIso) return null;
  const start = new Date(startIso).getTime();
  const end = endIso ? new Date(endIso).getTime() : Date.now();
  return (end - start) / 1000;
}

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* Multilingual search normalisation */
function normalise(s) {
  return String(s ?? '').normalize('NFC').toLocaleLowerCase();
}

/* --------------------------------------------------------
   Duration cells on dashboard
   -------------------------------------------------------- */
function renderDurations() {
  elems('[data-start]').forEach(td => {
    const start = td.dataset.start;
    const end   = td.dataset.end || null;
    const status = td.dataset.status;
    if (!start) return;
    const secs = calcDuration(start, end || (status === 'running' ? null : undefined));
    const span = td.querySelector('.duration-cell');
    if (span) span.textContent = fmtDuration(secs);
  });
}

/* --------------------------------------------------------
   Dashboard auto-refresh
   -------------------------------------------------------- */
function initDashboardRefresh(intervalMs) {
  // Render durations immediately
  renderDurations();

  // Tick running durations live
  setInterval(renderDurations, 1000);

  // Full page refresh when any running job exists
  const hasRunning = document.querySelector('.badge-running') !== null;
  if (hasRunning) {
    setTimeout(() => location.reload(), intervalMs);
  }
}

/* --------------------------------------------------------
   Collapsible sections
   -------------------------------------------------------- */
function initCollapsibles() {
  elems('.collapsible-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const bodyId = btn.getAttribute('aria-controls');
      const body = $(bodyId);
      if (!body) return;
      const expanded = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', String(!expanded));
      body.hidden = expanded;
    });
  });
}

/* --------------------------------------------------------
   Slider ↔ number-input synchronisation
   -------------------------------------------------------- */
function initSliders() {
  elems('.slider').forEach(slider => {
    const targetId = slider.dataset.target;
    const numInput = $(targetId);
    if (!numInput) return;

    slider.addEventListener('input', () => {
      numInput.value = slider.value;
    });
    numInput.addEventListener('input', () => {
      slider.value = numInput.value;
    });
  });
}

/* --------------------------------------------------------
   Crawl mode – show/hide comparison settings
   -------------------------------------------------------- */
function initModeRadios() {
  const radios = elems('input[name="crawl_mode"]');
  const compareSection = $('compare-section');
  if (!radios.length || !compareSection) return;

  function update() {
    const selected = radios.find(r => r.checked);
    if (!selected) return;
    compareSection.hidden = (selected.value !== 'compare');
  }

  radios.forEach(r => r.addEventListener('change', update));
  update();
}

/* --------------------------------------------------------
   Seeds file upload preview
   -------------------------------------------------------- */
function initSeedsUpload() {
  const fileInput = $('seeds_file');
  const preview   = $('seeds-file-preview');
  if (!fileInput || !preview) return;

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) { preview.classList.add('hidden'); return; }
    const reader = new FileReader();
    reader.onload = e => {
      const lines = e.target.result.split('\n').filter(l => l.trim());
      preview.textContent = `✔ ${file.name} — ${lines.length} URL(s) detected`;
      preview.classList.remove('hidden');
    };
    reader.readAsText(file);
  });
}

/* --------------------------------------------------------
   Form validation
   -------------------------------------------------------- */
function initFormValidation() {
  const form = $('crawl-form');
  if (!form) return;

  form.addEventListener('submit', e => {
    const seedsText = form.querySelector('#seeds_text');
    const seedsFile = form.querySelector('#seeds_file');
    const hasText = seedsText && seedsText.value.trim().length > 0;
    const hasFile = seedsFile && seedsFile.files.length > 0;

    if (!hasText && !hasFile) {
      e.preventDefault();
      alert('Please provide at least one seed URL or upload a seeds file.');
      return;
    }

    const btn = $('submit-btn');
    const lbl = $('submit-label');
    if (btn && lbl) {
      btn.disabled = true;
      lbl.textContent = '⏳ Starting…';
    }
  });
}

/* --------------------------------------------------------
   New Crawl form – master init
   -------------------------------------------------------- */
function initNewCrawlForm() {
  initCollapsibles();
  initSliders();
  initModeRadios();
  initSeedsUpload();
  initFormValidation();
}

/* --------------------------------------------------------
   Job log poller
   -------------------------------------------------------- */
function initJobPoller(jobId, initialStatus, initialStart, initialEnd) {
  const terminal   = $('log-terminal');
  const badge      = $('status-badge');
  const spinner    = $('log-spinner');
  const metaDur    = $('meta-duration');
  const autoscroll = $('autoscroll-toggle');
  const resultsLink = $('results-link');
  const summaryCard = $('summary-card');

  let status = initialStatus;
  let startTime = initialStart;
  let endTime   = initialEnd;
  let lastLen   = 0;
  let pollTimer = null;

  initCollapsibles();

  function scrollTerminal() {
    if (terminal && autoscroll && autoscroll.checked) {
      terminal.scrollTop = terminal.scrollHeight;
    }
  }

  function updateDuration() {
    if (!metaDur || !startTime) return;
    metaDur.textContent = fmtDuration(calcDuration(startTime, endTime));
  }

  function updateBadge(s) {
    if (!badge) return;
    badge.textContent = s;
    badge.className = `badge badge-${s} badge-lg`;
  }

  function extractSummaryFromLogs(lines) {
    if (!summaryCard) return;
    const text = lines.join('\n');
    const stats = {};

    const patterns = [
      [/pages? crawled[:\s]+(\d+)/i,       'Pages Crawled'],
      [/links? found[:\s]+(\d+)/i,          'Links Found'],
      [/links? enqueued[:\s]+(\d+)/i,       'Links Enqueued'],
      [/duration[:\s]+([\d.]+)\s*s/i,       'Duration (s)'],
      [/errors?[:\s]+(\d+)/i,               'Errors'],
      [/quantum circuits?[:\s]+(\d+)/i,     'Quantum Circuits'],
    ];

    patterns.forEach(([re, label]) => {
      const m = text.match(re);
      if (m) stats[label] = m[1];
    });

    if (Object.keys(stats).length === 0) return;

    const grid = $('summary-content');
    if (!grid) return;
    grid.innerHTML = Object.entries(stats).map(([k, v]) =>
      `<div class="summary-item"><div class="summary-val">${escapeHtml(v)}</div><div class="summary-lbl">${escapeHtml(k)}</div></div>`
    ).join('');
    summaryCard.hidden = false;
  }

  function renderLog(lines) {
    if (!terminal) return;
    if (lines.length === 0) return;
    if (lines.length === lastLen) return;
    lastLen = lines.length;
    terminal.textContent = lines.join('\n');
    scrollTerminal();
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    if (spinner) spinner.hidden = true;
  }

  async function poll() {
    try {
      const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/logs`);
      if (!resp.ok) return;
      const data = await resp.json();

      status   = data.status;
      const lines = data.log || [];
      renderLog(lines);
      updateBadge(status);

      if (status === 'completed' || status === 'failed') {
        stopPolling();
        // Try to get end time from full job detail
        try {
          const jr = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
          if (jr.ok) {
            const jd = await jr.json();
            endTime = jd.end_time || endTime;
            startTime = jd.start_time || startTime;
          }
        } catch {}
        updateDuration();
        extractSummaryFromLogs(lines);

        if (status === 'completed' && resultsLink) {
          resultsLink.style.opacity = '';
          resultsLink.style.pointerEvents = '';
        }
      }
    } catch {}
    updateDuration();
  }

  // Kick off
  if (status === 'completed' || status === 'failed') {
    stopPolling();
    poll(); // one-shot to populate log
    updateDuration();
  } else {
    poll();
    pollTimer = setInterval(poll, 2000);
    // Live duration tick
    setInterval(updateDuration, 1000);
  }
}

/* --------------------------------------------------------
   Results Explorer
   -------------------------------------------------------- */
const PAGE_SIZE = 50;

function initResultsExplorer(jobId) {
  const loadingEl = $('results-loading');
  const sectionEl = $('results-section');
  const emptyEl   = $('results-empty');
  const countEl   = $('results-count');
  const searchEl  = $('results-search');
  const tbody     = $('results-tbody');
  const exportBtn = $('export-csv-btn');
  const prevBtn   = $('page-prev');
  const nextBtn   = $('page-next');
  const pageInfo  = $('page-info');

  let allData      = [];
  let filtered     = [];
  let currentPage  = 1;
  let sortCol      = null;
  let sortDir      = 'asc';
  let expandedRow  = null;

  const COLUMNS = [
    { key: 'url',             label: 'URL' },
    { key: 'final_url',       label: 'Final URL' },
    { key: 'depth',           label: 'Depth' },
    { key: 'status',          label: 'Status' },
    { key: 'http_status',     label: 'HTTP' },
    { key: 'title',           label: 'Title' },
    { key: 'links_found',     label: 'Links Found' },
    { key: 'links_enqueued',  label: 'Enqueued' },
    { key: 'fetch_time',      label: 'Fetch Time' },
    { key: 'timestamp',       label: 'Timestamp' },
  ];

  /* Load */
  fetch(`/api/jobs/${encodeURIComponent(jobId)}/results`)
    .then(r => r.json())
    .then(data => {
      allData  = data;
      filtered = [...allData];
      loadingEl.hidden = true;

      if (allData.length === 0) {
        emptyEl.hidden = false;
        return;
      }

      sectionEl.hidden = false;
      renderTable();
    })
    .catch(() => {
      loadingEl.hidden = true;
      emptyEl.hidden = false;
    });

  /* Search */
  if (searchEl) {
    searchEl.addEventListener('input', () => {
      const q = normalise(searchEl.value);
      if (!q) {
        filtered = [...allData];
      } else {
        filtered = allData.filter(row =>
          COLUMNS.some(c => normalise(row[c.key]).includes(q)) ||
          normalise(row.snippet || '').includes(q)
        );
      }
      currentPage = 1;
      renderTable();
    });
  }

  /* Sort headers */
  elems('.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortCol === col) {
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortCol = col;
        sortDir = 'asc';
      }
      elems('.sortable').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
      th.classList.add(`sort-${sortDir}`);

      filtered.sort((a, b) => {
        const av = a[col] ?? '';
        const bv = b[col] ?? '';
        const n = Number(av) - Number(bv);
        if (!isNaN(n)) return sortDir === 'asc' ? n : -n;
        const cmp = String(av).localeCompare(String(bv));
        return sortDir === 'asc' ? cmp : -cmp;
      });
      currentPage = 1;
      renderTable();
    });
  });

  /* Pagination */
  if (prevBtn) prevBtn.addEventListener('click', () => { currentPage--; renderTable(); });
  if (nextBtn) nextBtn.addEventListener('click', () => { currentPage++; renderTable(); });

  /* CSV Export */
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      const headers = COLUMNS.map(c => c.key);
      const rows = [headers.join(',')];
      filtered.forEach(row => {
        rows.push(headers.map(k => {
          const v = String(row[k] ?? '').replace(/"/g, '""');
          return `"${v}"`;
        }).join(','));
      });
      const blob = new Blob([rows.join('\r\n')], { type: 'text/csv;charset=utf-8;' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href = url; a.download = `results-${String(jobId).slice(0, 8)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  /* Render table */
  function renderTable() {
    if (!tbody) return;

    const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const start = (currentPage - 1) * PAGE_SIZE;
    const slice = filtered.slice(start, start + PAGE_SIZE);

    tbody.innerHTML = '';
    expandedRow = null;

    slice.forEach((row, i) => {
      const tr = document.createElement('tr');
      tr.className = 'result-row';
      tr.dataset.idx = String(start + i);

      const httpStatus = row.http_status ?? row.status_code ?? '';
      const httpClass  = httpStatus >= 400 ? 'text-muted' : (httpStatus >= 300 ? '' : '');

      tr.innerHTML = [
        `<td class="monospace" style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(row.url)}"><a href="${escapeHtml(row.url)}" target="_blank" rel="noopener">${escapeHtml(truncate(row.url, 40))}</a></td>`,
        `<td class="monospace text-muted" style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(row.final_url || '')}">` +
          (row.final_url && row.final_url !== row.url ? escapeHtml(truncate(row.final_url, 35)) : '<em>—</em>') + `</td>`,
        `<td>${escapeHtml(row.depth ?? '—')}</td>`,
        `<td>${escapeHtml(row.status ?? '—')}</td>`,
        `<td class="${httpClass}">${escapeHtml(httpStatus || '—')}</td>`,
        `<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(row.title || '')}">${escapeHtml(truncate(row.title, 45) || '—')}</td>`,
        `<td>${escapeHtml(row.links_found ?? '—')}</td>`,
        `<td>${escapeHtml(row.links_enqueued ?? '—')}</td>`,
        `<td>${row.fetch_time != null ? escapeHtml(Number(row.fetch_time).toFixed(3)) + 's' : '—'}</td>`,
        `<td class="text-muted">${escapeHtml(row.timestamp ? String(row.timestamp).replace('T', ' ').slice(0, 19) : '—')}</td>`,
      ].join('');

      // Row expand on click
      tr.addEventListener('click', e => {
        if (e.target.tagName === 'A') return;
        toggleSnippet(tr, row);
      });
      tr.style.cursor = 'pointer';

      tbody.appendChild(tr);
    });

    // Update count + pagination
    if (countEl) countEl.textContent = `${filtered.length.toLocaleString()} results`;
    if (pageInfo) pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
    if (prevBtn) prevBtn.disabled = currentPage <= 1;
    if (nextBtn) nextBtn.disabled = currentPage >= totalPages;
  }

  function toggleSnippet(tr, row) {
    // Remove existing snippet row
    const existing = tbody.querySelector('.snippet-row');
    if (existing) {
      existing.remove();
      const prev = tbody.querySelector('.row-expanded');
      if (prev) prev.classList.remove('row-expanded');
      if (expandedRow === tr) { expandedRow = null; return; }
    }

    expandedRow = tr;
    tr.classList.add('row-expanded');

    const snippetTr = document.createElement('tr');
    snippetTr.className = 'snippet-row';
    const snippet = row.snippet || row.text || row.content || '';
    snippetTr.innerHTML = `<td colspan="10">${escapeHtml(snippet ? truncate(snippet, 600) : '(no snippet available)')}</td>`;
    tr.insertAdjacentElement('afterend', snippetTr);
  }

  function truncate(s, n) {
    if (!s) return '';
    s = String(s);
    return s.length > n ? s.slice(0, n) + '…' : s;
  }
}
