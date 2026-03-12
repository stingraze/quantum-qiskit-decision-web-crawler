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
   Dashboard auto-refresh (Ajax-based, no page reload)
   -------------------------------------------------------- */
function initDashboardRefresh(intervalMs) {
  // Render durations immediately and keep ticking
  renderDurations();
  setInterval(renderDurations, 1000);

  // Start polling only when there are active jobs
  const hasActive = document.querySelector('.badge-running, .badge-pending, .badge-paused') !== null;
  if (hasActive) {
    _startDashboardPoller(intervalMs);
  }
}

function stopJobFromDashboard(jobId, btn) {
  if (!confirm('Are you sure you want to stop this crawl?')) return;
  btn.disabled = true;
  fetch('/api/jobs/' + encodeURIComponent(jobId) + '/stop', { method: 'POST' })
    .then(r => r.json())
    .then(() => { /* dashboard poller will update the row */ })
    .catch(err => {
      console.error('[dashboard] stop error:', err);
      btn.disabled = false;
    });
}

function _startDashboardPoller(intervalMs) {
  let pollTimer = setInterval(async () => {
    try {
      const resp = await fetch('/api/jobs');
      if (!resp.ok) return;
      const jobs = await resp.json();
      _updateDashboard(jobs);

      // Stop polling when no more active jobs
      const stillActive = jobs.some(j => j.status === 'running' || j.status === 'pending' || j.status === 'paused');
      if (!stillActive) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    } catch (err) {
      console.error('[dashboard] poll error:', err);
    }
  }, intervalMs);
}

function _updateDashboard(jobs) {
  // Update stats cards
  const total     = jobs.length;
  const running   = jobs.filter(j => j.status === 'running' || j.status === 'paused').length;
  const completed = jobs.filter(j => j.status === 'completed').length;
  const failed    = jobs.filter(j => j.status === 'failed' || j.status === 'stopped').length;

  const statTotal     = $('stat-total');
  const statRunning   = $('stat-running');
  const statCompleted = $('stat-completed');
  const statFailed    = $('stat-failed');
  if (statTotal)     statTotal.textContent     = total;
  if (statRunning)   statRunning.textContent   = running;
  if (statCompleted) statCompleted.textContent = completed;
  if (statFailed)    statFailed.textContent    = failed;

  // Handle empty-state → table transition when jobs first appear
  const emptyState  = $('jobs-empty-state');
  const tableWrapper = $('jobs-table-wrapper');
  if (jobs.length > 0 && emptyState && !tableWrapper) {
    // Build the table from scratch and insert it
    const section = emptyState.parentElement;
    emptyState.remove();
    const wrapper = document.createElement('div');
    wrapper.className = 'table-responsive';
    wrapper.id = 'jobs-table-wrapper';
    wrapper.innerHTML =
      '<table class="table" id="jobs-table">' +
        '<thead><tr>' +
          '<th>Job ID</th><th>Status</th><th>Mode</th>' +
          '<th>Start Time</th><th>Duration</th><th>Actions</th>' +
        '</tr></thead>' +
        '<tbody></tbody>' +
      '</table>';
    section.appendChild(wrapper);
  }

  // Update / insert rows
  const tbody = document.querySelector('#jobs-table tbody');
  if (!tbody) return;

  jobs.forEach(job => {
    const existingRow = Array.from(tbody.querySelectorAll('tr[data-job-id]'))
      .find(r => r.dataset.jobId === job.id);
    if (existingRow) {
      _updateJobRow(existingRow, job);
    } else {
      const newRow = _buildJobRow(job);
      // Prepend so newest jobs appear at the top (matching server render order)
      tbody.insertBefore(newRow, tbody.firstChild);
    }
  });
}

function _modeTag(mode) {
  if (mode === 'compare') return '<span class="mode-tag mode-compare">Compare</span>';
  if (mode === 'base')    return '<span class="mode-tag mode-base">Base</span>';
  return '<span class="mode-tag mode-hybrid">Hybrid v4</span>';
}

function _fmtStartTime(iso) {
  if (!iso) return '<em>pending</em>';
  const s = String(iso);
  return escapeHtml(s.length >= 19 ? s.slice(0, 19).replace('T', ' ') + ' UTC' : s + ' UTC');
}

function _actionsHtml(job) {
  const viewHref    = `/crawl/${encodeURIComponent(job.id)}`;
  const resultsHref = `/crawl/${encodeURIComponent(job.id)}/results`;
  let html = `<a href="${escapeHtml(viewHref)}" class="btn btn-secondary btn-xs">View</a>`;
  if (job.status === 'completed') {
    html += ` <a href="${escapeHtml(resultsHref)}" class="btn btn-primary btn-xs">Results</a>`;
  }
  if (job.status === 'running' || job.status === 'paused') {
    const safeId = escapeHtml(job.id);
    html += ` <button type="button" class="btn btn-danger btn-xs" onclick="stopJobFromDashboard('${safeId}', this)">⏹</button>`;
  }
  return html;
}

function _buildJobRow(job) {
  const tr = document.createElement('tr');
  tr.dataset.jobId = job.id;
  const mode = (job.params && job.params.crawl_mode) || 'hybrid';
  tr.innerHTML =
    `<td class="monospace text-muted" title="${escapeHtml(job.id)}">${escapeHtml(job.id.slice(0, 8))}…</td>` +
    `<td><span class="badge badge-${escapeHtml(job.status)}" id="badge-${escapeHtml(job.id)}">${escapeHtml(job.status)}</span></td>` +
    `<td>${_modeTag(mode)}</td>` +
    `<td class="text-muted" id="start-time-${escapeHtml(job.id)}">${_fmtStartTime(job.start_time)}</td>` +
    `<td class="text-muted" data-job-id="${escapeHtml(job.id)}" data-start="${escapeHtml(job.start_time || '')}" data-end="${escapeHtml(job.end_time || '')}" data-status="${escapeHtml(job.status)}">` +
      (job.start_time ? '<span class="duration-cell">…</span>' : '—') +
    `</td>` +
    `<td id="actions-${escapeHtml(job.id)}">${_actionsHtml(job)}</td>`;
  return tr;
}

function _updateJobRow(tr, job) {
  // Badge
  const badge = $(`badge-${job.id}`);
  if (badge) {
    badge.textContent = job.status;
    badge.className = `badge badge-${job.status}`;
  }

  // Start time
  const startTd = $(`start-time-${job.id}`);
  if (startTd) startTd.innerHTML = _fmtStartTime(job.start_time);

  // Duration cell – update data attributes so renderDurations() keeps working
  const durTd = tr.querySelector(`td[data-job-id]`);
  if (durTd) {
    durTd.dataset.start  = job.start_time || '';
    durTd.dataset.end    = job.end_time   || '';
    durTd.dataset.status = job.status;
    if (job.start_time && !durTd.querySelector('.duration-cell')) {
      durTd.innerHTML = '<span class="duration-cell">…</span>';
    }
  }

  // Actions – add Results button when job completes
  const actionsTd = $(`actions-${job.id}`);
  if (actionsTd) actionsTd.innerHTML = _actionsHtml(job);
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
function initJobPoller(jobId, initialStatus, initialStart, initialEnd, initialPaused) {
  const terminal   = $('log-terminal');
  const badge      = $('status-badge');
  const spinner    = $('log-spinner');
  const metaDur    = $('meta-duration');
  const autoscroll = $('autoscroll-toggle');
  const resultsLink = $('results-link');
  const summaryCard = $('summary-card');
  const pauseBtn   = $('pause-btn');
  const stopBtn    = $('stop-btn');
  const controlsCard = $('job-controls');

  let status = initialStatus;
  let paused = !!initialPaused;
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

  function updateControls() {
    const isActive = status === 'running' || status === 'paused';
    if (controlsCard) controlsCard.style.display = isActive ? '' : 'none';
    if (pauseBtn) {
      pauseBtn.disabled = !isActive;
      pauseBtn.textContent = paused ? '▶ Resume' : '⏸ Pause';
    }
    if (stopBtn) stopBtn.disabled = !isActive;
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

      status = data.status;
      paused = !!data.paused;
      const lines = data.log || [];
      renderLog(lines);
      updateBadge(status);
      updateControls();

      if (status === 'completed' || status === 'failed' || status === 'stopped') {
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

  // Pause / Resume button
  if (pauseBtn) {
    pauseBtn.addEventListener('click', async () => {
      pauseBtn.disabled = true;
      try {
        const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/pause`, { method: 'POST' });
        if (resp.ok) {
          const data = await resp.json();
          status = data.status;
          paused = !!data.paused;
          updateBadge(status);
          updateControls();
        } else {
          console.error('[poller] pause/resume failed, status:', resp.status);
        }
      } catch (err) {
        console.error('[poller] pause/resume error:', err);
      }
      pauseBtn.disabled = false;
    });
  }

  // Stop button
  if (stopBtn) {
    stopBtn.addEventListener('click', async () => {
      if (!confirm('Are you sure you want to stop this crawl?')) return;
      stopBtn.disabled = true;
      if (pauseBtn) pauseBtn.disabled = true;
      try {
        const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/stop`, { method: 'POST' });
        if (resp.ok) {
          status = 'stopped';
          paused = false;
          updateBadge(status);
          updateControls();
          stopPolling();
          // Fetch final state
          try {
            const jr = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
            if (jr.ok) {
              const jd = await jr.json();
              endTime = jd.end_time || endTime;
              startTime = jd.start_time || startTime;
            }
          } catch (err) {
            console.error('[poller] failed to fetch final job state:', err);
          }
          updateDuration();
        } else {
          console.error('[poller] stop failed, status:', resp.status);
          stopBtn.disabled = false;
          if (pauseBtn) pauseBtn.disabled = false;
        }
      } catch (err) {
        console.error('[poller] stop error:', err);
        stopBtn.disabled = false;
        if (pauseBtn) pauseBtn.disabled = false;
      }
    });
  }

  // Kick off
  if (status === 'completed' || status === 'failed' || status === 'stopped') {
    stopPolling();
    poll(); // one-shot to populate log
    updateDuration();
    updateControls();
  } else {
    updateControls();
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
