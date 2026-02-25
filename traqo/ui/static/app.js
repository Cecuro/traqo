/* traqo trace viewer */
'use strict';

// ── Icons ────────────────────────────────────────────────────────────────────

const FOLDER_SVG = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M1 3.5A1.5 1.5 0 012.5 2h3.172a1.5 1.5 0 011.06.44l.658.658a.5.5 0 00.354.147H13.5A1.5 1.5 0 0115 4.75v7.75A1.5 1.5 0 0113.5 14h-11A1.5 1.5 0 011 12.5v-9z"/></svg>';
const TRACE_SVG = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M2 3.5A1.5 1.5 0 013.5 2h5.672a1.5 1.5 0 011.06.44l2.829 2.828a1.5 1.5 0 01.439 1.06V12.5A1.5 1.5 0 0112 14H3.5A1.5 1.5 0 012 12.5v-9zM3.5 3a.5.5 0 00-.5.5v9a.5.5 0 00.5.5H12a.5.5 0 00.5-.5V6.328a.5.5 0 00-.146-.353L9.525 3.146A.5.5 0 009.172 3H3.5z"/></svg>';

// ── State ────────────────────────────────────────────────────────────────────

const state = {
  traces: [],         // all trace summaries from /api/traces
  dir: '',            // current directory path in browse view
  traceFile: null,    // currently open trace file path
  events: [],         // raw events for current trace
  parsed: null,       // parsed trace: { spans, logEvents, traceStart, traceEnd }
  selectedSpan: null,
  statusFilter: 'all',
  selectedTags: new Set(),
};

const _copy = { data: new Map(), id: 0 };

// ── DOM Helpers ──────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function showLoading(on) { $('loading').classList.toggle('active', on); }

function showError(msg) {
  $('error-msg').textContent = msg;
  $('error-banner').classList.add('active');
}

function dismissError() { $('error-banner').classList.remove('active'); }

// ── Formatting ───────────────────────────────────────────────────────────────

function fmtDur(s) {
  if (s == null) return '\u2013';
  if (s < 0.001) return '<1ms';
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  if (s < 60) return `${s.toFixed(2)}s`;
  return `${(s / 60).toFixed(1)}m`;
}

function fmtTime(ts) {
  if (!ts) return '\u2013';
  return new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function fmtTimeFull(ts) {
  if (!ts) return '\u2013';
  return new Date(ts).toISOString().replace('T', ' ').replace('Z', ' UTC');
}

function fmtN(n) {
  if (n == null) return '0';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ── Data Helpers ─────────────────────────────────────────────────────────────

function kindBadge(kind) {
  if (kind === 'llm') return 'badge-llm';
  if (kind === 'tool') return 'badge-tool';
  return 'badge-kind';
}

function summarizeInput(input) {
  if (input == null) return '';
  if (typeof input === 'string') return input.length > 80 ? input.slice(0, 80) + '\u2026' : input;
  if (typeof input === 'object') {
    for (const key of ['query', 'question', 'prompt', 'message', 'text', 'input']) {
      if (input[key] && typeof input[key] === 'string') {
        const v = input[key];
        return v.length > 80 ? v.slice(0, 80) + '\u2026' : v;
      }
    }
    const keys = Object.keys(input);
    return keys.length ? keys.join(', ') : '';
  }
  return String(input).slice(0, 80);
}

// ── JSON Display ─────────────────────────────────────────────────────────────

function syntaxHighlight(obj, indent) {
  indent = indent || 0;
  if (obj === null) return '<span class="json-null">null</span>';
  if (typeof obj === 'boolean') return `<span class="json-bool">${obj}</span>`;
  if (typeof obj === 'number') return `<span class="json-number">${obj}</span>`;
  if (typeof obj === 'string') {
    const display = obj.length > 2000 ? obj.slice(0, 2000) + '\u2026' : obj;
    const escaped = esc(display).replace(/\n/g, '\n' + ' '.repeat(indent + 2));
    return `<span class="json-string">"${escaped}"</span>`;
  }
  if (Array.isArray(obj)) {
    if (!obj.length) return '[]';
    const pad = ' '.repeat(indent + 2);
    const end = ' '.repeat(indent);
    return '[\n' + obj.map(item => pad + syntaxHighlight(item, indent + 2)).join(',\n') + '\n' + end + ']';
  }
  if (typeof obj === 'object') {
    const keys = Object.keys(obj);
    if (!keys.length) return '{}';
    const pad = ' '.repeat(indent + 2);
    const end = ' '.repeat(indent);
    return '{\n' + keys.map(k =>
      pad + `<span class="json-key">"${esc(k)}"</span>: ` + syntaxHighlight(obj[k], indent + 2)
    ).join(',\n') + '\n' + end + '}';
  }
  return esc(String(obj));
}

function renderJson(value) {
  if (typeof value === 'string') {
    return value.includes('\n') || value.length > 100
      ? `<div class="json-viewer">${esc(value)}</div>`
      : `<div class="json-viewer"><span class="json-string">"${esc(value)}"</span></div>`;
  }
  return `<div class="json-viewer">${syntaxHighlight(value)}</div>`;
}

function jsonSection(title, value, open) {
  if (open === undefined) open = true;
  const cid = 'c' + (++_copy.id);
  const sid = 'sc' + _copy.id;
  _copy.data.set(cid, value);
  return `<div class="detail-section">
    <div class="section-header" data-action="toggle-section" data-target="${sid}">
      <span class="toggle${open ? '' : ' collapsed'}">\u25BC</span>${esc(title)}
      <button class="copy-btn" data-action="copy" data-copy-id="${cid}">Copy</button>
    </div>
    <div id="${sid}"${open ? '' : ' style="display:none"'}>${renderJson(value)}</div>
  </div>`;
}

// ── API ──────────────────────────────────────────────────────────────────────

async function fetchTraces() {
  showLoading(true);
  try {
    const res = await fetch('/api/traces');
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    state.traces = await res.json();
  } catch (e) {
    showError('Failed to load traces: ' + e.message);
    state.traces = [];
  } finally {
    showLoading(false);
  }
}

async function loadTrace(file, targetSpanId) {
  showLoading(true);
  dismissError();
  try {
    const res = await fetch('/api/trace?file=' + encodeURIComponent(file));
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    state.events = data.events || [];
    state.traceFile = file;
    state.parsed = parseEvents(state.events, file);
    showTraceDetail(targetSpanId);
  } catch (e) {
    showError('Failed to load trace: ' + e.message);
  } finally {
    showLoading(false);
  }
}

// ── Data Processing ──────────────────────────────────────────────────────────

function resolveChildKey(parentFileKey, childData) {
  if (!parentFileKey) return childData.child_file || childData.child_name + '.jsonl';
  const parts = parentFileKey.split('/');
  parts[parts.length - 1] = childData.child_file || childData.child_name + '.jsonl';
  return parts.join('/');
}

function parseEvents(events, parentFileKey) {
  const starts = new Map();
  const ends = new Map();
  const logEvents = [];
  const childTracers = new Map();
  let traceStart = null;
  let traceEnd = null;

  for (const ev of events) {
    switch (ev.type) {
      case 'trace_start': traceStart = ev; break;
      case 'trace_end':   traceEnd = ev; break;
      case 'span_start':  starts.set(ev.id, ev); break;
      case 'span_end':    ends.set(ev.id, ev); break;
      case 'event': {
        if (ev.name === 'child_started' && ev.data) {
          const name = ev.data.child_name;
          const file = resolveChildKey(parentFileKey, ev.data);
          childTracers.set(name, {
            name,
            file,
            startedTs: ev.ts,
            stats: null,
            parsed: null,
            expanded: false,
            fetchError: null,
            loading: false,
          });
        } else if (ev.name === 'child_ended' && ev.data) {
          const name = ev.data.child_name;
          const entry = childTracers.get(name);
          if (entry) {
            entry.stats = {
              spans: ev.data.spans,
              total_input_tokens: ev.data.total_input_tokens,
              total_output_tokens: ev.data.total_output_tokens,
              duration_s: ev.data.duration_s,
            };
            if (ev.data.child_file) entry.file = resolveChildKey(parentFileKey, ev.data);
          }
        } else {
          logEvents.push(ev);
        }
        break;
      }
    }
  }

  const spans = [];
  for (const [id, end] of ends) {
    const start = starts.get(id) || {};
    spans.push({
      id,
      parent_id: end.parent_id,
      name: end.name,
      kind: end.kind || start.kind,
      status: end.status,
      duration_s: end.duration_s,
      input: start.input,
      output: end.output,
      metadata: end.metadata || start.metadata,
      tags: end.tags || start.tags || [],
      error: end.error,
      ts_start: start.ts,
      ts_end: end.ts,
    });
  }

  return { spans, logEvents, traceStart, traceEnd, childTracers };
}

function tracesInDir(prefix) {
  if (!prefix) return state.traces;
  return state.traces.filter(t => t.file.startsWith(prefix + '/'));
}

function itemsAtDir(dirPrefix, traces) {
  const dirs = new Map();
  const direct = [];
  const prefix = dirPrefix ? dirPrefix + '/' : '';

  for (const t of traces) {
    const rel = dirPrefix ? t.file.slice(prefix.length) : t.file;
    const slash = rel.indexOf('/');
    if (slash === -1) {
      direct.push(t);
    } else {
      const name = rel.slice(0, slash);
      if (!dirs.has(name)) dirs.set(name, prefix + name);
    }
  }

  const dirList = [];
  for (const [name, path] of dirs) {
    const sub = tracesInDir(path);
    const agg = aggregateStats(sub);
    let latest = '';
    for (const t of sub) if (t.ts && t.ts > latest) latest = t.ts;
    dirList.push({ type: 'folder', name, path, latestTs: latest, ...agg });
  }
  dirList.sort((a, b) => a.name.localeCompare(b.name));

  return { dirs: dirList, traces: direct };
}

function aggregateStats(traces) {
  let spans = 0, errors = 0, tokIn = 0, tokOut = 0, dur = 0;
  for (const t of traces) {
    const s = t.stats || {};
    spans += s.spans || 0;
    errors += s.errors || 0;
    tokIn += s.total_input_tokens || 0;
    tokOut += s.total_output_tokens || 0;
    dur += t.duration_s || 0;
  }
  return {
    traceCount: traces.length,
    totalSpans: spans,
    totalErrors: errors,
    totalIn: tokIn,
    totalOut: tokOut,
    totalDuration: dur,
    avgDuration: traces.length ? dur / traces.length : 0,
  };
}

// ── Row Renderers ────────────────────────────────────────────────────────────

function renderFolderRow(d) {
  const hasErr = d.totalErrors > 0;
  return `<div class="item-row" data-action="navigate" data-path="${esc(d.path)}">
    <div class="row-icon folder">${FOLDER_SVG}</div>
    <div class="row-main">
      <div class="row-name">${esc(d.name)}</div>
      <div class="row-desc">${d.traceCount} trace${d.traceCount !== 1 ? 's' : ''}${d.latestTs ? ' &middot; latest ' + fmtTime(d.latestTs) : ''}</div>
    </div>
    <div class="row-meta">
      <div class="row-meta-item">
        <span class="val tok"><span class="in">${fmtN(d.totalIn)}</span> / <span class="out">${fmtN(d.totalOut)}</span></span>
        <span class="lbl">tokens</span>
      </div>
      <div class="row-meta-item"><span class="val">${fmtDur(d.avgDuration)}</span><span class="lbl">avg</span></div>
      ${hasErr
        ? `<div class="row-meta-item"><span class="val" style="color:var(--red)">${d.totalErrors}</span><span class="lbl">errors</span></div>`
        : `<div class="row-meta-item"><span class="badge badge-ok">ok</span></div>`}
    </div>
    <div class="row-chevron">&rsaquo;</div>
  </div>`;
}

function renderTraceRow(t) {
  const stats = t.stats || {};
  const hasErr = (stats.errors || 0) > 0;
  const displayName = t.file.split('/').pop().replace(/\.jsonl$/, '');
  const desc = summarizeInput(t.input);
  return `<div class="item-row" data-action="open-trace" data-file="${esc(t.file)}">
    <div class="row-icon ${hasErr ? 'trace-error' : 'trace'}">${TRACE_SVG}</div>
    <div class="row-main">
      <div class="row-name">${esc(displayName)}</div>
      <div class="row-desc">${esc(desc)}${t.tags?.length ? ' &middot; ' + t.tags.slice(0, 3).map(x => `<span class="badge badge-tag">${esc(x)}</span>`).join('') : ''}</div>
    </div>
    <div class="row-meta">
      <div class="row-meta-item">
        <span class="val tok"><span class="in">${fmtN(stats.total_input_tokens || 0)}</span> / <span class="out">${fmtN(stats.total_output_tokens || 0)}</span></span>
        <span class="lbl">tokens</span>
      </div>
      <div class="row-meta-item"><span class="val">${fmtDur(t.duration_s)}</span><span class="lbl">duration</span></div>
      <div class="row-meta-item"><span class="val">${stats.spans || 0}</span><span class="lbl">spans</span></div>
      <div class="row-meta-item"><span class="val">${fmtTime(t.ts)}</span><span class="lbl">time</span></div>
      <div class="row-meta-item">
        ${hasErr
          ? `<span class="badge badge-error">${stats.errors} error${stats.errors !== 1 ? 's' : ''}</span>`
          : `<span class="badge badge-ok">ok</span>`}
      </div>
    </div>
    <div class="row-chevron">&rsaquo;</div>
  </div>`;
}

function renderSpanNode(span, depth, t0, dur, qualifiedId) {
  const s0 = span.ts_start ? new Date(span.ts_start).getTime() : t0;
  const s1 = span.ts_end ? new Date(span.ts_end).getTime() : s0;
  const left = ((s0 - t0) / dur * 100).toFixed(1);
  const width = Math.max((s1 - s0) / dur * 100, 1).toFixed(1);
  const isErr = span.status === 'error';
  const spanId = qualifiedId || span.id;

  return `<div class="span-node${isErr ? ' error' : ''}" style="--depth:${depth}" data-span-id="${esc(spanId)}" data-action="select-span">
    <div class="span-info">
      <span class="status-dot ${isErr ? 'error' : 'ok'}"></span>
      ${span.kind ? `<span class="badge ${kindBadge(span.kind)}">${esc(span.kind)}</span>` : ''}
      <span class="name">${esc(span.name)}</span>
    </div>
    <div class="span-timing">
      <div class="waterfall-container"><div class="waterfall-bg"></div><div class="waterfall-bar" data-kind="${span.kind || ''}" style="left:${left}%;width:${width}%"></div></div>
      <span class="duration">${fmtDur(span.duration_s)}</span>
    </div>
  </div>`;
}

// ── Browse View ──────────────────────────────────────────────────────────────

function showBrowseView(dir) {
  state.dir = dir || '';
  state.traceFile = null;
  state.events = [];
  state.parsed = null;
  _copy.data.clear();
  _copy.id = 0;

  $('browse-view').style.display = 'block';
  $('trace-detail').style.display = 'none';
  $('trace-detail').classList.remove('active');
  $('back-btn').style.display = 'none';
  renderBreadcrumb();
  populateTagFilter();
  renderBrowseContent();
}

function populateTagFilter() {
  const scoped = state.dir ? tracesInDir(state.dir) : [...state.traces];
  const counts = new Map();
  for (const t of scoped) {
    for (const tag of t.tags || []) {
      counts.set(tag, (counts.get(tag) || 0) + 1);
    }
  }
  const tags = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const menu = $('tag-menu');

  if (!tags.length) {
    menu.innerHTML = '<div class="dropdown-item" style="color:var(--text-dim)">No tags</div>';
    updateTagToggleLabel();
    return;
  }

  menu.innerHTML = tags.map(([tag, count]) => {
    const sel = state.selectedTags.has(tag);
    return `<div class="dropdown-item${sel ? ' selected' : ''}" data-action="toggle-tag" data-tag="${esc(tag)}"><span class="check">${sel ? '&#10003;' : ''}</span>${esc(tag)} <span style="color:var(--text-dim);margin-left:auto">${count}</span></div>`;
  }).join('');
  updateTagToggleLabel();
}

function updateTagToggleLabel() {
  const btn = $('tag-dropdown').querySelector('.dropdown-toggle');
  const n = state.selectedTags.size;
  if (n === 0) {
    btn.innerHTML = 'All tags <span class="caret">&#9662;</span>';
  } else {
    btn.innerHTML = `Tags <span class="tag-count-badge">${n}</span> <span class="caret">&#9662;</span>`;
  }
}

function renderFolderHeader() {
  const el = $('folder-header');
  if (!state.dir) {
    el.classList.remove('visible');
    el.innerHTML = '';
    return;
  }
  el.classList.add('visible');
  const folderName = state.dir.split('/').pop();
  const parentDir = state.dir.includes('/') ? state.dir.slice(0, state.dir.lastIndexOf('/')) : '';
  el.innerHTML = `
    <button class="folder-back-btn" data-action="navigate" data-path="${esc(parentDir)}" title="Go back">&larr;</button>
    <div class="folder-icon">${FOLDER_SVG}</div>
    <div class="folder-title">${esc(folderName)}</div>
  `;
}

function renderBrowseContent() {
  renderFolderHeader();
  const search = $('search-input').value.toLowerCase();

  const scoped = state.dir ? tracesInDir(state.dir) : [...state.traces];

  // Apply filters
  const filtered = scoped.filter(t => {
    if (search) {
      const hay = [t.file, t.thread_id || '', ...(t.tags || []), JSON.stringify(t.input || '')].join(' ').toLowerCase();
      if (!hay.includes(search)) return false;
    }
    if (state.selectedTags.size && !(t.tags || []).some(tag => state.selectedTags.has(tag))) return false;
    if (state.statusFilter === 'ok' && (t.stats?.errors || 0) > 0) return false;
    if (state.statusFilter === 'error' && (t.stats?.errors || 0) === 0) return false;
    return true;
  });

  const items = itemsAtDir(state.dir, filtered);

  // Summary strip (unfiltered scope)
  const agg = aggregateStats(scoped);
  $('summary-strip').innerHTML = `
    <div class="stat"><span class="val">${agg.traceCount}</span><span class="lbl">traces</span></div>
    <div class="stat"><span class="val" style="color:var(--blue)">${fmtN(agg.totalIn)}</span><span class="lbl">in</span></div>
    <div class="stat"><span class="val" style="color:var(--orange)">${fmtN(agg.totalOut)}</span><span class="lbl">out</span></div>
    <div class="stat"><span class="val">${fmtDur(agg.avgDuration)}</span><span class="lbl">avg</span></div>
    <div class="stat"><span class="val" style="color:${agg.totalErrors > 0 ? 'var(--red)' : 'var(--text)'}">${agg.totalErrors}</span><span class="lbl">errors</span></div>
  `;

  // Filter info
  const info = $('filter-info');
  if (filtered.length < scoped.length) {
    info.textContent = `Showing ${filtered.length} of ${scoped.length} traces`;
    info.classList.add('active');
  } else {
    info.classList.remove('active');
  }

  // Sort by newest first (default)
  const sorted = [...items.traces].sort((a, b) => (b.ts || '') > (a.ts || '') ? 1 : -1);

  const listEl = $('item-list');
  const emptyEl = $('empty-list');

  if (!items.dirs.length && !sorted.length) {
    listEl.style.display = 'none';
    emptyEl.style.display = 'flex';
    return;
  }
  listEl.style.display = 'block';
  emptyEl.style.display = 'none';
  listEl.innerHTML = items.dirs.map(renderFolderRow).join('') + sorted.map(renderTraceRow).join('');
}

// ── Breadcrumb ───────────────────────────────────────────────────────────────

function renderBreadcrumb() {
  let html = '<a data-action="navigate" data-path="">traces</a>';
  if (state.dir) {
    const parts = state.dir.split('/');
    let path = '';
    for (let i = 0; i < parts.length; i++) {
      path += (i > 0 ? '/' : '') + parts[i];
      html += '<span class="sep">/</span>';
      if (i === parts.length - 1 && !state.traceFile) {
        html += `<span class="current">${esc(parts[i])}</span>`;
      } else {
        html += `<a data-action="navigate" data-path="${esc(path)}">${esc(parts[i])}</a>`;
      }
    }
  }
  if (state.traceFile) {
    html += `<span class="sep">/</span><span class="current">${esc(state.traceFile.split('/').pop())}</span>`;
  }
  $('breadcrumb').innerHTML = html;
}

// ── Trace Detail View ────────────────────────────────────────────────────────

function showTraceDetail(targetSpanId) {
  _copy.data.clear();
  _copy.id = 0;

  $('browse-view').style.display = 'none';
  const detail = $('trace-detail');
  detail.style.display = 'flex';
  detail.classList.add('active');
  $('back-btn').style.display = 'flex';

  // Set dir from trace path
  const parts = state.traceFile.split('/');
  if (parts.length > 1) state.dir = parts.slice(0, -1).join('/');
  renderBreadcrumb();

  const { traceStart, traceEnd, spans } = state.parsed;
  const stats = traceEnd?.stats || {};

  $('summary-bar').innerHTML = `
    <button class="trace-back-btn" data-action="navigate" data-path="${esc(state.dir)}">&larr; Back</button>
    <div class="summary-stat"><span class="label">Duration</span><span class="value">${fmtDur(traceEnd?.duration_s)}</span></div>
    <div class="summary-stat"><span class="label">Spans</span><span class="value">${stats.spans || 0}</span></div>
    <div class="summary-stat"><span class="label">Input tokens</span><span class="value" style="color:var(--blue)">${fmtN(stats.total_input_tokens || 0)}</span></div>
    <div class="summary-stat"><span class="label">Output tokens</span><span class="value" style="color:var(--orange)">${fmtN(stats.total_output_tokens || 0)}</span></div>
    <div class="summary-stat"><span class="label">Events</span><span class="value">${stats.events || 0}</span></div>
    <div class="summary-stat"><span class="label">Errors</span><span class="value" style="color:${(stats.errors || 0) > 0 ? 'var(--red)' : 'var(--text)'}">${stats.errors || 0}</span></div>
    ${traceStart?.tags?.length ? `<div class="summary-stat"><span class="label">Tags</span><span class="value">${traceStart.tags.map(t => `<span class="badge badge-tag">${esc(t)}</span>`).join(' ')}</span></div>` : ''}
    ${traceStart?.thread_id ? `<div class="summary-stat"><span class="label">Thread</span><span class="value" style="font-size:13px">${esc(traceStart.thread_id)}</span></div>` : ''}
  `;

  renderSpanTree();

  // Select target span or auto-select first meaningful node
  if (targetSpanId) {
    const found = state.parsed.spans.find(s => s.id === targetSpanId);
    if (found) {
      selectSpan(targetSpanId);
    } else if (targetSpanId === '__trace__') {
      selectTraceIO();
    }
  } else if (traceStart?.input || traceEnd?.output) {
    selectTraceIO();
  } else if (spans.length) {
    const roots = spans.filter(s => !s.parent_id).sort((a, b) => (a.ts_start || '') < (b.ts_start || '') ? -1 : 1);
    if (roots.length) selectSpan(roots[0].id);
  }
}

const _childCache = new Map();

function renderChildNode(child, depth, t0, dur) {
  const chevron = child.expanded ? '\u25BC' : '\u25B6';
  const statsHtml = child.stats
    ? `<span class="tok"><span class="in">${fmtN(child.stats.total_input_tokens)}</span>/<span class="out">${fmtN(child.stats.total_output_tokens)}</span></span> &middot; ${fmtDur(child.stats.duration_s)} &middot; ${child.stats.spans} spans`
    : '<span style="color:var(--yellow)">running\u2026</span>';

  let h = `<div class="child-trace-node${child.expanded ? ' expanded' : ''}" style="--depth:${depth}" data-child-name="${esc(child.name)}" data-action="toggle-child">
    <div class="span-info">
      <span class="child-chevron">${chevron}</span>
      <span class="badge badge-child">child</span>
      <span class="name">${esc(child.name)}</span>
      <span class="child-stats">${statsHtml}</span>
    </div>
    <div class="span-timing">
      <a class="child-open-link" data-action="open-child" data-file="${esc(child.file)}" title="Open full trace">open \u2197</a>
    </div>
  </div>`;

  if (child.expanded) {
    h += `<div class="child-trace-content" style="--depth:${depth}">`;
    if (child.loading) {
      h += '<div class="child-loading">Loading child trace\u2026</div>';
    } else if (child.fetchError) {
      h += `<div class="child-error">Failed to load: ${esc(child.fetchError)}</div>`;
    } else if (child.parsed) {
      h += renderChildSpanTree(child, depth + 1, t0, dur);
    }
    h += '</div>';
  }
  return h;
}

function renderChildSpanTree(child, depth, parentT0, parentDur) {
  const { spans, logEvents, traceStart, traceEnd, childTracers } = child.parsed;
  const t0 = traceStart?.ts ? new Date(traceStart.ts).getTime() : parentT0;
  const t1 = traceEnd?.ts ? new Date(traceEnd.ts).getTime() : t0;
  const dur = t1 - t0 || 1;

  const roots = [];
  const spanChildren = new Map();
  for (const s of spans) {
    if (!s.parent_id) {
      roots.push(s);
    } else {
      if (!spanChildren.has(s.parent_id)) spanChildren.set(s.parent_id, []);
      spanChildren.get(s.parent_id).push(s);
    }
  }
  const sortByStart = arr => arr.sort((a, b) => (a.ts_start || '') < (b.ts_start || '') ? -1 : 1);
  for (const c of spanChildren.values()) sortByStart(c);
  sortByStart(roots);

  function buildSubtree(span, d) {
    let h = renderSpanNode(span, d, t0, dur, 'child:' + child.name + ':' + span.id);
    for (const ch of spanChildren.get(span.id) || []) h += buildSubtree(ch, d + 1);
    return h;
  }

  // Build unified items: root spans + grandchildren
  const items = [];
  for (const r of roots) items.push({ type: 'span', span: r, ts: r.ts_start || '' });
  if (childTracers) {
    for (const [, gc] of childTracers) items.push({ type: 'child', child: gc, ts: gc.startedTs || '' });
  }
  items.sort((a, b) => (a.ts || '') < (b.ts || '') ? -1 : 1);

  let h = '';
  for (const item of items) {
    if (item.type === 'span') h += buildSubtree(item.span, depth);
    else h += renderChildNode(item.child, depth, t0, dur);
  }

  for (const ev of logEvents) {
    h += `<div class="span-node" style="--depth:${depth};opacity:0.6" data-span-id="ev-${esc(ev.id)}" data-action="select-event" data-event-id="${esc(ev.id)}" data-child-context="${esc(child.name)}">
      <div class="span-info"><span class="status-dot ok" style="background:var(--yellow)"></span><span class="badge badge-kind">event</span>
      <span class="name">${esc(ev.name)}</span></div>
      <div class="span-timing"></div></div>`;
  }

  return h || '<div style="padding:8px 0 8px calc(12px + ' + depth + ' * 20px);color:var(--text-dim);font-size:12px">No spans</div>';
}

function renderSpanTree() {
  const { spans, logEvents, traceStart, traceEnd, childTracers } = state.parsed;
  const t0 = traceStart?.ts ? new Date(traceStart.ts).getTime() : 0;
  const t1 = traceEnd?.ts ? new Date(traceEnd.ts).getTime() : 0;
  const dur = t1 - t0 || 1;

  // Build parent-child tree
  const roots = [];
  const children = new Map();
  for (const s of spans) {
    if (!s.parent_id) {
      roots.push(s);
    } else {
      if (!children.has(s.parent_id)) children.set(s.parent_id, []);
      children.get(s.parent_id).push(s);
    }
  }
  const sortByStart = arr => arr.sort((a, b) => (a.ts_start || '') < (b.ts_start || '') ? -1 : 1);
  for (const c of children.values()) sortByStart(c);
  sortByStart(roots);

  function buildSubtree(span, depth) {
    let h = renderSpanNode(span, depth, t0, dur);
    for (const child of children.get(span.id) || []) h += buildSubtree(child, depth + 1);
    return h;
  }

  let html = '';

  // Trace I/O node
  if (traceStart?.input || traceEnd?.output) {
    html += `<div class="span-node" style="--depth:0;opacity:0.7" data-span-id="__trace__" data-action="select-trace-io">
      <div class="span-info"><span class="status-dot ok"></span><span class="badge badge-kind">trace</span>
      <span class="name">Trace I/O</span></div>
      <div class="span-timing"><span class="duration">${fmtDur(traceEnd?.duration_s)}</span></div></div>`;
  }

  // Build unified items: root spans + child tracers, sorted by timestamp
  const items = [];
  for (const r of roots) items.push({ type: 'span', span: r, ts: r.ts_start || '' });
  if (childTracers) {
    for (const [, ct] of childTracers) items.push({ type: 'child', child: ct, ts: ct.startedTs || '' });
  }
  items.sort((a, b) => (a.ts || '') < (b.ts || '') ? -1 : 1);

  for (const item of items) {
    if (item.type === 'span') html += buildSubtree(item.span, 0);
    else html += renderChildNode(item.child, 0, t0, dur);
  }

  // Log event nodes
  for (const ev of logEvents) {
    html += `<div class="span-node" style="--depth:0;opacity:0.6" data-span-id="ev-${esc(ev.id)}" data-action="select-event" data-event-id="${esc(ev.id)}">
      <div class="span-info"><span class="status-dot ok" style="background:var(--yellow)"></span><span class="badge badge-kind">event</span>
      <span class="name">${esc(ev.name)}</span></div>
      <div class="span-timing"></div></div>`;
  }

  $('span-tree').innerHTML = html || '<div class="empty-state"><div class="msg">No spans</div></div>';
}

async function fetchChildTrace(child) {
  child.loading = true;
  renderSpanTree();
  try {
    const res = await fetch('/api/trace?file=' + encodeURIComponent(child.file));
    if (!res.ok) throw new Error(res.status === 404 ? 'Child trace not found' : `Server error ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    const parsed = parseEvents(data.events || [], child.file);
    child.parsed = parsed;
    child.fetchError = null;
    _childCache.set(child.file, parsed);
  } catch (e) {
    child.fetchError = e.message;
  } finally {
    child.loading = false;
    renderSpanTree();
  }
}

function toggleChild(childName) {
  if (!state.parsed?.childTracers) return;
  const child = state.parsed.childTracers.get(childName);
  if (!child) return;

  if (child.expanded) {
    child.expanded = false;
    renderSpanTree();
  } else {
    child.expanded = true;
    if (_childCache.has(child.file)) {
      child.parsed = _childCache.get(child.file);
      renderSpanTree();
    } else {
      fetchChildTrace(child);
    }
  }
}

function findChildByName(name, parsed) {
  if (!parsed?.childTracers) return null;
  return parsed.childTracers.get(name) || null;
}

function findNestedChild(qualifiedName) {
  // qualifiedName can be "childA" or nested "childA>childB"
  const parts = qualifiedName.split('>');
  let current = state.parsed;
  let child = null;
  for (const part of parts) {
    child = findChildByName(part, current);
    if (!child?.parsed) return child;
    current = child.parsed;
  }
  return child;
}

// ── Span Detail ──────────────────────────────────────────────────────────────

function renderSpanDetail(span) {
  let h = `<div class="detail-header">
    <span class="name">${esc(span.name)}</span>
    ${span.kind ? `<span class="badge ${kindBadge(span.kind)}">${esc(span.kind)}</span>` : ''}
    <span class="badge ${span.status === 'error' ? 'badge-error' : 'badge-ok'}">${span.status || 'ok'}</span>
    <span class="duration">${fmtDur(span.duration_s)}</span>
  </div>`;

  if (span.ts_start || span.ts_end) {
    h += `<div class="detail-section"><div class="section-header">Timing</div><table class="meta-table">
      ${span.ts_start ? `<tr><td>Start</td><td>${fmtTimeFull(span.ts_start)}</td></tr>` : ''}
      ${span.ts_end ? `<tr><td>End</td><td>${fmtTimeFull(span.ts_end)}</td></tr>` : ''}
      ${span.duration_s != null ? `<tr><td>Duration</td><td>${span.duration_s.toFixed(3)}s</td></tr>` : ''}
    </table></div>`;
  }

  if (span.tags?.length) {
    h += `<div class="detail-section"><div class="section-header">Tags</div>
      <div class="tags-row">${span.tags.map(t => `<span class="badge badge-tag">${esc(t)}</span>`).join('')}</div></div>`;
  }

  if (span.error) {
    h += `<div class="detail-section"><div class="section-header">Error</div>
      <div class="error-box"><div class="error-type">${esc(span.error.type || 'Error')}</div>
      <div class="error-msg">${esc(span.error.message || '')}</div></div></div>`;
  }

  if (span.metadata?.token_usage) {
    const tu = span.metadata.token_usage;
    const it = tu.input_tokens || 0;
    const ot = tu.output_tokens || 0;
    const tot = it + ot || 1;
    h += `<div class="detail-section"><div class="section-header">Token Usage</div>
      <div class="tok" style="margin-bottom:6px"><span class="in">${fmtN(it)} input</span> / <span class="out">${fmtN(ot)} output</span></div>
      <div class="token-bar"><div class="input-bar" style="width:${(it / tot * 100).toFixed(1)}%"></div><div class="output-bar" style="width:${(ot / tot * 100).toFixed(1)}%"></div></div>
      ${tu.reasoning_tokens ? `<div style="margin-top:4px;font-size:12px;color:var(--text-muted)">Reasoning: ${fmtN(tu.reasoning_tokens)}</div>` : ''}
      ${tu.cache_read_tokens ? `<div style="margin-top:2px;font-size:12px;color:var(--text-muted)">Cache read: ${fmtN(tu.cache_read_tokens)}</div>` : ''}
    </div>`;
  }

  if (span.input != null) h += jsonSection('Input', span.input);
  if (span.output != null) h += jsonSection('Output', span.output);

  if (span.metadata && Object.keys(span.metadata).length) {
    const m = { ...span.metadata };
    delete m.token_usage;
    if (Object.keys(m).length) h += jsonSection('Metadata', m);
  }

  return h;
}

function selectSpan(spanId) {
  if (!state.parsed) return;

  // Handle child span qualified IDs: child:{childName}:{spanId}
  if (spanId.startsWith('child:')) {
    selectChildSpan(spanId);
    return;
  }

  const span = state.parsed.spans.find(s => s.id === spanId);
  if (!span) return;
  state.selectedSpan = spanId;
  document.querySelectorAll('.span-node, .child-trace-node').forEach(n => n.classList.toggle('selected', n.dataset.spanId === spanId));
  _copy.data.clear();
  _copy.id = 0;

  // Update URL without creating history entry
  if (state.traceFile) {
    history.replaceState(null, '', '#trace:' + encodeURIComponent(state.traceFile) + '@' + encodeURIComponent(spanId));
  }

  $('span-detail').innerHTML = renderSpanDetail(span);
}

function selectChildSpan(qualifiedId) {
  // Format: child:{childName}:{spanId}
  const parts = qualifiedId.split(':');
  if (parts.length < 3) return;
  const childName = parts[1];
  const spanId = parts.slice(2).join(':');

  const child = state.parsed?.childTracers?.get(childName);
  if (!child?.parsed) return;

  const span = child.parsed.spans.find(s => s.id === spanId);
  if (!span) return;

  state.selectedSpan = qualifiedId;
  document.querySelectorAll('.span-node, .child-trace-node').forEach(n => n.classList.toggle('selected', n.dataset.spanId === qualifiedId));
  _copy.data.clear();
  _copy.id = 0;

  $('span-detail').innerHTML = renderSpanDetail(span);
}

function selectTraceIO() {
  if (!state.parsed) return;
  state.selectedSpan = '__trace__';
  document.querySelectorAll('.span-node, .child-trace-node').forEach(n => n.classList.toggle('selected', n.dataset.spanId === '__trace__'));
  _copy.data.clear();
  _copy.id = 0;

  if (state.traceFile) {
    history.replaceState(null, '', '#trace:' + encodeURIComponent(state.traceFile) + '@__trace__');
  }

  const { traceStart, traceEnd } = state.parsed;
  let h = '<div class="detail-header"><span class="name">Trace I/O</span><span class="badge badge-kind">trace</span></div>';
  if (traceStart?.input != null) h += jsonSection('Trace Input', traceStart.input);
  if (traceEnd?.output != null) h += jsonSection('Trace Output', traceEnd.output);
  if (traceStart?.metadata && Object.keys(traceStart.metadata).length) h += jsonSection('Trace Metadata', traceStart.metadata);
  $('span-detail').innerHTML = h;
}

function selectEvent(evId) {
  if (!state.parsed) return;
  const ev = state.parsed.logEvents.find(e => e.id === evId);
  if (!ev) return;
  state.selectedSpan = 'ev-' + evId;
  document.querySelectorAll('.span-node, .child-trace-node').forEach(n => n.classList.toggle('selected', n.dataset.spanId === state.selectedSpan));
  _copy.data.clear();
  _copy.id = 0;

  let h = `<div class="detail-header"><span class="name">${esc(ev.name)}</span><span class="badge badge-kind">event</span></div>`;
  if (ev.ts) h += `<div class="detail-section"><div class="section-header">Timestamp</div><div style="font-family:var(--font-mono);font-size:13px">${fmtTimeFull(ev.ts)}</div></div>`;
  if (ev.data != null) h += jsonSection('Data', ev.data);
  $('span-detail').innerHTML = h;
}

// ── Keyboard Navigation ─────────────────────────────────────────────────────

function navigateSpanTree(direction) {
  const nodes = [...document.querySelectorAll('.span-node, .child-trace-node')];
  if (!nodes.length) return;

  const idx = nodes.findIndex(n => n.classList.contains('selected'));
  let next;
  if (direction === 'up') {
    next = idx <= 0 ? nodes.length - 1 : idx - 1;
  } else {
    next = idx >= nodes.length - 1 ? 0 : idx + 1;
  }

  const node = nodes[next];
  const action = node.dataset.action;
  if (action === 'select-span') selectSpan(node.dataset.spanId);
  else if (action === 'select-trace-io') selectTraceIO();
  else if (action === 'select-event') selectEvent(node.dataset.eventId);
  else if (action === 'toggle-child') toggleChild(node.dataset.childName);

  node.scrollIntoView({ block: 'nearest' });
}

// ── Routing ──────────────────────────────────────────────────────────────────

function goto(hash) { location.hash = hash; }

function route() {
  const raw = decodeURIComponent(location.hash.slice(1));

  if (raw.startsWith('trace:')) {
    const rest = raw.slice(6);
    const at = rest.lastIndexOf('@');
    const file = at > 0 ? rest.slice(0, at) : rest;
    const spanId = at > 0 ? rest.slice(at + 1) : null;

    // If trace is already loaded, just select the span
    if (file === state.traceFile && state.parsed) {
      if (spanId) {
        const found = state.parsed.spans.find(s => s.id === spanId);
        if (found) selectSpan(spanId);
        else if (spanId === '__trace__') selectTraceIO();
      }
    } else {
      loadTrace(file, spanId);
    }
  } else {
    showBrowseView(raw);
  }
}

window.addEventListener('hashchange', route);

// ── Event Handlers ───────────────────────────────────────────────────────────

function clearFilters() {
  $('search-input').value = '';
  state.statusFilter = 'all';
  state.selectedTags.clear();
  updateStatusToggleLabel();
  populateTagFilter();
  updateClearBtn();
  renderBrowseContent();
}

function updateClearBtn() {
  const hasFilters = $('search-input').value
    || state.statusFilter !== 'all'
    || state.selectedTags.size > 0;
  $('clear-filters').classList.toggle('visible', hasFilters);
}

function updateStatusToggleLabel() {
  const btn = $('status-dropdown').querySelector('.dropdown-toggle');
  const labels = { all: 'All statuses', ok: 'OK only', error: 'Errors only' };
  btn.innerHTML = `${labels[state.statusFilter]} <span class="caret">&#9662;</span>`;
  $('status-dropdown').querySelectorAll('.dropdown-item').forEach(el => {
    const sel = el.dataset.value === state.statusFilter;
    el.classList.toggle('selected', sel);
    el.querySelector('.check').innerHTML = sel ? '&#10003;' : '';
  });
}

function initEventListeners() {
  // Delegated click handler
  $('app').addEventListener('click', (e) => {
    // Close open dropdowns when clicking outside
    const clickedDropdown = e.target.closest('.dropdown');
    document.querySelectorAll('.dropdown.open').forEach(d => {
      if (d !== clickedDropdown) d.classList.remove('open');
    });

    const el = e.target.closest('[data-action]');
    if (!el) return;
    e.preventDefault();
    switch (el.dataset.action) {
      case 'navigate':       goto(el.dataset.path || ''); break;
      case 'go-root':        goto(''); break;
      case 'open-trace':     goto('trace:' + el.dataset.file); break;
      case 'select-span':    selectSpan(el.dataset.spanId); break;
      case 'select-trace-io': selectTraceIO(); break;
      case 'select-event':   selectEvent(el.dataset.eventId); break;
      case 'toggle-child':   { e.stopPropagation(); toggleChild(el.dataset.childName); break; }
      case 'open-child':     { e.stopPropagation(); goto('trace:' + el.dataset.file); break; }
      case 'toggle-section': {
        const t = document.getElementById(el.dataset.target);
        const tog = el.querySelector('.toggle');
        if (t.style.display === 'none') { t.style.display = ''; if (tog) tog.classList.remove('collapsed'); }
        else { t.style.display = 'none'; if (tog) tog.classList.add('collapsed'); }
        break;
      }
      case 'toggle-guide':   $('guide-overlay').classList.toggle('open'); break;
      case 'toggle-dropdown': {
        const dd = $(el.dataset.dropdown);
        dd.classList.toggle('open');
        break;
      }
      case 'select-status': {
        state.statusFilter = el.dataset.value;
        updateStatusToggleLabel();
        $('status-dropdown').classList.remove('open');
        updateClearBtn();
        renderBrowseContent();
        break;
      }
      case 'toggle-tag': {
        const tag = el.dataset.tag;
        if (state.selectedTags.has(tag)) state.selectedTags.delete(tag);
        else state.selectedTags.add(tag);
        populateTagFilter();
        updateClearBtn();
        renderBrowseContent();
        break;
      }
      case 'clear-filters':  clearFilters(); break;
      case 'go-back':        history.back(); break;
      case 'dismiss-error':  dismissError(); break;
      case 'copy': {
        const value = _copy.data.get(el.dataset.copyId);
        if (value === undefined) break;
        const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
        navigator.clipboard.writeText(text).then(() => {
          el.textContent = 'Copied!';
          setTimeout(() => { el.textContent = 'Copy'; }, 1200);
        });
        break;
      }
    }
  });

  // Close guide overlay on backdrop click
  $('guide-overlay').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) $('guide-overlay').classList.remove('open');
  });

  // Filter handlers
  $('search-input').addEventListener('input', debounce(() => { updateClearBtn(); renderBrowseContent(); }, 150));

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;

    if (e.key === 'Escape') {
      const openDD = document.querySelector('.dropdown.open');
      if (openDD) { openDD.classList.remove('open'); return; }
      const guide = $('guide-overlay');
      if (guide.classList.contains('open')) guide.classList.remove('open');
      else if (state.traceFile) history.back();
    }
    if (e.key === '?') $('guide-overlay').classList.toggle('open');
    if (e.key === 'r' && !state.traceFile) {
      fetchTraces().then(() => renderBrowseContent());
    }
    if (state.traceFile) {
      if (e.key === 'ArrowUp') { e.preventDefault(); navigateSpanTree('up'); }
      if (e.key === 'ArrowDown') { e.preventDefault(); navigateSpanTree('down'); }
    }
  });
}

// ── Init ─────────────────────────────────────────────────────────────────────

(async () => {
  initEventListeners();
  await fetchTraces();
  route();
})();
