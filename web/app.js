const state = {
  index: null,
  results: [],
  selectedQualifiedName: null,
};

const els = {
  search: document.getElementById("search-input"),
  kind: document.getElementById("kind-filter"),
  publicOnly: document.getElementById("public-only"),
  stats: document.getElementById("stats"),
  results: document.getElementById("results"),
  count: document.getElementById("result-count"),
  details: document.getElementById("details"),
};

const KIND_ORDER = {
  class: 1,
  function: 2,
  method: 3,
  classmethod: 3,
  staticmethod: 3,
  property: 4,
  field: 5,
  module: 6,
  alias: 7,
  constant: 8,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayKind(kind) {
  if (kind === "classmethod") return "class method";
  if (kind === "staticmethod") return "static method";
  return kind;
}

function displaySignature(item) {
  if (!item.signature) return item.qualifiedName;
  if (item.kind === "field" || item.kind === "alias") {
    return `${item.name}${item.signature}`;
  }
  if (item.kind === "module" || item.kind === "constant") {
    return item.qualifiedName;
  }
  return `${item.name}${item.signature}`;
}

function sourceText(item) {
  const source = item.source || {};
  return `${source.path || ""}:${source.line || 1}`;
}

function parseHash() {
  const params = new URLSearchParams(window.location.hash.slice(1));
  return {
    q: params.get("q") || "",
    kind: params.get("kind") || "all",
    selected: params.get("item") || "",
    publicOnly: params.get("private") !== "1",
  };
}

function writeHash() {
  const params = new URLSearchParams();
  const query = els.search.value.trim();
  if (query) params.set("q", query);
  if (els.kind.value !== "all") params.set("kind", els.kind.value);
  if (!els.publicOnly.checked) params.set("private", "1");
  if (state.selectedQualifiedName) params.set("item", state.selectedQualifiedName);
  const next = params.toString();
  const hash = next ? `#${next}` : window.location.pathname;
  window.history.replaceState(null, "", hash);
}

function tokensFor(query) {
  return query
    .toLowerCase()
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
}

function scoreItem(item, tokens, rawQuery) {
  const haystack = item.searchText || "";
  if (tokens.some((token) => !haystack.includes(token))) return 0;

  const qname = item.qualifiedName.toLowerCase();
  const name = item.name.toLowerCase();
  const signature = (item.signature || "").toLowerCase();
  const summary = (item.summary || "").toLowerCase();
  const docstring = (item.docstring || "").toLowerCase();
  let score = item.public ? 4 : 0;

  if (!tokens.length) {
    score += 40 - (KIND_ORDER[item.kind] || 20);
    if (item.kind === "module") score -= 10;
    return score;
  }

  if (qname === rawQuery) score += 120;
  if (qname.includes(rawQuery)) score += 45;
  if (name.includes(rawQuery)) score += 28;

  for (const token of tokens) {
    if (name === token) score += 60;
    else if (name.startsWith(token)) score += 36;
    else if (name.includes(token)) score += 24;

    if (qname.includes(token)) score += 18;
    if (signature.includes(token)) score += 10;
    if (summary.includes(token)) score += 8;
    if (docstring.includes(token)) score += 3;
  }

  score += 10 - (KIND_ORDER[item.kind] || 10);
  return score;
}

function compareItems(a, b) {
  if (b.score !== a.score) return b.score - a.score;
  const kindDelta = (KIND_ORDER[a.item.kind] || 20) - (KIND_ORDER[b.item.kind] || 20);
  if (kindDelta !== 0) return kindDelta;
  return a.item.qualifiedName.localeCompare(b.item.qualifiedName);
}

function filteredItems() {
  const query = els.search.value.trim().toLowerCase();
  const tokens = tokensFor(query);
  const kind = els.kind.value;
  const publicOnly = els.publicOnly.checked;

  return state.index.items
    .filter((item) => {
      if (kind === "all") return true;
      if (kind === "method") return ["method", "classmethod", "staticmethod"].includes(item.kind);
      return item.kind === kind;
    })
    .filter((item) => !publicOnly || item.public)
    .map((item) => ({ item, score: scoreItem(item, tokens, query) }))
    .filter((entry) => entry.score > 0)
    .sort(compareItems)
    .map((entry) => entry.item);
}

function renderStats() {
  const stats = state.index.stats || {};
  const entries = [
    ["items", state.index.items.length],
    ["classes", stats.class || 0],
    ["functions", stats.function || 0],
    ["methods", (stats.method || 0) + (stats.classmethod || 0) + (stats.staticmethod || 0)],
    ["fields", stats.field || 0],
  ];

  els.stats.innerHTML = entries
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`)
    .join("");
}

function renderResults() {
  state.results = filteredItems();
  const visible = state.results.slice(0, 250);
  els.count.textContent = `${state.results.length} match${state.results.length === 1 ? "" : "es"}`;

  if (!visible.length) {
    els.results.innerHTML = "";
    els.details.innerHTML = '<p class="empty-state">No API definitions match the current filters.</p>';
    return;
  }

  if (!state.selectedQualifiedName || !state.results.some((item) => item.qualifiedName === state.selectedQualifiedName)) {
    state.selectedQualifiedName = visible[0].qualifiedName;
  }

  els.results.innerHTML = visible
    .map((item) => {
      const active = item.qualifiedName === state.selectedQualifiedName ? " active" : "";
      const summary = item.summary ? `<span class="result-summary">${escapeHtml(item.summary)}</span>` : "";
      return `
        <li>
          <button class="result-button${active}" type="button" data-qname="${escapeHtml(item.qualifiedName)}">
            <span class="result-title">${escapeHtml(item.qualifiedName)}</span>
            <span class="result-meta">
              <span class="badge">${escapeHtml(displayKind(item.kind))}</span>
              <span class="badge ${item.public ? "public" : "private"}">${item.public ? "public" : "private"}</span>
              <span>${escapeHtml(sourceText(item))}</span>
            </span>
            ${summary}
          </button>
        </li>
      `;
    })
    .join("");

  renderDetails(state.results.find((item) => item.qualifiedName === state.selectedQualifiedName) || visible[0]);
}

function renderDetails(item) {
  state.selectedQualifiedName = item.qualifiedName;
  const source = item.source || {};
  const decorators = item.decorators?.length
    ? `<div class="detail-section"><h3>Decorators</h3><pre class="signature"><code>${escapeHtml(item.decorators.join("\n"))}</code></pre></div>`
    : "";
  const bases = item.bases?.length
    ? `<div class="detail-section"><h3>Bases</h3><p>${escapeHtml(item.bases.join(", "))}</p></div>`
    : "";
  const exports = item.exports?.length
    ? `<div class="detail-section"><h3>Exports</h3><p>${escapeHtml(item.exports.join(", "))}</p></div>`
    : "";
  const docstring = item.docstring
    ? escapeHtml(item.docstring)
    : "No docstring found in source.";

  els.details.innerHTML = `
    <header class="details-header">
      <div class="details-meta">
        <span class="badge">${escapeHtml(displayKind(item.kind))}</span>
        <span class="badge ${item.public ? "public" : "private"}">${item.public ? "public" : "private"}</span>
      </div>
      <h2 class="details-title">${escapeHtml(item.qualifiedName)}</h2>
      <div class="source-link">${escapeHtml(source.path || "")}:${escapeHtml(source.line || 1)}</div>
    </header>
    <pre class="signature"><code>${escapeHtml(displaySignature(item))}</code></pre>
    <section class="detail-section">
      <h3>Docstring</h3>
      <pre class="docstring">${docstring}</pre>
    </section>
    ${bases}
    ${decorators}
    ${exports}
  `;

  for (const button of els.results.querySelectorAll(".result-button")) {
    button.classList.toggle("active", button.dataset.qname === item.qualifiedName);
  }
  writeHash();
}

function attachEvents() {
  els.search.addEventListener("input", () => {
    state.selectedQualifiedName = null;
    renderResults();
    writeHash();
  });
  els.kind.addEventListener("change", () => {
    state.selectedQualifiedName = null;
    renderResults();
    writeHash();
  });
  els.publicOnly.addEventListener("change", () => {
    state.selectedQualifiedName = null;
    renderResults();
    writeHash();
  });
  els.results.addEventListener("click", (event) => {
    const button = event.target.closest(".result-button");
    if (!button) return;
    const item = state.results.find((entry) => entry.qualifiedName === button.dataset.qname);
    if (item) renderDetails(item);
  });
}

async function loadIndex() {
  const response = await fetch("api.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load api.json (${response.status})`);
  }
  return response.json();
}

async function main() {
  attachEvents();
  const hash = parseHash();
  els.search.value = hash.q;
  els.kind.value = hash.kind;
  els.publicOnly.checked = hash.publicOnly;
  state.selectedQualifiedName = hash.selected || null;

  try {
    state.index = await loadIndex();
    renderStats();
    renderResults();
  } catch (error) {
    els.count.textContent = "Unavailable";
    els.details.innerHTML = `
      <p class="error-state">
        ${escapeHtml(error.message)}. Serve the web folder over HTTP, or deploy it with GitHub Pages.
      </p>
    `;
  }
}

main();
