(() => {
  const API = ""; // same origin via nginx proxy
  const form = document.getElementById("search-form");
  const input = document.getElementById("query");
  const status = document.getElementById("status");
  const results = document.getElementById("results");
  const pager = document.getElementById("pager");
  const prevBtn = document.getElementById("prev");
  const nextBtn = document.getElementById("next");
  const pageInfo = document.getElementById("page-info");

  const PAGE_SIZE = 12;
  let currentQuery = "";
  let page = 1;
  let totalPages = 0;

  async function search(q, p) {
    status.textContent = "检索中…";
    results.hidden = true;
    pager.hidden = true;

    const url = `${API}/api/search?q=${encodeURIComponent(q)}&page=${p}&page_size=${PAGE_SIZE}`;
    const resp = await fetch(url);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  function render(data) {
    currentQuery = data.query;
    page = data.page;
    totalPages = data.total_pages || 0;

    status.textContent = data.total
      ? `共 ${data.total} 条候选 · 第 ${data.page} / ${data.total_pages} 页`
      : "没有找到相关 meme";

    results.innerHTML = "";
    data.results.forEach((item, i) => {
      const card = document.createElement("article");
      card.className = "card";
      card.style.animationDelay = `${i * 0.04}s`;
      card.innerHTML = `
        <img src="${item.image_url}" alt="${escapeHtml(item.gt || item.meme_id)}" loading="lazy" />
        <div class="meta">
          <span class="rank">#${item.rank}</span>
          <span class="gt">${escapeHtml(item.gt || item.meme_id)}</span>
          <span class="score">相似度 ${(item.score * 100).toFixed(1)}%</span>
        </div>
      `;
      results.appendChild(card);
    });

    results.hidden = data.results.length === 0;
    pager.hidden = data.total === 0;
    pageInfo.textContent = `${page} / ${Math.max(totalPages, 1)}`;
    prevBtn.disabled = page <= 1;
    nextBtn.disabled = page >= totalPages;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    try {
      const data = await search(q, 1);
      render(data);
    } catch (err) {
      status.textContent = `出错：${err.message}`;
    }
  });

  prevBtn.addEventListener("click", async () => {
    if (page <= 1) return;
    try {
      render(await search(currentQuery, page - 1));
    } catch (err) {
      status.textContent = `出错：${err.message}`;
    }
  });

  nextBtn.addEventListener("click", async () => {
    if (page >= totalPages) return;
    try {
      render(await search(currentQuery, page + 1));
    } catch (err) {
      status.textContent = `出错：${err.message}`;
    }
  });

  // health check
  fetch(`${API}/api/health`)
    .then((r) => r.json())
    .then((h) => {
      if (!h.es) status.textContent = "后端未连接 Elasticsearch";
      else if (!h.indexed) status.textContent = "索引为空，请稍候（首次启动会自动建库）";
      else status.textContent = `就绪 · 已索引 ${h.indexed} 张 meme`;
    })
    .catch(() => {
      status.textContent = "无法连接后端，请确认服务已启动";
    });
})();
