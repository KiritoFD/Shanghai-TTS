const state = {
  rebuildTimer: null,
  sidebarWidth: 360,
};

const els = {
  sidebar: document.getElementById("sidebar"),
  main: document.getElementById("main"),
  sidebarToggle: document.getElementById("sidebar-toggle"),
  desktopMenu: document.getElementById("desktop-menu"),
  mobileMenu: document.getElementById("mobile-menu"),
  sidebarResizer: document.getElementById("sidebar-resizer"),
  chatWindow: document.getElementById("chat-window"),
  userInput: document.getElementById("userInput"),
  sendBtn: document.getElementById("sendBtn"),
  directModelChip: document.getElementById("direct-model-chip"),
  mobileDirectModel: document.getElementById("mobile-direct-model"),
  rebuildButton: document.getElementById("rebuild-button"),
  rebuildModal: document.getElementById("rebuild-modal"),
  rebuildLog: document.getElementById("rebuild-log"),
  closeRebuild: document.getElementById("close-rebuild"),
  doneRebuild: document.getElementById("done-rebuild"),
};

const SIDEBAR_WIDTH_KEY = "shanghaiTts.sidebarWidth";
const SIDEBAR_MIN = 320;
const SIDEBAR_MAX = 520;
const SIDEBAR_DEFAULT = 360;

function isOverlayMode() {
  return window.matchMedia("(max-width: 900px)").matches;
}

function isPhoneMode() {
  return window.matchMedia("(max-width: 640px)").matches;
}

function clampSidebarWidth(width) {
  const numeric = Number(width);
  if (!Number.isFinite(numeric)) return SIDEBAR_DEFAULT;
  return Math.max(SIDEBAR_MIN, Math.min(SIDEBAR_MAX, Math.round(numeric)));
}

function getStoredSidebarWidth() {
  return clampSidebarWidth(window.localStorage.getItem(SIDEBAR_WIDTH_KEY) || SIDEBAR_DEFAULT);
}

function overlaySidebarWidth(width) {
  if (isPhoneMode()) return "100vw";
  return width > window.innerWidth / 2 ? "100vw" : `${Math.min(width, Math.floor(window.innerWidth / 2))}px`;
}

function applySidebarWidth(width) {
  state.sidebarWidth = clampSidebarWidth(width);
  document.documentElement.style.setProperty("--sidebar-width", `${state.sidebarWidth}px`);
  document.documentElement.style.setProperty("--overlay-sidebar-width", overlaySidebarWidth(state.sidebarWidth));
  window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(state.sidebarWidth));
  syncSidebarResizer();
}

function syncSidebarResizer() {
  if (!els.sidebarResizer) return;
  const hidden = isOverlayMode() || els.sidebar.classList.contains("collapsed");
  els.sidebarResizer.classList.toggle("hidden", hidden);
}

function setSidebarCollapsed(collapsed) {
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  els.sidebar.classList.toggle("collapsed", collapsed);
  els.main.classList.toggle("collapsed", collapsed);
  if (els.sidebarToggle) {
    els.sidebarToggle.innerHTML = collapsed
      ? `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>`
      : `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>`;
    els.sidebarToggle.setAttribute("aria-label", collapsed ? "展开侧边栏" : "收起侧边栏");
  }
  syncSidebarResizer();
}

function toggleSidebar() {
  setSidebarCollapsed(!els.sidebar.classList.contains("collapsed"));
}

function nowLabel() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
  els.chatWindow.scrollTop = els.chatWindow.scrollHeight;
}

function addMessage(role, text, options = {}) {
  const row = document.createElement("div");
  row.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "我" : "旦";

  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (options.html) bubble.innerHTML = text;
  else bubble.textContent = text;
  wrap.appendChild(bubble);

  if (options.audio) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = options.audio;
    wrap.appendChild(audio);
  }

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = options.meta || nowLabel();
  wrap.appendChild(meta);

  row.append(avatar, wrap);
  els.chatWindow.appendChild(row);
  scrollToBottom();
  return row;
}

function fillPrompt(text) {
  els.userInput.value = text;
  els.userInput.focus();
}

async function fetchJson(path, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const data = await response.json();
    return { response, data };
  } finally {
    window.clearTimeout(timer);
  }
}

function modelMetaText(status) {
  const ready = status.ready ? "资源齐全" : "缺少文件";
  const device = status.device || "unloaded";
  const step = status.checkpoint_step || "?";
  return `${ready} / 状态 ${device} / step ${step}`;
}

function setChipText(el, text, error = false) {
  if (!el) return;
  el.textContent = text;
  
  // Clear status classes
  el.classList.remove("error", "cuda", "cpu", "unloaded");
  
  if (error) {
    el.classList.add("error");
  } else {
    const cleanText = text.toLowerCase();
    if (cleanText.includes("cuda") || cleanText.includes("gpu")) {
      el.classList.add("cuda");
    } else if (cleanText.includes("cpu")) {
      el.classList.add("cpu");
    } else if (cleanText.includes("unloaded")) {
      el.classList.add("unloaded");
    }
  }
}

function renderTtsDisconnected(message) {
  for (const model of ["shanghai", "shaoxing"]) {
    setChipText(document.getElementById(`status-${model}`), "OFFLINE", true);
    const meta = document.getElementById(`meta-${model}`);
    if (meta) meta.textContent = message || "未连接";
    const card = document.getElementById(`card-${model}`);
    if (card) {
      card.querySelectorAll("button").forEach((button) => {
        button.classList.remove("active", "direct-active");
      });
    }
  }
  els.directModelChip.innerHTML = "<strong>拼音直读</strong> 未连接";
  els.directModelChip.classList.add("error");
  els.mobileDirectModel.textContent = "未连接";
}

function renderTtsStatus(tts) {
  const models = tts.models || {};
  const activeDirect = tts.active_direct_model || "";
  const labels = { shanghai: "上海话", shaoxing: "绍兴话" };

  for (const model of ["shanghai", "shaoxing"]) {
    const status = models[model];
    const statusEl = document.getElementById(`status-${model}`);
    const metaEl = document.getElementById(`meta-${model}`);
    const card = document.getElementById(`card-${model}`);
    const directBtn = document.getElementById(`direct-btn-${model}`);

    if (!status) {
      setChipText(statusEl, "不可用", true);
      if (metaEl) metaEl.textContent = "未发现模型配置";
      continue;
    }

    const badge = status.device === "unloaded" ? "UNLOADED" : String(status.device || "unloaded").toUpperCase();
    setChipText(statusEl, badge, false);
    if (metaEl) metaEl.textContent = modelMetaText(status);

    if (directBtn) {
      directBtn.classList.remove("active");
      directBtn.classList.toggle("direct-active", activeDirect === model);
      directBtn.textContent = activeDirect === model ? "当前拼音直读" : "设为拼音直读";
    }

    if (card) {
      card.querySelectorAll("[data-load]").forEach((button) => {
        const [, device] = button.dataset.load.split(":");
        button.classList.toggle("active", status.device === device);
      });
    }
  }

  const label = labels[activeDirect] || activeDirect || "未设置";
  els.directModelChip.innerHTML = `<strong>拼音直读</strong> ${label}`;
  els.directModelChip.classList.remove("error");
  els.mobileDirectModel.textContent = label;
}

async function refreshTtsStatus() {
  const { data } = await fetchJson("/api/tts/status", {}, 8000);
  if (!data.ok) throw new Error(data.error || "读取 TTS 状态失败");
  renderTtsStatus(data.tts || {});
}

async function loadTtsModel(model, device) {
  const button = document.querySelector(`[data-load="${model}:${device}"]`);
  const originalText = button ? button.textContent : "";
  if (button) {
    button.disabled = true;
    button.textContent = "加载中";
  }
  try {
    const { response, data } = await fetchJson(
      "/api/tts/load",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, device }),
      },
      90000,
    );
    if (!response.ok || !data.ok) throw new Error(data.error || "模型切换失败");
    renderTtsStatus(data.tts || {});
    addMessage("ai", `${model} ${device === "unloaded" ? "已卸载" : `已切到 ${device.toUpperCase()}`}`);
  } catch (error) {
    addMessage("ai", `模型加载失败：${error.message || error}`);
    await refreshTtsStatus().catch(() => null);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = originalText;
    }
  }
}

async function setDirectModel(model) {
  try {
    const { response, data } = await fetchJson(
      "/api/tts/direct_model",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      },
      15000,
    );
    if (!response.ok || !data.ok) throw new Error(data.error || "设置拼音直读失败");
    renderTtsStatus(data.tts || {});
  } catch (error) {
    addMessage("ai", `切换拼音直读模型失败：${error.message || error}`);
  }
}

async function readHeadword(button) {
  const headword = button.dataset.headword || "";
  const source = button.dataset.source || "shanghai_csv";
  const originalHtml = button.innerHTML;
  button.disabled = true;
  button.innerHTML = "<span class='voice-icon'>...</span><span>生成中</span>";
  try {
    const { response, data } = await fetchJson(
      "/api/tts/read_headword",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ headword, source }),
      },
      90000,
    );
    if (!response.ok) throw new Error(data.text || data.error || "词条朗读失败");
    addMessage("ai", data.text || "已生成语音。", { html: true, audio: data.audio || "" });
  } catch (error) {
    addMessage("ai", `词条朗读失败：${error.message || error}`);
  } finally {
    button.disabled = false;
    button.innerHTML = originalHtml;
  }
}

async function sendMessage(customText = null) {
  const text = (customText ?? els.userInput.value).trim();
  if (!text) return;
  addMessage("user", text);
  els.userInput.value = "";
  els.sendBtn.disabled = true;
  const pending = addMessage("ai", "思考中...");
  try {
    const { response, data } = await fetchJson(
      "/api/chat",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      },
      60000,
    );
    pending.remove();
    if (!response.ok) throw new Error(data.text || data.error || "请求失败");
    addMessage("ai", data.text || "没有返回内容。", { html: true, audio: data.audio || "" });
  } catch (error) {
    pending.remove();
    addMessage("ai", `请求失败：${error.message || error}`);
  } finally {
    els.sendBtn.disabled = false;
    if (window.matchMedia("(min-width: 901px)").matches) els.userInput.focus();
  }
}

function closeRebuildModal() {
  els.rebuildModal.classList.remove("show");
  els.rebuildModal.setAttribute("aria-hidden", "true");
  if (state.rebuildTimer) {
    window.clearTimeout(state.rebuildTimer);
    state.rebuildTimer = null;
  }
}

async function pollRebuildStatus() {
  try {
    const { data } = await fetchJson("/api/rebuild_index/status", {}, 8000);
    const status = data.status || {};
    const lines = [];
    if (Array.isArray(status.progress)) lines.push(...status.progress);
    if (status.error) lines.push(`ERROR: ${status.error}`);
    if (status.done) lines.push("完成。");
    els.rebuildLog.textContent = lines.join("\n") || "处理中...";
    if (status.running) state.rebuildTimer = window.setTimeout(pollRebuildStatus, 1200);
  } catch (error) {
    els.rebuildLog.textContent = `状态查询失败：${error.message || error}`;
  }
}

async function startRebuild() {
  els.rebuildModal.classList.add("show");
  els.rebuildModal.setAttribute("aria-hidden", "false");
  els.rebuildLog.textContent = "准备启动重建...";
  try {
    const { data } = await fetchJson("/api/rebuild_index", { method: "POST" }, 15000);
    els.rebuildLog.textContent = data.message || data.error || "已启动";
    if (data.ok) pollRebuildStatus();
  } catch (error) {
    els.rebuildLog.textContent = `启动失败：${error.message || error}`;
  }
}

function bindEvents() {
  els.sidebarToggle.addEventListener("click", toggleSidebar);
  els.desktopMenu.addEventListener("click", () => setSidebarCollapsed(false));
  els.mobileMenu.addEventListener("click", () => setSidebarCollapsed(false));
  els.sendBtn.addEventListener("click", () => sendMessage());
  els.rebuildButton.addEventListener("click", startRebuild);
  els.closeRebuild.addEventListener("click", closeRebuildModal);
  els.doneRebuild.addEventListener("click", closeRebuildModal);

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => fillPrompt(button.dataset.prompt || ""));
  });
  document.querySelectorAll("[data-load]").forEach((button) => {
    button.addEventListener("click", () => {
      const [model, device] = button.dataset.load.split(":");
      loadTtsModel(model, device);
    });
  });
  document.querySelectorAll("[data-direct]").forEach((button) => {
    button.addEventListener("click", () => setDirectModel(button.dataset.direct));
  });

  els.chatWindow.addEventListener("click", (event) => {
    const button = event.target.closest(".voice-btn");
    if (button) readHeadword(button);
  });

  els.userInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  if (els.sidebarResizer) {
    els.sidebarResizer.addEventListener("pointerdown", (event) => {
      if (isOverlayMode()) return;
      const startX = event.clientX;
      const startWidth = state.sidebarWidth;
      els.sidebarResizer.classList.add("dragging");
      els.sidebarResizer.setPointerCapture(event.pointerId);

      const onMove = (moveEvent) => {
        applySidebarWidth(startWidth + moveEvent.clientX - startX);
      };
      const onUp = () => {
        els.sidebarResizer.classList.remove("dragging");
        window.removeEventListener("pointermove", onMove);
      };

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp, { once: true });
    });
  }

  window.addEventListener("resize", () => applySidebarWidth(state.sidebarWidth));
}

function init() {
  applySidebarWidth(getStoredSidebarWidth());
  setSidebarCollapsed(isOverlayMode());
  bindEvents();
  refreshTtsStatus().catch((error) => renderTtsDisconnected(error.message || "未连接"));
}

init();
