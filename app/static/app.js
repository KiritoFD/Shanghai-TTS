(() => {
    const chatWindow = document.getElementById("chat-window");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const rebuildBtn = document.getElementById("rebuildBtn");
    const rebuildModal = document.getElementById("rebuild-modal");
    const rebuildLog = document.getElementById("rebuild-log");
    const closeRebuildBtn = document.getElementById("closeRebuildBtn");
    const finishRebuildBtn = document.getElementById("finishRebuildBtn");
    const directModelChip = document.getElementById("direct-model-chip");
    const liveStatusChip = document.getElementById("live-status-chip");
    const composerStatus = document.getElementById("composer-status");
    const emptyState = document.getElementById("empty-state");
    const pipelineButtons = Array.from(document.querySelectorAll("#pipeline-mode button"));
    const staticActionButtons = Array.from(
        document.querySelectorAll(".topbar button, .sidebar button, .composer button, .modal button"),
    );

    let rebuildTimer = null;
    let pipelineMode = localStorage.getItem("shanghai_tts_pipeline_mode") || "lookup";
    let activeUiRequests = 0;

    function nowLabel() {
        return new Date().toLocaleTimeString("zh-CN", {
            hour: "2-digit",
            minute: "2-digit",
        });
    }

    function scrollToBottom() {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function nextFrame() {
        return new Promise((resolve) => requestAnimationFrame(() => resolve()));
    }

    function syncEmptyState() {
        if (!emptyState) {
            return;
        }
        const realRows = chatWindow.querySelectorAll(".row");
        emptyState.classList.toggle("hidden", realRows.length > 1);
    }

    function fillPrompt(text) {
        userInput.value = text;
        userInput.focus();
    }

    function shouldAutoFocusComposer() {
        return window.matchMedia("(min-width: 901px)").matches;
    }

    function setPipelineMode(mode) {
        pipelineMode = mode === "rag_mt" ? "rag_mt" : "lookup";
        localStorage.setItem("shanghai_tts_pipeline_mode", pipelineMode);
        pipelineButtons.forEach((button) => {
            button.classList.toggle("active", button.dataset.mode === pipelineMode);
        });
        if (composerStatus) {
            composerStatus.textContent =
                pipelineMode === "rag_mt"
                    ? "当前为翻译增强模式，响应会更慢。按 Enter 发送，Shift+Enter 换行。"
                    : "当前为查询优先模式。按 Enter 发送，Shift+Enter 换行。";
        }
    }

    function setLiveStatus(label, detail = "") {
        if (!liveStatusChip) {
            return;
        }
        const stateMap = {
            "处理中": "busy",
            "完成": "done",
            "失败": "error",
            "空闲": "idle",
        };
        liveStatusChip.dataset.state = stateMap[label] || "idle";
        liveStatusChip.innerHTML = `<strong>请求</strong> ${label}${detail ? ` · ${detail}` : ""}`;
    }

    function setUiBusy(isBusy) {
        activeUiRequests = Math.max(0, activeUiRequests + (isBusy ? 1 : -1));
        const busy = activeUiRequests > 0;
        document.body.classList.toggle("busy", busy);
        staticActionButtons.forEach((button) => {
            button.disabled = busy;
        });
        chatWindow.querySelectorAll(".voice-btn").forEach((button) => {
            button.disabled = busy;
        });
        userInput.disabled = busy;
        if (!busy && shouldAutoFocusComposer()) {
            userInput.focus();
        }
    }

    function createPendingMessage(kind = "lookup") {
        const stagesByKind = {
            lookup: ["规范化查询...", "词典命中与召回...", "整理结果..."],
            rag_mt: ["规范化查询...", "翻译增强中...", "补充词典结果...", "整理结果..."],
            tts: ["准备请求...", "加载模型或查词...", "生成音频..."],
            status: ["处理中..."],
        };
        const stages = stagesByKind[kind] || stagesByKind.status;
        let stageIndex = 0;
        const row = addMessage(
            "ai",
            `<div>处理中...</div><div class="pending-stage">${stages[0]}</div><div class="pending-bar"><span></span></div>`,
            { isHtml: true, trustedHtml: true },
        );
        const stageEl = row.querySelector(".pending-stage");
        const timer = setInterval(() => {
            stageIndex = Math.min(stageIndex + 1, stages.length - 1);
            if (stageEl) {
                stageEl.textContent = stages[stageIndex];
            }
        }, 850);
        return {
            finish() {
                clearInterval(timer);
                row.remove();
                syncEmptyState();
            },
            fail(message) {
                clearInterval(timer);
                if (stageEl) {
                    stageEl.textContent = message;
                }
            },
        };
    }

    function timingEntriesFromPayload(data) {
        if (data && data.timings_ms && typeof data.timings_ms === "object") {
            return Object.entries(data.timings_ms);
        }
        if (data && data.dictionary_result && data.dictionary_result.timings_ms) {
            return Object.entries(data.dictionary_result.timings_ms);
        }
        return [];
    }

    function decorateResponseHtml(html, data) {
        const entries = timingEntriesFromPayload(data)
            .filter(([, value]) => Number.isFinite(Number(value)))
            .slice(0, 6);
        if (!entries.length) {
            return html;
        }
        const labels = {
            total: "总耗时",
            parse: "解析",
            build_terms: "展词",
            exact_hits: "精确命中",
            shanghai_recall: "上海召回",
            shaoxing_lookup: "绍兴检索",
            translate: "翻译",
            context_lookup: "上下文",
            rag_mt_total: "翻译链路",
            lookup_total: "词典链路",
        };
        const pills = entries
            .map(
                ([key, value]) =>
                    `<span class="timing-pill">${labels[key] || key} ${Math.round(Number(value))}ms</span>`,
            )
            .join("");
        return `${html}<div class="timing-strip">${pills}</div>`;
    }

    function sanitizeHtmlFragment(html) {
        const template = document.createElement("template");
        template.innerHTML = html;
        const allowedTags = new Set(["A", "B", "BR", "BUTTON", "DIV", "HR", "SPAN", "STRONG"]);
        const allowedClasses = new Set([
            "pending-stage",
            "pending-bar",
            "timing-strip",
            "timing-pill",
            "voice-btn",
            "result-badge",
        ]);

        function cleanNode(node) {
            if (node.nodeType === Node.TEXT_NODE) {
                return document.createTextNode(node.textContent || "");
            }
            if (node.nodeType !== Node.ELEMENT_NODE) {
                return document.createDocumentFragment();
            }

            const tag = node.tagName.toUpperCase();
            if (!allowedTags.has(tag)) {
                const fragment = document.createDocumentFragment();
                Array.from(node.childNodes).forEach((child) => fragment.appendChild(cleanNode(child)));
                return fragment;
            }

            const el = document.createElement(tag.toLowerCase());
            if (tag === "A") {
                const href = node.getAttribute("href") || "";
                if (href.startsWith("/download/")) {
                    el.setAttribute("href", href);
                    el.setAttribute("target", "_blank");
                    el.setAttribute("rel", "noopener");
                }
            }
            if (tag === "BUTTON") {
                el.setAttribute("type", "button");
                const headword = node.getAttribute("data-headword") || "";
                const source = node.getAttribute("data-source") || "";
                if (headword) {
                    el.setAttribute("data-headword", headword);
                }
                if (source) {
                    el.setAttribute("data-source", source);
                }
            }

            const className = node.getAttribute("class") || "";
            const keptClasses = className
                .split(/\s+/)
                .filter((name) => allowedClasses.has(name))
                .join(" ");
            if (keptClasses) {
                el.setAttribute("class", keptClasses);
            }

            Array.from(node.childNodes).forEach((child) => el.appendChild(cleanNode(child)));
            return el;
        }

        const fragment = document.createDocumentFragment();
        Array.from(template.content.childNodes).forEach((child) => fragment.appendChild(cleanNode(child)));
        return fragment;
    }

    function renderBubbleHtml(bubble, html, trustedHtml = false) {
        bubble.replaceChildren();
        if (trustedHtml) {
            bubble.innerHTML = html;
            return;
        }
        bubble.appendChild(sanitizeHtmlFragment(html));
    }

    function addMessage(role, html, opts = {}) {
        const row = document.createElement("div");
        row.className = `row ${role}`;

        const avatar = document.createElement("div");
        avatar.className = `avatar ${role}`;
        avatar.textContent = role === "user" ? "我" : "SH";

        const wrap = document.createElement("div");
        wrap.className = "bubble-wrap";

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        if (opts.isHtml) {
            renderBubbleHtml(bubble, html, Boolean(opts.trustedHtml));
        } else {
            bubble.textContent = html;
        }

        wrap.appendChild(bubble);

        if (opts.audio) {
            const audio = document.createElement("audio");
            audio.controls = true;
            audio.src = opts.audio;
            wrap.appendChild(audio);
        }

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML = `<span>${nowLabel()}</span>`;
        wrap.appendChild(meta);

        row.appendChild(avatar);
        row.appendChild(wrap);
        chatWindow.appendChild(row);
        syncEmptyState();
        scrollToBottom();
        return row;
    }

    async function sendMessage(customText = null) {
        const text = (customText ?? userInput.value).trim();
        if (!text) {
            return;
        }

        addMessage("user", text);
        userInput.value = "";
        setUiBusy(true);
        const pending = createPendingMessage(pipelineMode === "rag_mt" ? "rag_mt" : "lookup");
        setLiveStatus("处理中", pipelineMode === "rag_mt" ? "翻译增强" : "查询优先");
        await nextFrame();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, pipeline_mode: pipelineMode }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.text || "请求失败");
            }
            pending.finish();
            addMessage("ai", decorateResponseHtml(data.text || "没有返回内容。", data), {
                isHtml: true,
                audio: data.audio || "",
            });
            if (data.audio) {
                refreshTtsStatus().catch(() => {});
            }
            if (composerStatus) {
                const totalMs = data.timings_ms && data.timings_ms.total ? Number(data.timings_ms.total) : null;
                const modeLabel = pipelineMode === "rag_mt" ? "翻译增强" : "查询优先";
                composerStatus.textContent = totalMs
                    ? `${modeLabel} | ${Math.round(totalMs)}ms`
                    : `${modeLabel} | 已完成`;
            }
            const totalMs = data.timings_ms && data.timings_ms.total ? Math.round(Number(data.timings_ms.total)) : null;
            setLiveStatus("完成", totalMs ? `${totalMs}ms` : "");
        } catch (error) {
            pending.finish();
            addMessage("ai", `请求失败：${error.message || error}`, { isHtml: false });
            if (composerStatus) {
                composerStatus.textContent = `请求失败：${error.message || error}`;
            }
            setLiveStatus("失败");
        } finally {
            setUiBusy(false);
        }
    }

    function modelMetaText(status) {
        if (!status) {
            return "状态未知";
        }
        const ready = status.ready ? "资源齐全" : "缺少文件";
        const device = status.device || "unloaded";
        const step = status.checkpoint_step || "?";
        return `${ready} / 状态 ${device} / step ${step}`;
    }

    function renderTtsStatus(tts) {
        const models = (tts && tts.models) || {};
        const activeDirect = (tts && tts.active_direct_model) || "";

        for (const modelName of ["shanghai", "shaoxing"]) {
            const status = models[modelName];
            const statusEl = document.getElementById(`status-${modelName}`);
            const metaEl = document.getElementById(`meta-${modelName}`);
            const directBtn = document.getElementById(`direct-btn-${modelName}`);
            const cardEl = document.getElementById(`card-${modelName}`);
            const loadButtons = Array.from(document.querySelectorAll(`[data-model-load="${modelName}"]`));
            if (!statusEl || !metaEl || !directBtn) {
                continue;
            }
            if (!status) {
                statusEl.innerHTML = "<strong>不可用</strong>";
                statusEl.dataset.state = "error";
                metaEl.textContent = "未发现模型配置";
                directBtn.classList.remove("direct-active");
                if (cardEl) {
                    cardEl.dataset.device = "unloaded";
                }
                loadButtons.forEach((button) => {
                    button.classList.remove("active");
                    button.setAttribute("aria-pressed", "false");
                });
                continue;
            }
            const badge = status.device === "unloaded" ? "UNLOADED" : `${status.device.toUpperCase()}`;
            statusEl.innerHTML = `<strong>${badge}</strong>`;
            statusEl.dataset.state = status.device === "unloaded" ? "idle" : "done";
            metaEl.textContent = modelMetaText(status);
            if (cardEl) {
                cardEl.dataset.device = status.device || "unloaded";
            }
            loadButtons.forEach((button) => {
                const isActive = button.getAttribute("data-device") === status.device;
                button.classList.toggle("active", isActive);
                button.setAttribute("aria-pressed", isActive ? "true" : "false");
            });
            directBtn.classList.toggle("direct-active", activeDirect === modelName);
            directBtn.setAttribute("aria-pressed", activeDirect === modelName ? "true" : "false");
            directBtn.textContent = activeDirect === modelName ? "当前拼音直读" : "设为拼音直读";
        }

        if (directModelChip) {
            const labelMap = { shanghai: "上海话", shaoxing: "绍兴话" };
            directModelChip.innerHTML = `<strong>拼音直读</strong> ${labelMap[activeDirect] || activeDirect || "未设置"}`;
        }
    }

    async function refreshTtsStatus() {
        const response = await fetch("/api/tts/status");
        const data = await response.json();
        if (!data.ok) {
            throw new Error(data.error || "读取 TTS 状态失败");
        }
        renderTtsStatus(data.tts || {});
        return data.tts || {};
    }

    async function loadTtsModel(model, device) {
        setUiBusy(true);
        const pending = createPendingMessage("status");
        setLiveStatus("处理中", `${model} ${device}`);
        await nextFrame();
        try {
            const response = await fetch("/api/tts/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model, device }),
            });
            const data = await response.json();
            if (!response.ok || !data.ok) {
                throw new Error(data.error || `切换 ${model} 到 ${device} 失败`);
            }
            pending.finish();
            renderTtsStatus(data.tts || {});
            const action = device === "unloaded" ? "已卸载" : `已切到 ${device.toUpperCase()}`;
            addMessage("ai", `${model} ${action}`, { isHtml: false });
            setLiveStatus("完成", action);
        } catch (error) {
            pending.finish();
            addMessage("ai", `模型加载失败：${error.message || error}`, { isHtml: false });
            setLiveStatus("失败");
        } finally {
            setUiBusy(false);
        }
    }

    async function setDirectModel(model) {
        setUiBusy(true);
        const pending = createPendingMessage("status");
        setLiveStatus("处理中", "拼音直读切换");
        await nextFrame();
        try {
            const response = await fetch("/api/tts/direct_model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model }),
            });
            const data = await response.json();
            if (!response.ok || !data.ok) {
                throw new Error(data.error || `设置 ${model} 为拼音直读模型失败`);
            }
            pending.finish();
            renderTtsStatus(data.tts || {});
            setLiveStatus("完成", `${model} 拼音直读`);
        } catch (error) {
            pending.finish();
            addMessage("ai", `切换拼音直读模型失败：${error.message || error}`, { isHtml: false });
            setLiveStatus("失败");
        } finally {
            setUiBusy(false);
        }
    }

    async function quickRead(headword, source = "shanghai_csv") {
        setUiBusy(true);
        const pending = createPendingMessage("tts");
        setLiveStatus("处理中", "词条朗读");
        await nextFrame();
        try {
            const response = await fetch("/api/tts/read_headword", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ headword, source }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.text || "词条朗读失败");
            }
            pending.finish();
            addMessage("ai", decorateResponseHtml(data.text || "没有返回内容。", data), {
                isHtml: true,
                audio: data.audio || "",
            });
            refreshTtsStatus().catch(() => {});
            const totalMs = data.timings_ms && data.timings_ms.total ? Math.round(Number(data.timings_ms.total)) : null;
            setLiveStatus("完成", totalMs ? `${totalMs}ms` : "词条朗读");
        } catch (error) {
            pending.finish();
            addMessage("ai", `词条朗读失败：${error.message || error}`, { isHtml: false });
            setLiveStatus("失败");
        } finally {
            setUiBusy(false);
        }
    }

    function closeRebuildModal() {
        rebuildModal.classList.remove("show");
        if (rebuildTimer) {
            clearTimeout(rebuildTimer);
            rebuildTimer = null;
        }
    }

    async function pollRebuildStatus() {
        try {
            const response = await fetch("/api/rebuild_index/status");
            const data = await response.json();
            const status = data.status || {};
            const logs = Array.isArray(status.progress) ? status.progress : [];
            const pieces = [];
            if (logs.length) {
                pieces.push(logs.join("\n"));
            }
            if (status.error) {
                pieces.push(`\nERROR: ${status.error}`);
            }
            if (status.done) {
                pieces.push("\n完成。");
            }
            rebuildLog.textContent = pieces.join("\n") || "处理中...";
            if (status.running) {
                rebuildTimer = setTimeout(pollRebuildStatus, 1200);
            }
        } catch (error) {
            rebuildLog.textContent = `状态查询失败：${error.message || error}`;
        }
    }

    async function startRebuild() {
        rebuildModal.classList.add("show");
        rebuildLog.textContent = "准备启动重建...";
        try {
            const response = await fetch("/api/rebuild_index", { method: "POST" });
            const data = await response.json();
            if (data.ok) {
                rebuildLog.textContent = data.message || "已启动";
                pollRebuildStatus();
            } else {
                rebuildLog.textContent = data.error || "启动失败";
            }
        } catch (error) {
            rebuildLog.textContent = `启动失败：${error.message || error}`;
        }
    }

    sendBtn.addEventListener("click", () => {
        sendMessage();
    });

    rebuildBtn.addEventListener("click", () => {
        startRebuild();
    });

    closeRebuildBtn.addEventListener("click", () => {
        closeRebuildModal();
    });

    finishRebuildBtn.addEventListener("click", () => {
        closeRebuildModal();
    });

    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    document.addEventListener("click", (event) => {
        const pipelineBtn = event.target.closest("#pipeline-mode button");
        if (pipelineBtn) {
            setPipelineMode(pipelineBtn.dataset.mode || "lookup");
            return;
        }

        const promptBtn = event.target.closest("[data-fill-prompt]");
        if (promptBtn) {
            fillPrompt(promptBtn.getAttribute("data-fill-prompt") || "");
            return;
        }

        const loadBtn = event.target.closest("[data-model-load]");
        if (loadBtn) {
            loadTtsModel(loadBtn.getAttribute("data-model-load") || "", loadBtn.getAttribute("data-device") || "cpu");
            return;
        }

        const directBtn = event.target.closest("[data-direct-model]");
        if (directBtn) {
            setDirectModel(directBtn.getAttribute("data-direct-model") || "");
            return;
        }

        const voiceBtn = event.target.closest(".voice-btn");
        if (voiceBtn && !voiceBtn.disabled) {
            const headword = voiceBtn.getAttribute("data-headword") || "";
            const source = voiceBtn.getAttribute("data-source") || "shanghai_csv";
            if (headword) {
                quickRead(headword, source);
            }
        }
    });

    setPipelineMode(pipelineMode);
    setLiveStatus("空闲");
    syncEmptyState();
    refreshTtsStatus().catch((error) => {
        addMessage("ai", `TTS 状态读取失败：${error.message || error}`, { isHtml: false });
    });
})();
