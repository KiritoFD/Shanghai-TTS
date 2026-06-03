const chatWindow = document.getElementById("chat-window");
            const userInput = document.getElementById("userInput");
            const sendBtn = document.getElementById("sendBtn");
            const rebuildModal = document.getElementById("rebuild-modal");
            const rebuildLog = document.getElementById("rebuild-log");
            const directModelChip = document.getElementById("direct-model-chip");
            let rebuildTimer = null;
            let ttsStatusCache = null;
            const sidebar = document.getElementById("sidebar");
            const sidebarHeader = document.getElementById("sidebar-header");
            const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
            const topbar = document.getElementById("topbar");
            const chatPanel = document.getElementById("chat-panel");
            const composer = document.getElementById("composer");
            const mobileTopbar = document.getElementById("mobile-topbar");

            function isMobile() {
                return window.matchMedia("(max-width: 640px)").matches;
            }

            function toggleSidebar() {
                sidebar.classList.toggle("collapsed");
                sidebarToggleBtn.classList.toggle("collapsed");
                topbar.classList.toggle("collapsed");
                chatPanel.classList.toggle("collapsed");
                composer.classList.toggle("collapsed");
                if (isMobile()) {
                    const isCollapsed = sidebar.classList.contains("collapsed");
                    if (mobileTopbar) mobileTopbar.classList.toggle("hidden", !isCollapsed);
                    if (sidebarHeader) sidebarHeader.classList.toggle("hidden", isCollapsed);
                }
            }

            // 手机端默认关闭侧边栏
            if (isMobile()) {
                sidebar.classList.add("collapsed");
                sidebarToggleBtn.classList.add("collapsed");
                chatPanel.classList.add("collapsed");
                composer.classList.add("collapsed");
                if (sidebarHeader) sidebarHeader.classList.add("hidden");
            }

            function nowLabel() {
                return new Date().toLocaleTimeString("zh-CN", {
                    hour: "2-digit",
                    minute: "2-digit",
                });
            }

            function scrollToBottom() {
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }

            function fillPrompt(text) {
                userInput.value = text;
                userInput.focus();
            }

            function shouldAutoFocusComposer() {
                return window.matchMedia("(min-width: 901px)").matches;
            }

            function addMessage(role, html, opts = {}) {
                const row = document.createElement("div");
                row.className = `row ${role}`;

                const avatar = document.createElement("div");
                avatar.className = `avatar ${role}`;
                avatar.textContent = role === "user" ? "我" : "旦";

                const wrap = document.createElement("div");
                wrap.className = "bubble-wrap";

                const bubble = document.createElement("div");
                bubble.className = "bubble";
                if (opts.isHtml) {
                    bubble.innerHTML = html;
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
                scrollToBottom();
                return row;
            }

            function addThinking() {
                return addMessage("ai", "思考中……", { isHtml: false, className: "thinking" });
            }

            async function sendMessage(customText = null) {
                const text = (customText ?? userInput.value).trim();
                if (!text) {
                    return;
                }

                addMessage("user", text);
                userInput.value = "";
                sendBtn.disabled = true;
                const pending = addMessage("ai", "思考中……", { isHtml: false, className: "thinking" });

                try {
                    const response = await fetch("/api/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ message: text }),
                    });
                    const data = await response.json();
                    pending.remove();
                    addMessage("ai", data.text || "没有返回内容。", {
                        isHtml: true,
                        audio: data.audio || "",
                    });
                } catch (error) {
                    pending.remove();
                    addMessage("ai", `请求失败：${error.message || error}`, { isHtml: false });
                } finally {
                    sendBtn.disabled = false;
                    if (shouldAutoFocusComposer()) {
                        userInput.focus();
                    }
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
                ttsStatusCache = tts || null;
                const models = (tts && tts.models) || {};
                const activeDirect = (tts && tts.active_direct_model) || "";

                for (const modelName of ["shanghai", "shaoxing"]) {
                    const status = models[modelName];
                    const statusEl = document.getElementById(`status-${modelName}`);
                    const metaEl = document.getElementById(`meta-${modelName}`);
                    const directBtn = document.getElementById(`direct-btn-${modelName}`);
                    const modelCard = document.getElementById(`card-${modelName}`);
                    
                    if (!statusEl || !metaEl || !directBtn || !modelCard) {
                        continue;
                    }
                    
                    if (!status) {
                        statusEl.innerHTML = "<strong>不可用</strong>";
                        metaEl.textContent = "未发现模型配置";
                        directBtn.classList.remove("active");
                        const deviceBtns = modelCard.querySelectorAll(".button-grid .small-btn");
                        deviceBtns.forEach((btn, idx) => btn.classList.remove("active"));
                        continue;
                    }
                    
                    const badge = status.device === "unloaded" ? "UNLOADED" : (status.device ? status.device.toUpperCase() : "UNLOADED");
                    statusEl.innerHTML = `<strong>${badge}</strong>`;
                    metaEl.textContent = modelMetaText(status);
                    directBtn.classList.toggle("active", activeDirect === modelName);
                    directBtn.textContent = activeDirect === modelName ? "当前拼音直读" : "设为拼音直读";
                    
                    // 高亮当前设备按钮
                    const deviceBtns = modelCard.querySelectorAll(".button-grid .small-btn");
                    deviceBtns.forEach((btn, idx) => {
                        let isActive = false;
                        if (status.device === "cuda" && idx === 0) isActive = true;
                        if (status.device === "cpu" && idx === 1) isActive = true;
                        if (status.device === "unloaded" && idx === 2) isActive = true;
                        btn.classList.toggle("active", isActive);
                    });
                }

                if (directModelChip) {
                    const labelMap = { shanghai: "上海话", shaoxing: "绍兴话" };
                    const modelLabel = labelMap[activeDirect] || activeDirect || "未设置";
                    directModelChip.innerHTML = `<strong>拼音直读</strong> ${modelLabel}`;

                    // 更新手机端顶栏的拼音直读显示
                    const mobileDirectLabel = document.getElementById("mobile-direct-model-label");
                    if (mobileDirectLabel) {
                        mobileDirectLabel.textContent = modelLabel;
                    }
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
                const card = document.getElementById(`card-${model}`);
                const deviceBtnIndex = device === "cuda" ? 0 : device === "cpu" ? 1 : 2;
                const btn = card ? card.querySelectorAll(".button-grid .small-btn")[deviceBtnIndex] : null;
                let originalText = "";
                let originalDisabled = false;

                try {
                    if (btn) {
                        originalText = btn.textContent;
                        originalDisabled = btn.disabled;
                        btn.textContent = "加载中...";
                        btn.disabled = true;
                        btn.classList.add("active");
                    }

                    const response = await fetch("/api/tts/load", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ model, device }),
                    });
                    const data = await response.json();
                    if (!response.ok || !data.ok) {
                        throw new Error(data.error || `切换 ${model} 到 ${device} 失败`);
                    }
                    renderTtsStatus(data.tts || {});
                    const action = device === "unloaded" ? "已卸载" : `已切到 ${device.toUpperCase()}`;
                    addMessage("ai", `${model} ${action}`, { isHtml: false });
                } catch (error) {
                    addMessage("ai", `模型加载失败：${error.message || error}`, { isHtml: false });
                } finally {
                    if (btn) {
                        btn.textContent = originalText;
                        btn.disabled = originalDisabled;
                        btn.classList.remove("active");
                        refreshTtsStatus();
                    }
                }
            }

            async function setDirectModel(model) {
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
                    renderTtsStatus(data.tts || {});
                } catch (error) {
                    addMessage("ai", `切换拼音直读模型失败：${error.message || error}`, { isHtml: false });
                }
            }

            async function quickRead(headword, source = "shanghai_csv") {
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
                    addMessage("ai", data.text || "没有返回内容。", {
                        isHtml: true,
                        audio: data.audio || "",
                    });
                } catch (error) {
                    addMessage("ai", `词条朗读失败：${error.message || error}`, { isHtml: false });
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

            userInput.addEventListener("keydown", (event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            });

            refreshTtsStatus().catch((error) => {
                addMessage("ai", `TTS 状态读取失败：${error.message || error}`, { isHtml: false });
            });

            if (shouldAutoFocusComposer()) {
                userInput.focus();
            }