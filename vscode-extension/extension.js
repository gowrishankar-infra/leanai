// LeanAI VS Code Extension
// Connects to the LeanAI server (localhost:8000) for AI-powered coding assistance.
// Run `python run_server.py` first, then use LeanAI commands in VS Code.

const vscode = require('vscode');
const http = require('http');
const https = require('https');

// ── API Client ───────────────────────────────────────────────────

function getServerUrl() {
    return vscode.workspace.getConfiguration('leanai').get('serverUrl', 'http://127.0.0.1:8000');
}

function apiCall(endpoint, method, body) {
    return new Promise((resolve, reject) => {
        const url = new URL(endpoint, getServerUrl());
        const options = {
            hostname: url.hostname,
            port: url.port,
            path: url.pathname,
            method: method || 'GET',
            headers: { 'Content-Type': 'application/json' },
        };

        const client = url.protocol === 'https:' ? https : http;
        const req = client.request(options, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(data));
                } catch (e) {
                    resolve({ error: data });
                }
            });
        });

        req.on('error', (e) => {
            reject(new Error(
                `Cannot connect to LeanAI server at ${getServerUrl()}.\n` +
                `Start it with: python run_server.py\n\n` +
                `Error: ${e.message}`
            ));
        });

        req.setTimeout(120000); // 2 min timeout for model inference

        if (body) {
            req.write(JSON.stringify(body));
        }
        req.end();
    });
}

// ── Output Channel ───────────────────────────────────────────────

let outputChannel;

function getOutput() {
    if (!outputChannel) {
        outputChannel = vscode.window.createOutputChannel('LeanAI');
    }
    return outputChannel;
}

function showOutput(title, content) {
    const out = getOutput();
    out.clear();
    out.appendLine(`═══ ${title} ═══`);
    out.appendLine('');
    out.appendLine(content);
    out.show(true);
}

// ── Chat Panel (Webview) ─────────────────────────────────────────

let chatPanel;

function getChatPanelHtml() {
    const serverUrl = getServerUrl();
    return `<!DOCTYPE html>
<html>
<head>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
        background: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
        height: 100vh;
        display: flex;
        flex-direction: column;
    }

    /* Header */
    .header {
        padding: 12px 16px;
        border-bottom: 1px solid var(--vscode-panel-border, rgba(255,255,255,0.08));
        display: flex;
        align-items: center;
        gap: 10px;
        flex-shrink: 0;
    }
    .header-logo {
        width: 28px; height: 28px; border-radius: 8px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; font-weight: 700; color: white;
    }
    .header-title { font-size: 14px; font-weight: 600; }
    .header-badge {
        font-size: 10px; padding: 2px 8px; border-radius: 10px;
        background: rgba(99,102,241,0.15); color: #818cf8;
        font-weight: 500;
    }
    .header-status {
        margin-left: auto; font-size: 11px;
        color: var(--vscode-descriptionForeground, #888);
        display: flex; align-items: center; gap: 6px;
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #22c55e; display: inline-block;
    }
    .status-dot.offline { background: #ef4444; }

    /* Messages area */
    #messages {
        flex: 1; overflow-y: auto; padding: 16px;
        scroll-behavior: smooth;
    }
    #messages::-webkit-scrollbar { width: 6px; }
    #messages::-webkit-scrollbar-track { background: transparent; }
    #messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

    /* Welcome screen */
    .welcome {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; height: 100%; opacity: 0.7;
        text-align: center; padding: 40px;
    }
    .welcome-icon { font-size: 48px; margin-bottom: 16px; }
    .welcome h2 { font-size: 18px; font-weight: 600; margin-bottom: 8px; }
    .welcome p { font-size: 13px; color: var(--vscode-descriptionForeground, #888); max-width: 320px; line-height: 1.6; }
    .quick-actions {
        display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px;
        justify-content: center;
    }
    .quick-btn {
        font-size: 11px; padding: 5px 12px; border-radius: 14px;
        border: 1px solid var(--vscode-panel-border, rgba(255,255,255,0.12));
        background: transparent; color: var(--vscode-editor-foreground);
        cursor: pointer; transition: all 0.15s;
    }
    .quick-btn:hover {
        background: rgba(99,102,241,0.1); border-color: #6366f1;
        color: #818cf8;
    }

    /* Message containers */
    .msg-container {
        display: flex; gap: 10px; margin-bottom: 20px;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .msg-container.user { flex-direction: row-reverse; }

    /* Avatars */
    .avatar {
        width: 30px; height: 30px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; font-weight: 600; flex-shrink: 0;
        margin-top: 2px;
    }
    .avatar.user-avatar {
        background: rgba(59,130,246,0.15); color: #60a5fa;
    }
    .avatar.ai-avatar {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white; font-size: 11px;
    }

    /* Message bubbles */
    .msg-content {
        max-width: 85%; min-width: 60px;
    }
    .msg-sender {
        font-size: 11px; font-weight: 600; margin-bottom: 4px;
        color: var(--vscode-descriptionForeground, #888);
    }
    .msg-container.user .msg-sender { text-align: right; }
    
    .msg-bubble {
        padding: 10px 14px; border-radius: 12px;
        font-size: 13px; line-height: 1.65;
        word-wrap: break-word; overflow-wrap: break-word;
    }
    .msg-container.user .msg-bubble {
        background: #6366f1; color: #fff;
        border-bottom-right-radius: 4px;
    }
    .msg-container.ai .msg-bubble {
        background: var(--vscode-textBlockQuote-background, rgba(255,255,255,0.05));
        border: 1px solid var(--vscode-panel-border, rgba(255,255,255,0.08));
        border-bottom-left-radius: 4px;
    }

    /* Code blocks inside messages */
    .msg-bubble pre {
        background: rgba(0,0,0,0.3); border-radius: 8px;
        padding: 12px; margin: 8px 0; overflow-x: auto;
        font-family: 'Fira Code', 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
        font-size: 12px; line-height: 1.5;
        border: 1px solid rgba(255,255,255,0.06);
        position: relative;
    }
    .msg-bubble code {
        font-family: 'Fira Code', 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
        font-size: 12px;
    }
    .msg-bubble p code {
        background: rgba(99,102,241,0.15); padding: 2px 6px;
        border-radius: 4px; font-size: 12px; color: #c4b5fd;
    }
    .msg-container.user .msg-bubble p code {
        background: rgba(255,255,255,0.2); color: #fff;
    }

    /* Code block header with language + copy */
    .code-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 6px 12px; background: rgba(0,0,0,0.2);
        border-radius: 8px 8px 0 0; border: 1px solid rgba(255,255,255,0.06);
        border-bottom: none; margin-top: 8px; margin-bottom: -1px;
    }
    .code-lang {
        font-size: 11px; color: #818cf8; font-weight: 500;
        font-family: -apple-system, sans-serif;
    }
    .copy-btn {
        font-size: 11px; padding: 2px 10px; border-radius: 4px;
        background: transparent; color: var(--vscode-descriptionForeground, #888);
        border: 1px solid rgba(255,255,255,0.1); cursor: pointer;
        transition: all 0.15s; font-family: -apple-system, sans-serif;
    }
    .copy-btn:hover { background: rgba(255,255,255,0.1); color: #fff; }
    .copy-btn.copied { color: #22c55e; border-color: #22c55e; }
    .code-header + pre { border-radius: 0 0 8px 8px; margin-top: 0; }

    /* Syntax highlighting */
    .kw { color: #c792ea; }
    .fn { color: #82aaff; }
    .str { color: #c3e88d; }
    .num { color: #f78c6c; }
    .cmt { color: #546e7a; font-style: italic; }
    .op { color: #89ddff; }
    .cls { color: #ffcb6b; }
    .dec { color: #c792ea; }
    .bi { color: #82aaff; }

    /* Message actions */
    .msg-actions {
        display: flex; gap: 4px; margin-top: 6px; opacity: 0;
        transition: opacity 0.15s;
    }
    .msg-container:hover .msg-actions { opacity: 1; }
    .msg-action-btn {
        font-size: 11px; padding: 3px 10px; border-radius: 4px;
        background: transparent; border: 1px solid rgba(255,255,255,0.08);
        color: var(--vscode-descriptionForeground, #888);
        cursor: pointer; transition: all 0.15s;
        font-family: -apple-system, sans-serif;
    }
    .msg-action-btn:hover {
        background: rgba(255,255,255,0.05); color: var(--vscode-editor-foreground);
    }

    /* Meta info bar */
    .msg-meta {
        font-size: 10px; margin-top: 6px; display: flex; gap: 8px;
        color: var(--vscode-descriptionForeground, #666);
    }
    .meta-badge {
        padding: 1px 8px; border-radius: 8px;
        background: rgba(99,102,241,0.1); color: #818cf8;
        font-weight: 500;
    }
    .meta-badge.high { background: rgba(34,197,94,0.1); color: #4ade80; }
    .meta-badge.low { background: rgba(251,191,36,0.1); color: #fbbf24; }

    /* Loading animation */
    .loading-dots { display: flex; gap: 4px; padding: 8px 0; }
    .loading-dots span {
        width: 6px; height: 6px; border-radius: 50%;
        background: #818cf8; animation: bounce 1.4s infinite;
    }
    .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
    .loading-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* Input area */
    #input-area {
        padding: 12px 16px; flex-shrink: 0;
        border-top: 1px solid var(--vscode-panel-border, rgba(255,255,255,0.08));
        background: var(--vscode-editor-background);
    }
    .input-wrapper {
        display: flex; gap: 8px; align-items: flex-end;
    }
    #input {
        flex: 1; padding: 10px 14px; min-height: 40px; max-height: 120px;
        border: 1px solid var(--vscode-panel-border, rgba(255,255,255,0.12));
        background: var(--vscode-input-background);
        color: var(--vscode-input-foreground);
        border-radius: 12px; font-family: inherit; font-size: 13px;
        resize: none; outline: none; line-height: 1.4;
        transition: border-color 0.15s;
    }
    #input:focus { border-color: #6366f1; }
    #input::placeholder { color: var(--vscode-input-placeholderForeground, #666); }
    
    .send-btn {
        width: 38px; height: 38px; border-radius: 10px;
        background: #6366f1; border: none; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        transition: all 0.15s; flex-shrink: 0;
    }
    .send-btn:hover { background: #4f46e5; transform: scale(1.05); }
    .send-btn:active { transform: scale(0.95); }
    .send-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .send-btn svg { fill: white; width: 16px; height: 16px; }

    .input-hint {
        font-size: 10px; color: var(--vscode-descriptionForeground, #666);
        margin-top: 6px; text-align: center;
    }

    /* Markdown formatting */
    .msg-bubble h1, .msg-bubble h2, .msg-bubble h3 {
        margin: 12px 0 6px 0; font-weight: 600;
    }
    .msg-bubble h1 { font-size: 16px; }
    .msg-bubble h2 { font-size: 15px; }
    .msg-bubble h3 { font-size: 14px; }
    .msg-bubble p { margin: 6px 0; }
    .msg-bubble ul, .msg-bubble ol { margin: 6px 0; padding-left: 20px; }
    .msg-bubble li { margin: 3px 0; }
    .msg-bubble strong { font-weight: 600; color: #c4b5fd; }
    .msg-container.user .msg-bubble strong { color: #fff; }
    .msg-bubble hr { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 12px 0; }
    .msg-bubble blockquote {
        border-left: 3px solid #6366f1; padding: 4px 12px;
        margin: 8px 0; opacity: 0.85;
    }
    .msg-bubble table { border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 12px; }
    .msg-bubble th, .msg-bubble td {
        padding: 6px 10px; border: 1px solid rgba(255,255,255,0.1);
        text-align: left;
    }
    .msg-bubble th { background: rgba(99,102,241,0.1); font-weight: 600; }
</style>
</head>
<body>

<div class="header">
    <div class="header-logo">L</div>
    <span class="header-title">LeanAI</span>
    <span class="header-badge">LOCAL</span>
    <div class="header-status">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">Connecting...</span>
    </div>
</div>

<div id="messages">
    <div class="welcome" id="welcome">
        <div class="welcome-icon">&#x1f9e0;</div>
        <h2>LeanAI</h2>
        <p>Your local AI coding assistant. I understand your entire codebase, remember past sessions, and verify my own code.</p>
        <div class="quick-actions">
            <button class="quick-btn" onclick="sendQuick('/brain .')">Scan Project</button>
            <button class="quick-btn" onclick="sendQuick('/status')">System Status</button>
            <button class="quick-btn" onclick="sendQuick('/git activity')">Git Activity</button>
            <button class="quick-btn" onclick="sendQuick('/sessions')">Past Sessions</button>
        </div>
    </div>
</div>

<div id="input-area">
    <div class="input-wrapper">
        <textarea id="input" placeholder="Ask about your code, explain a file, find bugs..." rows="1"
            onkeydown="if(event.key==='Enter' && !event.shiftKey){event.preventDefault();send()}"
            oninput="this.style.height='auto';this.style.height=Math.min(this.scrollHeight,120)+'px'"></textarea>
        <button class="send-btn" id="sendBtn" onclick="send()">
            <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
    </div>
    <div class="input-hint">Enter to send &middot; Shift+Enter for new line &middot; Try /brain . to scan your project</div>
</div>

<script>
const SERVER = '${serverUrl}';
let busy = false;

// Check server status
fetch(SERVER + '/status').then(r => r.json()).then(d => {
    document.getElementById('statusDot').className = 'status-dot';
    document.getElementById('statusText').textContent = d.model_loaded ? d.model : 'Ready';
}).catch(() => {
    document.getElementById('statusDot').className = 'status-dot offline';
    document.getElementById('statusText').textContent = 'Offline — run python run_server.py';
});

function sendQuick(msg) {
    document.getElementById('input').value = msg;
    send();
}

function highlightCode(code, lang) {
    // Skip highlighting for non-Python languages
    const pyLangs = ['python', 'py', '', 'code'];
    if (!pyLangs.includes(lang.toLowerCase())) {
        return code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    let h = code
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/(#.*$)/gm, '<span class="cmt">$1</span>')
        .replace(/\\b(def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|as|yield|async|await|raise|pass|break|continue|lambda|and|or|not|in|is|global|nonlocal)\\b/g, '<span class="kw">$1</span>')
        .replace(/\\b(print|len|range|str|int|float|list|dict|set|type|isinstance|hasattr|getattr|open|super|enumerate|zip|map|filter|sorted|True|False|None)\\b/g, '<span class="bi">$1</span>')
        .replace(/\\b(self)\\b/g, '<span class="kw">$1</span>')
        .replace(/^(@\\w+)/gm, '<span class="dec">$1</span>')
        .replace(/(["'])(?:(?!\\\\1).)*?\\1/g, '<span class="str">$&</span>')
        .replace(/\\b(\\d+\\.?\\d*)\\b/g, '<span class="num">$1</span>');
    return h;
}

function renderMarkdown(text) {
    let html = text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    // Code blocks with language
    html = html.replace(/\`\`\`(\\w*)\\n([\\s\\S]*?)\`\`\`/g, function(m, lang, code) {
        lang = lang || 'code';
        const highlighted = highlightCode(code.trim(), lang);
        const id = 'code-' + Math.random().toString(36).substr(2, 6);
        return '<div class="code-header"><span class="code-lang">' + lang + '</span>' +
            '<button class="copy-btn" onclick="copyCode(\\'' + id + '\\')">Copy</button></div>' +
            '<pre id="' + id + '"><code>' + highlighted + '</code></pre>';
    });
    
    // Inline code
    html = html.replace(/\`([^\`]+)\`/g, '<code>$1</code>');
    
    // Bold
    html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
    
    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    
    // Horizontal rule
    html = html.replace(/^---$/gm, '<hr>');
    
    // List items
    html = html.replace(/^[\\-\\*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/^(\\d+)\\. (.+)$/gm, '<li>$2</li>');
    
    // Blockquote
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
    
    // Line breaks (but not inside pre/code)
    html = html.replace(/\\n/g, '<br>');
    
    return html;
}

function copyCode(id) {
    const el = document.getElementById(id);
    const text = el.textContent || el.innerText;
    navigator.clipboard.writeText(text);
    const btn = el.previousElementSibling.querySelector('.copy-btn');
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
}

function copyMessage(btn) {
    const bubble = btn.closest('.msg-container').querySelector('.msg-bubble');
    navigator.clipboard.writeText(bubble.textContent || bubble.innerText);
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
}

function addMessage(type, content, meta) {
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();
    
    const container = document.createElement('div');
    container.className = 'msg-container ' + type;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar ' + (type === 'user' ? 'user-avatar' : 'ai-avatar');
    avatar.textContent = type === 'user' ? 'U' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';
    
    const sender = document.createElement('div');
    sender.className = 'msg-sender';
    sender.textContent = type === 'user' ? 'You' : 'LeanAI';
    
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    
    if (type === 'ai') {
        bubble.innerHTML = renderMarkdown(content);
    } else {
        bubble.textContent = content;
    }
    
    contentDiv.appendChild(sender);
    contentDiv.appendChild(bubble);
    
    if (type === 'ai' && meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'msg-meta';
        
        const conf = meta.confidence || 0;
        const confClass = conf >= 80 ? 'high' : conf >= 50 ? '' : 'low';
        
        if (meta.tier) metaDiv.innerHTML += '<span class="meta-badge">' + meta.tier + '</span>';
        metaDiv.innerHTML += '<span class="meta-badge ' + confClass + '">' + Math.round(conf) + '%</span>';
        if (meta.latency) metaDiv.innerHTML += '<span>' + meta.latency + '</span>';
        
        contentDiv.appendChild(metaDiv);
        
        const actions = document.createElement('div');
        actions.className = 'msg-actions';
        actions.innerHTML = '<button class="msg-action-btn" onclick="copyMessage(this)">Copy</button>';
        contentDiv.appendChild(actions);
    }
    
    container.appendChild(avatar);
    container.appendChild(contentDiv);
    
    document.getElementById('messages').appendChild(container);
    container.scrollIntoView({ behavior: 'smooth' });
    return bubble;
}

function addLoading() {
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();
    
    const container = document.createElement('div');
    container.className = 'msg-container ai';
    container.id = 'loading-msg';
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar ai-avatar';
    avatar.textContent = 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';
    
    const sender = document.createElement('div');
    sender.className = 'msg-sender';
    sender.textContent = 'LeanAI';
    
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
    
    contentDiv.appendChild(sender);
    contentDiv.appendChild(bubble);
    container.appendChild(avatar);
    container.appendChild(contentDiv);
    
    document.getElementById('messages').appendChild(container);
    container.scrollIntoView({ behavior: 'smooth' });
}

function formatLatency(ms) {
    if (ms < 1000) return Math.round(ms) + 'ms';
    if (ms < 60000) return (ms/1000).toFixed(1) + 's';
    return Math.floor(ms/60000) + 'm ' + Math.round((ms%60000)/1000) + 's';
}

async function send() {
    if (busy) return;
    const input = document.getElementById('input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    input.style.height = 'auto';
    
    addMessage('user', msg);
    addLoading();
    
    busy = true;
    document.getElementById('sendBtn').disabled = true;
    
    try {
        const res = await fetch(SERVER + '/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });
        const data = await res.json();
        
        const loading = document.getElementById('loading-msg');
        if (loading) loading.remove();
        
        addMessage('ai', data.text || data.error || 'No response', {
            tier: data.tier,
            confidence: data.confidence,
            latency: formatLatency(data.latency_ms || 0),
        });
    } catch(e) {
        const loading = document.getElementById('loading-msg');
        if (loading) loading.remove();
        addMessage('ai', 'Could not connect to LeanAI server.\\n\\nMake sure the server is running:\\n\`\`\`bash\\npython run_server.py\\n\`\`\`');
    }
    
    busy = false;
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('input').focus();
}
</script>
</body>
</html>`;
}

function openChatPanel(context) {
    if (chatPanel) {
        chatPanel.reveal();
        return;
    }
    chatPanel = vscode.window.createWebviewPanel(
        'leanaiChat', 'LeanAI Chat',
        vscode.ViewColumn.Two,
        { enableScripts: true, retainContextWhenHidden: true }
    );
    chatPanel.webview.html = getChatPanelHtml();
    chatPanel.onDidDispose(() => { chatPanel = null; });
}

// ── Commands ─────────────────────────────────────────────────────

async function chatCommand() {
    const question = await vscode.window.showInputBox({
        prompt: 'Ask LeanAI anything',
        placeHolder: 'e.g., explain how this project handles authentication',
    });
    if (!question) return;

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'LeanAI: Thinking...' },
        async () => {
            try {
                const result = await apiCall('/chat', 'POST', { message: question });
                showOutput('LeanAI Response', 
                    `Q: ${question}\n\n${result.text || result.error}\n\n` +
                    `Tier: ${result.tier} | Confidence: ${Math.round(result.confidence)}% | ${Math.round(result.latency_ms)}ms`
                );
            } catch (e) {
                vscode.window.showErrorMessage(e.message);
            }
        }
    );
}

async function explainSelection() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const selection = editor.document.getText(editor.selection);
    if (!selection) {
        vscode.window.showWarningMessage('Select some code first.');
        return;
    }

    const fileName = editor.document.fileName;
    const language = editor.document.languageId;
    const prompt = `Explain this ${language} code from ${fileName}:\n\n${selection}`;

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'LeanAI: Explaining code...' },
        async () => {
            try {
                const result = await apiCall('/chat', 'POST', { message: prompt });
                showOutput('Code Explanation', result.text || result.error);
            } catch (e) {
                vscode.window.showErrorMessage(e.message);
            }
        }
    );
}

async function fixSelection() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const selection = editor.document.getText(editor.selection);
    if (!selection) {
        vscode.window.showWarningMessage('Select some code first.');
        return;
    }

    const prompt = `Fix this code. Return ONLY the fixed code, no explanation:\n\n${selection}`;

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'LeanAI: Fixing code...' },
        async () => {
            try {
                const result = await apiCall('/chat', 'POST', { message: prompt });
                const fixed = (result.text || '').replace(/```\w*\n?/g, '').replace(/```$/g, '').trim();
                if (fixed) {
                    await editor.edit(editBuilder => {
                        editBuilder.replace(editor.selection, fixed);
                    });
                    vscode.window.showInformationMessage('LeanAI: Code fixed!');
                }
            } catch (e) {
                vscode.window.showErrorMessage(e.message);
            }
        }
    );
}

async function generateTests() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const fileContent = editor.document.getText();
    const fileName = editor.document.fileName;
    const baseName = require('path').basename(fileName, '.py');

    const prompt = `Write pytest tests for this file (${fileName}):\n\n${fileContent.substring(0, 3000)}\n\nOutput ONLY the test code.`;

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'LeanAI: Generating tests...' },
        async () => {
            try {
                const result = await apiCall('/chat', 'POST', { message: prompt });
                const testCode = (result.text || '').replace(/```\w*\n?/g, '').replace(/```$/g, '').trim();
                if (testCode) {
                    const doc = await vscode.workspace.openTextDocument({
                        content: testCode,
                        language: 'python',
                    });
                    await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
                    vscode.window.showInformationMessage(`LeanAI: Tests generated for ${baseName}`);
                }
            } catch (e) {
                vscode.window.showErrorMessage(e.message);
            }
        }
    );
}

async function swarmCommand() {
    const question = await vscode.window.showInputBox({
        prompt: 'Ask with Swarm Consensus (3 passes, highest accuracy)',
    });
    if (!question) return;

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'LeanAI: Swarm consensus (3 passes)...' },
        async () => {
            try {
                const result = await apiCall('/swarm', 'POST', { message: question });
                const status = result.unanimous ? 'UNANIMOUS' : `${Math.round(result.consensus_score * 100)}% agreement`;
                showOutput('Swarm Consensus',
                    `Q: ${question}\n\n${result.text}\n\n` +
                    `${status} | Confidence: ${Math.round(result.confidence)}% | ${result.num_passes} passes | ${Math.round(result.latency_ms)}ms`
                );
            } catch (e) {
                vscode.window.showErrorMessage(e.message);
            }
        }
    );
}

async function runCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    const code = editor.document.getText(editor.selection) || editor.document.getText();

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'LeanAI: Running code...' },
        async () => {
            try {
                const result = await apiCall('/run', 'POST', { code: code });
                const status = result.success ? 'PASSED' : 'FAILED';
                showOutput(`Code Execution — ${status}`,
                    (result.output || result.error || 'No output') +
                    `\n\n${status} | ${result.execution_time_ms}ms`
                );
            } catch (e) {
                vscode.window.showErrorMessage(e.message);
            }
        }
    );
}

async function scanProject() {
    const folders = vscode.workspace.workspaceFolders;
    if (!folders) {
        vscode.window.showWarningMessage('Open a folder first.');
        return;
    }
    const projectPath = folders[0].uri.fsPath;

    await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'LeanAI: Scanning project...' },
        async () => {
            try {
                const result = await apiCall('/brain/scan', 'POST', { path: projectPath });
                showOutput('Project Brain',
                    `Scanned: ${result.files_found || 0} files\n` +
                    `Functions: ${result.graph?.functions || 0}\n` +
                    `Classes: ${result.graph?.classes || 0}\n` +
                    `Edges: ${result.graph?.edges || 0}\n` +
                    `Time: ${result.scan_time_ms || 0}ms`
                );
                vscode.window.showInformationMessage(`LeanAI: Scanned ${result.files_found || 0} files`);
            } catch (e) {
                vscode.window.showErrorMessage(e.message);
            }
        }
    );
}

async function findRefs() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    // Get word under cursor
    const position = editor.selection.active;
    const wordRange = editor.document.getWordRangeAtPosition(position);
    const word = wordRange ? editor.document.getText(wordRange) : '';
    
    const symbol = await vscode.window.showInputBox({
        prompt: 'Find all references to:',
        value: word,
    });
    if (!symbol) return;

    try {
        const result = await apiCall('/refs', 'POST', { query: symbol });
        showOutput(`References: ${symbol}`, result.references || 'No references found');
    } catch (e) {
        vscode.window.showErrorMessage(e.message);
    }
}

async function gitActivity() {
    try {
        const result = await apiCall('/git/activity', 'GET');
        showOutput('Git Activity', result.activity || 'No activity');
    } catch (e) {
        vscode.window.showErrorMessage(e.message);
    }
}

async function gitHotspots() {
    try {
        const result = await apiCall('/git/hotspots', 'GET');
        showOutput('Git Hotspots', result.hotspots || 'No data');
    } catch (e) {
        vscode.window.showErrorMessage(e.message);
    }
}

// ── Status Bar ───────────────────────────────────────────────────

let statusBarItem;

function createStatusBar() {
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = '$(hubot) LeanAI';
    statusBarItem.tooltip = 'LeanAI — Local AI Assistant';
    statusBarItem.command = 'leanai.showPanel';
    statusBarItem.show();
    
    // Check server status
    apiCall('/status', 'GET').then(data => {
        if (data.model_loaded) {
            statusBarItem.text = '$(hubot) LeanAI ●';
            statusBarItem.tooltip = `LeanAI — ${data.model} | ${data.memory_episodes} memories`;
        } else {
            statusBarItem.text = '$(hubot) LeanAI ○';
            statusBarItem.tooltip = 'LeanAI — Ready (model loads on first query)';
        }
    }).catch(() => {
        statusBarItem.text = '$(hubot) LeanAI ✗';
        statusBarItem.tooltip = 'LeanAI — Server not running. Run: python run_server.py';
    });
}

// ── Extension Lifecycle ──────────────────────────────────────────

function activate(context) {
    console.log('LeanAI extension activated');

    // Register all commands
    context.subscriptions.push(
        vscode.commands.registerCommand('leanai.chat', chatCommand),
        vscode.commands.registerCommand('leanai.explainSelection', explainSelection),
        vscode.commands.registerCommand('leanai.fixSelection', fixSelection),
        vscode.commands.registerCommand('leanai.generateTests', generateTests),
        vscode.commands.registerCommand('leanai.swarm', swarmCommand),
        vscode.commands.registerCommand('leanai.runCode', runCode),
        vscode.commands.registerCommand('leanai.scanProject', scanProject),
        vscode.commands.registerCommand('leanai.findRefs', findRefs),
        vscode.commands.registerCommand('leanai.gitActivity', gitActivity),
        vscode.commands.registerCommand('leanai.gitHotspots', gitHotspots),
        vscode.commands.registerCommand('leanai.showPanel', () => openChatPanel(context)),
    );

    // Status bar
    createStatusBar();
    context.subscriptions.push(statusBarItem);

    // Welcome message
    if (vscode.workspace.getConfiguration('leanai').get('autoStart', false)) {
        apiCall('/status', 'GET').catch(() => {
            vscode.window.showInformationMessage(
                'LeanAI server not running. Start it with: python run_server.py',
                'OK'
            );
        });
    }
}

function deactivate() {
    if (outputChannel) outputChannel.dispose();
    if (chatPanel) chatPanel.dispose();
}

module.exports = { activate, deactivate };
