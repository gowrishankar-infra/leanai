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
    body { font-family: var(--vscode-font-family); background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); padding: 12px; margin: 0; }
    #messages { height: calc(100vh - 80px); overflow-y: auto; }
    .msg { margin: 8px 0; padding: 8px 12px; border-radius: 6px; white-space: pre-wrap; word-wrap: break-word; font-size: 13px; line-height: 1.5; }
    .user { background: var(--vscode-button-background); color: var(--vscode-button-foreground); margin-left: 20%; }
    .ai { background: var(--vscode-editor-inactiveSelectionBackground); margin-right: 20%; }
    .meta { font-size: 11px; opacity: 0.7; margin-top: 4px; }
    pre { background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px; overflow-x: auto; }
    #input-area { position: fixed; bottom: 0; left: 0; right: 0; padding: 8px 12px; background: var(--vscode-editor-background); display: flex; gap: 8px; }
    #input { flex: 1; padding: 8px; border: 1px solid var(--vscode-input-border); background: var(--vscode-input-background); color: var(--vscode-input-foreground); border-radius: 4px; font-family: inherit; }
    button { padding: 8px 16px; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background: var(--vscode-button-hoverBackground); }
    .loading { opacity: 0.6; font-style: italic; }
</style>
</head>
<body>
<div id="messages"></div>
<div id="input-area">
    <input type="text" id="input" placeholder="Ask LeanAI..." autofocus onkeydown="if(event.key==='Enter')send()">
    <button onclick="send()">Send</button>
</div>
<script>
const SERVER = '${serverUrl}';
let busy = false;

function addMsg(cls, text) {
    const d = document.createElement('div');
    d.className = 'msg ' + cls;
    d.textContent = text;
    document.getElementById('messages').appendChild(d);
    d.scrollIntoView();
    return d;
}

async function send() {
    if (busy) return;
    const input = document.getElementById('input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';
    addMsg('user', msg);
    const loading = addMsg('ai', 'Thinking...');
    loading.classList.add('loading');
    busy = true;
    try {
        const res = await fetch(SERVER + '/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });
        const data = await res.json();
        loading.classList.remove('loading');
        loading.textContent = data.text || data.error || 'No response';
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = data.tier + ' | ' + Math.round(data.confidence) + '% | ' + Math.round(data.latency_ms) + 'ms';
        loading.appendChild(meta);
    } catch(e) {
        loading.textContent = 'Error: ' + e.message;
        loading.classList.remove('loading');
    }
    busy = false;
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
