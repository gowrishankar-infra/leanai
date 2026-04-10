/**
 * LeanAI Inline Completion Provider
 * 
 * Add this to your existing extension.js to get autocomplete.
 * 
 * HOW TO ADD:
 * 1. Copy the registerInlineCompletion function into extension.js
 * 2. Call registerInlineCompletion(context) in your activate() function
 * 3. Reload VS Code
 * 
 * HOW IT WORKS:
 * - As you type, it sends the current word prefix to LeanAI's /complete endpoint
 * - LeanAI searches the brain's indexed functions (instant, no model call)
 * - Returns completions from YOUR actual codebase
 * - Appears as ghost text inline (like Copilot)
 */

const vscode = require('vscode');
const http = require('http');

const LEANAI_URL = 'http://localhost:8000';
const DEBOUNCE_MS = 150; // wait 150ms after typing stops

let debounceTimer = null;

function registerInlineCompletion(context) {
    // Register inline completion provider for all languages
    const provider = vscode.languages.registerInlineCompletionItemProvider(
        { pattern: '**' }, // all files
        {
            provideInlineCompletionItems: async (document, position, context, token) => {
                // Get the current line text up to cursor
                const linePrefix = document.lineAt(position).text.substring(0, position.character);
                
                // Skip if line is empty or just whitespace
                if (!linePrefix.trim()) return [];
                
                // Extract the current word/prefix being typed
                const prefix = extractPrefix(linePrefix);
                if (!prefix || prefix.length < 2) return [];
                
                // Skip if in a comment or string
                if (isInCommentOrString(linePrefix)) return [];
                
                // Detect language
                const language = detectLanguage(document.languageId);
                
                try {
                    const completions = await fetchCompletions(prefix, language, document.fileName, linePrefix);
                    
                    if (!completions || completions.length === 0) return [];
                    
                    // Convert to VS Code inline completion items
                    return completions.map(c => {
                        // Calculate what text to insert after the prefix
                        let insertText = c.insertText || c.text;
                        
                        // Remove the prefix part that's already typed
                        if (insertText.toLowerCase().startsWith(prefix.toLowerCase())) {
                            insertText = insertText.substring(prefix.length);
                        }
                        
                        return new vscode.InlineCompletionItem(
                            insertText,
                            new vscode.Range(position, position)
                        );
                    }).filter(item => item.insertText.length > 0);
                    
                } catch (err) {
                    // Silently fail — don't interrupt typing
                    return [];
                }
            }
        }
    );
    
    context.subscriptions.push(provider);
    console.log('[LeanAI] Inline completion provider registered');
}

function extractPrefix(linePrefix) {
    // Extract the current word being typed, including dots for method access
    const match = linePrefix.match(/[\w.]+$/);
    return match ? match[0] : '';
}

function isInCommentOrString(line) {
    const trimmed = line.trim();
    // Simple heuristic — not perfect but fast
    if (trimmed.startsWith('#') || trimmed.startsWith('//') || trimmed.startsWith('*')) return true;
    // Count quotes — if odd number, we're inside a string
    const singleQuotes = (line.match(/'/g) || []).length;
    const doubleQuotes = (line.match(/"/g) || []).length;
    return (singleQuotes % 2 !== 0) || (doubleQuotes % 2 !== 0);
}

function detectLanguage(languageId) {
    const langMap = {
        'python': 'python',
        'javascript': 'javascript',
        'typescript': 'typescript',
        'javascriptreact': 'javascript',
        'typescriptreact': 'typescript',
        'go': 'go',
        'rust': 'rust',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
    };
    return langMap[languageId] || 'python';
}

function fetchCompletions(prefix, language, filePath, line) {
    return new Promise((resolve, reject) => {
        const data = JSON.stringify({
            prefix: prefix,
            language: language,
            file_path: filePath || '',
            line: line || '',
            max_results: 5,
        });
        
        const options = {
            hostname: 'localhost',
            port: 8000,
            path: '/complete',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(data),
            },
            timeout: 200, // 200ms timeout — if server is slow, skip
        };
        
        const req = http.request(options, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try {
                    const result = JSON.parse(body);
                    resolve(result.completions || []);
                } catch (e) {
                    resolve([]);
                }
            });
        });
        
        req.on('error', () => resolve([]));
        req.on('timeout', () => { req.destroy(); resolve([]); });
        
        req.write(data);
        req.end();
    });
}

module.exports = { registerInlineCompletion };
