# LeanAI VS Code Extension

Local AI coding assistant that runs entirely on your machine.

## Install

### Quick Install (Development Mode)

1. **Start the LeanAI server** (keep this terminal open):
   ```
   cd C:\Users\adity\Downloads\LeanAi\leanai-phase1\leanai
   python run_server.py
   ```

2. **Install the extension in VS Code**:
   - Open VS Code
   - Press `Ctrl+Shift+P` → type "Developer: Install Extension from Location..."
   - If that option doesn't exist, use this method instead:
     ```
     # Open a terminal and run:
     cd C:\Users\adity\Downloads\LeanAi\leanai-phase1\leanai\vscode-extension
     npm install
     ```
   - Then in VS Code: `Ctrl+Shift+P` → "Extensions: Install from VSIX..."
   - Or simply: copy the `vscode-extension` folder to:
     ```
     %USERPROFILE%\.vscode\extensions\leanai
     ```
   - Restart VS Code

### Easiest Method (Symlink)

```powershell
# Run in PowerShell as Administrator:
cmd /c mklink /D "%USERPROFILE%\.vscode\extensions\leanai" "C:\Users\adity\Downloads\LeanAi\leanai-phase1\leanai\vscode-extension"
```

Then restart VS Code. Done.

## Usage

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+L` | Ask LeanAI a question |
| `Ctrl+Shift+E` | Explain selected code |
| `Ctrl+Shift+F` | Fix selected code (replaces in-place) |

### Right-Click Menu
Select any code → Right-click → You'll see:
- **LeanAI: Explain Selected Code**
- **LeanAI: Fix Selected Code**
- **LeanAI: Run Selected Code**
- **LeanAI: Generate Tests for This File**

### Command Palette (Ctrl+Shift+P)
Type "LeanAI" to see all commands:
- **LeanAI: Ask a Question** — chat with AI
- **LeanAI: Ask with Swarm Consensus** — 3-pass verified answer
- **LeanAI: Scan Project (Brain)** — index entire project
- **LeanAI: Find All References** — find symbol across project
- **LeanAI: Git Activity** — recent commits
- **LeanAI: Git Hotspots** — most changed files
- **LeanAI: Open Chat Panel** — sidebar chat UI

### Status Bar
Bottom-right shows LeanAI status:
- `LeanAI ●` — server connected, model loaded
- `LeanAI ○` — server connected, model not yet loaded
- `LeanAI ✗` — server not running

## Requirements

- LeanAI server running (`python run_server.py`)
- VS Code 1.80+
- No npm packages needed — pure VS Code API
