"""
LeanAI — Terminal UI
Beautiful, colorful terminal output with ANSI escape codes.
Works on Windows 10+, macOS, and Linux.
"""

import os
import sys
import re

# ── Enable ANSI colors on Windows ─────────────────────────────

if sys.platform == "win32":
    os.system("")  # enables ANSI escape codes on Windows 10+


# ── Color codes ───────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET     = "\033[0m"
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    ITALIC    = "\033[3m"
    UNDERLINE = "\033[4m"

    # Regular colors
    BLACK     = "\033[30m"
    RED       = "\033[31m"
    GREEN     = "\033[32m"
    YELLOW    = "\033[33m"
    BLUE      = "\033[34m"
    MAGENTA   = "\033[35m"
    CYAN      = "\033[36m"
    WHITE     = "\033[37m"

    # Bright colors
    BRED      = "\033[91m"
    BGREEN    = "\033[92m"
    BYELLOW   = "\033[93m"
    BBLUE     = "\033[94m"
    BMAGENTA  = "\033[95m"
    BCYAN     = "\033[96m"
    BWHITE    = "\033[97m"

    # Background
    BG_BLACK  = "\033[40m"
    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_BLUE   = "\033[44m"
    BG_MAGENTA= "\033[45m"
    BG_CYAN   = "\033[46m"
    BG_WHITE  = "\033[47m"
    BG_GRAY   = "\033[100m"


# ── Banner ────────────────────────────────────────────────────

def print_banner():
    """Print the startup banner with colors."""
    purple = C.BMAGENTA
    blue = C.BBLUE
    cyan = C.BCYAN
    dim = C.DIM
    r = C.RESET
    b = C.BOLD

    print(f"""
{purple}  ╔══════════════════════════════════════════════════════════════╗{r}
{purple}  ║{r}  {b}{cyan}██╗     ███████╗ █████╗ ███╗   ██╗ █████╗ ██╗{r}             {purple}║{r}
{purple}  ║{r}  {b}{cyan}██║     ██╔════╝██╔══██╗████╗  ██║██╔══██╗██║{r}             {purple}║{r}
{purple}  ║{r}  {b}{cyan}██║     █████╗  ███████║██╔██╗ ██║███████║██║{r}             {purple}║{r}
{purple}  ║{r}  {b}{cyan}██║     ██╔══╝  ██╔══██║██║╚██╗██║██╔══██║██║{r}             {purple}║{r}
{purple}  ║{r}  {b}{cyan}███████╗███████╗██║  ██║██║ ╚████║██║  ██║██║{r}             {purple}║{r}
{purple}  ║{r}  {b}{cyan}╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝{r}             {purple}║{r}
{purple}  ║{r}                                                            {purple}║{r}
{purple}  ║{r}  {dim}Project-Aware AI Coding System — 100% Local & Private{r}     {purple}║{r}
{purple}  ║{r}  {dim}Brain · Git · TDD · Memory · Swarm · Fine-Tune{r}           {purple}║{r}
{purple}  ╚══════════════════════════════════════════════════════════════╝{r}
""")


# ── Status display ────────────────────────────────────────────

def print_status(model_name, model_mode, memory_count, mem_backend,
                 profile_count, training_pairs, session_count,
                 exchange_count, git_branch, finetune_pairs):
    """Print system status with colors."""
    g = C.BGREEN
    b = C.BBLUE
    c = C.BCYAN
    y = C.BYELLOW
    m = C.BMAGENTA
    d = C.DIM
    r = C.RESET
    w = C.BWHITE

    print(f"  {d}{'─' * 58}{r}")
    print(f"  {b}Model{r}    : {w}{model_name}{r} {d}| mode: {model_mode}{r}")
    print(f"  {c}Memory{r}   : {w}{memory_count}{r} episodes {d}| {mem_backend}{r}")
    print(f"  {m}Profile{r}  : {w}{profile_count}{r} fields")
    print(f"  {y}Training{r} : {w}{training_pairs}{r} pairs")
    print(f"  {g}Sessions{r} : {w}{session_count}{r} past {d}| {exchange_count} exchanges{r}")
    print(f"  {b}Git{r}      : {g}active{r} {d}| branch: {git_branch}{r}")
    print(f"  {m}FineTune{r} : {w}{finetune_pairs}{r} training pairs")
    print(f"  {d}{'─' * 58}{r}")
    print()


# ── Commands help ─────────────────────────────────────────────

def print_commands():
    """Print available commands with colors."""
    h = C.BBLUE
    c = C.CYAN
    d = C.DIM
    r = C.RESET

    print(f"  {h}Commands:{r}")
    print(f"    {c}Chat{r}     : just type {d}| /swarm <q> | /run <code>{r}")
    print(f"    {c}Build{r}    : /build <task> {d}| /tdd <tests> | /tdd-desc <desc>{r}")
    print(f"    {c}Reason{r}   : /reason <q> {d}| /plan <task> | /decompose <problem>{r}")
    print(f"    {c}Write{r}    : /write <doc> {d}| /essay <topic> | /report <topic>{r}")
    print(f"    {c}Project{r}  : /brain <path> {d}| /describe <file> | /deps <file>{r}")
    print(f"    {c}Git{r}      : /git activity {d}| /git hotspots | /git history <file>{r}")
    print(f"    {c}Verify{r}   : /fuzz <code> {d}| /bisect <bug>{r}")
    print(f"    {c}Complete{r} : /complete <prefix>")
    print(f"    {c}Track{r}    : /evolution narrative {d}| insights | predict{r}")
    print(f"    {c}Memory{r}   : /remember <fact> {d}| /profile | /sessions{r}")
    print(f"    {c}System{r}   : /model <cmd> {d}| /speed | /status | /help | /quit{r}")
    print()


# ── Input prompt ──────────────────────────────────────────────

def get_prompt():
    """Return the styled input prompt."""
    return f"{C.BMAGENTA}  ▶{C.RESET} "


# ── Response formatting ──────────────────────────────────────

def format_response(text):
    """Format AI response with syntax highlighting for code blocks."""
    lines = text.split("\n")
    result = []
    in_code = False
    code_lang = ""

    for line in lines:
        # Code block start
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line.strip().replace("```", "").strip()
                lang_display = code_lang if code_lang else "code"
                result.append(f"  {C.DIM}┌─ {lang_display} {'─' * (40 - len(lang_display))}┐{C.RESET}")
            else:
                in_code = False
                result.append(f"  {C.DIM}└{'─' * 44}┘{C.RESET}")
            continue

        if in_code:
            highlighted = highlight_code_line(line, code_lang)
            result.append(f"  {C.DIM}│{C.RESET} {highlighted}")
        else:
            # Markdown formatting
            formatted = format_markdown_line(line)
            result.append(f"  {formatted}")

    return "\n".join(result)


def highlight_code_line(line, lang=""):
    """Apply syntax highlighting to a code line."""
    if not line.strip():
        return line

    # Python / generic highlighting
    highlighted = line

    # Comments
    if re.match(r"^\s*#", line):
        return f"{C.DIM}{C.GREEN}{line}{C.RESET}"

    # Keywords
    keywords = r"\b(def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|as|yield|async|await|raise|pass|break|continue|lambda|and|or|not|in|is)\b"
    highlighted = re.sub(keywords, f"{C.BMAGENTA}\\1{C.RESET}", highlighted)

    # Builtins
    builtins = r"\b(print|len|range|str|int|float|list|dict|set|type|True|False|None|self)\b"
    highlighted = re.sub(builtins, f"{C.BCYAN}\\1{C.RESET}", highlighted)

    # Strings (simple detection)
    highlighted = re.sub(r'(\"[^\"]*\")', f"{C.BGREEN}\\1{C.RESET}", highlighted)
    highlighted = re.sub(r"(\'[^\']*\')", f"{C.BGREEN}\\1{C.RESET}", highlighted)

    # Numbers
    highlighted = re.sub(r"\b(\d+\.?\d*)\b", f"{C.BYELLOW}\\1{C.RESET}", highlighted)

    # Function definitions
    highlighted = re.sub(r"(def\s+)(\w+)", f"{C.BMAGENTA}\\1{C.RESET}{C.BBLUE}\\2{C.RESET}", highlighted)

    # Class definitions
    highlighted = re.sub(r"(class\s+)(\w+)", f"{C.BMAGENTA}\\1{C.RESET}{C.BYELLOW}\\2{C.RESET}", highlighted)

    return highlighted


def format_markdown_line(line):
    """Format a markdown line for terminal display."""
    stripped = line.strip()

    # Headers
    if stripped.startswith("### "):
        return f"{C.BOLD}{C.BCYAN}{stripped[4:]}{C.RESET}"
    if stripped.startswith("## "):
        return f"{C.BOLD}{C.BBLUE}{stripped[3:]}{C.RESET}"
    if stripped.startswith("# "):
        return f"{C.BOLD}{C.BMAGENTA}{stripped[2:]}{C.RESET}"

    # Bullet points
    if stripped.startswith("- ") or stripped.startswith("* "):
        return f"  {C.BCYAN}•{C.RESET} {stripped[2:]}"

    # Numbered lists
    m = re.match(r"^(\d+)\. (.+)", stripped)
    if m:
        return f"  {C.BCYAN}{m.group(1)}.{C.RESET} {m.group(2)}"

    # Blockquotes
    if stripped.startswith("> "):
        return f"  {C.DIM}{C.YELLOW}▌{C.RESET} {C.ITALIC}{stripped[2:]}{C.RESET}"

    # Bold
    line = re.sub(r"\*\*(.+?)\*\*", f"{C.BOLD}\\1{C.RESET}", line)

    # Inline code
    line = re.sub(r"`([^`]+)`", f"{C.BG_GRAY}{C.BWHITE} \\1 {C.RESET}", line)

    return line


# ── Confidence bar ────────────────────────────────────────────

def format_confidence(confidence, conf_label):
    """Format confidence bar with colors."""
    if confidence >= 90:
        color = C.BGREEN
    elif confidence >= 70:
        color = C.BCYAN
    elif confidence >= 50:
        color = C.BYELLOW
    else:
        color = C.BRED

    filled = int(confidence / 5)
    bar = f"{color}{'█' * filled}{C.DIM}{'░' * (20 - filled)}{C.RESET}"
    return f"  {C.DIM}Confidence{C.RESET}  [{bar}] {color}{confidence:.0f}%{C.RESET}  {conf_label}"


# ── Meta info ─────────────────────────────────────────────────

def format_meta(tier, latency_str, from_mem=False, verified=False,
                mem_active=False, code_verified=False, enriched=False):
    """Format metadata line with colors."""
    parts = [f"{C.DIM}Tier:{C.RESET} {C.BBLUE}{tier}{C.RESET}"]
    parts.append(f"{C.DIM}Latency:{C.RESET} {C.BWHITE}{latency_str}{C.RESET}")

    if from_mem:
        parts.append(f"{C.BGREEN}from memory{C.RESET}")
    if verified:
        parts.append(f"{C.BGREEN}verified{C.RESET}")
    if mem_active:
        parts.append(f"{C.BCYAN}memory active{C.RESET}")
    if code_verified:
        parts.append(f"{C.BGREEN}code verified{C.RESET}")
    if enriched:
        parts.append(f"{C.BMAGENTA}context enriched{C.RESET}")

    return "  " + f" {C.DIM}·{C.RESET} ".join(parts)


# ── Separator ─────────────────────────────────────────────────

def separator():
    """Print a styled separator."""
    print(f"  {C.DIM}{'─' * 58}{C.RESET}")


# ── Section headers ───────────────────────────────────────────

def section_header(title, icon=""):
    """Print a section header."""
    print(f"\n  {C.BBLUE}{icon}{C.RESET} {C.BOLD}{title}{C.RESET}")
    print(f"  {C.DIM}{'─' * 58}{C.RESET}")


# ── Code execution result ────────────────────────────────────

def format_code_result(passed, output="", error=""):
    """Format code execution result."""
    if passed:
        print(f"\n  {C.BGREEN}✓ Code: PASSED{C.RESET}")
        if output:
            for line in output.strip().split("\n")[:10]:
                print(f"    {C.DIM}{line}{C.RESET}")
    else:
        if error:
            print(f"\n  {C.BRED}✗ Code: {error[:120]}{C.RESET}")
        else:
            print(f"\n  {C.BRED}✗ Code: execution error{C.RESET}")


# ── Brain scan result ────────────────────────────────────────

def format_brain_scan(result):
    """Format brain scan result beautifully."""
    g = C.BGREEN
    b = C.BBLUE
    c = C.BCYAN
    y = C.BYELLOW
    m = C.BMAGENTA
    d = C.DIM
    r = C.RESET
    w = C.BWHITE

    print(f"\n  {b}Project:{r} {w}{result.get('project', 'unknown')}{r}")
    print(f"  {d}Path:{r}    {result.get('path', '')}")
    print(f"  {d}{'─' * 42}{r}")
    print(f"  {c}Files:{r}     {w}{result.get('files_found', 0)}{r} indexed")
    print(f"  {g}Functions:{r} {w}{result.get('graph', {}).get('functions', 0)}{r}")
    print(f"  {m}Classes:{r}   {w}{result.get('graph', {}).get('classes', 0)}{r}")
    print(f"  {y}Edges:{r}     {w}{result.get('graph', {}).get('edges', 0)}{r}")
    print(f"  {d}Lines:{r}     {result.get('total_lines', 0):,}")
    print(f"  {d}Scan time:{r} {result.get('scan_time_ms', 0):.0f}ms")


# ── Autocomplete result ──────────────────────────────────────

def format_completions(prefix, results, elapsed_ms, stats):
    """Format autocomplete results."""
    if results:
        print(f"\n  {C.BBLUE}Completions for{C.RESET} '{C.BWHITE}{prefix}{C.RESET}' {C.DIM}({elapsed_ms:.1f}ms){C.RESET}")
        for r in results:
            icon = {"function": f"{C.BCYAN}ƒ{C.RESET}", "class": f"{C.BYELLOW}◆{C.RESET}",
                    "keyword": f"{C.BMAGENTA}⚡{C.RESET}", "snippet": f"{C.BGREEN}✂{C.RESET}"}.get(r.kind, "·")
            print(f"    {icon} {C.BWHITE}{r.label:<40}{C.RESET} {C.DIM}{r.detail}{C.RESET}")
    else:
        print(f"\n  {C.DIM}No completions for '{prefix}'{C.RESET}")
    print(f"  {C.DIM}Index: {stats['functions_indexed']} functions, {stats['classes_indexed']} classes{C.RESET}")


# ── Loading indicator ─────────────────────────────────────────

def print_thinking(model_name=""):
    """Print thinking indicator."""
    if model_name:
        print(f"\n  {C.DIM}⟳ Thinking ({model_name})...{C.RESET}", end="", flush=True)
    else:
        print(f"\n  {C.DIM}⟳ Thinking...{C.RESET}", end="", flush=True)


def clear_thinking():
    """Clear the thinking line."""
    print(f"\r{' ' * 60}\r", end="", flush=True)


# ── Response header ───────────────────────────────────────────

def print_response_header():
    """Print the response header."""
    print(f"\n  {C.BMAGENTA}LeanAI{C.RESET}")
    print(f"  {C.DIM}{'─' * 58}{C.RESET}")
