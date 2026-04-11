"""
LeanAI — Terminal UI v2
Stunning terminal output with 256-color gradients, rich Unicode, and visual polish.
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
    """ANSI color codes + 256-color support."""
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

    # 256-color helpers
    @staticmethod
    def fg(n): return f"\033[38;5;{n}m"

    @staticmethod
    def bg(n): return f"\033[48;5;{n}m"


# ── Accent colors (256-color palette) ─────────────────────────

ACCENT    = C.fg(141)   # soft purple
ACCENT2   = C.fg(75)    # soft blue
ACCENT3   = C.fg(114)   # soft green
ACCENT4   = C.fg(222)   # soft gold
ACCENT5   = C.fg(210)   # soft coral
DIMMED    = C.fg(245)   # gray
BRIGHT    = C.fg(255)   # white


# ── Banner ────────────────────────────────────────────────────

def print_banner():
    """Print a stunning startup banner with gradient colors."""
    r = C.RESET
    box = C.fg(141)
    t1 = C.fg(147)
    t2 = C.fg(141)
    t3 = C.fg(135)
    t4 = C.fg(99)
    t5 = C.fg(63)
    t6 = C.fg(69)
    b = C.BOLD

    print()
    print(f"  {box}╔{'═' * 62}╗{r}")
    print(f"  {box}║{r}                                                                {box}║{r}")
    print(f"  {box}║{r}   {b}{t1}██╗{r}     {b}{t1}███████╗{r} {b}{t2}█████╗{r} {b}{t2}███╗   ██╗{r} {b}{t3}█████╗{r} {b}{t3}██╗{r}          {box}║{r}")
    print(f"  {box}║{r}   {b}{t1}██║{r}     {b}{t1}██╔════╝{r}{b}{t2}██╔══██╗{r}{b}{t2}████╗  ██║{r}{b}{t3}██╔══██╗{r}{b}{t3}██║{r}          {box}║{r}")
    print(f"  {box}║{r}   {b}{t2}██║{r}     {b}{t2}█████╗{r}  {b}{t3}███████║{r}{b}{t3}██╔██╗ ██║{r}{b}{t4}███████║{r}{b}{t4}██║{r}          {box}║{r}")
    print(f"  {box}║{r}   {b}{t3}██║{r}     {b}{t3}██╔══╝{r}  {b}{t4}██╔══██║{r}{b}{t4}██║╚██╗██║{r}{b}{t5}██╔══██║{r}{b}{t5}██║{r}          {box}║{r}")
    print(f"  {box}║{r}   {b}{t4}███████╗{r}{b}{t4}███████╗{r}{b}{t5}██║  ██║{r}{b}{t5}██║ ╚████║{r}{b}{t6}██║  ██║{r}{b}{t6}██║{r}          {box}║{r}")
    print(f"  {box}║{r}   {b}{t5}╚══════╝{r}{b}{t5}╚══════╝{r}{b}{t6}╚═╝  ╚═╝{r}{b}{t6}╚═╝  ╚═══╝{r}{b}{t6}╚═╝  ╚═╝{r}{b}{t6}╚═╝{r}          {box}║{r}")
    print(f"  {box}║{r}                                                                {box}║{r}")
    print(f"  {box}║{r}   {ACCENT}◆{r} {BRIGHT}Project-Aware AI Coding System{r}                           {box}║{r}")
    print(f"  {box}║{r}   {DIMMED}100% Local  ·  100% Private  ·  $0 Forever{r}                 {box}║{r}")
    print(f"  {box}║{r}                                                                {box}║{r}")
    print(f"  {box}║{r}   {C.fg(114)}⧫ Brain{r}  {C.fg(75)}⧫ Git{r}  {C.fg(222)}⧫ TDD{r}  {C.fg(210)}⧫ Memory{r}  {C.fg(141)}⧫ Swarm{r}  {C.fg(69)}⧫ GPU{r}      {box}║{r}")
    print(f"  {box}╚{'═' * 62}╝{r}")
    print()


# ── Status display ────────────────────────────────────────────

def print_status(model_name, model_mode, memory_count, mem_backend,
                 profile_count, training_pairs, session_count,
                 exchange_count, git_branch, finetune_pairs):
    """Print system status with icons and colors."""
    r = C.RESET

    print(f"  {DIMMED}{'─' * 62}{r}")
    print(f"  {C.fg(75)}⚙{r}  {DIMMED}Model{r}     {BRIGHT}{model_name}{r} {DIMMED}│ mode: {model_mode}{r}")
    print(f"  {C.fg(114)}◉{r}  {DIMMED}Memory{r}    {BRIGHT}{memory_count}{r} episodes {DIMMED}│ {mem_backend}{r}")
    print(f"  {C.fg(222)}◎{r}  {DIMMED}Profile{r}   {BRIGHT}{profile_count}{r} fields")
    print(f"  {C.fg(210)}◈{r}  {DIMMED}Training{r}  {BRIGHT}{training_pairs}{r} pairs")
    print(f"  {C.fg(141)}◆{r}  {DIMMED}Sessions{r}  {BRIGHT}{session_count}{r} past {DIMMED}│ {exchange_count} exchanges{r}")
    print(f"  {C.fg(69)}◇{r}  {DIMMED}Git{r}       {C.fg(114)}active{r} {DIMMED}│ branch: {git_branch}{r}")
    print(f"  {C.fg(135)}◈{r}  {DIMMED}FineTune{r}  {BRIGHT}{finetune_pairs}{r} training pairs")
    print(f"  {DIMMED}{'─' * 62}{r}")
    print()


# ── Commands help ─────────────────────────────────────────────

def print_commands():
    """Print available commands with icons and colors."""
    r = C.RESET

    print(f"  {ACCENT}Commands{r}")
    print(f"    {C.fg(75)}▸{r} {C.fg(75)}Chat{r}      just type {DIMMED}│ /swarm <q> │ /run <code>{r}")
    print(f"    {C.fg(114)}▸{r} {C.fg(114)}Build{r}     /build <task> {DIMMED}│ /tdd <tests> │ /tdd-desc <desc>{r}")
    print(f"    {C.fg(222)}▸{r} {C.fg(222)}Reason{r}    /reason <q> {DIMMED}│ /plan <task> │ /decompose <problem>{r}")
    print(f"    {C.fg(210)}▸{r} {C.fg(210)}Write{r}     /write <doc> {DIMMED}│ /essay <topic> │ /report <topic>{r}")
    print(f"    {C.fg(141)}▸{r} {C.fg(141)}Project{r}   /brain <path> {DIMMED}│ /describe <file> │ /deps <file>{r}")
    print(f"    {C.fg(69)}▸{r} {C.fg(69)}Git{r}       /git activity {DIMMED}│ /git hotspots │ /git history <file>{r}")
    print(f"    {C.fg(135)}▸{r} {C.fg(135)}Verify{r}    /fuzz <code> {DIMMED}│ /bisect <bug>{r}")
    print(f"    {C.fg(210)}▸{r} {C.fg(210)}Debug{r}     /explain <error> {DIMMED}│ /diff │ /security <file>{r}")
    print(f"    {C.fg(222)}▸{r} {C.fg(222)}Test{r}      /test <function>")
    print(f"    {C.fg(81)}▸{r} {C.fg(81)}Complete{r}  /complete <prefix>")
    print(f"    {C.fg(147)}▸{r} {C.fg(147)}Track{r}     /evolution narrative {DIMMED}│ insights │ predict{r}")
    print(f"    {C.fg(114)}▸{r} {C.fg(114)}Memory{r}    /remember <fact> {DIMMED}│ /profile │ /sessions{r}")
    print(f"    {DIMMED}▸{r} {DIMMED}System{r}    /model <cmd> {DIMMED}│ /speed │ /status │ /help │ /quit{r}")
    print()


# ── Input prompt ──────────────────────────────────────────────

def get_prompt():
    """Return a styled input prompt with gradient arrow."""
    return f"\n  {C.fg(141)}❯{C.fg(135)}❯{C.fg(99)}❯{C.RESET} "


# ── Response formatting ──────────────────────────────────────

def format_response(text):
    """Format AI response with syntax highlighting for code blocks."""
    lines = text.split("\n")
    result = []
    in_code = False
    code_lang = ""
    line_num = 0

    for line in lines:
        # Code block start
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line.strip().replace("```", "").strip()
                lang_display = code_lang if code_lang else "code"
                line_num = 0
                # Styled code block header
                header = f" {lang_display} "
                pad = 50 - len(header)
                result.append(f"  {C.fg(238)}╭─{C.fg(75)}{C.BOLD}{header}{C.RESET}{C.fg(238)}{'─' * pad}╮{C.RESET}")
            else:
                in_code = False
                result.append(f"  {C.fg(238)}╰{'─' * 52}╯{C.RESET}")
            continue

        if in_code:
            line_num += 1
            highlighted = highlight_code_line(line, code_lang)
            num_str = f"{C.fg(240)}{line_num:3}{C.RESET}"
            result.append(f"  {C.fg(238)}│{C.RESET} {num_str} {C.fg(238)}│{C.RESET} {highlighted}")
        else:
            # Markdown formatting
            formatted = format_markdown_line(line)
            result.append(f"  {formatted}")

    return "\n".join(result)


def highlight_code_line(line, lang=""):
    """Return code line as-is. Clean code is better than corrupted colors."""
    return line


def format_markdown_line(line):
    """Format a markdown line for terminal display."""
    stripped = line.strip()

    # Headers
    if stripped.startswith("#### "):
        return f"    {C.fg(81)}◇{C.RESET} {C.fg(81)}{stripped[5:]}{C.RESET}"
    if stripped.startswith("### "):
        return f"\n  {C.fg(75)}{C.BOLD}▸ {stripped[4:]}{C.RESET}"
    if stripped.startswith("## "):
        return f"\n  {C.fg(141)}{C.BOLD}■ {stripped[3:]}{C.RESET}"
    if stripped.startswith("# "):
        return f"\n  {C.fg(147)}{C.BOLD}█ {stripped[2:]}{C.RESET}"

    # Bullet points
    if stripped.startswith("- ") or stripped.startswith("* "):
        return f"    {C.fg(141)}●{C.RESET} {stripped[2:]}"

    # Numbered lists
    m = re.match(r"^(\d+)\. (.+)", stripped)
    if m:
        return f"    {C.fg(75)}{m.group(1)}.{C.RESET} {m.group(2)}"

    # Blockquotes
    if stripped.startswith("> "):
        return f"    {C.fg(222)}▐{C.RESET} {C.ITALIC}{C.fg(250)}{stripped[2:]}{C.RESET}"

    # Bold
    line = re.sub(r"\*\*(.+?)\*\*", lambda m: C.BOLD + BRIGHT + m.group(1) + C.RESET, line)

    # Inline code
    line = re.sub(r"`([^`]+)`", lambda m: C.fg(81) + C.bg(236) + " " + m.group(1) + " " + C.RESET, line)

    return line


# ── Confidence bar ────────────────────────────────────────────

def format_confidence(confidence, conf_label):
    """Format confidence bar with gradient colors."""
    r = C.RESET

    if confidence >= 90:
        colors = [C.fg(46), C.fg(82), C.fg(118), C.fg(154)]
        label_color = C.fg(82)
    elif confidence >= 70:
        colors = [C.fg(75), C.fg(81), C.fg(87), C.fg(123)]
        label_color = C.fg(81)
    elif confidence >= 50:
        colors = [C.fg(220), C.fg(221), C.fg(222), C.fg(228)]
        label_color = C.fg(222)
    else:
        colors = [C.fg(196), C.fg(203), C.fg(210), C.fg(217)]
        label_color = C.fg(210)

    filled = int(confidence / 5)
    bar_chars = []
    for i in range(20):
        if i < filled:
            color = colors[min(i // 5, len(colors) - 1)]
            bar_chars.append(f"{color}━{r}")
        else:
            bar_chars.append(f"{C.fg(238)}╌{r}")

    bar = "".join(bar_chars)
    return f"  {DIMMED}Confidence{r}  [{bar}] {label_color}{C.BOLD}{confidence:.0f}%{r}  {label_color}{conf_label}{r}"


# ── Meta info ─────────────────────────────────────────────────

def format_meta(tier, latency_str, from_mem=False, verified=False,
                mem_active=False, code_verified=False, enriched=False):
    """Format metadata line with icons and colors."""
    r = C.RESET
    parts = []

    tier_icons = {"small": "⚡", "medium": "◆", "full": "★", "unknown": "○"}
    icon = tier_icons.get(tier, "○")
    parts.append(f"{C.fg(75)}{icon} {tier}{r}")
    parts.append(f"{DIMMED}⏱ {r}{BRIGHT}{latency_str}{r}")

    if from_mem:
        parts.append(f"{C.fg(114)}◉ memory{r}")
    if verified:
        parts.append(f"{C.fg(82)}✓ verified{r}")
    if mem_active:
        parts.append(f"{C.fg(141)}◈ context{r}")
    if code_verified:
        parts.append(f"{C.fg(82)}✓ code ok{r}")
    if enriched:
        parts.append(f"{C.fg(222)}⧫ enriched{r}")

    return "  " + f" {DIMMED}·{r} ".join(parts)


# ── Separator ─────────────────────────────────────────────────

def separator():
    """Print a styled gradient separator."""
    r = C.RESET
    colors_sep = [236, 238, 240, 242, 244, 246, 244, 242, 240, 238, 236]
    seg_len = 62 // len(colors_sep)
    line = ""
    for c in colors_sep:
        line += f"{C.fg(c)}{'─' * seg_len}"
    print(f"  {line}{r}")


# ── Section headers ───────────────────────────────────────────

def section_header(title, icon="◆"):
    """Print a section header."""
    r = C.RESET
    print(f"\n  {C.fg(141)}{icon}{r} {C.BOLD}{BRIGHT}{title}{r}")
    separator()


# ── Code execution result ────────────────────────────────────

def format_code_result(passed, output="", error=""):
    """Format code execution result with icons."""
    r = C.RESET
    if passed:
        print(f"\n  {C.fg(82)}✓ Code verified{r} {DIMMED}— runs successfully{r}")
        if output:
            for line in output.strip().split("\n")[:10]:
                print(f"    {C.fg(242)}{line}{r}")
    else:
        if error:
            print(f"\n  {C.fg(210)}✗ Code issue:{r} {C.fg(250)}{error[:120]}{r}")
        else:
            print(f"\n  {C.fg(210)}✗ Code: execution error{r}")


# ── Brain scan result ────────────────────────────────────────

def format_brain_scan(result):
    """Format brain scan result."""
    r = C.RESET
    print(f"\n  {C.fg(141)}◆{r} {C.BOLD}Project:{r} {BRIGHT}{result.get('project', 'unknown')}{r}")
    print(f"  {DIMMED}  Path:    {result.get('path', '')}{r}")
    separator()
    print(f"  {C.fg(75)}  Files:{r}     {BRIGHT}{result.get('files_found', 0)}{r} indexed")
    print(f"  {C.fg(114)}  Functions:{r} {BRIGHT}{result.get('graph', {}).get('functions', 0)}{r}")
    print(f"  {C.fg(222)}  Classes:{r}   {BRIGHT}{result.get('graph', {}).get('classes', 0)}{r}")
    print(f"  {C.fg(210)}  Edges:{r}     {BRIGHT}{result.get('graph', {}).get('edges', 0)}{r}")
    print(f"  {DIMMED}  Lines:     {result.get('total_lines', 0):,}{r}")
    print(f"  {DIMMED}  Scan time: {result.get('scan_time_ms', 0):.0f}ms{r}")


# ── Autocomplete result ──────────────────────────────────────

def format_completions(prefix, results, elapsed_ms, stats):
    """Format autocomplete results with icons."""
    r = C.RESET
    if results:
        print(f"\n  {C.fg(75)}Completions for{r} '{C.BOLD}{BRIGHT}{prefix}{r}' {DIMMED}({elapsed_ms:.1f}ms){r}")
        for item in results:
            icon = {
                "function": f"{C.fg(81)}ƒ{r}",
                "class": f"{C.fg(222)}◆{r}",
                "keyword": f"{C.fg(141)}⚡{r}",
                "snippet": f"{C.fg(114)}✂{r}",
            }.get(item.kind, f"{DIMMED}·{r}")
            print(f"    {icon} {BRIGHT}{item.label:<40}{r} {DIMMED}{item.detail}{r}")
    else:
        print(f"\n  {DIMMED}No completions for '{prefix}'{r}")
    print(f"  {DIMMED}Index: {stats['functions_indexed']} functions, {stats['classes_indexed']} classes{r}")


# ── Loading indicator ─────────────────────────────────────────

def print_thinking(model_name=""):
    """Print thinking indicator."""
    r = C.RESET
    if model_name:
        print(f"\n  {C.fg(141)}◌{r} {DIMMED}Thinking ({model_name})...{r}", end="", flush=True)
    else:
        print(f"\n  {C.fg(141)}◌{r} {DIMMED}Thinking...{r}", end="", flush=True)


def clear_thinking():
    """Clear the thinking line."""
    print(f"\r{' ' * 60}\r", end="", flush=True)


# ── Response header ───────────────────────────────────────────

def print_response_header():
    """Print the response header with gradient name."""
    r = C.RESET
    print(f"\n  {C.fg(147)}{C.BOLD}L{C.fg(141)}e{C.fg(135)}a{C.fg(99)}n{C.fg(75)}A{C.fg(69)}I{r}")
    separator()
