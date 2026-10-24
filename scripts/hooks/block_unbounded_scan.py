"""Claude Code PreToolUse hook: refuse recursive scans of MNLM's large trees before they start.

Inherited from Poly (scripts/hooks/block_unbounded_scan.py, PC-13). A `grep -r`/`find`/`du`/`Get-ChildItem -Recurse`
over a data root does not finish inside a tool call; when the shell dies the scan is orphaned and keeps eating CPU and
disk. The Grep and Glob tools use ripgrep, which honours .gitignore and so skips var/ and data roots; a scan scoped to a
specific small directory is fine.

Blocks (exit 2, reason on stderr) when a Bash/PowerShell command recursively scans a path that is, or contains, one of
LARGE: the repo root (or any ancestor of it), var/, data/, D:\\MNLMData, or the Poly tree. A command wrapped in
`timeout N` (Bash) is bounded and allowed. Never blocks anything else; any error parsing the hook input allows the call.
"""
import json
import os
from pathlib import Path
import re
import shlex
import sys

R = Path(__file__).resolve().parents[2]
LARGE = [R, R/'var', R/'data', R/'tmp', Path('D:/MNLMData'), Path('D:/PolyData'), Path('C:/code/poly'),
         Path('C:/code/poly/var')]
GREPS = {'grep', 'egrep', 'fgrep'}
WALKERS = {'find', 'du', 'tree'}


def norm(raw, cwd):
    raw = raw.strip('"\'')
    m = re.match(r'^/([a-zA-Z])/(.*)$', raw)  # Git Bash /c/code -> C:/code
    if m:
        raw = f'{m.group(1)}:/{m.group(2)}'
    if raw == '~' or raw.startswith('~/'):
        raw = str(Path.home()) + raw[1:]
    p = Path(raw)
    if not p.is_absolute():
        p = Path(cwd)/p
    return Path(os.path.normcase(os.path.abspath(p)))


def too_large(path):
    for big in LARGE:
        big = Path(os.path.normcase(os.path.abspath(big)))
        if path == big or path in big.parents:
            return big
    return None


def segments(command):
    """Simple commands of a shell line, split on | || && ; & and newlines OUTSIDE quotes. Redirections are dropped."""
    lexer = shlex.shlex(command.replace('\n', ';'), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    words = []
    try:
        for token in lexer:
            if token and set(token) <= set('|&;()'):
                if words:
                    yield words
                words = []
            elif token and set(token) <= set('<>'):
                words.append(None)
            else:
                words.append(token)
    except ValueError:
        return
    if words:
        yield words


def clean(words):
    out, skip = [], False
    for w in words:
        if w is None:
            skip = True
            continue
        if skip:
            skip = False
            continue
        if re.match(r'^\d*[<>]', w):
            continue
        out.append(w)
    return out


def bash_offence(command, cwd):
    for words in segments(command):
        words = clean(words)
        if not words:
            continue
        if words[0] == 'timeout':
            continue
        if words[0] in ('cd', 'pushd') and len(words) > 1:
            cwd = str(norm(words[1], cwd))
            continue
        while words and ('=' in words[0] and not words[0].startswith('-')):
            words = words[1:]
        if not words:
            continue
        name = os.path.basename(words[0]).lower().removesuffix('.exe')
        if name in GREPS:
            args = words[1:]
            rec = any(a in ('--recursive', '--dereference-recursive') or
                      (a.startswith('-') and not a.startswith('--') and any(c in a[1:] for c in 'rR')) for a in args)
            if not rec:
                continue
            positional, skip = [], False
            for a in args:
                if skip:
                    skip = False
                    continue
                if a in ('-e', '-f', '--include', '--exclude', '--exclude-dir', '-m', '-A', '-B', '-C'):
                    skip = True
                    continue
                if a.startswith('-'):
                    continue
                positional.append(a)
            paths = positional[1:] if not any(a in ('-e', '-f') for a in args) else positional
            for target in (paths or ['.']):
                big = too_large(norm(target, cwd))
                if big:
                    return f'recursive {name} over {big}'
        elif name in WALKERS:
            args = words[1:]
            if name == 'find' and '-maxdepth' in args:
                try:
                    if int(args[args.index('-maxdepth')+1]) <= 3:
                        continue
                except (ValueError, IndexError):
                    pass
            roots = [a for a in args if not a.startswith('-')][:1] if name == 'find' else [a for a in args if not a.startswith('-')]
            for target in (roots or ['.']):
                big = too_large(norm(target, cwd))
                if big:
                    return f'{name} over {big}'
    return None


def powershell_offence(command, cwd):
    if not re.search(r'-Recurse\b', command, re.I):
        return None
    for m in re.finditer(r'(?:-Path|-LiteralPath)?\s+["\']?([A-Za-z]:[\\/][^"\'\s|;]*|\.{1,2}[\\/]?|[\w.\\/-]+)["\']?', command):
        candidate = m.group(1)
        if re.match(r'^[A-Za-z]:[\\/]|^\.', candidate):
            big = too_large(norm(candidate, cwd))
            if big:
                return f'recursive PowerShell walk over {big}'
    if re.search(r'(Get-ChildItem|gci|ls|dir|Select-String|sls)\b[^|;]*-Recurse', command, re.I) and \
            not re.search(r'[A-Za-z]:[\\/]|\s\.[\\/]?\w', command):
        big = too_large(norm('.', cwd))
        if big:
            return f'recursive PowerShell walk over {big} (current directory)'
    return None


def main():
    try:
        event = json.load(sys.stdin)
        tool = event.get('tool_name')
        command = (event.get('tool_input') or {}).get('command') or ''
        cwd = event.get('cwd') or os.getcwd()
        offence = bash_offence(command, cwd) if tool == 'Bash' else powershell_offence(command, cwd) if tool == 'PowerShell' else None
    except Exception:
        return 0
    if offence:
        print(f'MNLM_UNBOUNDED_SCAN_BLOCKED: {offence}. Data roots hold millions of files; the scan outlives the tool '
              'call and is orphaned. Use the Grep or Glob tool (ripgrep, skips gitignored var/ and data/), scope the '
              'scan to a specific small directory, or bound it with `timeout N`.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
