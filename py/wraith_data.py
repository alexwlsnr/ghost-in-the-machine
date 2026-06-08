#!/usr/bin/env python3
"""
wraith_data.py — tldr-pages data builder for the Wraith Linux-guru model.

Fetches tldr pages for common Linux commands and parses them into Q|A
training pairs WITHOUT any LLM teacher calls.

Output: data/wraith_training_pairs.txt in QUERY|RESPONSE format.
Case is preserved — flags, commands, and paths must remain lowercase.

Usage:
    python3 py/wraith_data.py [--output data/wraith_training_pairs.txt]
                              [--limit 500]
                              [--tldr-dir ~/tldr]
                              [--log-missing]
"""

import argparse
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Tuple

# ─── 500 most common Linux/Unix commands ────────────────────────────────────

COMMANDS = [
    # File and directory
    "ls", "cd", "pwd", "mkdir", "rmdir", "rm", "cp", "mv", "ln", "touch",
    "find", "locate", "which", "whereis", "file", "stat", "du", "df",
    "tree", "dirname", "basename", "realpath", "readlink",
    # Text processing
    "cat", "less", "more", "head", "tail", "grep", "egrep", "fgrep",
    "sed", "awk", "cut", "sort", "uniq", "wc", "tr", "tee", "diff",
    "patch", "xargs", "paste", "join", "column", "fold", "fmt",
    "strings", "od", "xxd", "hexdump",
    # Compression and archives
    "tar", "gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz",
    "zip", "unzip", "zcat", "zgrep", "7z",
    # Process management
    "ps", "top", "htop", "kill", "killall", "pkill", "pgrep",
    "nice", "renice", "nohup", "bg", "fg", "jobs", "wait",
    "sleep", "watch", "timeout", "strace", "ltrace",
    # System information
    "uname", "hostname", "uptime", "who", "w", "last", "id",
    "whoami", "groups", "date", "cal", "timedatectl", "locale",
    "lscpu", "lsmem", "lsblk", "lspci", "lsusb", "lshw",
    "dmesg", "journalctl", "sysctl",
    # Networking
    "ping", "traceroute", "tracepath", "ip", "ifconfig", "iwconfig",
    "netstat", "ss", "nmap", "curl", "wget", "scp", "sftp", "rsync",
    "ssh", "ssh-keygen", "ssh-copy-id", "telnet", "nc", "ncat",
    "dig", "nslookup", "host", "whois", "arp", "route", "iptables",
    "ufw", "firewall-cmd",
    # Permissions and ownership
    "chmod", "chown", "chgrp", "umask", "chattr", "lsattr",
    "setfacl", "getfacl",
    # Users and groups
    "useradd", "userdel", "usermod", "groupadd", "groupdel", "groupmod",
    "passwd", "su", "sudo", "visudo", "newgrp",
    # Package management (common distros)
    "apt", "apt-get", "apt-cache", "dpkg", "snap",
    "yum", "dnf", "rpm", "pacman", "zypper",
    "pip", "pip3", "npm", "gem",
    # Disk and filesystem
    "mount", "umount", "fdisk", "parted", "mkfs", "fsck", "blkid",
    "lsof", "fuser", "sync", "swapoff", "swapon", "dd",
    # Shell and scripting
    "echo", "printf", "read", "export", "env", "set", "unset",
    "alias", "source", "exec", "eval", "test", "expr",
    "bc", "awk", "bash", "sh", "zsh", "fish",
    # Editors
    "vim", "vi", "nano", "emacs", "ed",
    # Git
    "git",
    # System control
    "systemctl", "service", "chkconfig", "init", "shutdown",
    "reboot", "halt", "poweroff", "cron", "crontab", "at",
    # Development tools
    "gcc", "g++", "make", "cmake", "gdb", "valgrind",
    "python3", "python", "perl", "ruby", "node",
    "strace", "objdump", "nm", "ldd", "ar",
    # Security
    "openssl", "gpg", "md5sum", "sha1sum", "sha256sum", "sha512sum",
    "base64", "certbot",
    # Monitoring and performance
    "sar", "iostat", "vmstat", "mpstat", "pidstat", "free",
    "iftop", "nethogs", "iotop", "perf",
    # Misc utilities
    "xargs", "parallel", "time", "seq", "yes", "true", "false",
    "tput", "stty", "screen", "tmux", "byobu",
    "man", "info", "help", "type", "command",
    "history", "fc",
    "curl", "jq", "yq",
    "convert", "ffmpeg", "imagemagick",
    "rsync", "rclone",
    "cmp", "comm",
    "iconv", "recode",
    "logger", "wall", "write", "mesg",
    "ntp", "ntpdate", "chronyc",
    "lsmod", "modprobe", "rmmod", "insmod", "modinfo",
    "blkid", "hdparm", "smartctl",
    "ip6tables", "nft",
    "syslog", "logrotate",
    "getenforce", "setenforce", "sestatus",
    "ausearch", "auditctl",
    "tcpdump", "wireshark", "tshark",
    "docker", "podman", "kubectl", "helm",
    "ansible", "terraform",
    "vim", "nvim",
    "zcat", "bzcat", "xzcat",
    "pv", "mbuffer",
    "socat", "netcat",
    "hping3", "mtr",
    "fping", "arping",
    "ethtool", "mii-tool",
    "brctl", "bridge",
    "tunctl", "openvpn",
    "cryptsetup", "veracrypt",
    "lvm", "pvs", "vgs", "lvs", "pvcreate", "vgcreate", "lvcreate",
    "mdadm",
    "btrfs", "zfs",
    "quota", "repquota",
    "xinetd", "inetd",
    "postfix", "sendmail", "mutt",
    "nginx", "apache2", "httpd",
    "mysql", "psql", "sqlite3",
    "redis-cli", "mongo",
    "influx",
    "systemd-analyze", "bootctl",
    "dracut", "update-initramfs",
    "grub2-mkconfig", "grub-install",
    "update-grub",
]

# Deduplicate while preserving order
seen = set()
_deduped = []
for c in COMMANDS:
    if c not in seen:
        seen.add(c)
        _deduped.append(c)
COMMANDS = _deduped

TLDR_COMMON_URL = (
    "https://raw.githubusercontent.com/tldr-pages/tldr/main/pages/common/{}.md"
)
TLDR_LINUX_URL = (
    "https://raw.githubusercontent.com/tldr-pages/tldr/main/pages/linux/{}.md"
)


# ─── Parser ─────────────────────────────────────────────────────────────────

def parse_tldr_page(content: str, command: str) -> Optional[dict]:
    """Parse a tldr Markdown page into structured data.

    Returns dict with keys: command, description, examples (list of dicts with
    'desc' and 'cmd' keys). Returns None if parsing fails.

    tldr format:
        # command
        > Short description.
        > See also: other-cmd
        - Example description:
          `example command`
    """
    lines = content.splitlines()
    if not lines:
        return None

    description = None
    examples: List[dict] = []

    i = 0
    # Skip the title line (# command)
    while i < len(lines) and not lines[i].startswith("#"):
        i += 1
    i += 1  # skip the # line

    # Collect description lines (> lines)
    desc_parts = []
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">"):
            text = line[1:].strip()
            # Skip "See also:" and "More information:" lines (contain URLs, not useful)
            if not text.lower().startswith("see also") and \
               not text.lower().startswith("more information"):
                desc_parts.append(text)
        elif line.startswith("-") or (desc_parts and line.startswith("`")):
            break
        i += 1

    if desc_parts:
        description = " ".join(desc_parts).rstrip(".")

    if not description:
        return None

    # Collect examples: alternating "- desc:" and "`cmd`" lines
    current_desc = None
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("-"):
            # Strip leading "- " and trailing ":"
            current_desc = line[1:].strip().rstrip(":")
            # Remove trailing period if present
            current_desc = current_desc.rstrip(".")
        elif line.startswith("`") and line.endswith("`") and current_desc:
            cmd = line[1:-1].strip()
            if cmd:
                examples.append({"desc": current_desc, "cmd": cmd})
            current_desc = None
        i += 1

    return {"command": command, "description": description, "examples": examples}


def fetch_tldr_page(command: str, tldr_dir: Optional[str] = None) -> Optional[str]:
    """Fetch a tldr page from local dir or GitHub. Returns raw content or None."""

    # Try local tldr clone first
    if tldr_dir:
        for subdir in ("common", "linux", "osx", "windows"):
            path = Path(tldr_dir) / "pages" / subdir / f"{command}.md"
            if path.exists():
                return path.read_text(encoding="utf-8")

    # Fetch from GitHub (try common first, then linux)
    for url_template in (TLDR_COMMON_URL, TLDR_LINUX_URL):
        url = url_template.format(command)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "wraith-data-builder/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            pass
        # Small delay between requests to be polite
        time.sleep(0.05)

    return None


# ─── Pair generation ─────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal spaces."""
    return " ".join(text.split())


def _placeholder_safe(template_cmd: str) -> str:
    """Replace tldr placeholder tokens like {{filename}} with readable text.

    Also normalises the newer tldr optional-flag format: [-a|--all] → -a
    """
    # Replace {{something}} with just "something"
    result = re.sub(r"\{\{([^}]+)\}\}", r"\1", template_cmd)
    # Normalise [-short|--long] optional flag syntax → take the short form
    result = re.sub(r"\[-([a-zA-Z0-9-]+)(?:\|[^\]]+)?\]", r"-\1", result)
    return result


def generate_pairs(page: dict) -> List[Tuple[str, str]]:
    """Generate Q|A training pairs from a parsed tldr page.

    Pair types:
      1. "what does CMD do?" | description
      2. "what is CMD?" | description
      3. "CMD" | description. Common: top-2 examples
      4. "how do I EXAMPLE_DESC with CMD?" | command
      5. "how to EXAMPLE_DESC?" | command  (short form)
    """
    pairs = []
    cmd = page["command"]
    desc = _clean(page["description"])
    examples = page["examples"]

    # Type 1: what does X do?
    pairs.append((f"what does {cmd} do?", desc))

    # Type 2: what is X?
    pairs.append((f"what is {cmd}?", desc))

    # Type 3: bare command → description + top examples
    if examples:
        top = examples[:2]
        example_str = ". ".join(
            f"Use {_placeholder_safe(ex['cmd'])} to {ex['desc'].lower()}"
            for ex in top
        )
        combined = f"{desc}. {example_str}"
        # Keep responses under ~120 chars for the 128-ctx window
        if len(combined) > 120:
            combined = desc
        pairs.append((cmd, combined))
    else:
        pairs.append((cmd, desc))

    # Type 4 + 5: per-example pairs
    for ex in examples:
        ex_desc = _clean(ex["desc"])
        ex_cmd = _placeholder_safe(_clean(ex["cmd"]))

        # Skip very long commands (won't fit in context)
        if len(ex_cmd) > 80:
            continue

        # "how do I X with CMD?" | command
        pairs.append((
            f"how do I {ex_desc.lower()} with {cmd}?",
            ex_cmd,
        ))

        # "how to X?" | command (shorter, more natural query)
        pairs.append((
            f"how to {ex_desc.lower()}?",
            ex_cmd,
        ))

    return pairs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Wraith training data from tldr-pages (no teacher calls)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/wraith_training_pairs.txt",
        help="Output file path (default: data/wraith_training_pairs.txt)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=500,
        help="Max number of commands to process (default: 500)",
    )
    parser.add_argument(
        "--tldr-dir",
        default=None,
        help="Path to a local tldr-pages clone (skips GitHub fetches if set)",
    )
    parser.add_argument(
        "--log-missing",
        action="store_true",
        help="Print commands that had no tldr page",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count pairs without writing output",
    )
    args = parser.parse_args()

    # Expand tldr-dir path
    tldr_dir = None
    if args.tldr_dir:
        tldr_dir = os.path.expanduser(args.tldr_dir)
        if not os.path.isdir(tldr_dir):
            print(f"WARNING: --tldr-dir {tldr_dir!r} does not exist; falling back to GitHub",
                  file=sys.stderr)
            tldr_dir = None

    commands = COMMANDS[: args.limit]
    print(f"Processing {len(commands)} commands...", file=sys.stderr)

    all_pairs: List[Tuple[str, str]] = []
    found = []
    missing = []

    for i, cmd in enumerate(commands, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(commands)} commands processed "
                  f"({len(all_pairs)} pairs so far)", file=sys.stderr)

        raw = fetch_tldr_page(cmd, tldr_dir)
        if raw is None:
            missing.append(cmd)
            continue

        page = parse_tldr_page(raw, cmd)
        if page is None:
            missing.append(cmd)
            continue

        pairs = generate_pairs(page)
        if pairs:
            found.append(cmd)
            all_pairs.extend(pairs)

    # Summary
    print(f"\nResults:", file=sys.stderr)
    print(f"  Commands found:   {len(found)}", file=sys.stderr)
    print(f"  Commands missing: {len(missing)}", file=sys.stderr)
    print(f"  Total pairs:      {len(all_pairs)}", file=sys.stderr)

    if args.log_missing and missing:
        print(f"\nMissing commands:", file=sys.stderr)
        for cmd in missing:
            print(f"  {cmd}", file=sys.stderr)

    if args.dry_run:
        print("\nDry run — no output written.", file=sys.stderr)
        # Show a sample
        print("\nSample pairs (first 10):", file=sys.stderr)
        for q, r in all_pairs[:10]:
            print(f"  Q: {q!r}", file=sys.stderr)
            print(f"  R: {r!r}", file=sys.stderr)
        return

    # Write output
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q, r in all_pairs:
            q = q.strip()
            r = r.strip()
            if not q or not r:
                continue
            # Reject pairs with pipe chars that would corrupt the format
            if "|" in q or "|" in r:
                q = q.replace("|", "/")
                r = r.replace("|", "/")
            f.write(f"{q}|{r}\n")
            written += 1

    print(f"\nWrote {written} pairs to {out_path}", file=sys.stderr)
    print(f"Commands found: {len(found)}/{len(commands)}", file=sys.stderr)


if __name__ == "__main__":
    main()
