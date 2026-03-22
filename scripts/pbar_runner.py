#!/usr/bin/env python3
"""
pbar_runner.py — PBAR loop helper for sub-agents.

KEY STEPS FOR EVERY R1 BRANCH CALL:
  1. health_check()     — verify Framework1 / R1 state before calling
  2. call_r1()          — call DeepSeek R1 via r1_query.py on Framework1
  3. sanitize()         — sanitize R1 output before Claude reads it
  4. heartbeat()        — report gen progress back to parent session

USAGE (from sub-agent):
  import subprocess, sys
  sys.path.insert(0, '/Users/garrett/.openclaw/workspace/skills/pbar/scripts')
  import pbar_runner as pbar

  ok, status = pbar.health_check()
  if ok:
      raw = pbar.call_r1("Your branch prompt here")
      result = pbar.sanitize(raw, task="branch_generation")
      if result["blocked"]:
          # fall back to web_search
      else:
          branch_text = result["text"]  # XML-wrapped, safe for Claude to read

  pbar.heartbeat("Hormuz v4", gen=1, total=5, top=[("Military", 0.85), ("Economic", 0.78)], r1_count=3)
"""

import subprocess
import sys
import os

# ── Config ─────────────────────────────────────────────────────────────────────
SSH_KEY     = os.path.expanduser("~/.ssh/framework_key")
SSH_HOST    = "gk@100.112.143.23"
R1_SCRIPT   = "/tmp/r1_query.py"
PARENT_SESSION = "agent:vera:direct:784460676068409394"
SANITIZER   = "/Users/garrett/.openclaw/workspace/skills/osint/model_output_sanitizer.py"


# ── 1. Framework1 Health Check ─────────────────────────────────────────────────
def health_check(retries=3) -> tuple[bool, str]:
    """
    Check R1 state on Framework1.
    Returns (ready: bool, status_string: str)

    States:
      "ready"    — R1 loaded, no issues
      "cold"     — no model loaded, will cold-start (~2-3 min on first call)
      "stopping" — stuck in unload cycle, needs kill or wait
      "ssh_fail" — can't reach Framework1
    """
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["ssh", "-i", SSH_KEY, "-o", "ConnectTimeout=10", SSH_HOST,
                 "ollama ps 2>/dev/null"],
                capture_output=True, text=True, timeout=15
            )
            out = result.stdout.lower()
            if result.returncode != 0:
                return False, "ssh_fail"
            if "stopping" in out:
                if attempt < retries - 1:
                    import time; time.sleep(30)
                    continue
                return False, "stopping"
            if "deepseek-r1" in out:
                return True, "ready"
            return True, "cold"  # Empty = will cold-load, still callable
        except Exception as e:
            return False, f"ssh_fail: {e}"
    return False, "stopping"


# ── 2. Call R1 ─────────────────────────────────────────────────────────────────
def call_r1(prompt: str, timeout: int = 180) -> str | None:
    """
    Send prompt to DeepSeek R1 via r1_query.py on Framework1.
    Returns response string, or None on failure.
    """
    try:
        result = subprocess.run(
            ["ssh", "-i", SSH_KEY, "-o", "ConnectTimeout=10", SSH_HOST,
             f"echo {subprocess.list2cmdline([prompt])} | python3 {R1_SCRIPT} 2>/dev/null"],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ── 3. Sanitize R1 Output ──────────────────────────────────────────────────────
def sanitize(raw: str, task: str = "pbar_branch") -> dict:
    """
    Sanitize R1 output before Claude reads it.
    Returns sanitizer result dict: {safe, blocked, text, ...}
    Falls back to a simple block if sanitizer unavailable.
    """
    if not raw:
        return {"blocked": True, "safe": False, "text": "", "reason": "empty output"}
    try:
        sys.path.insert(0, os.path.dirname(SANITIZER))
        from model_output_sanitizer import sanitize_model_output
        return sanitize_model_output(raw, source_model="deepseek-r1:70b", task=task)
    except ImportError:
        # Sanitizer unavailable — apply minimal inline check
        triggers = ["ignore previous", "system:", "you are now", "disregard", "new instructions"]
        lower = raw.lower()
        if any(t in lower for t in triggers):
            return {"blocked": True, "safe": False, "text": "", "reason": "injection pattern detected (fallback check)"}
        # Wrap in XML to signal it passed
        wrapped = f"<r1_output source='deepseek-r1:70b' task='{task}'>\n{raw}\n</r1_output>"
        return {"blocked": False, "safe": True, "text": wrapped}


# ── 4. Heartbeat ───────────────────────────────────────────────────────────────
def heartbeat(topic: str, gen: int, total: int, top: list[tuple], r1_count: int, fw1_status: str = "unknown"):
    """
    Send gen progress back to the parent session.
    top: list of (branch_name, score) tuples, e.g. [("Military endurance", 0.85), ...]
    """
    top_str = ", ".join(f"{name} ({score:.2f})" for name, score in top)
    msg = (
        f"🔬 PBAR {topic} — Gen {gen}/{total} complete. "
        f"Framework1: {fw1_status}. "
        f"Top: {top_str}. "
        f"R1: {r1_count}/4 branches."
    )
    # sessions_send is an OpenClaw tool — sub-agents call it via their tool interface.
    # This function prints the formatted message for the sub-agent to send.
    print(f"[HEARTBEAT] {msg}")
    return msg


# ── Quick self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Checking Framework1...")
    ok, status = health_check()
    print(f"  Status: {status} ({'ready' if ok else 'NOT ready'})")

    if "--test-r1" in sys.argv and ok:
        print("Testing R1 call (simple prompt)...")
        out = call_r1("Reply with exactly: R1_OK")
        if out:
            result = sanitize(out, task="self_test")
            print(f"  R1 response received. Blocked: {result['blocked']}")
            if not result["blocked"]:
                print(f"  Safe output: {result['text'][:100]}...")
        else:
            print("  R1 call failed or timed out.")
