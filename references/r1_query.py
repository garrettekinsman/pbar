"""
r1_query.py — Deploy to Framework1 at /tmp/r1_query.py before running PBAR loops.
Reads prompt from stdin, calls DeepSeek R1 via LiteLLM, prints response to stdout.

Deploy:
    scp -i ~/.ssh/framework_key r1_query.py gk@100.112.143.23:/tmp/r1_query.py

Call:
    ssh -i ~/.ssh/framework_key gk@100.112.143.23 "echo 'YOUR PROMPT' | python3 /tmp/r1_query.py"

Timeout: 180s (R1:70b is slow — budget ~2-3 min per branch call)
"""

import sys
import json
import urllib.request

LITELLM_URL = "http://localhost:4000/v1/chat/completions"
API_KEY = os.environ.get("LITELLM_MASTER_KEY", "")  # load from ~/.openclaw/secrets.env
MODEL = "deepseek-r1"
MAX_TOKENS = 2000
TIMEOUT = 180

prompt = sys.stdin.read().strip()
if not prompt:
    print("ERROR: empty prompt", file=sys.stderr)
    sys.exit(1)

payload = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": MAX_TOKENS,
    "stream": False
}).encode()

req = urllib.request.Request(
    LITELLM_URL,
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
)

try:
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        d = json.load(resp)
        print(d["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    print(f"ERROR: HTTP {e.code} — {e.read().decode()}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(28)  # Match curl timeout exit code for consistent fallback logic
