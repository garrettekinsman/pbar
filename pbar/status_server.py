"""Lightweight SSE status server for PBAR.

Agent: Jarvis

Each branch thread pushes status updates via put_status().
Clients connect to /events for real-time SSE stream.

Usage:
    # Start server (in orchestrator or separate process)
    server = StatusServer(port=8766)
    server.start()
    
    # From branch threads
    from pbar.status_server import put_status
    put_status(branch_id=0, event="experiment_start", data={...})
    
    # Client connects to http://localhost:8766/events
"""

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

# Global event queue — threads push here, SSE server reads
_event_queue: queue.Queue = queue.Queue(maxsize=1000)
_status_lock = threading.Lock()
_current_status: Dict[str, Any] = {
    "generation": 0,
    "temperature": 2.0,
    "branches": {},
    "global_best": float("inf"),
    "total_experiments": 0,
    "started_at": None,
    "last_update": None,
}


@dataclass
class StatusEvent:
    """A status update event from a branch thread."""
    
    timestamp: float = field(default_factory=time.time)
    branch_id: Optional[int] = None
    event: str = "update"  # experiment_start, experiment_end, generation_end, etc.
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_sse(self) -> str:
        """Format as SSE message."""
        payload = {
            "timestamp": self.timestamp,
            "branch_id": self.branch_id,
            "event": self.event,
            **self.data
        }
        return f"event: {self.event}\ndata: {json.dumps(payload)}\n\n"


def put_status(
    branch_id: Optional[int] = None,
    event: str = "update",
    **data
) -> None:
    """Push a status update from any thread.
    
    This is the main API for branch threads to report progress.
    
    Args:
        branch_id: Which branch is reporting (None for orchestrator-level events)
        event: Event type (experiment_start, experiment_end, generation_end, etc.)
        **data: Event-specific data (score, description, duration, etc.)
    """
    evt = StatusEvent(branch_id=branch_id, event=event, data=data)
    
    # Update global status
    with _status_lock:
        _current_status["last_update"] = evt.timestamp
        
        if branch_id is not None:
            if branch_id not in _current_status["branches"]:
                _current_status["branches"][branch_id] = {
                    "experiments": 0,
                    "best_score": float("inf"),
                    "status": "idle",
                    "last_event": None,
                }
            
            branch = _current_status["branches"][branch_id]
            branch["last_event"] = event
            branch["last_update"] = evt.timestamp
            
            if event == "experiment_start":
                branch["status"] = "running"
                branch["current_description"] = data.get("description", "")
            
            elif event == "experiment_end":
                branch["status"] = "idle"
                branch["experiments"] += 1
                _current_status["total_experiments"] += 1
                
                score = data.get("score")
                if score is not None and score < branch["best_score"]:
                    branch["best_score"] = score
                if score is not None and score < _current_status["global_best"]:
                    _current_status["global_best"] = score
        
        if event == "generation_start":
            _current_status["generation"] = data.get("generation", 0)
            _current_status["temperature"] = data.get("temperature", 0)
        
        if event == "run_start":
            _current_status["started_at"] = evt.timestamp
    
    # Push to SSE queue (non-blocking)
    try:
        _event_queue.put_nowait(evt)
    except queue.Full:
        pass  # Drop oldest events if queue is full


def get_status() -> Dict[str, Any]:
    """Get current aggregated status (for polling/API)."""
    with _status_lock:
        return dict(_current_status)


class SSEHandler(BaseHTTPRequestHandler):
    """HTTP handler for SSE and status endpoints."""
    
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def do_GET(self):
        if self.path == "/events":
            self._handle_sse()
        elif self.path == "/status":
            self._handle_status()
        elif self.path == "/":
            self._handle_dashboard()
        else:
            self.send_error(404)
    
    def _handle_sse(self):
        """SSE event stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        # Send initial status
        initial = StatusEvent(event="status", data=get_status())
        self.wfile.write(initial.to_sse().encode())
        self.wfile.flush()
        
        # Stream events
        while True:
            try:
                evt = _event_queue.get(timeout=30)
                self.wfile.write(evt.to_sse().encode())
                self.wfile.flush()
            except queue.Empty:
                # Send heartbeat
                self.wfile.write(": heartbeat\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
    
    def _handle_status(self):
        """JSON status endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(get_status(), default=str).encode())
    
    def _handle_dashboard(self):
        """Minimal HTML dashboard."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(DASHBOARD_HTML.encode())


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>PBAR Monitor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'SF Mono', 'Fira Code', monospace; 
            background: #0f0f0f; 
            color: #e0e0e0;
            padding: 20px;
        }
        h1 { color: #38bdf8; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
        }
        .card.running { border-color: #a855f7; box-shadow: 0 0 10px rgba(168, 85, 247, 0.3); }
        .card h2 { font-size: 14px; color: #888; margin-bottom: 10px; }
        .card .value { font-size: 24px; color: #38bdf8; }
        .card .value.best { color: #22c55e; }
        .card .status { 
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin-top: 8px;
        }
        .card .status.running { background: #a855f7; color: white; }
        .card .status.idle { background: #333; color: #888; }
        .events {
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
            font-size: 12px;
        }
        .event { 
            padding: 5px 10px; 
            border-left: 3px solid #333;
            margin: 2px 0;
        }
        .event.experiment_end { border-color: #38bdf8; }
        .event.generation_end { border-color: #a855f7; background: #1f1f2e; }
        .meta { display: flex; gap: 20px; margin-bottom: 20px; }
        .meta .item { }
        .meta .label { color: #666; font-size: 11px; }
        .meta .val { color: #38bdf8; font-size: 18px; }
    </style>
</head>
<body>
    <h1>🧬 PBAR Monitor</h1>
    
    <div class="meta">
        <div class="item">
            <div class="label">GENERATION</div>
            <div class="val" id="generation">0</div>
        </div>
        <div class="item">
            <div class="label">TEMPERATURE</div>
            <div class="val" id="temperature">2.000</div>
        </div>
        <div class="item">
            <div class="label">GLOBAL BEST</div>
            <div class="val" id="global_best">∞</div>
        </div>
        <div class="item">
            <div class="label">EXPERIMENTS</div>
            <div class="val" id="total_experiments">0</div>
        </div>
    </div>
    
    <div class="grid" id="branches"></div>
    
    <div class="events" id="events"></div>
    
    <script>
        const branchesEl = document.getElementById('branches');
        const eventsEl = document.getElementById('events');
        
        function updateMeta(status) {
            document.getElementById('generation').textContent = status.generation || 0;
            document.getElementById('temperature').textContent = (status.temperature || 0).toFixed(3);
            const best = status.global_best;
            document.getElementById('global_best').textContent = 
                (best === null || best === Infinity || best > 1e6) ? '∞' : best.toFixed(6);
            document.getElementById('total_experiments').textContent = status.total_experiments || 0;
        }
        
        function updateBranches(branches) {
            branchesEl.innerHTML = '';
            for (const [id, b] of Object.entries(branches || {}).sort((a,b) => a[0] - b[0])) {
                const card = document.createElement('div');
                card.className = 'card' + (b.status === 'running' ? ' running' : '');
                const best = b.best_score;
                const bestStr = (best === null || best === Infinity || best > 1e6) ? '∞' : best.toFixed(6);
                card.innerHTML = `
                    <h2>BRANCH ${id}</h2>
                    <div class="value ${best < Infinity ? 'best' : ''}">${bestStr}</div>
                    <div>${b.experiments || 0} experiments</div>
                    <span class="status ${b.status || 'idle'}">${(b.status || 'idle').toUpperCase()}</span>
                    ${b.current_description ? '<div style="color:#666;font-size:11px;margin-top:5px">' + b.current_description.slice(0,40) + '</div>' : ''}
                `;
                branchesEl.appendChild(card);
            }
        }
        
        function addEvent(evt) {
            const div = document.createElement('div');
            div.className = 'event ' + evt.event;
            const time = new Date(evt.timestamp * 1000).toLocaleTimeString();
            let text = `[${time}] `;
            if (evt.branch_id !== null && evt.branch_id !== undefined) {
                text += `B${evt.branch_id}: `;
            }
            text += evt.event;
            if (evt.score !== undefined) {
                text += ` → ${evt.score.toFixed(6)}`;
            }
            if (evt.description) {
                text += ` (${evt.description.slice(0, 30)})`;
            }
            div.textContent = text;
            eventsEl.insertBefore(div, eventsEl.firstChild);
            // Keep max 50 events
            while (eventsEl.children.length > 50) {
                eventsEl.removeChild(eventsEl.lastChild);
            }
        }
        
        // Connect to SSE
        const es = new EventSource('/events');
        
        es.addEventListener('status', (e) => {
            const data = JSON.parse(e.data);
            updateMeta(data);
            updateBranches(data.branches);
        });
        
        es.addEventListener('experiment_start', (e) => {
            const data = JSON.parse(e.data);
            addEvent(data);
            // Update branch status
            fetch('/status').then(r => r.json()).then(s => updateBranches(s.branches));
        });
        
        es.addEventListener('experiment_end', (e) => {
            const data = JSON.parse(e.data);
            addEvent(data);
            fetch('/status').then(r => r.json()).then(s => {
                updateMeta(s);
                updateBranches(s.branches);
            });
        });
        
        es.addEventListener('generation_start', (e) => {
            const data = JSON.parse(e.data);
            addEvent(data);
            updateMeta(data);
        });
        
        es.addEventListener('generation_end', (e) => {
            const data = JSON.parse(e.data);
            addEvent(data);
        });
        
        es.onerror = () => {
            console.log('SSE connection lost, reconnecting...');
        };
    </script>
</body>
</html>
"""


class StatusServer:
    """Threaded HTTP server for status updates."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8766):
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the server in a background thread."""
        self.server = ThreadingHTTPServer((self.host, self.port), SSEHandler)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        print(f"PBAR status server running at http://{self.host}:{self.port}/")
    
    def stop(self) -> None:
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            self.server = None


# Convenience: start server if run directly
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    server = StatusServer(port=port)
    server.start()
    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
