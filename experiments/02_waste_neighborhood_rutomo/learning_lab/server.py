"""Local, isolated LLMBox practice server for the A1 -> A2 learning lab.

Run from assignment02: .venv/bin/python a2/learning_lab/server.py
The server binds to loopback only. It reads experiment counts and metrics but
never writes into a2/runs, a2/data, or a2/labels.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llmbox"))
sys.path.insert(0, str(ROOT / "a2/scripts"))

from a2common import BASELINE, JSON_FORMAT_LINE  # noqa: E402
from src.batch import BatchRunner, SUBMODES  # noqa: E402
from src.generation import GenerationManager  # noqa: E402
from src.schema import ToolDef  # noqa: E402

PHI_LOCAL = REPO / "lab01/models/phi-4-mini-instruct"
CONFIGS = {
    "phi": (ROOT / "a2/runs/C1_structured/config.yaml" if PHI_LOCAL.is_dir() else ROOT / "a2/runs/BENCH_Linux/config.yaml",
            PHI_LOCAL if PHI_LOCAL.is_dir() else Path("/opt/95820-models/microsoft/Phi-4-mini-instruct")),
    "gemma": (ROOT / "a2/runs/E_gemma270m_structured/config.yaml",
              ROOT / "llmbox/models/llms/google/gemma-3-270m-it"),
}
RUNS = ["B1_generate_defaults", "B3_generate_tuned", "C1_structured", "C2_tools", "C3_guarded",
        "C4_fewshot", "E_gemma270m_structured", "E_nonllm_tfidf_lr"]
A1_LAB = REPO / "assignment01/assignment1/support/learning/learning_lab.html"


def state():
    runs = {}
    for name in RUNS:
        base = ROOT / "a2/runs" / name
        responses = base / "responses.jsonl"
        metrics = base / "metrics.json"
        runs[name] = {
            "rows": sum(bool(line.strip()) for line in responses.open(encoding="utf-8")) if responses.exists() else 0,
            "metrics": json.loads(metrics.read_text(encoding="utf-8")) if metrics.exists() else None,
        }
    partd_path = ROOT / "a2/results/partd_metrics.json"
    partd = json.loads(partd_path.read_text(encoding="utf-8")) if partd_path.exists() else None
    return {"runs": runs, "partd": ({"by_category": partd["by_category"],
                                      "probe_design": partd.get("probe_design")} if partd else None),
            "labels_ready": (ROOT / "a2/labels/human_labels_a2.jsonl").exists(),
            "probes_ready": (ROOT / "a2/data/probes/probes_seed.jsonl").exists(),
            "a1_lab_available": A1_LAB.is_file()}


class PracticeModel:
    def __init__(self):
        self.lock = threading.Lock()
        self.loaded = {}

    def _runner(self, name):
        if name not in CONFIGS:
            raise ValueError("Choose phi or gemma")
        if name not in self.loaded:
            config_path, weights = CONFIGS[name]
            if not config_path.exists() or not weights.is_dir():
                raise FileNotFoundError(f"Local {name} weights or configuration are unavailable")
            if config_path.name == "config.yaml" and config_path.parent.name == "BENCH_Linux":
                from hydra import compose, initialize_config_dir
                from src.schema import register_configs
                register_configs()
                with initialize_config_dir(config_dir=str((ROOT / "llmbox/conf").resolve()), version_base=None):
                    cfg = compose(config_name="config", overrides=["model=phi4_instruct", "mode=batch"])
            else:
                cfg = OmegaConf.load(config_path)
            cfg.model.source = "local"
            cfg.model.local_path = str(weights.resolve())
            cfg.guardrail.log_path = None
            cfg.guardrail.refusal_message = "This request cannot be completed by the 311 tagging API. Ask a staff member to review it."
            task_data = REPO / "assignment01/assignment1/support/data"
            if not task_data.is_dir():
                task_data = ROOT / "a2/data/a1"
            cfg.tool_calling.codebook_path = str(task_data / "311_issue_category_codebook.csv")
            cfg.tool_calling.issue_summary_path = str(task_data / "waste_issue_summary.csv")
            tool_spec = json.loads((ROOT / "llmbox/conf/tools/waste311_tools.json").read_text(encoding="utf-8"))
            cfg.tool_calling.tools_file = None
            cfg.tool_calling.tools = [ToolDef(**t) for t in tool_spec["tools"]
                                      if t["name"] in {"lookup_codebook", "get_issue_stats"}]
            cfg.tool_calling.tool_choice = "required"
            cfg.tool_calling.review_queue_path = None
            cfg.structured_output.pydantic_model = "src.pydantic_models.waste_tags:WasteTags"
            generator = GenerationManager()
            runner = BatchRunner(generator, cfg)
            runner.model, runner.tokenizer, runner.device = generator._load_model_and_tokenizer(cfg)
            self.loaded[name] = runner
        return self.loaded[name]

    def run(self, request):
        model = request.get("model", "phi")
        mode = request.get("mode", "generate")
        if mode not in {"generate", "structured", "guarded", "tools"}:
            raise ValueError("Choose generate, structured, tools, or guarded")
        if mode == "tools" and model != "phi":
            raise ValueError("The local Gemma practice model does not support this tool mode; choose Phi")
        record = request.get("record", "")
        if not isinstance(record, str) or not 1 <= len(record.strip()) <= 3000:
            raise ValueError("Enter a record of 1 to 3000 characters")
        max_tokens = request.get("max_new_tokens", 128)
        if type(max_tokens) is not int or not 16 <= max_tokens <= 256:
            raise ValueError("max_new_tokens must be an integer from 16 to 256")
        do_sample = request.get("do_sample", False)
        if type(do_sample) is not bool:
            raise ValueError("do_sample must be true or false")
        temperature = request.get("temperature", 1.0)
        top_p = request.get("top_p", 0.95)
        if not isinstance(temperature, (int, float)) or not 0 < temperature <= 2:
            raise ValueError("temperature must be greater than 0 and at most 2")
        if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        with self.lock:
            runner = self._runner(model)
            cfg = runner.cfg
            old = OmegaConf.to_container(cfg.generation, resolve=True)
            try:
                cfg.generation.max_new_tokens = max_tokens
                cfg.generation.do_sample = do_sample
                cfg.generation.temperature = float(temperature)
                cfg.generation.top_p = float(top_p)
                row = {"id": "practice", "instruction": BASELINE + JSON_FORMAT_LINE,
                       "record": record.strip()}
                result = SUBMODES[mode](runner, row, 0)
            finally:
                for key, value in old.items():
                    cfg.generation[key] = value
        return {key: result.get(key) for key in ("text", "parsed", "violation", "retries", "blocked", "review", "guard", "tool_calls", "tool_rounds", "prompt_tokens", "completion_tokens", "latency_s", "finish_reason", "seed") if key in result}


PRACTICE = PracticeModel()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # HTTP request lines can contain practice content; keep only the method/path.
        print(f"lab: {self.command} {self.path.split('?')[0]}", file=sys.stderr)

    def send(self, code, payload, mime="application/json; charset=utf-8"):
        data = json.dumps(payload, ensure_ascii=False).encode() if not isinstance(payload, bytes) else payload
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        assets = {"/": (HERE / "index.html", "text/html; charset=utf-8"),
                  "/style.css": (HERE / "style.css", "text/css; charset=utf-8"),
                  "/app.js": (HERE / "app.js", "text/javascript; charset=utf-8"),
                  "/a1": (A1_LAB, "text/html; charset=utf-8")}
        if self.path == "/api/state":
            return self.send(200, state())
        if self.path in assets and assets[self.path][0].is_file():
            path, mime = assets[self.path]
            return self.send(200, path.read_bytes(), mime)
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/run":
            return self.send(404, {"error": "Not found"})
        if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}"):
            return self.send(403, {"error": "Local origin required"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("Request body must be 1 to 8192 bytes")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON object")
            return self.send(200, PRACTICE.run(request))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception:
            self.send(500, {"error": "Model run failed; check the local terminal"})
            raise


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8762)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{server.server_port}/ (local only)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
