"""
batch.py  [A2 M1]

Batch runner for mode=batch. Upstream LLMBox inference modes take one prompt
from the command line and print one answer to stdout. Nothing is saved, and
nothing is timed or counted. An evaluation of 50 inputs needs more:

  * load the model once, then run every row of a JSONL input file;
  * save each response with its measurements (tokens, latency, seed,
    finish_reason) to runs/<run_name>/responses.jsonl;
  * write a manifest (input hash, model fingerprint, git sha, library
    versions, resolved config) so a run can be reproduced and audited;
  * resume after an interruption without redoing rows that are already done.

Input rows (JSONL):
  {"id": "...", "prompt": "..."}                        free-form request
  {"id": "...", "instruction": "...", "record": "..."}  task over a corpus record
plus an optional "meta" dict that is copied through unchanged.

How a row is answered depends on batch.submode. Each submode is a function
registered in SUBMODES that takes (runner, row, index) and returns a dict.
"""

import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

SUBMODES = {}


def register_submode(name):
    def decorator(func):
        SUBMODES[name] = func
        return func
    return decorator


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_fingerprint(local_path):
    """Hash the config/tokenizer JSON files and the *names and sizes* of the
    weight shards. Hashing 7 GB of weights on every run is too slow; the
    size plus config hash is enough to catch swapping in a different checkpoint."""
    root = Path(local_path)
    if not root.is_dir():
        return None
    parts = {}
    for p in sorted(root.iterdir()):
        if p.suffix == ".json":
            parts[p.name] = _sha_file(p)
        elif p.suffix == ".safetensors":
            parts[p.name] = f"size:{p.stat().st_size}"
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - git is optional (e.g. copied onto the VM)
        return None


def redact(text):
    """Run files get copied to the course VM, where the student works under a
    pseudonym. Replace the home directory and never record the hostname."""
    home = str(Path.home())
    return str(text).replace(home, "~")


def compose_user_message(row, spotlight=False):
    """Build the user turn from a row. With spotlight=True the corpus record
    is fenced in <record> tags so the guardrail can tell data from instructions."""
    if row.get("prompt"):
        return row["prompt"]
    record = row.get("record") or ""
    if spotlight:
        record = f"<record>\n{record}\n</record>"
    return f"{row.get('instruction', '').strip()}\n\nRecord:\n{record}".strip()


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {path}")
    return rows


class BatchRunner:
    def __init__(self, generator, cfg):
        self.generator = generator
        self.cfg = cfg
        self.model = self.tokenizer = self.device = None

    # ---- shared helpers used by submodes ---------------------------------
    def seed_for(self, index):
        return self.cfg.seed + index if self.cfg.batch.apply_seed else None

    def base_messages(self, user_content, system_prompt=None, shots=None):
        system_prompt = self.cfg.system_prompt if system_prompt is None else system_prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(shots or [])          # [A2 M5] few-shot example turns, if any
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(self, messages, index, **template_kwargs):
        return self.generator._generate_measured(
            self.model, self.tokenizer, self.device, messages, self.cfg,
            seed=self.seed_for(index), **template_kwargs,
        )

    # ---- run -------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        bcfg = cfg.batch
        if bcfg.submode not in SUBMODES:
            raise ValueError(f"Unknown batch.submode '{bcfg.submode}'. Choose one of: {sorted(SUBMODES)}")
        if not bcfg.input_path:
            raise ValueError("mode=batch requires batch.input_path=path/to/inputs.jsonl")
        rows = load_rows(bcfg.input_path)
        if bcfg.limit:
            rows = rows[: bcfg.limit]

        run_name = bcfg.run_name or f"{bcfg.submode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        run_dir = Path(bcfg.runs_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "responses.jsonl"

        done = {}
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    done[(rec["id"], rec.get("repeat", 0))] = rec
        todo = [(i, r, k) for k in range(bcfg.repeat) for i, r in enumerate(rows) if (r["id"], k) not in done]
        log.info("batch %s: %d rows x %d repeats, %d already done, %d to run",
                 run_name, len(rows), bcfg.repeat, len(done), len(todo))

        (run_dir / "config.yaml").write_text(redact(OmegaConf.to_yaml(cfg)), encoding="utf-8")
        manifest = self._manifest(rows, run_name, started=True)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if todo:
            self.model, self.tokenizer, self.device = self.generator._load_model_and_tokenizer(cfg)
            submode = SUBMODES[bcfg.submode]
            with out_path.open("a", encoding="utf-8") as f:
                for n, (index, row, repeat) in enumerate(todo, 1):
                    result = submode(self, row, index)
                    record = {"id": row["id"], "repeat": repeat, "index": index,
                              "submode": bcfg.submode, **result, "meta": row.get("meta", {})}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    log.info("[%d/%d] %s  %.2fs  %s+%s tok", n, len(todo), row["id"],
                             result.get("latency_s", 0), result.get("prompt_tokens"), result.get("completion_tokens"))

        manifest = self._manifest(rows, run_name, started=False)
        manifest["responses_sha256"] = _sha_file(out_path) if out_path.exists() else None
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[info] batch run written to {run_dir}")

    def _manifest(self, rows, run_name, started):
        import torch
        import transformers
        cfg = self.cfg
        key = "started_at" if started else "finished_at"
        existing = {}
        path = Path(cfg.batch.runs_dir) / run_name / "manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.update({
            "run_name": run_name,
            "submode": cfg.batch.submode,
            "n_rows": len(rows),
            "repeat": cfg.batch.repeat,
            "input_path": redact(cfg.batch.input_path),
            "input_sha256": _sha_file(cfg.batch.input_path),
            "row_ids": [r["id"] for r in rows],
            "model_name": cfg.model.name,
            "model_source": cfg.model.source,
            "model_fingerprint": model_fingerprint(cfg.model.local_path) if cfg.model.local_path else None,
            "generation": OmegaConf.to_container(cfg.generation, resolve=True),
            "seed": cfg.seed,
            "apply_seed": cfg.batch.apply_seed,
            "git_sha": existing.get("git_sha") or git_sha(),   # code version at run start
            "platform": f"{platform.system()}-{platform.machine()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            key: datetime.now(timezone.utc).isoformat(),
        })
        return existing


@register_submode("generate")
def _generate(runner, row, index):
    """Baseline behaviour: the same single-turn call mode=generate makes."""
    messages = runner.base_messages(compose_user_message(row))
    return runner.generate(messages, index)


@register_submode("structured")
def _structured(runner, row, index):
    """[A2 M2] generate + structured output: Pydantic schema in the system
    prompt, validation, and error-feedback retries (src/structured.py)."""
    from src.structured import load_pydantic_model, schema_instruction, structured_generate
    so = runner.cfg.structured_output
    if not so.pydantic_model:
        raise ValueError("batch.submode=structured requires structured_output.pydantic_model")
    model_cls = load_pydantic_model(so.pydantic_model)
    system = (runner.cfg.system_prompt or "") + schema_instruction(model_cls)
    messages = runner.base_messages(compose_user_message(row), system_prompt=system.strip(), shots=row.get("_shots"))
    return structured_generate(lambda m: runner.generate(m, index), messages, model_cls, so.max_retries)


def load_tool_spec(cfg):
    """[A2 M3] Offered tools (name -> ToolDef dict) and the optional system hint."""
    tc = cfg.tool_calling
    if tc.tools_file:
        spec = json.loads(Path(tc.tools_file).read_text(encoding="utf-8"))
        tools, hint = spec["tools"], spec.get("system_hint", "")
    else:
        tools, hint = OmegaConf.to_container(tc.tools, resolve=True), ""
    return {t["name"]: t for t in tools}, hint


@register_submode("tools")
def _tools(runner, row, index):
    """[A2 M3] generate + tool_calling (Phi functools convention), multi-round,
    allowlisted and argument-validated. The record id is set by the runner,
    never by the model."""
    from src import tools as toolmod
    if not getattr(runner, "_tools_ready", False):
        toolmod.configure_waste311(runner.cfg)
        runner._offered, runner._tool_hint = load_tool_spec(runner.cfg)
        runner._tools_ready = True
    toolmod.CONTEXT["doc_id"] = row["id"]
    base = "\n\n".join(x for x in [runner.cfg.system_prompt, runner._tool_hint] if x)
    choice = runner.cfg.tool_calling.tool_choice
    if choice == "none":
        messages = runner.base_messages(compose_user_message(row), system_prompt=base)
        return {**runner.generate(messages, index), "tool_calls": [], "tool_rounds": 0, "nudged": False}
    system = runner.generator.build_functools_system_prompt(base, list(runner._offered.values()), choice)
    messages = runner.base_messages(compose_user_message(row), system_prompt=system)
    return runner.generator.run_tool_loop(lambda m: runner.generate(m, index), messages,
                                          runner._offered, runner.cfg.tool_calling.max_rounds,
                                          require_call=(choice == "required"))


@register_submode("guarded")
def _guarded(runner, row, index):
    """[A2 M4] Wrap another submode with the guardrail (src/guardrail.py)."""
    from src.guardrail import Guardrail
    if getattr(runner, "_guard", None) is None:
        runner._guard = Guardrail(runner.cfg, runner.generator, runner.model, runner.tokenizer, runner.device)
    gcfg = runner.cfg.guardrail
    inner_name = gcfg.inner_submode if row.get("record") else gcfg.prompt_submode

    def inner(masked_row, system_extra):
        saved = runner.cfg.system_prompt
        runner.cfg.system_prompt = "\n\n".join(x for x in [saved, system_extra] if x)
        try:
            spotlit = dict(masked_row)
            if spotlit.get("record"):
                spotlit["record"] = f"<record>\n{spotlit['record']}\n</record>"
            return SUBMODES[inner_name](runner, spotlit, index)
        finally:
            runner.cfg.system_prompt = saved

    return runner._guard.run(row, index, inner)


@register_submode("fewshot")
def _fewshot(runner, row, index):
    """[A2 M5] Retrieve k labelled DEV examples (src/retrieval.py) and run the
    inner submode with them as prior turns. Retrieved ids are recorded."""
    from src.retrieval import FewShotPool
    rc = runner.cfg.retrieval
    if getattr(runner, "_pool", None) is None:
        if not rc.pool_path:
            raise ValueError("batch.submode=fewshot requires retrieval.pool_path")
        runner._pool = FewShotPool(rc.pool_path)
    shots, retrieved = runner._pool.shots(row, rc.k, compose_user_message)
    result = SUBMODES[rc.inner_submode](runner, {**row, "_shots": shots}, index)
    return {**result, "retrieved": retrieved}


@register_submode("routed")
def _routed(runner, row, index):
    """[A2 Part D baseline] The same API as 'guarded' with the guard removed:
    corpus-record rows go to guardrail.inner_submode, free-form requests to
    guardrail.prompt_submode. Comparing 'routed' with 'guarded' isolates the
    guardrail's effect."""
    gcfg = runner.cfg.guardrail
    return SUBMODES[gcfg.inner_submode if row.get("record") else gcfg.prompt_submode](runner, row, index)
