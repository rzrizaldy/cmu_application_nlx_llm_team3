'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

"""
schema.py

Structured (dataclass) configuration schema for the LLM configurator. Hydra
uses these dataclasses to type-check every config value, catch typos and
wrong types at composition time, and let every field be overridden from the
command line, e.g.:

    python configurator.py model=phi4_instruct mode=chat generation.temperature=0.7
    python configurator.py model=gemma3_270m mode=finetune training.enabled=true \
        training.method=lora optimizer=adamw optimizer.learning_rate=5e-5
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Which model to load and how. One YAML file per model under conf/model/.

    `source` picks where the weights actually come from:
      - "huggingface": `model_id` is a hub repo id (e.g. "google/gemma-3-270m-it").
        transformers will use a local cache if one exists and download
        otherwise -- normal from_pretrained behavior.
      - "local": `local_path` is a path to a directory already on disk
        (a `huggingface-cli download ... --local-dir` checkout, or any
        folder with the usual config/tokenizer/weight files in it).
        Loaded with local_files_only=True, so this never touches the
        network and fails fast with a clear error if the path is wrong,
        rather than transformers silently trying to interpret it as a
        hub repo id.
    """
    name: str = MISSING
    source: str = "local"              # huggingface | local
    model_id: str = MISSING            # HF hub repo id, used when source=huggingface
    local_path: Optional[str] = None   # path to a local checkout, used when source=local
    architecture: str = MISSING        # informational only, e.g. "gemma3_text", "phi3"
    dtype: str = "bfloat16"            # bfloat16 | float16 | float32
    device: str = "auto"               # auto | mps | cuda | cpu
    trust_remote_code: bool = False
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    # generic: mode=tool_calling passes tool defs to apply_chat_template via
    #   a `tools=` kwarg, the way most function-calling chat templates expect.
    # functools_prompt: for models whose template doesn't do that (e.g.
    #   Phi-4-mini-instruct) -- tool defs go in the system prompt as
    #   <|tool|>[...]<|/tool|>, and a "functools[...]"-prefixed reply is
    #   parsed and dispatched locally. See GenerationManager.run_tool_turn.
    tool_calling_format: str = "generic"       # generic | functools_prompt
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    # How to find assistant tokens for loss masking (see TrainingDataLoader):
        #   template        -> template must contain {% generation %} blocks
        #   verified_prefix -> infer spans from exact token-prefix matches
        # Gemma 3 and Phi-4-mini templates have no generation blocks.
    assistant_mask_strategy: str = "verified_prefix"   # template | verified_prefix
    # prepare_data: model-specific data formatting options
    add_prefix: Optional[str] = None           # for prepare_data - string to prepend to prompt
    add_suffix: Optional[str] = None           # for prepare_data - string to append to completion
    add_specialtokens: Optional[List[str]] = field(default_factory=list)  # special tokens to add to each record if needed

@dataclass
class GenerationConfig:
    """Sampling / decoding hyperparameters. Used by every inference mode
    (chat, generate, tool_calling, structured_output)."""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    do_sample: bool = True
    repetition_penalty: float = 1.0

@dataclass
class ToolDef:
    """One callable tool, described the way chat templates that support
    function calling generally expect (JSON-schema-style parameters)."""
    name: str = MISSING
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolCallingConfig:
    enabled: bool = False
    tool_choice: str = "auto"          # auto | required | none
    tools: List[ToolDef] = field(default_factory=list)
    tools_file: Optional[str] = None       # [A2 M3] JSON {"system_hint": str, "tools": [ToolDef...]}; replaces `tools`
    max_rounds: int = 3                    # [A2 M3] tool rounds before the model must answer
    codebook_path: Optional[str] = None    # [A2 M3] 311 issue/category codebook CSV (lookup_codebook)
    issue_summary_path: Optional[str] = None   # [A2 M3] waste issue closure-time summary CSV (get_issue_stats)
    review_queue_path: Optional[str] = None    # [A2 M3] JSONL queue written by queue_tag_update

@dataclass
class StructuredOutputConfig:
    enabled: bool = False
    schema_path: Optional[str] = None  # path to a JSON Schema file describing the desired output
    strict: bool = True                # if true, fail loudly (well, warn) when output isn't valid JSON
    pydantic_model: Optional[str] = None   # [A2 M2] "module.path:ClassName"; overrides schema_path and validates
    max_retries: int = 2                   # [A2 M2] re-ask with the validation errors up to this many times

@dataclass
class BatchConfig:
    """[A2 M1] mode=batch: run every row of a JSONL file through one submode."""
    input_path: Optional[str] = None   # JSONL rows: {id, prompt} or {id, instruction, record}
    submode: str = "generate"          # generate | structured | tools | guarded | fewshot
    run_name: Optional[str] = None     # folder name under runs_dir (default: submode + timestamp)
    runs_dir: str = "../a2/runs"
    limit: int = 0                     # 0 = all rows
    repeat: int = 1                    # >1 re-runs each row (determinism checks)
    apply_seed: bool = True            # seed each row with cfg.seed + row index

@dataclass
class GuardrailConfig:
    """[A2 M4] batch.submode=guarded. See src/guardrail.py for the four decisions."""
    inner_submode: str = "structured"          # submode used for corpus-record rows
    prompt_submode: str = "generate"           # submode used for free-form request rows
    injection_patterns: List[str] = field(default_factory=lambda: [
        r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|directions)",
        r"disregard (all |any )?(the )?(previous|prior|above|earlier|your) (instructions|rules)",
        r"(reveal|print|show|repeat|output) (me )?(your|the) (system )?(prompt|instructions)",
        r"you are now\b",
        r"new instructions?:",
        r"\bsystem prompt\b",
        r"\bdeveloper mode\b",
        r"do not (tag|classify|follow)",
    ])
    input_judge: str = "none"                  # none | llm  (scope/harm judge for free-form requests)
    scope_judge_prompt: str = (
        "You screen requests sent to the City of Pittsburgh 311 knowledge-base tagging API. "
        "The API only tags 311 service records (service focus, information type, responsible department) "
        "and looks up 311 issue codes and historical closure times.\n"
        "Request:\n<<<\n{text}\n>>>\n"
        "Answer with exactly one word: IN_SCOPE if the request is that kind of work, "
        "OUT_OF_SCOPE if it asks for anything else, HARMFUL if it asks for abusive, demeaning, "
        "unsafe or privacy-violating content.")
    toxicity: str = "none"                     # none | lexicon | llm
    lexicon_path: Optional[str] = None         # one term per line, written by the student
    toxicity_judge_prompt: str = (
        "Does the following text contain abusive, demeaning, threatening or unsafe content about "
        "any person or group?\nText:\n<<<\n{text}\n>>>\nAnswer with exactly one word: SAFE or UNSAFE.")
    judge_max_new_tokens: int = 6
    secret_markers: List[str] = field(default_factory=list)
    block_rules: List[str] = field(default_factory=lambda: [
        "injection_pattern", "scope_judge", "secret_leak", "pii_in_output", "toxicity_lexicon", "toxicity_judge"])
    review_rules: List[str] = field(default_factory=lambda: ["ungrounded_department", "schema_invalid", "masked"])
    refusal_message: str = ("This request cannot be completed by the 311 tagging API. "
                            "It has been logged, and a staff member can review it.")
    log_path: Optional[str] = None

@dataclass
class RetrievalConfig:
    """[A2 M5] batch.submode=fewshot: k labelled DEV examples shown before each input."""
    pool_path: Optional[str] = None            # JSONL {id, record, label}; DEV records only
    k: int = 3
    inner_submode: str = "structured"

@dataclass
class OptimizerConfig:
    """One YAML file per optimizer under conf/optimizer/."""
    name: str = "adamw"                # adamw | sgd | adafactor
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    momentum: float = 0.0              # only used by sgd

@dataclass
class TrainingConfig:
    """Shared by both the 'train' (continued pretraining, full weights only)
    and 'finetune' (parameter-efficient or full) modes.
    New: method supports additional techniques beyond LoRA.
    """
    enabled: bool = False
    method: str = "lora"               # full | lora | adapters | bitfit | freeze | prefix | qlora
    epochs: float = 3.0
    batch_size: int = 2
    grad_accum_steps: int = 8
    warmup_ratio: float = 0.03
    max_length: int = 1024
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    prefix_length: int = 30            # For prefix-tuning
    adapters_dim: int = 64             # For adapters
    bitfit_bias_params: Optional[List[str]] = field(default_factory=lambda: [])  # For BitFit, e.g. ['bias']
    freeze_modules: Optional[List[str]] = field(default_factory=lambda: [])  # For freeze, e.g. ['embed', 'ln_f']
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    output_dir: str = "./finetuned"
    merge_adapter: bool = False        # After PEFT training, also save an adapter-free merged copy
    # Added for budget metrics:
    token_budget: int = 1000000        # Total allowed tokens (settable)
    # These fields are not used by training itself, but are used for reporting
    # Inserted to accommodate budget/cost tracking and metric reporting.
    report_metrics: bool = True


@dataclass
class DataConfig:
    """Where training/finetuning data comes from. One YAML file per source
    type under conf/data/.
    For prepare_data mode, allow additional attributes for raw/original path,
    text/target columns, label maps, instruction, output_dir, etc."""
    type: str = MISSING                # chat_log | jsonl | other future source types
    path: str = MISSING
    min_turns: int = 1                 # chat_log only: skip sessions with fewer turns than this
    eval_split: float = 0.1            # fraction of conversations held out for eval (0 disables it)
    # Extra fields for prepare_data
    raw_path: Optional[str] = None     # optional: path to original data to reformat
    original_path: Optional[str] = None # optional alias for raw_path
    text_columns: Optional[List[str]] = field(default_factory=list)    # For prepare_data/source conversion
    target_columns: Optional[List[str]] = field(default_factory=list)  # For prepare_data/source conversion
    label_maps: Optional[Dict[str, Dict[Any, Any]]] = field(default_factory=dict)   # for prepare_data
    instruction: Optional[str] = ""    # for prepare_data, optional prompt-instruction
    output_dir: Optional[str] = None   # for prepare_data: custom output dir for storing transformed files
    train_fraction: Optional[float] = None  # for prepare_data - train/test split
    split: Optional[float] = None          # alias for train_fraction

@dataclass
class ModeConfig:
    """Which of the seven run modes to execute. One YAML file per mode under
    conf/mode/.
    Now supports prepare_data in addition to chat, generate, tool_calling, structured_output, train, finetune.
    """
    name: str = MISSING                # chat | generate | tool_calling | structured_output | train | finetune | prepare_data

@dataclass
class TrainingMetrics:
    """Track and record metrics and resource use for a training job."""
    # MLflow-logged scalar metrics
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    # Resource and cost metrics
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_time: Optional[float] = None
    max_memory_bytes: Optional[int] = None
    avg_cpu_util_percent: Optional[float] = None
    avg_gpu_util_percent: Optional[float] = None
    flops_estimate: Optional[float] = None
    carbon_kg_estimate: Optional[float] = None
    tokens_processed: int = 0
    token_budget: int = 0

    @property
    def tokens_remaining(self) -> int:
        return (self.token_budget or 0) - (self.tokens_processed or 0)

@dataclass
class Config:
    model: ModelConfig = MISSING
    mode: ModeConfig = MISSING
    data: DataConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tool_calling: ToolCallingConfig = field(default_factory=ToolCallingConfig)
    structured_output: StructuredOutputConfig = field(default_factory=StructuredOutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)          # [A2 M1]
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)  # [A2 M4]
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)  # [A2 M5]
    # Single-turn modes (generate / tool_calling / structured_output) read
    # their input from one of these; chat mode ignores them and reads stdin.
    prompt: Optional[str] = None
    prompt_file: Optional[str] = None
    username: Optional[str] = None
    system_prompt: str = "You are a helpful assistant."
    seed: int = 42
    output_dir: str = "outputs"        # chat session logs land under <output_dir>/chat_log
    metrics: Optional[TrainingMetrics] = None

def register_configs() -> None:
    """Register the schema so conf/config.yaml's `- config_schema` defaults
    entry can pull it in for validation."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
