# Vendored LLMBox

Upstream: https://github.com/sarakingsley/llmbox, commit `d924984` (2026-09-21).
This is the same release the course VM has at `/opt/95820/SRC/llmbox-20260921-d924984/`.

The first commit adding this folder is the untouched upstream code. It leaves out
`finetuned/` (a 112 MB LoRA adapter) and `outputs/` (upstream run logs), because
neither is needed to run inference. Every Assignment 2 change is a separate
commit after it, so `git log -- assignment02/llmbox` and `git diff <vendor-commit> -- assignment02/llmbox`
show the API design changes exactly.

License: GPL-3.0 (see upstream README).
