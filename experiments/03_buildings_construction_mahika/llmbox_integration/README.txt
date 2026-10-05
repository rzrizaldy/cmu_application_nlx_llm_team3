# Drop these files into the professor's LLMBox tree to run the custom
# features through the LLMBox CLI directly. They are NEW files; they do not
# edit any of her existing modules.
#
#   llmbox_integration/pydantic_models/triage.py
#       -> copy to  llmbox-main/src/pydantic_models/triage.py
#   llmbox_integration/conf/tool_calling/route_request.yaml
#       -> copy to  llmbox-main/conf/tool_calling/route_request.yaml
#
# See cli_examples.md for the exact commands. The batch 50-input runs use the
# wrapper in ../triage_api.py (which imports LLMBox's own GenerationManager);
# these CLI files are the native-LLMBox proof and the structured_output schema.
