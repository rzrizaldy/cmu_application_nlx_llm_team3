### Tool calling

| Run | Rows with executed call | Calls | Rejected | Queued writes | Final text is an unexecuted call | Nudged | Calls by tool |
|---|---:|---:|---:|---:|---:|---:|---|
| C2_tools | 90% | 76 | 0 | 44 | 10% | 10% | {'get_issue_stats': 11, 'lookup_codebook': 21, 'queue_tag_update': 44} |
| C2abl_dev_tools_auto | 2% | 2 | 0 | 0 | 0% | 0% | {'get_issue_stats': 1, 'lookup_codebook': 1} |
| C2abl_dev_tools_required | 70% | 54 | 0 | 36 | 30% | 30% | {'get_issue_stats': 6, 'lookup_codebook': 12, 'queue_tag_update': 36} |
