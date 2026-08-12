# Dataset Card

## Summary

The current dataset is a discovery set of real AI-inference engineering workloads recommended by maintainers. It is intended to seed task curation for AI Infra Bench; it is not itself an executable benchmark.

| Item | Current value |
| --- | ---: |
| Records | 24 |
| vLLM records | 23 |
| Cross-repository FlashInfer records | 1 |
| Pull requests | 21 |
| Issues | 3 |
| Validated benchmark tasks | 0 |

The GitHub metadata snapshot date is August 8, 2026.

## Collection

Maintainers were asked to nominate memorable or meaningful issues and pull requests, explain why the work was difficult or important, classify it, and provide an informal effort estimate. Duplicate nominations were merged while preserving each endorsement.

The source file is [`data/vllm_survey_results.jsonl`](../data/vllm_survey_results.jsonl). The record schema and privacy rules are documented in the [data guide](../data/README.md).

## Intended use

- identify task candidates with real maintainer value;
- study the kinds of debugging, performance, feature, and review work found in inference systems;
- construct reproducible, execution-graded benchmark tasks;
- compare representative community work with maintainer-nominated hard cases.

## Out-of-scope uses

- treating survey records as runnable tasks;
- ranking contributors or maintainers;
- converting informal effort directly to hours;
- estimating the complete vLLM workload distribution from this convenience sample;
- training on held-out benchmark solutions before evaluation.

## Limitations

The data is a small, non-random survey sample and overrepresents memorable, recent, and difficult work. It does not yet provide meaningful heterogeneous-accelerator coverage. GitHub activity cannot recover unrecorded debugging, private discussion, or true human effort.

Representative benchmark tasks will therefore be sampled separately from a frozen 200-PR candidate pool derived from the broader repository history.

## Privacy

Public records exclude respondent names, email addresses, response timestamps, personal profile links, and links to identity-bearing individual comments. Names mentioned inside free-text responses are replaced with role-neutral placeholders. Repository-level issue and pull-request links remain because they are necessary technical provenance.
