# RQ1 conclusion audit: May 18 to July 31

This audit records how the RQ1 conclusions changed after replacing the May 18 snapshot analysis with the cutoff-consistent merged database through **2026-07-31 23:59:59 UTC**. The July analysis uses seven complete 2026 months; the earlier estimates used the shorter January–May observation window. The values are therefore successive descriptive estimates, not a statistical before/after experiment.

## Result changes

| Claim or indicator | Earlier May analysis | July 31 analysis | Assessment |
|---|---:|---:|---|
| Monthly new PRs in 2026 | 1,836.3 | 2,102.6 | Demand-growth claim strengthened |
| Monthly merged PRs in 2026 | 909.0 | 974.4 | Throughput rose much more slowly than intake |
| Active roster reviewers per month | 60.3 | 58.0 | No corresponding reviewer expansion |
| Submitted reviews per new PR | 1.55 | 1.35 | Review-density decline strengthened |
| Open PRs at cutoff | 3,037 | 4,194 | Integration queue grew materially |
| Issue roster response within 7 days | 27.2% | 23.0% | Responsiveness decline strengthened |
| PR roster response within 7 days | 62.2% | 57.6% | Responsiveness decline strengthened |
| Submitted roster review within 7 days | 55.0% | 50.8% | Formal-review decline strengthened |
| Active roster reviewers in 2026 | 75 | 77 | Participation broadened slightly |
| Top-five share of roster reviews | 39.7% | 35.0% | Concentration claim revised downward |

## Conclusions that remain robust

1. **PR demand is growing faster than visible review and merge capacity.** Relative to 2025, monthly PR intake in 2026 Jan–Jul increased 96.8%, while merged PRs increased 35.3%, reviewer-days 24.4%, and submitted reviews 17.7%.
2. **External contributors provide most implementation intake, while final integration remains permission-gated.** External humans authored 75.1% of human PRs in 2026 Jan–Jul; May-18 write-capable actors performed 96.1% of human-authored merges with a known actor.
3. **Merged PRs are not the whole workload.** Open and closed-unmerged 2026 PRs received 18.6% of roster submitted reviews and 25.9% of roster inline comments.
4. **AI-inference maintenance is technically specialized.** Distributed execution, attention/kernels, V1 runtime, KV cache, quantization, MoE, speculative decoding, CUDA, ROCm, XPU, CPU, and cross-backend work all form material benchmark strata.
5. **A merged-PR-only benchmark would overstate coverage of repository maintenance.** The benchmark also needs diagnosis, review, verifier construction, open/closed-unmerged work, and specialist hardware tasks.

## Conclusion revised

The July data do **not** support saying that review became more concentrated. The reviewer population expanded from 75 to 77 and the top-five review share fell from 39.7% to 35.0%. The defensible conclusion is narrower: review participation broadened slightly, but observable capacity grew far more slowly than incoming PR demand, so visible review per PR declined.

## Stability checks

The merged collector sometimes observes the current text or open-PR file list after the analytical cutoff. Recomputing the taxonomies after excluding those records changes the largest category share by 0.75 percentage points for issue intent, 0.28 for PR work type, 0.28 for hardware, and 1.00 for inference topic. The workload-composition conclusions are stable to this limitation.

The merge reconstruction uses the union of an observed merge event and the materialized cutoff merge timestamp. This retains 279 PRs whose generic artifact state says `CLOSED` despite a merge event and 168 whose merge timestamp is available but whose merge event is absent. The latter actors remain unknown, so actor shares use only observed actors.

The analysis is byte-reproducible from the released database: a clean rerun produced the same `summary.json`, all 19 figures, and all 57 aggregate CSV tables. All 19 database release validations passed.

## Interpretation boundary

The roster is the collaborator permission snapshot from May 18, not a July roster or a historical membership table. Roster-response results are therefore sensitivity estimates of visible maintenance capacity. Any-human response is the primary response estimand. GitHub event counts are not hours and do not include private discussion, security work, local debugging, vendor coordination, or work in other repositories.

This dataset uses Simon Mo's [*vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran)*](https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1) as its base and extends the data through 2026-07-31.
