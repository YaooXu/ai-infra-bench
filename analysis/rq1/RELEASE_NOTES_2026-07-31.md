# vLLM GitHub workload dataset through July 31, 2026

This prerelease contains the public SQLite dataset used to extend AI Infra Bench's RQ1 workload study through the inclusive UTC cutoff `2026-07-31T23:59:59Z`.

Download `vllm_github_2026-07-31.sqlite.zst`, verify SHA-256 `a15e30ab5d187a46b46a4fc493b157fe00d13d3135d87ab94b990e2df04383e0`, and run `unzstd vllm_github_2026-07-31.sqlite.zst`. The decompressed SQLite file is 2,935,083,008 bytes with SHA-256 `2ac86507a95f9b8785e6ce0bbf2745e3fbba67c747e37b54020a7e57ce80f8b5`.

The database preserves the original Fivetran tables, adds the full `delta_*` normalized API layer, and provides deduplicated `canonical_*` tables plus queryable provenance and validation. All 19 release checks pass, including SQLite integrity, foreign keys, union counts, and cutoff leakage checks. See `MERGED_DATABASE_2026-07-31.json` for counts, anomalies, checksums, and limitations.

## Source and citation

The base snapshot is Simon Mo's [*vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran)*](https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1), shared May 18, 2026. Please cite that gist when using this dataset:

> Simon Mo. “vLLM GitHub Gym: vLLM GitHub Snapshot (Fivetran).” GitHub Gist, May 18, 2026. <https://gist.github.com/simon-mo/2b0f4e9f872d479a08ae53edac51ecb1>.

## Data notice

This database contains public GitHub text, usernames, actor identifiers, and commit metadata including names and emails. It is not de-identified survey data. No new license is asserted over third-party GitHub content; users should follow the source and GitHub terms when redistributing or using it.
