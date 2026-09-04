Work in `/workspace/repo`.

We are seeing an accuracy regression for Mamba-family models in NIXL-style
disaggregated prefill/decode serving with FULL CUDA graphs. Requests finish and
there is no exception, but decode-side tokens can differ from eager serving
after recurrent state has already been produced or transferred.

Please restore FULL-CG accuracy without changing genuine first-token prompts,
ordinary decode, or supported speculative-serving paths.

I do not have a reduced reproduction or a metadata-level diagnosis. Build
focused tests that distinguish this topology from neighboring working cases,
then implement the fix.
