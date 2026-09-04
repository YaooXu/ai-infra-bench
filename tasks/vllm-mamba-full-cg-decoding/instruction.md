Work in `/workspace/repo`.

I am seeing an accuracy regression with Mamba-family models in NIXL-style
disaggregated prefill/decode serving when FULL CUDA graphs are enabled. Requests
finish without an exception, but after recurrent state has been produced or
transferred, some decode-side tokens differ from eager serving.

Please restore the FULL-CG result without changing genuine first-token prompts,
ordinary decode, or supported speculative-serving paths.

I do not have a reduced reproduction or a metadata-level diagnosis. Please
build focused tests that distinguish this deployment from the neighboring
working cases, then implement and validate the fix.
