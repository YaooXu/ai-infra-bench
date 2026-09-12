# Local environment rebuild

The hardening image uses the real upstream Base commit c7560af42487b1570c4e6f4cea5df1605a4d59fc with a shallow boundary at that commit. The previous full-history fetch was replaced by a depth-one fetch. No later commits, remote refs or task artifacts are retained in the solver image.

The digest-pinned v0.20.2 runtime supplies nine native libraries and two generated Python files. Real runner construction exposed a missing generated FlashMLA interface; it is now copied from that same pinned runtime and checked in native.sha256. No newer vLLM source or native package was introduced.

Build from the worktree root with:

```sh
docker build --network host --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY -t pr56-hardening:base tasks/vllm-runner-v2-selection/environment
```

The image ID, canonical local tag, build-input hashes and dependency identity are recorded in environment/image-manifest.json. This was a local build; the tag has not been published to a registry.

Final-image checks cover exact Git HEAD/tree, clean status, absent remote/tag/unreachable history, writable source as agent, all native hashes, source and extension import paths, and a real CUDA allocation. Tests, fixtures and Oracle are mounted only for verification, not baked into the environment. The full log and build output are retained with the local evidence bundle. Historical build results are archived under validation/history and do not certify this image.
