Work in `/workspace/repo`.

We are trying to move a deployment that uses Decode Context Parallelism to GPU
Model Runner V2. The same model works without DCP, but with both options enabled
some requests fail or produce different results. We can also reproduce the
problem when CUDA graphs are enabled, so we cannot roll V2 out to this serving
configuration.

Implement complete Model Runner V2 support for the existing DCP configuration.
The result must remain correct across supported paged-KV-cache layouts and
successive decode steps, in eager and CUDA-graph execution. Serving without DCP
must remain backward compatible.

I do not have a reduced reproduction or an internal diagnosis. Please inspect
the production path, construct focused tests for the configuration, and make
the change suitable for production.
