Work in `/workspace/repo`.

I am trying to deploy vLLM's data-parallel workers behind an external load
balancer. At the moment I have to launch every local rank separately, so
Kubernetes has no single process or endpoint that represents the health of the
whole node. We have already seen cases where one rank died while the other
ports remained up and the node continued receiving traffic.

Could `vllm serve` support a node-local multi-port mode? One invocation should
start one API endpoint per local DP rank, using consecutive ports from
`--port`, and supervise them as one group. Rank and accelerator assignment
should continue to follow the existing data-parallel, tensor-parallel and
pipeline-parallel settings.

The command-line interface should expose
`--data-parallel-multi-port-external-lb` and
`--data-parallel-supervisor-port`. The supervisor port should provide
aggregated `/health`, `/ready`, and `/readyz` endpoints. Readiness must stay
unavailable until every local rank is healthy.

Please also make startup and shutdown safe: reject incompatible settings or
overlapping ports before leaving a partially started group behind, stop the
whole group if a child exits or becomes unhealthy, and forward termination to
all children so no process or listening socket is orphaned.

The exact internal design is up to you. You can use lightweight local servers
to exercise the lifecycle; model loading, Kubernetes integration, and
multi-node routing are outside the scope of this report.
