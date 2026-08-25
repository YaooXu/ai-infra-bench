Work in `/workspace/repo`.

Implement the node-local supervisor used by multi-port external-load-balancer data parallel serving. It must validate incompatible launch arguments, derive one child rank and port per local DP rank, assign each child its device visibility, launch each API server in a separate spawned process, and publish aggregate `/health`, `/ready`, and `/readyz` status on the supervisor port.

The supervisor must remain unready until every child is healthy. Once serving, a child process exit or failed health probe must stop the whole local group, forward shutdown to surviving children, wait for graceful exit, force-kill stragglers, stop its own HTTP server, and release all ports. Preserve clean SIGINT/SIGTERM propagation and avoid orphan processes.

The required scope is the production node-local supervisor module and its lifecycle contract. Kubernetes resources, multi-node routing, CLI wiring, model loading, and inference throughput are outside this task.
