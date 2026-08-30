# Reset CPU-offload cache state with in-flight transfers

The simple CPU offload connector cannot currently participate in prefix-cache
reset. Resetting scheduler-side state is unsafe while asynchronous store or
load DMA still owns CPU or GPU block references: freeing those blocks early can
let an old transfer write into storage that has already been reused.

Implement cache reset for the connector. Pending eager stores, lazy stores, and
loads must be abandoned without becoming cacheable, while their block
references remain pinned until completion arrives. Reset must report that it is
not yet complete while such work remains, then clear the CPU prefix cache once
all abandoned transfers have drained. Stale completion events must be harmless.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
