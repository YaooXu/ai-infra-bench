# Preserve structured tool arguments for union schemas

The tool-call parser mishandles parameters whose JSON Schema uses `anyOf`
without a top-level `type`. When the generated value is a JSON object, the
parser returns the object as an escaped string, so downstream callers receive a
string instead of structured data.

Update the implementation so object-valued union parameters retain their JSON
object semantics. Existing handling for strings, numbers, booleans, arrays,
nulls, malformed values, and schemas with an explicit `type` must remain
compatible.

Work in `/workspace/vllm`. Leave the source change in the working tree. Do not
modify task metadata or verifier files.
