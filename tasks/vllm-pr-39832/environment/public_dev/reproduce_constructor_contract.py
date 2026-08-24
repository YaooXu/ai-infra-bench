import sys
from types import ModuleType, SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)


class MinimalConnector(KVConnectorBase_V1):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(
        self, request, blocks, num_external_tokens
    ) -> None:
        pass

    def build_connector_meta(self, scheduler_output):
        return None

    def start_load_kv(self, forward_context, **kwargs) -> None:
        pass

    def wait_for_layer_load(self, layer_name) -> None:
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        pass

    def wait_for_save(self) -> None:
        pass


class LegacyConnector(MinimalConnector):
    constructed = 0

    def __init__(self, vllm_config, role):
        super().__init__(vllm_config=vllm_config, role=role)
        type(self).constructed += 1


class CurrentConnector(MinimalConnector):
    constructed = 0

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        type(self).constructed += 1


def make_config(connector_name: str, module_name: str):
    transfer = SimpleNamespace(
        kv_connector=connector_name,
        kv_connector_module_path=module_name,
        engine_id="constructor-contract-dev",
    )
    scheduler = SimpleNamespace(disable_hybrid_kv_cache_manager=True)
    return SimpleNamespace(
        kv_transfer_config=transfer,
        scheduler_config=scheduler,
    )


def main() -> None:
    module_name = "constructor_contract_external_connector"
    connector_module = ModuleType(module_name)
    connector_module.LegacyConnector = LegacyConnector
    connector_module.CurrentConnector = CurrentConnector
    sys.modules[module_name] = connector_module

    role = KVConnectorRole.SCHEDULER
    kv_cache_config = object()

    current = KVConnectorFactory.create_connector(
        make_config("CurrentConnector", module_name),
        role,
        kv_cache_config,
    )
    if CurrentConnector.constructed != 1:
        print("FAIL: current connector was not constructed exactly once")
        raise SystemExit(1)
    if current.role is not role or current._kv_cache_config is not kv_cache_config:
        print("FAIL: current connector did not preserve constructor arguments")
        raise SystemExit(1)

    legacy_config = make_config("LegacyConnector", module_name)
    factory_rejected = False
    try:
        KVConnectorFactory.create_connector(
            legacy_config,
            role,
            kv_cache_config,
        )
    except ValueError as exc:
        message = str(exc).lower()
        factory_rejected = "kv_cache_config" in message and "constructor" in message
    except TypeError:
        factory_rejected = True

    base_rejected = False
    try:
        LegacyConnector(legacy_config, role)
    except TypeError:
        base_rejected = True

    if factory_rejected and base_rejected and LegacyConnector.constructed == 0:
        print("PASS: current constructor works and legacy constructor is rejected")
        return

    print(
        "FAIL: legacy constructor remained accepted "
        f"(factory_rejected={factory_rejected}, "
        f"base_rejected={base_rejected}, "
        f"constructed={LegacyConnector.constructed})"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
