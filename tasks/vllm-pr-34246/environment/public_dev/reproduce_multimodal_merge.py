import torch

from vllm.model_executor.models.utils import _merge_multimodal_embeddings


class reject_cuda_sync:
    def __enter__(self):
        self.previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, exception_type, exception_value, traceback):
        torch.cuda.set_sync_debug_mode(self.previous)


def make_case(mask_device: str):
    num_tokens = 8192
    hidden_size = 512
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[::2] = True
    if mask_device == "cuda":
        mask = mask.cuda()
    inputs = torch.zeros(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    multimodal = [
        torch.ones(
            (num_tokens // 2, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
    ]
    return inputs, multimodal, mask


def measured_merge(mask_device: str, reject_sync: bool):
    inputs, multimodal, mask = make_case(mask_device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    if reject_sync:
        with reject_cuda_sync():
            _merge_multimodal_embeddings(inputs, multimodal, mask)
    else:
        _merge_multimodal_embeddings(inputs, multimodal, mask)
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - before
    target_bytes = inputs.numel() * inputs.element_size()
    correct = bool(
        torch.all(inputs[::2] == 1).item()
        and torch.all(inputs[1::2] == 0).item()
    )
    return peak_delta / target_bytes, correct


def check_cardinality_error() -> bool:
    inputs = torch.zeros((5, 8), dtype=torch.bfloat16, device="cuda")
    mask = torch.tensor([True, False, True, False, False], device="cpu")
    multimodal = [torch.ones((3, 8), dtype=torch.bfloat16, device="cuda")]
    try:
        _merge_multimodal_embeddings(inputs, multimodal, mask)
    except ValueError as exc:
        message = str(exc)
        return "3 multimodal tokens" in message and "2 placeholders" in message
    return False


def main() -> None:
    gpu_mask_ratio, gpu_mask_correct = measured_merge("cuda", reject_sync=False)
    print(f"gpu_mask_peak_ratio={gpu_mask_ratio:.3f}")
    if not gpu_mask_correct:
        print("FAIL: same-device merge corrupted embedding positions")
        raise SystemExit(1)

    try:
        cpu_mask_ratio, cpu_mask_correct = measured_merge("cpu", reject_sync=True)
    except Exception as exc:
        print(f"FAIL: CPU mask merge raised {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print(f"cpu_mask_peak_ratio={cpu_mask_ratio:.3f}")
    if not cpu_mask_correct:
        print("FAIL: CPU mask merge corrupted embedding positions")
        raise SystemExit(1)
    if cpu_mask_ratio >= 4.0:
        print("FAIL: CPU mask merge retained excessive temporary CUDA allocation")
        raise SystemExit(1)
    if not check_cardinality_error():
        print("FAIL: excess multimodal embeddings were not rejected precisely")
        raise SystemExit(1)

    print("PASS: CPU mask merge is correct, asynchronous, bounded, and strict")


if __name__ == "__main__":
    main()
