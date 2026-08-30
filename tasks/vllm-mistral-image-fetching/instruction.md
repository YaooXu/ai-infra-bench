After upgrading vLLM from 0.21.0 to 0.22.1, our Devstral Small vision server no longer reaches ready. The same launch configuration worked before the upgrade, and disabling image inputs lets the text-only server start, so this appears confined to the Mistral/Pixtral multimodal input path.

Startup now fails during dummy multimodal profiling with `ProcessorMixin.prepare_inputs_layout` calling `self.image_processor.fetch_images(images)`, followed by `AttributeError: 'MistralCommonImageProcessor' object has no attribute 'fetch_images'`. We also use this processor with decoded images, local image paths, URLs, and batched or nested image inputs.

Fix the regression so current Transformers can prepare those image forms through the Mistral processor, preserving input order and nesting and producing a clear error for unsupported values. Existing image processing behavior must continue to work.
