# coreml_models

The int8-quantised Core ML models the app runs. Xcode compiles each
`.mlpackage` here into a `.mlmodelc` in the app bundle's Resources, which is
what `AgarScopeKit`'s `Inference.load(dir:name:)` looks for first.

Quantised from the float32 originals in `~/bacteriaserius/coreml_models` by
`coreml_tools/quantize.py`: 627 MB to 178 MB, weights only, per-channel
symmetric. What that costs in colonies is measured in `AgarScopeKit/README.md`
rather than assumed.
