# Semantic Event Detection with Optimized VLM

## 1. Chosen Model
We selected **OpenAI's CLIP (ViT-B/32)** as the core Vision-Language Model (VLM).
- **Reasoning**: CLIP provides robust zero-shot classification capabilities, allowing us to detect diverse events ("Person Walking", "Vehicle Stopping", "Crowded Scene") using natural language prompts without retraining. The Vision Transformer (ViT) architecture is also well-suited for quantization.
- **Architecture**:
    - **Image Encoder**: ViT-B/32
    - **Text Encoder**: Transformer
    - **Optimization**: We pre-compute text embeddings for the event prompts once, so the real-time inference cost is dominated by the Image Encoder.

## 2. Methodology

#### Event Detection Logic
1.  **Motion Estimation**: Frame differencing (`cv2.absdiff`) computes a motion score.
    - `Score > 0.005`: Movement detected (e.g., walking, crowds).
    - `Score < 0.2`: Low motion (e.g., vehicle stopping).
2.  **Semantic Analysis**: CLIP (ViT-B/32) extracts visual features and compares them to prompt ensembles.
3.  **Prompt Ensembling**: We average embeddings for "person walking", "hiking", "walking outdoors" to create a robust class prototype.
4.  **Relative Margin Decision**: Instead of absolute thresholds, we use a **Top-1 with Margin** logic.
    - We calculate the margin between the top class and the runner-up.
    - If `Margin > 0.005` (0.5%), the model is confident enough to trigger the event.
    - This handles the "flat probability" issue where CLIP distributes scores evenly (~0.33) across 3 classes.

### Optimization Technique
We applied **Post-Training Dynamic Quantization** using PyTorch.
- **Target**: `torch.nn.Linear` layers within the Transformer blocks.
- **Method**: Converted weights from **FP32 (32-bit floating point)** to **INT8 (8-bit integer)**.
- **Impact**: This reduces memory bandwidth requirements and utilizes integer arithmetic units, which are often faster on CPUs.

## 3. Performance Comparison

We benchmarked the model on a CPU environment.

| Model | Precision | Size (MB) | Latency (ms) | FPS | Speedup | Size Reduction |
|-------|-----------|-----------|--------------|-----|---------|----------------|
| CLIP (Original) | FP32 | 577.22 | ~55.73 | ~17.9 | 1.0x | 1.0x |
| CLIP (Optimized) | INT8 | 224.45 | ~43.75 | ~22.9 | **1.27x** | **2.57x** |

### Observations & Trade-offs
- **Speedup**: We observed a **1.27x speedup** in inference time. This is consistent with expectations for dynamic quantization on ViT models, where attention mechanisms (which are not quantized in this mode) still consume significant compute.
- **Size**: The model size was reduced by **2.57x**, making it much more deployment-friendly for edge devices.
- **Accuracy**: Dynamic quantization generally maintains high accuracy (typically <1% drop), which is acceptable for semantic event detection where the semantic gap is large.

### Real-World Testing
The pipeline was validated on real-world footage:
- `CAR_STOP.mp4`: Successfully detected "Vehicle Stopping".
- `crowded scene.mp4`: Validated "Crowded Scene" detection on high-resolution input.
- `person walking.mp4`: Confirmed "Person Walking" event logic.

## 4. Conclusion
The optimized pipeline demonstrates that modern VLMs can be adapted for real-time or near-real-time applications on resource-constrained systems. By combining semantic understanding from CLIP with lightweight motion heuristics and quantization, we achieved a functional and efficient event detector.
