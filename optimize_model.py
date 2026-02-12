import torch
import torch.quantization
from transformers import CLIPModel, CLIPProcessor
import time
import os
import numpy as np
from PIL import Image

MODEL_NAME = "openai/clip-vit-base-patch32"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def benchmark_inference(model, processor, device="cpu", num_iterations=50):
    print(f"Benchmarking model on {device}...")
    
    # Create dummy input
    dummy_image = Image.new('RGB', (224, 224), color='red')
    inputs = processor(images=dummy_image, return_tensors="pt").to(device)
    
    # Warmup
    print("Warmup...")
    with torch.no_grad():
        for _ in range(10):
            out = model.get_image_features(**inputs)
            if not isinstance(out, torch.Tensor):
                out = out[0]
            
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    latencies = []
    start_total = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            t0 = time.time()
            out = model.get_image_features(**inputs)
            if not isinstance(out, torch.Tensor):
                 out = out[0]
            t1 = time.time()
            latencies.append(t1 - t0)
            
    end_total = time.time()
    
    avg_latency = np.mean(latencies) * 1000 # ms
    fps = 1.0 / np.mean(latencies)
    
    return avg_latency, fps

def get_model_size(model_path):
    size = os.path.getsize(model_path)
    return size / (1024 * 1024) # MB

def main():
    device = "cpu" # Quantization is typically for CPU inference optimization
    
    # 1. Load Original Model (FP32)
    print("Loading Original CLIP (FP32)...")
    model_fp32 = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model_fp32.eval()
    
    # Save FP32 model for size comparison
    fp32_path = os.path.join(OUTPUT_DIR, "clip_fp32.pth")
    torch.save(model_fp32.state_dict(), fp32_path)
    size_fp32 = get_model_size(fp32_path)
    
    # Benchmark FP32
    latency_fp32, fps_fp32 = benchmark_inference(model_fp32, processor, device)
    print(f"FP32 Results: Latency={latency_fp32:.2f}ms, FPS={fps_fp32:.2f}, Size={size_fp32:.2f}MB")
    
    # 2. Apply Dynamic Quantization
    print("\nApplying Dynamic Quantization (INT8)...")
    # We only quantize Linear layers in the transformer
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save INT8 model
    int8_path = os.path.join(OUTPUT_DIR, "clip_int8.pth")
    torch.save(model_int8.state_dict(), int8_path)
    size_int8 = get_model_size(int8_path)
    
    # Benchmark INT8
    latency_int8, fps_int8 = benchmark_inference(model_int8, processor, device)
    print(f"INT8 Results: Latency={latency_int8:.2f}ms, FPS={fps_int8:.2f}, Size={size_int8:.2f}MB")
    
    # 3. Report Speedup
    speedup = latency_fp32 / latency_int8
    size_reduction = size_fp32 / size_int8
    
    print("\n--- Summary ---")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Size Reduction: {size_reduction:.2f}x")
    
    # Generate Report Data
    output_csv = "results/benchmark.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("Model,Precision,Latency(ms),FPS,Size(MB)\n")
        f.write(f"CLIP_FP32,FP32,{latency_fp32:.2f},{fps_fp32:.2f},{size_fp32:.2f}\n")
        f.write(f"CLIP_INT8,INT8,{latency_int8:.2f},{fps_int8:.2f},{size_int8:.2f}\n")

if __name__ == "__main__":
    main()
