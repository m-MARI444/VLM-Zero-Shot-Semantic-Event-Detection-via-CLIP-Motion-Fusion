import cv2
import torch
import numpy as np
import argparse
import time
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
# Robust Prompt Ensembling
TARGET_EVENTS = {
    "Person Walking": [
        "a photo of a person walking",
        "a person walking outdoors", 
        "a person hiking", 
        "a person walking on a path", 
        "someone walking in nature"
    ],
    "Vehicle Stopping": [
        "a photo of a vehicle stopping", 
        "a car braking", 
        "brake lights on", 
        "traffic stopping", 
        "a car slowing down"
    ],
    "Crowded Scene": [
        "a photo of a crowded scene", 
        "many people gathering", 
        "a busy street", 
        "crowd of people", 
        "dense crowd"
    ]
}


class EventDetector:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading CLIP model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Pre-compute and Average Text Features (Prompt Ensembling)
        print("Pre-computing and averaging text embeddings...")
        self.event_classes = list(TARGET_EVENTS.keys())
        self.text_features = []
        
        with torch.no_grad():
            for event_name in self.event_classes:
                prompts = TARGET_EVENTS[event_name]
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                
                # Get embeddings for all prompts of this class
                class_features = self.model.get_text_features(**inputs)
                
                # Handle output type if necessary (reuse fix)
                if not isinstance(class_features, torch.Tensor):
                     if hasattr(class_features, 'text_embeds'):
                         class_features = class_features.text_embeds
                     elif hasattr(class_features, 'pooler_output'):
                         class_features = class_features.pooler_output
                     else:
                         try:
                             class_features = class_features[0]
                         except:
                             pass
                
                # Normalize individual prompts first (optional but good practice)
                class_features /= class_features.norm(dim=-1, keepdim=True)
                
                # Average them
                mean_feature = class_features.mean(dim=0, keepdim=True)
                
                # Normalize the averaged embedding
                mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
                
                self.text_features.append(mean_feature)
        
        # Stack into (3, 512) tensor
        self.text_features = torch.vstack(self.text_features)

        self.prev_frame_gray = None
        self.motion_score_ema = 0.0
        self.alpha = 0.2  # Smoothing factor for EMA
        

    def get_motion_score(self, current_frame):
        """Simple frame differencing to estimate global motion magnitude."""
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return 0.0
        
        frame_delta = cv2.absdiff(self.prev_frame_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = np.sum(thresh) / (gray.shape[0] * gray.shape[1]) # Normalize by area
        
        self.prev_frame_gray = gray
        return motion_score

    def detect(self, video_path, output_path=None, threshold=0.6):
        # Reset state for new video
        self.prev_frame_gray = None
        self.motion_score_ema = 0.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        inference_times = []

        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")

        print("Starting processing...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            pbar.update(1)
            start_time = time.time()
            
            # 1. Motion Estimation
            raw_motion = self.get_motion_score(frame)
            self.motion_score_ema = (self.alpha * raw_motion) + ((1 - self.alpha) * self.motion_score_ema)
            
            # 2. CLIP Inference (Image Encoder Only)
            # Resize for CLIP (224x224 usually handled by processor, but doing explicit resize can save transfer time if frame is huge)
            # However, processor handles normalization and resizing best.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                if not isinstance(image_features, torch.Tensor):
                     # Likely BaseModelOutputWithPooling
                     # Try to access pooler_output or index 0
                     if hasattr(image_features, 'image_embeds'):
                         image_features = image_features.image_embeds
                     elif hasattr(image_features, 'pooler_output'):
                         image_features = image_features.pooler_output
                     else:
                         try:
                             image_features = image_features[0]
                         except:
                             pass
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity
                similarity = (image_features @ self.text_features.T)
                
                # Softmax
                probs = torch.nn.functional.softmax(similarity, dim=-1).cpu().numpy()[0]

            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 3. Event Logic
            detected_events = []
            
            # Helper to get prob by name
            def get_prob(name):
                if name in self.event_classes:
                    return probs[self.event_classes.index(name)]
                return 0.0

            p_walking = get_prob("Person Walking")
            p_stopping = get_prob("Vehicle Stopping")
            p_crowded = get_prob("Crowded Scene")

            # Debug prints
            if frame_count % 30 == 0:
                 print(f"Frame {frame_count}: Walk={p_walking:.2f}, Stop={p_stopping:.2f}, Crowd={p_crowded:.2f}, Motion={self.motion_score_ema:.4f}")

            # Top-1 Relative Margin Logic
            # CLIP usually ranks correct class slightly higher, even if absolute probability is low (~0.35 vs 0.33)
            sorted_probs = np.sort(probs)
            margin = sorted_probs[-1] - sorted_probs[-2]
            
            top_index = np.argmax(probs)
            top_event = self.event_classes[top_index]
            top_prob = probs[top_index]

            # Logic: Only consider the winner if it beats the runner-up by a small margin
            # Lowered margin to 0.005 because 0.34 vs 0.33 is common for correct detections
            if margin > 0.005: 
                if top_event == "Person Walking" and self.motion_score_ema > 0.005:
                    detected_events.append(f"Person Walking ({top_prob:.2f})")
                
                elif top_event == "Vehicle Stopping" and self.motion_score_ema < 0.2:
                    detected_events.append(f"Vehicle Stopping ({top_prob:.2f})")

                elif top_event == "Crowded Scene":
                     detected_events.append(f"Crowded Scene ({top_prob:.2f})")

            # 4. Visualization
            annotated_frame = frame.copy()
            
            # Draw Labels (Events) - Increased Size
            y_offset = 50
            for event in detected_events:
                cv2.putText(annotated_frame, event, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                y_offset += 40
            
            # Draw Probabilities (Always visible for debugging) - Increased Size
            cv2.putText(annotated_frame, f"Walk: {p_walking:.2f} | Stop: {p_stopping:.2f} | Crowd: {p_crowded:.2f}", 
                        (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw Stats - Increased Size
            cv2.putText(annotated_frame, f"Motion: {self.motion_score_ema:.4f} | Inference: {inference_time*1000:.1f}ms", 
                        (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if output_path:
                out.write(annotated_frame)
            
            # Optional: Show frame (commented out for headless environment)
            # cv2.imshow('Event Detection', annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        if output_path:
            out.release()
        pbar.close()
        
        avg_time = np.mean(inference_times)
        print(f"Processing complete.")
        print(f"Average Inference Time per Frame: {avg_time*1000:.2f} ms")
        print(f"Average FPS: {1.0/avg_time:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--input_dir", type=str, help="Path to input directory containing videos")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output videos")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not args.video and not args.input_dir:
        parser.error("Must provide either --video or --input_dir")

    detector = EventDetector()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_dir:
        import glob
        video_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
        if not video_files:
            print(f"No .mp4 files found in {args.input_dir}")
        
        for video_path in video_files:
            filename = os.path.basename(video_path)
            output_path = os.path.join(args.output_dir, f"output_{filename}")
            print(f"\nProcessing {filename} -> {output_path}")
            detector.detect(video_path, output_path, args.threshold)
            
    else:
        # Single video mode
        output_path = args.output_dir if args.output_dir.endswith(".mp4") else os.path.join(args.output_dir, "output.mp4")
        if not args.output_dir.endswith(".mp4") and os.path.isdir(args.output_dir):
             filename = os.path.basename(args.video)
             output_path = os.path.join(args.output_dir, f"output_{filename}")
        
        print(f"Processing {args.video} -> {output_path}")
        detector.detect(args.video, output_path, args.threshold)
