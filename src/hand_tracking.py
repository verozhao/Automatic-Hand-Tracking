import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm
from mobile_sam import sam_model_registry, SamPredictor

def detect_hands(image: np.ndarray) -> List[Dict[str, np.ndarray]]:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.5
    )
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    hand_data = []
    if results.multi_hand_landmarks:
        height, width = image.shape[:2]
        
        for landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * width for lm in landmarks.landmark]
            y_coords = [lm.y * height for lm in landmarks.landmark]
            
            bbox = np.array([
                min(x_coords), min(y_coords),
                max(x_coords), max(y_coords)
            ]).astype(int)
            
            points = []
            key_indices = [0, 4, 8, 12, 16, 20]
            for idx in key_indices:
                point = np.array([
                    landmarks.landmark[idx].x * width,
                    landmarks.landmark[idx].y * height
                ]).astype(int)
                points.append(point)
            
            hand_data.append({
                'bbox': bbox,
                'points': np.array(points)
            })
    
    hands.close()
    return hand_data
def track_hands(input_path: str, output_path: str, sam_checkpoint: str = 'mobile_sam.pt') -> None:
    sam = sam_model_registry.get("vit_t")(checkpoint=sam_checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (width, height))
    
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            hand_data = detect_hands(frame)
            
            if hand_data:
                predictor.set_image(frame)
                combined_mask = np.zeros((height, width), dtype=bool)
                
                for hand in hand_data:
                    points = hand['points']
                    point_labels = np.ones(len(points))
                    
                    masks, scores, _ = predictor.predict(
                        point_coords=points,
                        point_labels=point_labels,
                        box=hand['bbox'][None, :],
                        multimask_output=False
                    )
                    
                    combined_mask |= masks[0]
                
                frame[combined_mask] = frame[combined_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            out.write(frame)
            pbar.update(1)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    track_hands('test.mp4', 'output.mp4')
