import os
import csv
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import argparse
from sklearn import metrics as sk_metrics

# Model definition from test.py of the original CFM repository (see README).
from test import SCL


# ==================== PREPROCESSING ====================

def get_transform():
    """Returns the image transforms for model input."""
    return alb.Compose([
        alb.Resize(320, 320),
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


def preprocess_image(img, config):
    """
    Preprocess an image for inference.
    img: numpy array (H, W, C) in BGR format
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_transform()
    transformed = transform(image=img_rgb)
    img_tensor = transformed['image']
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# ==================== INFERENCE FUNCTIONS ====================

def load_model(checkpoint_path, device='cuda'):
    """Load the pretrained CFM model."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SCL()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def predict_frame(model, img_tensor, threshold=0.5, device='cuda'):
    """
    Predicts whether a frame is fake or real.
    Returns: dict with 'prediction', 'probability', 'logits'
    """
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        prob_fake = probs[0, 1].item()
        prediction = 'fake' if prob_fake > threshold else 'real'
        return {
            'prediction': prediction,
            'probability': prob_fake,
            'logits': logits.cpu().numpy()
        }


def collect_frame_paths(video_dir):
    """Collects all frame paths in a video directory."""
    video_path = Path(video_dir)
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    frame_paths = []

    for ext in extensions:
        frame_paths.extend(sorted(video_path.glob(f'*{ext}')))

    return frame_paths


# ==================== METRICS ====================

def compute_metrics(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=float).squeeze()
    y_true = np.array(y_true, dtype=int)
    y_true[y_true >= 1] = 1

    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = sk_metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    eer = float(fpr[np.nanargmin(np.absolute(fnr - fpr))])
    ap  = sk_metrics.average_precision_score(y_true, y_pred)
    pred_class = (y_pred > 0.5).astype(int)
    acc = (pred_class == np.clip(y_true, 0, 1)).sum().item() / len(pred_class)

    return {"acc": round(acc, 4), "auc": round(auc, 4),
            "eer": round(eer, 4), "ap": round(ap, 4), "video_auc": round(auc, 4)}


# ==================== RUN FOLDER ====================

def run_folder_frames(folder, model, config, threshold, true_label, test_mode=False):
    """
    Iterates over videos in folder preprocessed with DeepfakeBench.
    Expected structure: <folder>/frames/<video_name>/*.png
    Frames are already face-cropped (256x256), no face detection needed.
    Returns list of dicts {video, score, true_label, pred_label, error}.
    true_label: 0 = REAL, 1 = FAKE
    """
    frames_root = os.path.join(folder, 'frames')
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"'frames' folder not found in: {folder}")

    rows = []
    video_dirs = sorted([d for d in os.listdir(frames_root)
                         if os.path.isdir(os.path.join(frames_root, d))])

    if test_mode:
        video_dirs = video_dirs[:2]

    print(f"  Videos found: {len(video_dirs)}")

    for video_name in tqdm(video_dirs, desc=f"{'FAKE' if true_label == 1 else 'REAL'}"):
        video_path = os.path.join(frames_root, video_name)
        frame_paths = collect_frame_paths(video_path)

        if test_mode:
            frame_paths = frame_paths[:32]

        try:
            probs = []
            for frame_path in frame_paths:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue

                face_tensor = preprocess_image(img, config)
                result = predict_frame(model, face_tensor, threshold, config['device'])
                probs.append(result['probability'])

            if not probs:
                rows.append({
                    "video":      video_name,
                    "score":      None,
                    "true_label": true_label,
                    "pred_label": "NO_FACE",
                    "error":      "",
                })
                continue

            score = sum(probs) / len(probs)
            pred_label = 'fake' if score > threshold else 'real'
            rows.append({
                "video":      video_name,
                "score":      round(score, 4),
                "true_label": true_label,
                "pred_label": pred_label,
                "error":      "",
            })

        except Exception as e:
            print(f"  Error on {video_name}: {e}")
            rows.append({
                "video":      video_name,
                "score":      None,
                "true_label": true_label,
                "pred_label": "ERROR",
                "error":      str(e),
            })

    return rows


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='CFM Batch Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (models_params_28.tar)')
    parser.add_argument('--fake-folder', type=str, nargs='+', required=True,
                        help='One or more FAKE video folders (each subdir = video with frames)')
    parser.add_argument('--real-folder', type=str, required=True,
                        help='REAL video folder (each subdir = video with frames)')
    parser.add_argument('--output', type=str, nargs='+', default=None,
                        help='Output CSV name(s), one per fake folder. Default: folder basename.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for fake/real classification (default: 0.5)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode: process only 2 videos and 32 frames per video')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device for inference')

    args = parser.parse_args()

    fake_folders = [os.path.abspath(f) for f in args.fake_folder]
    real_folder  = os.path.abspath(args.real_folder)

    if args.output:
        if len(args.output) != len(fake_folders):
            raise ValueError("--output must have the same number of elements as --fake-folder")
        output_names = args.output
    else:
        output_names = [os.path.basename(f.rstrip("/")) for f in fake_folders]

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    config = {
        'resolution': 320,
        'device': device
    }

    model = load_model(args.checkpoint, device=device)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Run real inference once, reuse across all fake folders
    print(f"\n{'='*50}\n[REAL] {real_folder}\n{'='*50}")
    real_rows = run_folder_frames(real_folder, model, config,
                                  args.threshold, true_label=0, test_mode=args.test_mode)

    for fake_folder, out_name in zip(fake_folders, output_names):
        print(f"\n{'='*50}\n[FAKE] {fake_folder}\n{'='*50}")
        fake_rows = run_folder_frames(fake_folder, model, config,
                                      args.threshold, true_label=1, test_mode=args.test_mode)

        all_rows = fake_rows + real_rows

        # Metrics
        valid = [(r["score"], r["true_label"]) for r in all_rows if r["score"] is not None]
        m = None
        if len(valid) >= 2:
            y_pred = np.array([v[0] for v in valid])
            y_true = np.array([v[1] for v in valid])
            m = compute_metrics(y_pred, y_true)

        output_csv = os.path.join(results_dir, f"{out_name}.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            if m is not None:
                writer.writerow(["metric", "value"])
                for k, v in m.items():
                    writer.writerow([k, v])
                writer.writerow([])
            writer.writerow(["video", "score", "true_label", "pred_label", "error"])
            for r in all_rows:
                writer.writerow([r["video"], r["score"], r["true_label"], r["pred_label"], r["error"]])

        print(f"\nResults saved to: {output_csv}")

        if m is not None:
            print("\n========== Metrics ==========")
            for k, v in m.items():
                print(f"  {k}: {v}")
            print("==============================")

    print("\nDone.")


if __name__ == '__main__':
    main()
