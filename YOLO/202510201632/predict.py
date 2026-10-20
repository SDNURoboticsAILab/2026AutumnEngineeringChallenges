"""Detect training-external images using the submitted trained model."""
import argparse
import hashlib
from pathlib import Path
from common import ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='new_images')
    parser.add_argument('--weights', default='runs/manual_train/weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--name', default='reproduce_predict')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = Path(args.source)
    if not source.is_absolute(): source = ROOT / source
    suffixes = {'.jpg','.jpeg','.png','.bmp','.webp'}
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in suffixes) if source.is_dir() else [source]
    if not images or not all(p.is_file() for p in images): raise ValueError('No readable input paths')
    used = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'dataset/images').rglob('*.jpg')}
    duplicates = [p.name for p in images if hashlib.sha256(p.read_bytes()).hexdigest() in used]
    if duplicates: raise ValueError('Training/validation images are not new inputs: '+', '.join(duplicates))
    weights = Path(args.weights)
    if not weights.is_absolute(): weights = ROOT / weights
    if not weights.is_file(): raise FileNotFoundError(weights)
    if args.check:
        print('New images:',len(images),'No exact overlap with train/val. Model:',weights)
        return
    from ultralytics import YOLO
    results = YOLO(str(weights)).predict(source=[str(p) for p in images], imgsz=640, conf=args.conf, device='cpu',
              save=True, save_txt=True, save_conf=True, show_labels=True, show_conf=True,
              project=str(ROOT/'results'), name=args.name, exist_ok=False)
    print('Results:',results[0].save_dir)

if __name__ == '__main__':
    main()
