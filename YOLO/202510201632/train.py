"""Reproduce training from official pretrained weights, using manual labels."""
import argparse
from pathlib import Path
from common import ROOT, runtime_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--model', default='yolo11n.pt', help='Official pretrained model or a local initialization file')
    parser.add_argument('--name', default='reproduce_train')
    parser.add_argument('--check', action='store_true', help='Validate configuration and dataset paths without training')
    args = parser.parse_args()
    data = runtime_data()
    if args.check:
        for split in ['train', 'val']:
            images = list((ROOT/'dataset'/'images'/split).glob('*.jpg'))
            assert images and all((ROOT/'dataset'/'labels'/split/(p.stem+'.txt')).exists() for p in images)
            print(split, len(images))
        print('Configuration and dataset paths OK:', data)
        return
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(4)
    # No submitted best.pt is overwritten: reproduction uses its own run name.
    weights_dir = ROOT / 'weights'
    weights_dir.mkdir(exist_ok=True)
    model_path = Path(args.model)
    if args.model == 'yolo11n.pt' and not model_path.exists():
        model_path = weights_dir / 'yolo11n.pt'
    model = YOLO(str(model_path))
    model.train(data=str(data), epochs=args.epochs, imgsz=640, batch=4, device='cpu', workers=0,
                project=str(ROOT/'runs'), name=args.name, exist_ok=False, seed=42,
                deterministic=True, amp=False, cache=False, plots=True)

if __name__ == '__main__':
    main()
