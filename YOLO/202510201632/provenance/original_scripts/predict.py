"""Level 4: predict new images with the user's trained best.pt."""
from pathlib import Path
import os
import hashlib
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'config')

def main():
    suffixes={'.jpg','.jpeg','.png','.bmp','.webp'}
    images=sorted(p for p in (ROOT/'new_images').iterdir() if p.suffix.lower() in suffixes)
    if not images:
        print('No images found. Copy new pictures into G:/YOLO学习项目/new_images first.')
        return
    used={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'dataset'/'images').rglob('*') if p.is_file() and p.suffix.lower() in suffixes}
    duplicates=[p.name for p in images if hashlib.sha256(p.read_bytes()).hexdigest() in used]
    if duplicates:
        print('These images were used in training/validation; replace them:', ', '.join(duplicates))
        return
    from ultralytics import YOLO
    model=YOLO(str(ROOT/'runs'/'manual_train'/'weights'/'best.pt'))
    print('Using your trained best.pt. New images:',len(images),flush=True)
    results=model.predict(source=[str(p) for p in images],imgsz=640,conf=0.25,device='cpu',save=True,save_txt=True,save_conf=True,
                          show_labels=True,show_conf=True,project=str(ROOT/'results'),name='new_images',exist_ok=False)
    print('Detection complete. Results:',results[0].save_dir,flush=True)
    for result in results:
        print(Path(result.path).name,'detections:',len(result.boxes))

if __name__=='__main__': main()
