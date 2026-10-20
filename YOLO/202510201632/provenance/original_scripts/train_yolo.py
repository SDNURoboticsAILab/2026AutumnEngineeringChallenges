"""Train YOLO using the user's manual labels only."""
import os
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'config')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache'/'matplotlib')
os.environ['TEMP']=str(ROOT/'tmp'); os.environ['TMP']=str(ROOT/'tmp')

def main(epochs=1):
    from ultralytics import YOLO
    import torch
    weights=ROOT/'weights'/'yolo11n.pt'
    if not weights.exists():
        weights.parent.mkdir(exist_ok=True)
        proxies={k:v.replace('https://','http://',1) for k,v in urllib.request.getproxies().items() if k in ('http','https')}
        opener=urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
        print('Downloading official pretrained weights for initialization...',flush=True)
        import shutil
        partial=weights.with_suffix('.part')
        with opener.open(url,timeout=60) as response, partial.open('wb') as output:
            shutil.copyfileobj(response,output)
        partial.replace(weights)
    model=YOLO(str(weights))
    torch.set_num_threads(4)
    print('Training 51 manually labeled images; validating on 12 images.',flush=True)
    print('One epoch is a pipeline check, not the final model.',flush=True)
    return model.train(data=str(ROOT/'data.yaml'),epochs=int(epochs),imgsz=640,batch=4,device='cpu',workers=0,
                       project=str(ROOT/'runs'),name='manual_smoke' if int(epochs)==1 else 'manual_train',
                       exist_ok=False,seed=42,deterministic=True,amp=False,cache=False,plots=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=1)
    args=parser.parse_args()
    main(args.epochs)
