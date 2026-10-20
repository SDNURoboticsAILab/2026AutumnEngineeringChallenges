from pathlib import Path
import os
import yaml
ROOT = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.local' / 'ultralytics')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.local' / 'matplotlib')

def runtime_data():
    config = yaml.safe_load((ROOT / 'data.yaml').read_text(encoding='utf-8'))
    config['path'] = str(ROOT / config.get('path', 'dataset'))
    target = ROOT / '.local' / 'data_runtime.yaml'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    return target
