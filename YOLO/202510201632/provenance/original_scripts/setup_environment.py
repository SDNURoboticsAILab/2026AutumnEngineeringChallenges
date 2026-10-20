"""Install project dependencies into this project's Python environment."""
import os
import pathlib
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent

def main():
    env = os.environ.copy()
    cache = ROOT / 'cache'
    temp = ROOT / 'tmp'
    cache.mkdir(exist_ok=True)
    temp.mkdir(exist_ok=True)
    env.update(PIP_CACHE_DIR=str(cache), TEMP=str(temp), TMP=str(temp), YOLO_CONFIG_DIR=str(ROOT / 'config'))
    # Normalize Windows proxy schemes for the older pip bundled with Python 3.8.
    for key, value in urllib.request.getproxies().items():
        if key in ('http', 'https'):
            env[key.upper() + '_PROXY'] = value.replace('https://', 'http://', 1)
    packages = ['torch==2.4.1', 'torchvision==0.19.1', 'numpy==1.24.4', 'ultralytics']
    print('Installing PyTorch, torchvision, NumPy and Ultralytics into the G-drive project environment.', flush=True)
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', *packages], env=env)
    if result.returncode:
        print('Installation failed. Keep the error output for troubleshooting.')
        return
    with (ROOT / 'requirements-installed.txt').open('w', encoding='utf-8') as output:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=output, check=True, env=env)
    print('Installation complete. Close and reopen Python before verifying imports.')

if __name__ == '__main__':
    main()
