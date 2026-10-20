"""Install LabelImg with the same proxy handling used by project setup."""
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent

def main():
    env = os.environ.copy()
    for name in ('cache', 'tmp', '过程截图'):
        (ROOT / name).mkdir(exist_ok=True)
    env.update(PIP_CACHE_DIR=str(ROOT / 'cache'), TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'))
    for key, value in urllib.request.getproxies().items():
        if key in ('http', 'https'):
            env[key.upper() + '_PROXY'] = value.replace('https://', 'http://', 1)
    command = [sys.executable, '-m', 'pip', 'install', 'labelImg==1.8.6']
    logpath = ROOT / '过程截图' / 'L2_02_LabelImg安装日志.txt'
    print('Installing LabelImg. Please wait...', flush=True)
    with logpath.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
        message = 'LabelImg installation complete.' if code == 0 else 'Installation failed. Keep the error output.'
        print(message, flush=True)
        log.write('\n' + message + '\n')
    return code

if __name__ == '__main__':
    raise SystemExit(main())
