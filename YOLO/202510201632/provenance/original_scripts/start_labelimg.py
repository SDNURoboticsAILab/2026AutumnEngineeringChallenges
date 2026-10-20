from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
QT_ROOT = Path('G:/YOLO_runtime/Lib/site-packages/PyQt5/Qt5')
os.environ['QT_PLUGIN_PATH'] = str(QT_ROOT / 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(QT_ROOT / 'plugins' / 'platforms')
os.environ['PATH'] = str(QT_ROOT / 'bin') + os.pathsep + os.environ.get('PATH', '')

if '--check-qt' in sys.argv:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    print('Qt platform initialization OK', flush=True)
    sys.exit(0)

sys.argv = ['labelImg', str(ROOT / 'raw' / 'cola'), str(ROOT / 'classes.txt'), str(ROOT / 'annotations')]
from labelImg.labelImg import main
if __name__ == '__main__':
    main()
