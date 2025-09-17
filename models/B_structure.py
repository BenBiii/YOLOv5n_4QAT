import sys
from pathlib import Path

# 현재 경로를 PYTHONPATH에 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5n_FACE 루트 디렉토리
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torchinfo import summary
from models.yolo import Model

# 모델 config와 클래스 수 설정
cfg_path = 'models/yolov5n.yaml'  # YOLOv5n 설정 파일
model = Model(cfg=cfg_path, ch=3, nc=1)  # ch=입력 채널수, nc=클래스 수

# GPU 할당 (가능할 경우)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 요약 정보 출력
summary(model, input_size=(1, 3, 640, 640), col_names=["output_size", "num_params"])
