import cv2
import numpy as np

# 원본 이미지 경로
img_path = r"C:\python_work\tftrain\workspace\YOLOv5n_FACE\test_image\many.jpg"

# 출력 크기 (FHD)
target_width = 1920
target_height = 1080

# 이미지 읽기
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

# 원본 크기
h, w = img.shape[:2]
aspect_ratio = w / h
target_ratio = target_width / target_height

# 리사이징 (aspect ratio 유지)
if aspect_ratio > target_ratio:
    # 이미지가 더 넓은 경우 → 너비에 맞추고 높이를 패딩
    new_width = target_width
    new_height = int(target_width / aspect_ratio)
else:
    # 이미지가 더 높은 경우 → 높이에 맞추고 너비를 패딩
    new_height = target_height
    new_width = int(target_height * aspect_ratio)

# 리사이즈된 이미지 생성
resized_img = cv2.resize(img, (new_width, new_height))

# 패딩 계산
top = (target_height - new_height) // 2
bottom = target_height - new_height - top
left = (target_width - new_width) // 2
right = target_width - new_width - left

# 패딩 추가 (검정색으로)
padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

# 결과 저장 (또는 바로 보기)
cv2.imwrite("output_fhd.jpg", padded_img)
# cv2.imshow("Padded FHD Image", padded_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
