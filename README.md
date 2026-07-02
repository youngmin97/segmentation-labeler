# Segmentation Labeler

GE E10 B-mode 초음파 영상의 segmentation 라벨링을 위한 Windows 데스크톱 앱입니다.

## 설치

```bash
cd "e:\Barreleye\20260701 Labeler"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

```bash
python run.py
```

## Segmentation 조작

| 동작 | 방법 |
|------|------|
| 영역 그리기 | 펜/터치로 **누른 채 드래그** → 떼면 시작점과 자동 연결되어 채움 |
| 드래그 미리보기 | 경로 + 끝점↔시작점 직선으로 폐곡선 영역 실시간 표시 |
| 영역 추가 | 빈 영역만 채움 (다른 라벨 침범 없음) |
| 영역 제외 | 현재 클래스 마스크에서 드래그 영역 차집합 |
| 클래스 전체 제거 | Remove 버튼 한 번 — 선택 클래스 마스크 전부 삭제 |
| 취소 (그리기 중) | Esc / 우클릭 |
| 줌 | 마우스 휠 |
| 팬 | Ctrl+드래그 또는 휠클릭 드래그 |
| 실행 취소 | Ctrl+Z |
| 저장 | Ctrl+S |

드래그 경로는 OpenCV Douglas-Peucker(`approxPolyDP`)로 단순화한 뒤 폐곡선으로 채웁니다.

## 클래스 인덱스

Label PNG는 **오버레이 색상이 아닌** uint8 grayscale 이미지입니다.  
각 픽셀 값 = 클래스 index (0=background, 1=skin, 2=thyroid, …).  
LabelBox export 형식과 동일합니다.
