from __future__ import annotations

# ============================================================
# Pokemon Pure Python GUI Analyzer
# ------------------------------------------------------------
# HTML / Flask 없이 Tkinter만으로 실행되는 포켓몬 분석기입니다.
#
# 기능:
# 1. 이미지 삽입
# 2. 검은색 그림판 그리기
# 3. data/PokemonData 폴더 확인
# 4. 없으면 KaggleHub로 데이터셋 다운로드 시도
# 5. TensorFlow 없이 간단한 이미지 특징 기반 classifier 생성
# 6. Top-5 예측 결과 출력
#
# 주의:
# - 이 버전은 TensorFlow 전이학습 모델이 아닙니다.
# - Python 3.14 문제를 피하기 위해 TensorFlow 없이 동작하도록 만든 버전입니다.
# ============================================================

import importlib.util
import math
import os
import pickle
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox


# ------------------------------------------------------------
# 0. 필요한 라이브러리 자동 설치
# ------------------------------------------------------------
def ensure_package(module_name: str, package_name: str) -> None:
    """module_name을 import할 수 없으면 package_name을 pip로 설치합니다."""
    if importlib.util.find_spec(module_name) is not None:
        return

    print(f"[SETUP] {package_name} 설치 중...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        package_name,
    ])


# TensorFlow는 쓰지 않습니다.
# Pillow: 이미지 열기/저장/그림판 처리
# NumPy: 이미지 특징 계산과 거리 계산
ensure_package("PIL", "pillow")
ensure_package("numpy", "numpy")

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps, ImageTk
except Exception as exc:
    messagebox.showerror(
        "라이브러리 오류",
        f"필수 라이브러리를 불러오지 못했습니다.\n\n{exc}"
    )
    raise


# ------------------------------------------------------------
# 1. 경로와 기본 설정
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "pokemon_feature_model.pkl"
SUMMARY_PATH = MODEL_DIR / "training_summary.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# 학습 속도를 조절하고 싶으면 숫자를 낮추세요.
# None이면 모든 이미지를 사용합니다.
MAX_IMAGES_PER_CLASS: Optional[int] = None

FEATURE_IMAGE_SIZE = 64
THUMB_SIZE = 16
HIST_BINS = 8


@dataclass
class Prediction:
    label: str
    confidence: float


# ------------------------------------------------------------
# 2. 데이터셋 찾기 / 다운로드
# ------------------------------------------------------------
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images_under(path: Path, limit: Optional[int] = None) -> int:
    if not path.exists() or not path.is_dir():
        return 0

    count = 0
    for file in path.rglob("*"):
        if is_image_file(file):
            count += 1
            if limit is not None and count >= limit:
                return count
    return count


def find_dataset_root() -> Optional[Path]:
    """
    이미지 분류용 데이터셋 root를 찾습니다.

    필요한 구조:
    data/PokemonData/
    ├─ Abra/
    │  ├─ img1.jpg
    │  └─ img2.jpg
    ├─ Pikachu/
    │  ├─ img1.jpg
    │  └─ img2.jpg
    └─ ...

    즉, 포켓몬 이름별 폴더가 있어야 합니다.
    data 폴더 안에 이미지 파일만 덩그러니 있으면 학습할 수 없습니다.
    """
    candidate_starts = [
        DATA_DIR / "PokemonData",
        DATA_DIR / "archive" / "PokemonData",
        DATA_DIR / "archive",
        DATA_DIR,
        PROJECT_DIR / "PokemonData",
        PROJECT_DIR / "archive" / "PokemonData",
        PROJECT_DIR / "archive",
    ]

    candidates: list[tuple[int, int, Path]] = []

    for start in candidate_starts:
        if not start.exists() or not start.is_dir():
            continue

        possible_roots = [start]
        possible_roots.extend([p for p in start.rglob("*") if p.is_dir()])

        for root in possible_roots:
            class_dirs = [
                child for child in root.iterdir()
                if child.is_dir() and count_images_under(child, limit=1) > 0
            ]

            if len(class_dirs) < 2:
                continue

            image_count = sum(count_images_under(class_dir) for class_dir in class_dirs)
            candidates.append((len(class_dirs), image_count, root))

    if not candidates:
        return None

    # 클래스 수와 이미지 수가 가장 많은 폴더를 데이터셋 root로 선택합니다.
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def download_dataset_if_needed(progress: Callable[[str, int], None]) -> Optional[Path]:
    root = find_dataset_root()
    if root is not None:
        progress(f"기존 데이터셋 발견: {root}", 5)
        return root

    progress("PokemonData 폴더가 없어 KaggleHub 다운로드를 시도합니다.", 5)

    try:
        ensure_package("kagglehub", "kagglehub")
        import kagglehub

        DATA_DIR.mkdir(exist_ok=True)
        progress("KaggleHub 다운로드 중... 처음에는 오래 걸릴 수 있습니다.", 10)

        try:
            kagglehub.dataset_download(
                "lantian773030/pokemonclassification",
                output_dir=str(DATA_DIR / "archive"),
            )
        except TypeError:
            # 구버전 kagglehub 대응
            kagglehub.dataset_download("lantian773030/pokemonclassification")

        root = find_dataset_root()
        if root is not None:
            progress(f"데이터셋 다운로드 완료: {root}", 15)
            return root

        progress("다운로드는 되었지만 PokemonData 구조를 찾지 못했습니다.", 15)
        return None

    except Exception as exc:
        progress(f"KaggleHub 다운로드 실패: {exc}", 15)
        return None


# ------------------------------------------------------------
# 3. 이미지 특징 추출
# ------------------------------------------------------------
def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def prepare_image(image: Image.Image, size: int = FEATURE_IMAGE_SIZE) -> Image.Image:
    """
    이미지를 정사각형으로 맞춥니다.
    비율은 유지하고 남는 부분은 흰색으로 채웁니다.
    """
    image = image.convert("RGB")
    return ImageOps.pad(
        image,
        (size, size),
        color=(255, 255, 255),
        method=Image.Resampling.LANCZOS,
    )


def extract_feature(image: Image.Image) -> np.ndarray:
    """
    TensorFlow 없이 사용할 간단한 특징 벡터를 만듭니다.

    포함하는 정보:
    1. RGB 색상 히스토그램
    2. 작은 흑백 썸네일
    3. 간단한 edge/윤곽 정보
    """
    image = prepare_image(image, FEATURE_IMAGE_SIZE)
    rgb = np.asarray(image, dtype=np.float32) / 255.0

    # 1) 색상 히스토그램: R/G/B 각각 8개 bin
    hist_parts = []
    for ch in range(3):
        hist, _ = np.histogram(
            rgb[:, :, ch],
            bins=HIST_BINS,
            range=(0.0, 1.0),
            density=False,
        )
        hist = hist.astype(np.float32)
        hist = hist / max(hist.sum(), 1.0)
        hist_parts.append(hist)

    color_hist = np.concatenate(hist_parts)

    # 2) 형태 정보: 16x16 흑백 이미지
    gray = image.convert("L").resize(
        (THUMB_SIZE, THUMB_SIZE),
        Image.Resampling.LANCZOS,
    )
    gray_arr = np.asarray(gray, dtype=np.float32) / 255.0
    gray_flat = gray_arr.flatten()

    # 3) 간단한 윤곽 정보: 인접 픽셀 차이
    dx = np.abs(np.diff(gray_arr, axis=1))
    dy = np.abs(np.diff(gray_arr, axis=0))
    edge_stats = np.array([
        dx.mean(),
        dx.std(),
        dx.max(),
        dy.mean(),
        dy.std(),
        dy.max(),
        gray_arr.mean(),
        gray_arr.std(),
    ], dtype=np.float32)

    feature = np.concatenate([color_hist, gray_flat, edge_stats]).astype(np.float32)

    # 벡터 길이를 1에 가깝게 정규화
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm

    return feature


# ------------------------------------------------------------
# 4. 모델 만들기 / 예측
# ------------------------------------------------------------
def collect_image_paths(dataset_root: Path) -> tuple[list[str], list[tuple[Path, int]]]:
    class_dirs = [
        child for child in dataset_root.iterdir()
        if child.is_dir() and count_images_under(child, limit=1) > 0
    ]

    class_dirs.sort(key=lambda p: p.name.lower())

    class_names = [p.name for p in class_dirs]
    items: list[tuple[Path, int]] = []

    for label_idx, class_dir in enumerate(class_dirs):
        image_paths = [
            p for p in class_dir.rglob("*")
            if is_image_file(p)
        ]

        image_paths.sort()

        if MAX_IMAGES_PER_CLASS is not None:
            image_paths = image_paths[:MAX_IMAGES_PER_CLASS]

        for path in image_paths:
            items.append((path, label_idx))

    return class_names, items


def train_feature_model(progress: Callable[[str, int], None]) -> None:
    """
    data/PokemonData 이미지를 읽어서 특징 벡터 모델을 만듭니다.
    이 과정은 딥러닝 학습은 아니고, 이미지 유사도 검색용 모델 생성입니다.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    dataset_root = download_dataset_if_needed(progress)

    if dataset_root is None:
        raise RuntimeError(
            "데이터셋을 찾지 못했습니다.\n\n"
            "data/PokemonData/포켓몬이름폴더/이미지.jpg 구조인지 확인해주세요.\n"
            "data 폴더 안에 이미지 파일만 넣으면 학습할 수 없습니다."
        )

    class_names, items = collect_image_paths(dataset_root)

    if len(class_names) < 2:
        raise RuntimeError("포켓몬 클래스 폴더가 2개 이상 필요합니다.")

    if len(items) == 0:
        raise RuntimeError("학습할 이미지 파일을 찾지 못했습니다.")

    progress(f"클래스 {len(class_names)}개, 이미지 {len(items)}개 발견", 20)

    features = []
    labels = []

    total = len(items)
    skipped = 0

    for idx, (path, label_idx) in enumerate(items, start=1):
        image = safe_open_image(path)
        if image is None:
            skipped += 1
            continue

        features.append(extract_feature(image))
        labels.append(label_idx)

        if idx % 10 == 0 or idx == total:
            percent = 20 + int(idx / total * 70)
            progress(
                f"모델 생성 중... {idx}/{total} 이미지 처리",
                min(percent, 90),
            )

    if not features:
        raise RuntimeError("정상적으로 읽을 수 있는 이미지가 없습니다.")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)

    model_data = {
        "type": "feature_knn",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "features": X,
        "labels": y,
        "feature_image_size": FEATURE_IMAGE_SIZE,
        "thumb_size": THUMB_SIZE,
        "hist_bins": HIST_BINS,
        "skipped": skipped,
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model_data, f)

    SUMMARY_PATH.write_text(
        f"""Pokemon Pure Python Feature Model
=================================

Created at: {model_data['created_at']}
Dataset root: {dataset_root}
Class count: {len(class_names)}
Image count used: {len(features)}
Skipped images: {skipped}

Model file:
{MODEL_PATH}

주의:
이 모델은 TensorFlow 전이학습 모델이 아니라
색상/형태 특징 기반 KNN classifier입니다.
""",
        encoding="utf-8",
    )

    progress(f"모델 저장 완료: {MODEL_PATH}", 100)


def load_feature_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "모델 파일이 없습니다.\n먼저 [모델 만들기] 버튼을 눌러주세요."
        )

    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def predict_with_feature_model(image: Image.Image, top_k: int = 5) -> list[Prediction]:
    model_data = load_feature_model()

    class_names: list[str] = model_data["class_names"]
    X: np.ndarray = model_data["features"]
    y: np.ndarray = model_data["labels"]

    q = extract_feature(image)

    # 전체 이미지와의 거리 계산
    distances = np.linalg.norm(X - q[None, :], axis=1)

    # 가장 가까운 이미지 일부만 보고 class별 점수 집계
    nearest_count = min(180, len(distances))
    nearest_indices = np.argsort(distances)[:nearest_count]

    class_scores: dict[int, float] = {}

    for rank, idx in enumerate(nearest_indices):
        label_idx = int(y[idx])
        dist = float(distances[idx])

        # 거리가 가까울수록 높은 점수
        score = 1.0 / (dist + 1e-6)

        # 순위가 높을수록 약간 가중치
        score *= 1.0 / ((rank + 1) ** 0.15)

        class_scores[label_idx] = class_scores.get(label_idx, 0.0) + score

    if not class_scores:
        return []

    sorted_items = sorted(
        class_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    raw_scores = np.array([score for _, score in sorted_items], dtype=np.float32)

    # 보기 좋은 confidence로 변환
    if raw_scores.sum() > 0:
        probs = raw_scores / raw_scores.sum()
    else:
        probs = np.ones(len(raw_scores), dtype=np.float32) / len(raw_scores)

    predictions = [
        Prediction(
            label=class_names[label_idx],
            confidence=float(probs[i]),
        )
        for i, (label_idx, _) in enumerate(sorted_items)
    ]

    return predictions


# ------------------------------------------------------------
# 5. Tkinter GUI
# ------------------------------------------------------------
class PokemonAnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("포켓몬 분석기 - Pure Python GUI")
        self.root.geometry("1180x720")
        self.root.minsize(1040, 650)

        # 처음 실행 시에는 아무 입력 방식도 선택되지 않은 상태입니다.
        # 그래서 "이미지 삽입" / "그림판 그리기" 버튼 모두 회색으로 보입니다.
        self.mode: Optional[str] = None

        self.selected_image: Optional[Image.Image] = None
        self.preview_tk: Optional[ImageTk.PhotoImage] = None
        self.draw_tk: Optional[ImageTk.PhotoImage] = None

        self.draw_image = Image.new("RGB", (460, 430), "white")
        self.draw = ImageDraw.Draw(self.draw_image)
        self.is_drawing = False
        self.has_drawing = False
        self.last_x = 0
        self.last_y = 0

        self.current_predictions: Optional[list[Prediction]] = None

        self.queue: queue.Queue = queue.Queue()
        self.loading = False
        self.progress = 0
        self.spinner_angle = 0

        self._build_ui()
        self._check_initial_model()
        self._process_queue()
        self._animate_loader()

    # ---------------- UI 구성 ----------------
    def _build_ui(self) -> None:
        self.root.configure(bg="#f4f7fb")

        title = tk.Label(
            self.root,
            text="포켓몬 분석기",
            font=("맑은 고딕", 28, "bold"),
            bg="#f4f7fb",
            fg="#1d1f27",
        )
        title.pack(pady=(18, 4))

        subtitle = tk.Label(
            self.root,
            text="이미지를 삽입하거나 직접 그린 뒤 분석 버튼을 누르세요.",
            font=("맑은 고딕", 11),
            bg="#f4f7fb",
            fg="#6b7280",
        )
        subtitle.pack(pady=(0, 12))

        main = tk.Frame(self.root, bg="#f4f7fb")
        main.pack(fill="both", expand=True, padx=22, pady=12)

        left = tk.Frame(main, bg="white", bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        right = tk.Frame(main, bg="#1d1f27", bd=1, relief="solid")
        right.pack(side="right", fill="both", expand=True, padx=(12, 0))

        # 왼쪽 패널
        tk.Label(
            left,
            text="1. 입력",
            font=("맑은 고딕", 17, "bold"),
            bg="white",
            fg="#111827",
        ).pack(anchor="w", padx=18, pady=(16, 10))

        mode_frame = tk.Frame(left, bg="white")
        mode_frame.pack(fill="x", padx=18, pady=(0, 10))

        self.upload_btn = tk.Button(
            mode_frame,
            text="이미지 삽입",
            command=self.select_image,
            font=("맑은 고딕", 11, "bold"),
            bg="#e5e7eb",
            fg="#111827",
            relief="flat",
            height=2,
        )
        self.upload_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.draw_btn = tk.Button(
            mode_frame,
            text="그림판 그리기",
            command=self.switch_to_draw_mode,
            font=("맑은 고딕", 11, "bold"),
            bg="#e5e7eb",
            fg="#111827",
            relief="flat",
            height=2,
        )
        self.draw_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))

        self.input_canvas = tk.Canvas(
            left,
            width=460,
            height=430,
            bg="white",
            highlightthickness=2,
            highlightbackground="#cbd5e1",
        )
        self.input_canvas.pack(padx=18, pady=10, fill="both", expand=True)

        self.input_canvas.bind("<ButtonPress-1>", self._start_draw)
        self.input_canvas.bind("<B1-Motion>", self._draw_motion)
        self.input_canvas.bind("<ButtonRelease-1>", self._stop_draw)
        self.input_canvas.bind("<Configure>", self._on_input_canvas_resize)

        bottom_frame = tk.Frame(left, bg="white")
        bottom_frame.pack(fill="x", padx=18, pady=(8, 18))

        self.clear_btn = tk.Button(
            bottom_frame,
            text="지우기",
            command=self.clear_input,
            font=("맑은 고딕", 10, "bold"),
            bg="#e5e7eb",
            fg="#111827",
            relief="flat",
            width=10,
            height=2,
        )
        self.clear_btn.pack(side="left", padx=(0, 8))

        self.train_btn = tk.Button(
            bottom_frame,
            text="모델 만들기",
            command=self.start_training,
            font=("맑은 고딕", 10, "bold"),
            bg="#2563eb",
            fg="white",
            relief="flat",
            width=13,
            height=2,
        )
        self.train_btn.pack(side="left", padx=8)

        self.analyze_btn = tk.Button(
            bottom_frame,
            text="분석",
            command=self.start_prediction,
            font=("맑은 고딕", 12, "bold"),
            bg="#e63946",
            fg="white",
            relief="flat",
            width=12,
            height=2,
        )
        self.analyze_btn.pack(side="right")

        # 오른쪽 패널
        tk.Label(
            right,
            text="2. 분석 결과",
            font=("맑은 고딕", 17, "bold"),
            bg="#1d1f27",
            fg="white",
        ).pack(anchor="w", padx=18, pady=(16, 10))

        self.result_canvas = tk.Canvas(
            right,
            width=460,
            height=500,
            bg="#1d1f27",
            highlightthickness=0,
        )
        self.result_canvas.pack(padx=18, pady=10, fill="both", expand=True)
        self.result_canvas.bind("<Configure>", self._on_result_canvas_resize)

        self.status_label = tk.Label(
            right,
            text="준비 중...",
            font=("맑은 고딕", 10),
            bg="#1d1f27",
            fg="#93c5fd",
            wraplength=460,
            justify="center",
        )
        self.status_label.pack(padx=18, pady=(0, 18), fill="x")

        self._draw_empty_result()
        self._draw_upload_empty()


    def _canvas_size(self, canvas: tk.Canvas, default_w: int = 460, default_h: int = 430) -> tuple[int, int]:
        """현재 캔버스 크기를 안전하게 구합니다."""
        w = max(canvas.winfo_width(), default_w)
        h = max(canvas.winfo_height(), default_h)
        return w, h

    def _on_input_canvas_resize(self, event=None) -> None:
        """창 크기가 바뀌어도 입력 화면이 가운데에 유지되도록 다시 그립니다."""
        if self.mode == "upload" and self.selected_image is not None:
            self._show_image_on_input_canvas(self.selected_image)
        elif self.mode == "draw":
            self._ensure_draw_image_size()
            self._redraw_drawing_canvas()
        elif self.mode is None:
            self._draw_upload_empty()

    def _on_result_canvas_resize(self, event=None) -> None:
        """전체화면/창 크기 변경 시 오른쪽 결과 화면도 가운데 기준으로 다시 그립니다."""
        if self.loading:
            self._draw_loader()
        elif self.current_predictions is not None:
            self._show_predictions(self.current_predictions)
        else:
            self._draw_empty_result()

    def _ensure_draw_image_size(self) -> None:
        """
        그림판 내부 이미지 크기를 실제 캔버스 크기와 맞춥니다.
        이렇게 해야 전체화면에서 그려도 이미지가 잘리지 않고,
        분석할 때도 사용자가 그린 내용이 그대로 들어갑니다.
        """
        w, h = self._canvas_size(self.input_canvas)

        if self.draw_image.size == (w, h):
            return

        old_image = self.draw_image
        new_image = Image.new("RGB", (w, h), "white")

        # 기존 그림은 새 캔버스 가운데에 붙여서 최대한 유지합니다.
        paste_x = max((w - old_image.width) // 2, 0)
        paste_y = max((h - old_image.height) // 2, 0)
        new_image.paste(old_image, (paste_x, paste_y))

        self.draw_image = new_image
        self.draw = ImageDraw.Draw(self.draw_image)

    def _redraw_drawing_canvas(self) -> None:
        """PIL에 저장된 그림을 Tkinter 캔버스에 다시 표시합니다."""
        self.input_canvas.delete("all")
        self.draw_tk = ImageTk.PhotoImage(self.draw_image)
        self.input_canvas.create_image(0, 0, image=self.draw_tk, anchor="nw")

        if not self.has_drawing:
            w, h = self._canvas_size(self.input_canvas)
            self.input_canvas.create_text(
                w // 2,
                h // 2,
                text="여기에 검은색으로 그리세요",
                font=("맑은 고딕", 15, "bold"),
                fill="#cbd5e1",
                tags=("placeholder",),
            )

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.upload_btn.configure(state=state)
        self.draw_btn.configure(state=state)
        self.clear_btn.configure(state=state)
        self.train_btn.configure(state=state)
        self.analyze_btn.configure(state=state)

    # ---------------- 초기 상태 ----------------
    def _check_initial_model(self) -> None:
        dataset_root = find_dataset_root()

        if MODEL_PATH.exists():
            self.status_label.configure(
                text=f"모델 있음: {MODEL_PATH.name}\n바로 분석할 수 있습니다."
            )
        elif dataset_root is not None:
            self.status_label.configure(
                text=f"데이터셋 발견: {dataset_root}\n먼저 [모델 만들기]를 누르세요."
            )
        else:
            self.status_label.configure(
                text=(
                    "모델이 없습니다.\n"
                    "data/PokemonData/포켓몬이름폴더/이미지.jpg 구조로 데이터를 넣거나\n"
                    "[모델 만들기]를 눌러 자동 다운로드를 시도하세요."
                )
            )

    # ---------------- 왼쪽 입력 ----------------
    def _draw_upload_empty(self) -> None:
        self.input_canvas.delete("all")
        w, h = self._canvas_size(self.input_canvas)

        self.input_canvas.create_text(
            w // 2,
            h // 2 - 35,
            text="이미지를 삽입하거나\n그림판 모드로 직접 그리세요.",
            font=("맑은 고딕", 17, "bold"),
            fill="#6b7280",
            justify="center",
            tags=("placeholder",),
        )
        self.input_canvas.create_text(
            w // 2,
            h // 2 + 55,
            text="지원 형식: JPG, PNG, WEBP",
            font=("맑은 고딕", 10),
            fill="#94a3b8",
            tags=("placeholder",),
        )

    def select_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="포켓몬 이미지 선택",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("이미지 오류", f"이미지를 열 수 없습니다.\n\n{exc}")
            return

        self.mode = "upload"
        self.has_drawing = False
        self.selected_image = image
        self._update_mode_buttons()
        self._show_image_on_input_canvas(image)

    def _show_image_on_input_canvas(self, image: Image.Image) -> None:
        canvas_w, canvas_h = self._canvas_size(self.input_canvas)

        preview = ImageOps.contain(
            image,
            (max(canvas_w - 20, 1), max(canvas_h - 20, 1)),
            Image.Resampling.LANCZOS,
        )

        self.preview_tk = ImageTk.PhotoImage(preview)
        self.input_canvas.delete("all")
        self.input_canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            image=self.preview_tk,
            anchor="center",
        )

    def switch_to_draw_mode(self) -> None:
        was_draw_mode = self.mode == "draw"

        self.mode = "draw"
        self.selected_image = None
        self._update_mode_buttons()

        # 이미 그림판 모드였다면 기존 그림을 지우지 않습니다.
        # 사용자가 명시적으로 [지우기]를 눌렀을 때만 그림이 사라집니다.
        if was_draw_mode:
            self._redraw_drawing_canvas()
        else:
            self.clear_drawing()

    def _update_mode_buttons(self) -> None:
        inactive_bg = "#e5e7eb"
        inactive_fg = "#111827"
        active_bg = "#e63946"
        active_fg = "white"

        if self.mode == "upload":
            self.upload_btn.configure(bg=active_bg, fg=active_fg)
            self.draw_btn.configure(bg=inactive_bg, fg=inactive_fg)
        elif self.mode == "draw":
            self.upload_btn.configure(bg=inactive_bg, fg=inactive_fg)
            self.draw_btn.configure(bg=active_bg, fg=active_fg)
        else:
            # 처음 실행 상태: 둘 다 회색
            self.upload_btn.configure(bg=inactive_bg, fg=inactive_fg)
            self.draw_btn.configure(bg=inactive_bg, fg=inactive_fg)

    def clear_input(self) -> None:
        if self.mode == "draw":
            self.clear_drawing()
        elif self.mode == "upload":
            self.selected_image = None
            self.mode = None
            self._update_mode_buttons()
            self._draw_upload_empty()
        else:
            self.selected_image = None
            self.has_drawing = False
            self._update_mode_buttons()
            self._draw_upload_empty()

        self._draw_empty_result()

    def clear_drawing(self) -> None:
        self._ensure_draw_image_size()
        w, h = self.draw_image.size
        self.draw_image = Image.new("RGB", (w, h), "white")
        self.draw = ImageDraw.Draw(self.draw_image)
        self.has_drawing = False
        self.input_canvas.configure(bg="white")
        self._redraw_drawing_canvas()

    def _start_draw(self, event) -> None:
        if self.mode != "draw":
            return

        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y

        # 안내 문구만 제거합니다.
        # 기존에 그린 선은 지우지 않습니다.
        self.input_canvas.delete("placeholder")
        self._ensure_draw_image_size()

    def _draw_motion(self, event) -> None:
        if self.mode != "draw" or not self.is_drawing:
            return

        self._ensure_draw_image_size()
        x, y = event.x, event.y
        self.has_drawing = True
        self.input_canvas.create_line(
            self.last_x,
            self.last_y,
            x,
            y,
            fill="black",
            width=8,
            capstyle="round",
            smooth=True,
        )
        self.draw.line(
            [(self.last_x, self.last_y), (x, y)],
            fill="black",
            width=8,
        )

        self.last_x = x
        self.last_y = y

    def _stop_draw(self, event) -> None:
        self.is_drawing = False

    def _get_current_input_image(self) -> Image.Image:
        if self.mode == "upload":
            if self.selected_image is None:
                raise RuntimeError("먼저 포켓몬 이미지를 선택해주세요.")
            return self.selected_image

        if self.mode == "draw":
            return self.draw_image

        raise RuntimeError("먼저 이미지 삽입 또는 그림판 그리기를 선택해주세요.")

    # ---------------- 오른쪽 결과 ----------------
    def _draw_empty_result(self) -> None:
        self.loading = False
        self.current_predictions = None
        self.result_canvas.delete("all")
        w, h = self._canvas_size(self.result_canvas, 460, 500)

        self.result_canvas.create_text(
            w // 2,
            h // 2 - 25,
            text="왼쪽에서 이미지를 넣거나 그림을 그린 뒤",
            fill="#cbd5e1",
            font=("맑은 고딕", 15),
            justify="center",
        )
        self.result_canvas.create_text(
            w // 2,
            h // 2 + 15,
            text="[분석] 버튼을 눌러주세요.",
            fill="white",
            font=("맑은 고딕", 18, "bold"),
            justify="center",
        )

    def _draw_loader(self) -> None:
        self.result_canvas.delete("all")
        w, h = self._canvas_size(self.result_canvas, 460, 500)

        cx, cy = w // 2, h // 2 - 70
        r = 56

        # 몬스터볼 비슷한 로딩 그림
        self.result_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="white", outline="#111827", width=5)
        self.result_canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=180, fill="#e63946", outline="#111827", width=5)
        self.result_canvas.create_rectangle(cx - r, cy - 6, cx + r, cy + 6, fill="#111827", outline="#111827")
        self.result_canvas.create_oval(cx - 18, cy - 18, cx + 18, cy + 18, fill="white", outline="#111827", width=5)

        # 회전 느낌을 위한 작은 점
        angle = math.radians(self.spinner_angle)
        dot_x = cx + math.cos(angle) * 72
        dot_y = cy + math.sin(angle) * 72
        self.result_canvas.create_oval(dot_x - 7, dot_y - 7, dot_x + 7, dot_y + 7, fill="#93c5fd", outline="")

        self.result_canvas.create_text(
            w // 2,
            cy + 110,
            text=f"{self.progress}%",
            fill="white",
            font=("맑은 고딕", 24, "bold"),
        )

        # progress bar
        bar_w = 310
        bar_h = 16
        x0 = w // 2 - bar_w // 2
        y0 = cy + 145
        self.result_canvas.create_rectangle(x0, y0, x0 + bar_w, y0 + bar_h, fill="#374151", outline="")
        self.result_canvas.create_rectangle(x0, y0, x0 + int(bar_w * self.progress / 100), y0 + bar_h, fill="#60a5fa", outline="")

    def _show_predictions(self, predictions: list[Prediction]) -> None:
        self.loading = False
        self.current_predictions = predictions
        self.result_canvas.delete("all")

        w, _ = self._canvas_size(self.result_canvas, 460, 500)

        if not predictions:
            self.result_canvas.create_text(
                w // 2,
                230,
                text="예측 결과가 없습니다.",
                fill="#fecaca",
                font=("맑은 고딕", 18, "bold"),
            )
            return

        best = predictions[0]

        self.result_canvas.create_text(
            w // 2,
            55,
            text="분석 완료",
            fill="#bfdbfe",
            font=("맑은 고딕", 13, "bold"),
        )

        self.result_canvas.create_text(
            w // 2,
            105,
            text=best.label,
            fill="white",
            font=("맑은 고딕", 34, "bold"),
        )

        self.result_canvas.create_text(
            w // 2,
            148,
            text=f"신뢰도 {best.confidence * 100:.2f}%",
            fill="#d1d5db",
            font=("맑은 고딕", 12),
        )

        y = 205
        for i, pred in enumerate(predictions, start=1):
            x0 = 55
            y0 = y
            x1 = w - 55
            y1 = y + 48

            self.result_canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill="#2a2f41",
                outline="#4b5563",
            )

            self.result_canvas.create_text(
                x0 + 16,
                y0 + 17,
                text=f"{i}. {pred.label}",
                anchor="w",
                fill="white",
                font=("맑은 고딕", 11, "bold"),
            )

            self.result_canvas.create_text(
                x1 - 16,
                y0 + 17,
                text=f"{pred.confidence * 100:.2f}%",
                anchor="e",
                fill="#bfdbfe",
                font=("맑은 고딕", 11, "bold"),
            )

            bar_x0 = x0 + 16
            bar_y0 = y0 + 33
            bar_x1 = x1 - 16
            bar_y1 = y0 + 39

            self.result_canvas.create_rectangle(
                bar_x0,
                bar_y0,
                bar_x1,
                bar_y1,
                fill="#374151",
                outline="",
            )
            self.result_canvas.create_rectangle(
                bar_x0,
                bar_y0,
                bar_x0 + int((bar_x1 - bar_x0) * pred.confidence),
                bar_y1,
                fill="#93c5fd",
                outline="",
            )

            y += 60

    # ---------------- 비동기 작업 ----------------
    def start_training(self) -> None:
        if self.loading:
            return

        if MODEL_PATH.exists():
            answer = messagebox.askyesno(
                "모델 다시 만들기",
                "이미 모델 파일이 있습니다.\n다시 만들면 시간이 걸릴 수 있습니다.\n\n다시 만들까요?"
            )
            if not answer:
                return

        self.loading = True
        self.progress = 0
        self._set_buttons_enabled(False)
        self.status_label.configure(text="모델을 만드는 중입니다. 창을 닫지 마세요.")

        def worker():
            try:
                def progress(message: str, percent: int):
                    self.queue.put(("progress", message, percent))

                train_feature_model(progress)
                self.queue.put(("train_done",))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def start_prediction(self) -> None:
        if self.loading:
            return

        try:
            input_image = self._get_current_input_image()
        except Exception as exc:
            messagebox.showwarning("입력 필요", str(exc))
            return

        if not MODEL_PATH.exists():
            answer = messagebox.askyesno(
                "모델 없음",
                "모델 파일이 없습니다.\n먼저 모델을 만들어야 실제 분석이 가능합니다.\n\n지금 모델을 만들까요?"
            )
            if answer:
                self.start_training()
            return

        self.loading = True
        self.progress = 0
        self._set_buttons_enabled(False)
        self.status_label.configure(text="분석 중입니다.")

        def worker():
            try:
                self.queue.put(("progress", "이미지 특징 추출 중...", 25))
                time.sleep(0.2)

                predictions = predict_with_feature_model(input_image, top_k=5)

                self.queue.put(("progress", "Top-5 결과 계산 중...", 85))
                time.sleep(0.25)

                self.queue.put(("predict_done", predictions))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _process_queue(self) -> None:
        try:
            while True:
                event = self.queue.get_nowait()

                if event[0] == "progress":
                    _, message, percent = event
                    self.progress = max(0, min(100, int(percent)))
                    self.status_label.configure(text=message)

                elif event[0] == "train_done":
                    self.progress = 100
                    self.loading = False
                    self._set_buttons_enabled(True)
                    self.status_label.configure(
                        text=f"모델 생성 완료!\n이제 이미지를 넣고 [분석]을 누르세요."
                    )
                    messagebox.showinfo(
                        "완료",
                        f"모델 생성이 완료되었습니다.\n\n{MODEL_PATH}"
                    )
                    self._draw_empty_result()

                elif event[0] == "predict_done":
                    _, predictions = event
                    self.progress = 100
                    self.loading = False
                    self._set_buttons_enabled(True)
                    self.status_label.configure(
                        text="분석 완료. 이 모델은 색상/형태 특징 기반 classifier입니다."
                    )
                    self._show_predictions(predictions)

                elif event[0] == "error":
                    _, message = event
                    self.loading = False
                    self._set_buttons_enabled(True)
                    self.status_label.configure(text="오류 발생")
                    messagebox.showerror("오류", message)
                    self._draw_empty_result()

        except queue.Empty:
            pass

        self.root.after(100, self._process_queue)

    def _animate_loader(self) -> None:
        if self.loading:
            self.spinner_angle = (self.spinner_angle + 18) % 360
            self._draw_loader()

        self.root.after(80, self._animate_loader)


def main() -> None:
    root = tk.Tk()
    app = PokemonAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
