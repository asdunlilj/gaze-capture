#!/usr/bin/env python3
"""M0 环境体检：判断这台机器能否支撑项目的精度承诺。

只读脚本 —— 不录制、不保存任何画面，只输出一份 JSON 报告与环境结论。

用法：
    python tools/health_check.py                 # 自动选第一个可用摄像头
    python tools/health_check.py --camera 1      # 指定设备序号
    python tools/health_check.py --seconds 6     # 延长测帧时长
    python tools/health_check.py --no-iris       # 跳过虹膜测量（未装 mediapipe 时）

判定依据（来自交付方案 M0 Gate，三条按"最可能推翻结论"排序）：
    ① 720p 下虹膜实际像素直径 >= 22px     —— 不达则整张精度表下移一档
    ② 端到端延迟看 p95                    —— 平滑窗口是延迟主成本
    ③ 有效帧率 >= 20Hz 通过 / 12–20Hz 有条件通过 / <12Hz 不通过
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量：方案里的阈值，改这里等于改验收口径，务必同步文档
# ---------------------------------------------------------------------------
IRIS_MIN_PX = 22.0          # 720p 下虹膜直径下限
FPS_PASS = 20.0             # 有效帧率通过线
FPS_FLOOR = 12.0            # 有效帧率不可用线（低于此扫视类指标全废）
LATENCY_P95_PASS_MS = 150.0  # 端到端延迟 p95 通过线
LATENCY_P95_LIMIT_MS = 200.0  # 上限（叠加层旁路可放宽到 300）

IRIS_REAL_DIAMETER_MM = 11.7  # 人类虹膜平均直径，用于反推眼-镜头距离
ASSUMED_HFOV_DEG = 65.0       # 未知视场角时的保守假设

MODEL_HINT = (
    "需要 MediaPipe 模型文件：models/face_landmarker.task\n"
    "下载（约 4MB）：\n"
    "  mkdir models\n"
    "  curl -L -o models/face_landmarker.task "
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _pct(values: list[float], q: float) -> float:
    """简单分位数（nearest-rank），避免引入 numpy 依赖做这点小事。"""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _fmt(value: float, digits: int = 1, unit: str = "") -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{digits}f}{unit}"


# ---------------------------------------------------------------------------
# 设备枚举与能力探测
# ---------------------------------------------------------------------------
def _backend() -> int:
    """Windows 上走 DirectShow：取流延迟更低，且能读到更真实的设备能力。"""
    import cv2

    return cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY


def enumerate_cameras(max_index: int = 6) -> list[int]:
    import cv2

    found: list[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, _backend())
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(index)
        finally:
            cap.release()
    return found


def _read_back(cap, prop: int) -> float:
    """读回属性值；不支持时 OpenCV 通常返回 -1.0。"""
    try:
        return float(cap.get(prop))
    except Exception:
        return -1.0


def probe_capabilities(cap) -> dict:
    import cv2

    caps: dict[str, object] = {}
    name = "unknown"
    try:
        raw = cap.getBackendName()
        name = str(raw) if raw else "unknown"
    except Exception:
        pass
    caps["backend"] = name

    for label, prop in (
        ("width", cv2.CAP_PROP_FRAME_WIDTH),
        ("height", cv2.CAP_PROP_FRAME_HEIGHT),
        ("fps", cv2.CAP_PROP_FPS),
        ("fourcc", cv2.CAP_PROP_FOURCC),
    ):
        caps[label] = _read_back(cap, prop)

    # 曝光锁定能力：能读到非负值且能写回，才算"支持"
    auto_exp = _read_back(cap, cv2.CAP_PROP_AUTO_EXPOSURE)
    caps["auto_exposure_raw"] = auto_exp
    caps["exposure_lock_supported"] = auto_exp >= 0
    return caps


def try_lock_exposure(cap) -> dict:
    """尝试锁定曝光与自动白平衡，并读回验证。

    注意：不同驱动取值语义不同（DSHOW 常见 0.25=手动、0.75=自动），
    所以这里只判断"是否有变化"，不判断绝对语义。
    """
    import cv2

    result: dict[str, object] = {}
    before = _read_back(cap, cv2.CAP_PROP_EXPOSURE)
    set_ok = True
    try:
        # 0.25 在 DirectShow 后端表示手动曝光
        ok1 = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        ok2 = cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        set_ok = bool(ok1) and bool(ok2)
    except Exception:
        set_ok = False
    after = _read_back(cap, cv2.CAP_PROP_EXPOSURE)
    result["exposure_before"] = before
    result["exposure_after"] = after
    result["exposure_set_accepted"] = set_ok
    result["exposure_lock_effective"] = set_ok and before >= 0 and after == before

    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        result["auto_wb_lock_accepted"] = True
    except Exception:
        result["auto_wb_lock_accepted"] = False
    return result


def try_set_format(cap, width: int, height: int, fps: float) -> dict:
    """设置 MJPG + 目标分辨率/帧率，并读回实际生效值。

    MJPG 很重要：YUY2 在 720p 下常被 USB2 带宽卡到 5–10fps。
    """
    import cv2

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return {
        "req_width": width,
        "req_height": height,
        "req_fps": fps,
        "act_width": _read_back(cap, cv2.CAP_PROP_FRAME_WIDTH),
        "act_height": _read_back(cap, cv2.CAP_PROP_FRAME_HEIGHT),
        "act_fps": _read_back(cap, cv2.CAP_PROP_FPS),
    }


# ---------------------------------------------------------------------------
# 测帧：先测"只取流"的裸帧率，再测带推理的有效帧率
# ---------------------------------------------------------------------------
def measure_raw_fps(cap, seconds: float, warmup: int = 15) -> dict:
    for _ in range(warmup):
        cap.read()

    stamps: list[float] = []
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        stamps.append(time.perf_counter())
    elapsed = time.perf_counter() - start

    if len(stamps) < 2:
        return {"frames": len(stamps), "fps": 0.0, "interval_p95_ms": float("nan")}

    intervals = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
    return {
        "frames": len(stamps),
        "elapsed_s": round(elapsed, 2),
        "fps": round(len(stamps) / elapsed, 2),
        "interval_p50_ms": round(statistics.median(intervals), 1),
        "interval_p95_ms": round(_pct(intervals, 0.95), 1),
        "interval_max_ms": round(max(intervals), 1),
    }


def _iris_diameter_px(landmarks, width: int, height: int) -> tuple[float | None, float | None]:
    """从 478 点里取两组虹膜轮廓，返回 (A 组直径, B 组直径)，单位像素。

    索引 468–472 / 473–477 为两组虹膜（1 个中心 + 4 个轮廓点）。
    不依赖"哪组是左眼"，也不依赖中心点排在第几位 —— 直接取组内最大点距，
    这样即使不同版本索引顺序变化也不会算错。

    归一化坐标必须分别用 width / height 还原，两者不能混用：MediaPipe 的
    x 按宽度归一化、y 按高度归一化，用同一个尺度会把椭圆算成圆。
    """
    if landmarks is None or len(landmarks) < 478:
        return None, None

    groups = (range(468, 473), range(473, 478))
    diameters: list[float] = []
    for group in groups:
        pts = [(landmarks[i].x * width, landmarks[i].y * height) for i in group]
        max_dist = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                max_dist = max(max_dist, math.hypot(pts[i][0] - pts[j][0],
                                                    pts[i][1] - pts[j][1]))
        diameters.append(max_dist)
    return diameters[0], diameters[1]


def measure_with_inference(cap, frames_target: int = 90) -> dict:
    """带模型推理的帧率、延迟、虹膜像素直径。模型缺失时返回 skipped 说明。"""
    result: dict[str, object] = {"status": "skipped", "reason": ""}

    model_path = Path("models/face_landmarker.task")
    if not model_path.exists():
        result["reason"] = MODEL_HINT
        return result

    try:
        import cv2  # noqa: F401
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as exc:  # pragma: no cover
        result["reason"] = f"mediapipe 不可用：{exc}"
        return result

    try:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:  # pragma: no cover
        result["reason"] = f"模型加载失败：{exc}"
        return result

    latencies: list[float] = []
    iris_px: list[float] = []
    faces = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    start = time.perf_counter()
    frame_idx = 0

    while frame_idx < frames_target:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_idx += 1
        rgb = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - start) * 1000)

        t0 = time.perf_counter()
        try:
            res = landmarker.detect_for_video(mp_image, ts_ms)
        except Exception:
            continue
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if res.face_landmarks:
            faces += 1
            height = frame.shape[0]
            left, right = _iris_diameter_px(res.face_landmarks[0], width, height)
            for value in (left, right):
                if value is not None and value > 0:
                    iris_px.append(value)

    elapsed = time.perf_counter() - start
    landmarker.close()

    result["status"] = "ok"
    result["frames_processed"] = frame_idx
    result["faces_detected"] = faces
    result["effective_fps"] = round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0
    if latencies:
        result["latency_p50_ms"] = round(statistics.median(latencies), 1)
        result["latency_p95_ms"] = round(_pct(latencies, 0.95), 1)
        result["latency_max_ms"] = round(max(latencies), 1)
    if iris_px:
        median_px = statistics.median(iris_px)
        result["iris_diameter_px_median"] = round(median_px, 1)
        result["iris_samples"] = len(iris_px)
        # 反推眼-镜头距离：focal_px = W / (2·tan(HFOV/2))
        focal_px = width / (2 * math.tan(math.radians(ASSUMED_HFOV_DEG / 2)))
        result["estimated_distance_mm"] = round(
            focal_px * IRIS_REAL_DIAMETER_MM / median_px, 0)
        result["assumed_hfov_deg"] = ASSUMED_HFOV_DEG
    return result


# ---------------------------------------------------------------------------
# 判定与输出
# ---------------------------------------------------------------------------
def judge(iris: dict, fps_info: dict) -> dict:
    verdicts: dict[str, dict] = {}

    iris_px = iris.get("iris_diameter_px_median")
    if iris.get("status") != "ok":
        verdicts["iris"] = {"level": "unknown", "detail": "未测到（缺模型或未检出人脸）"}
    elif iris_px is None:
        verdicts["iris"] = {"level": "fail", "detail": "未检出人脸，无法测虹膜"}
    elif iris_px >= IRIS_MIN_PX:
        verdicts["iris"] = {"level": "pass",
                            "detail": f"{iris_px:.1f}px >= {IRIS_MIN_PX}px"}
    else:
        verdicts["iris"] = {"level": "fail",
                            "detail": f"{iris_px:.1f}px < {IRIS_MIN_PX}px，精度表须下移一档"}

    eff = fps_info.get("effective_fps") or fps_info.get("fps") or 0.0
    if eff >= FPS_PASS:
        verdicts["fps"] = {"level": "pass", "detail": f"{eff:.1f}Hz >= {FPS_PASS}Hz"}
    elif eff >= FPS_FLOOR:
        verdicts["fps"] = {"level": "conditional",
                           "detail": f"{eff:.1f}Hz 在 {FPS_FLOOR}–{FPS_PASS}Hz，须关闭扫视类指标"}
    else:
        verdicts["fps"] = {"level": "fail",
                           "detail": f"{eff:.1f}Hz < {FPS_FLOOR}Hz，扫视指标全废"}

    p95 = fps_info.get("latency_p95_ms")
    if p95 is None:
        verdicts["latency"] = {"level": "unknown", "detail": "需模型推理才有延迟数据"}
    elif p95 <= LATENCY_P95_PASS_MS:
        verdicts["latency"] = {"level": "pass", "detail": f"p95 {p95:.0f}ms"}
    elif p95 <= LATENCY_P95_LIMIT_MS:
        verdicts["latency"] = {"level": "conditional",
                               "detail": f"p95 {p95:.0f}ms 偏高，叠加层需降频"}
    else:
        verdicts["latency"] = {"level": "fail",
                               "detail": f"p95 {p95:.0f}ms 超上限，须降分辨率或上 GPU"}

    levels = [v["level"] for v in verdicts.values()]
    if "fail" in levels:
        overall = "FAIL"
    elif "unknown" in levels:
        overall = "CONDITIONAL"
    elif "conditional" in levels:
        overall = "CONDITIONAL"
    else:
        overall = "PASS"
    return {"overall": overall, "items": verdicts}


def main() -> int:
    parser = argparse.ArgumentParser(description="M0 环境体检")
    parser.add_argument("--camera", type=int, default=None, help="设备序号，默认自动选第一个")
    parser.add_argument("--seconds", type=float, default=5.0, help="裸帧率测量时长")
    parser.add_argument("--no-iris", action="store_true", help="跳过虹膜测量")
    parser.add_argument("--out", default=None, help="JSON 输出路径")
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[x] 缺少 opencv-python：pip install -r requirements.txt", file=sys.stderr)
        return 2

    report: dict[str, object] = {
        "tool": "gaze-capture health_check",
        "schema": 1,
        "generated_at": _now(),
        "host": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "cpu": platform.processor() or "unknown",
            "logical_cores": os.cpu_count(),
        },
    }

    print(f"== gaze-capture · M0 环境体检 ==  {report['generated_at']}")
    print(f"主机：{report['host']['os']} / {report['host']['logical_cores']} 逻辑核")

    cameras = [args.camera] if args.camera is not None else enumerate_cameras()
    if not cameras:
        print("[x] 未发现可用摄像头。检查：设备是否被其他程序占用 / 隐私设置是否允许桌面应用访问。")
        report["judgement"] = {"overall": "FAIL", "items": {"camera": {
            "level": "fail", "detail": "未发现可用摄像头"}}}
        _dump(report, args.out)
        return 1

    cam_index = cameras[0]
    print(f"选中设备：index {cam_index}（共发现 {cameras}）")

    cap = cv2.VideoCapture(cam_index, _backend())
    if not cap.isOpened():
        print("[x] 设备打开失败")
        report["judgement"] = {"overall": "FAIL", "items": {"camera": {
            "level": "fail", "detail": "open 失败"}}}
        _dump(report, args.out)
        return 1

    try:
        report["format"] = try_set_format(cap, 1280, 720, 30.0)
        report["capabilities"] = probe_capabilities(cap)
        report["exposure"] = try_lock_exposure(cap)
        report["raw_capture"] = measure_raw_fps(cap, args.seconds)

        if args.no_iris:
            iris_result: dict[str, object] = {"status": "skipped", "reason": "--no-iris"}
        else:
            iris_result = measure_with_inference(cap)
        report["inference"] = iris_result

        fps_info = {
            "fps": report["raw_capture"].get("fps"),
            "effective_fps": iris_result.get("effective_fps"),
            "latency_p95_ms": iris_result.get("latency_p95_ms"),
        }
        # 有效帧率优先；缺推理数据时退回裸帧率并注明
        if fps_info["effective_fps"] is None:
            fps_info["effective_fps"] = fps_info["fps"]
            report["fps_source"] = "raw_capture"
        else:
            report["fps_source"] = "inference"
        report["judgement"] = judge(iris_result, fps_info)
    finally:
        cap.release()

    _print_summary(report)
    _dump(report, args.out)
    return 0  # 体检本身不因判定结果返回非零：FAIL 是要读的结论，不是脚本错误


def _print_summary(report: dict) -> None:
    fmt = report.get("format", {})
    cap_info = report.get("capabilities", {})
    exp = report.get("exposure", {})
    raw = report.get("raw_capture", {})
    inf = report.get("inference", {})
    j = report.get("judgement", {})
    items = j.get("items", {})

    print("\n-- 采集 --")
    print(f"  实际分辨率：{fmt.get('act_width', 'n/a')}x{fmt.get('act_height', 'n/a')}"
          f"（请求 {fmt.get('req_width')}x{fmt.get('req_height')}）")
    print(f"  名义帧率：{_fmt(cap_info.get('fps', float('nan')))}   后端：{cap_info.get('backend')}")
    print(f"  裸取流：{_fmt(raw.get('fps', float('nan')))} fps"
          f"，帧间隔 p95 {_fmt(raw.get('interval_p95_ms', float('nan')))}ms")

    print("\n-- 曝光 --")
    lock = "是" if exp.get("exposure_lock_effective") else "否/未能验证"
    print(f"  可锁定曝光：{lock}（写入被接受：{exp.get('exposure_set_accepted')}）")
    if not exp.get("exposure_lock_effective"):
        print("  ⚠ 曝光锁不住时，自动曝光跳变会毁掉虹膜边缘 —— 这是比分辨率更致命的项")

    print("\n-- 推理与虹膜 --")
    if inf.get("status") == "ok":
        print(f"  有效帧率：{_fmt(inf.get('effective_fps', float('nan')))} Hz")
        print(f"  单帧延迟：p50 {_fmt(inf.get('latency_p50_ms', float('nan')))}ms"
              f" / p95 {_fmt(inf.get('latency_p95_ms', float('nan')))}ms")
        print(f"  虹膜直径：{_fmt(inf.get('iris_diameter_px_median', float('nan')))} px"
              f"（阈值 {IRIS_MIN_PX}px）")
        if inf.get("estimated_distance_mm"):
            print(f"  估算眼-镜头距离：约 {inf['estimated_distance_mm']:.0f} mm"
                  f"（按 HFOV {ASSUMED_HFOV_DEG}° 假设）")
    else:
        print(f"  已跳过：{inf.get('reason', '')}")

    print("\n-- M0 判定 --")
    for key, label in (("iris", "虹膜像素"), ("fps", "有效帧率"), ("latency", "延迟")):
        item = items.get(key, {})
        mark = {"pass": "PASS ", "conditional": "WARN ", "fail": "FAIL ",
                "unknown": "UNKNOWN"}.get(item.get("level"), "?")
        print(f"  [{mark}] {label}：{item.get('detail', '')}")
    print(f"\n  总体：{j.get('overall')}")

    if j.get("overall") != "PASS":
        print("\n  说明：CONDITIONAL 表示方案可继续但须按降级路径裁剪能力；"
              "\n  FAIL 表示先解决硬件/光照，不写业务代码。")


def _dump(report: dict, out: str | None) -> None:
    path = Path(out) if out else Path("reports") / (
        f"health_check_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入：{path.resolve()}")
    print("该文件不含任何画面数据，可直接贴给别人排查环境问题。")


if __name__ == "__main__":
    raise SystemExit(main())
