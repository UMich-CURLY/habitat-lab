#!/usr/bin/env python3
"""
从 TensorBoard 的 event 文件中导出 top_down_map 图像（每个 run 导出一张，取最后一帧）。
用法:
  python export_tensorboard_topdown.py [--logdir LOGDIR] [--out OUT] [--tag TAG]
  python export_tensorboard_topdown.py --step 100   # 可选：指定 step，不指定则用最后一帧
默认: logdir=runs/habitat_visualization, out=exported_topdown
若全部 [skip]，可先运行: python export_tensorboard_topdown.py --debug
查看 event 里实际有哪些 image tag；若 --debug 也无输出，可尝试安装 tensorflow 再运行。
"""

import argparse
import glob
import io
import os
import sys

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from tensorboard.backend.event_processing import event_file_loader
    HAS_TB = True
except ImportError:
    HAS_TB = False


def _extract_image_from_event(event, tag, tag_contains_fallback=True):
    """从 event.summary.value 里取出 tag 对应的图像 bytes，没有则返回 None。"""
    if not getattr(event, "summary", None) or not event.summary.value:
        return None
    for v in event.summary.value:
        tag_match = v.tag == tag or (tag_contains_fallback and tag in v.tag)
        if not tag_match:
            continue
        if not getattr(v, "image", None) or not getattr(v.image, "encoded_image_string", None):
            continue
        data = v.image.encoded_image_string
        if not data or len(data) == 0:
            continue
        return data
    return None


def _has_image_bytes(v):
    """检查 Summary.Value 是否包含 image 的 encoded 数据。"""
    img = getattr(v, "image", None)
    if img is None:
        return False
    data = getattr(img, "encoded_image_string", None)
    if data is None:
        return False
    return len(data) > 0


def _list_tags_from_event_file(event_path, max_events=50):
    """调试用：列出前 max_events 个事件里出现的 image 类 tag。"""
    seen = set()
    count = 0
    if HAS_TF:
        try:
            for event in tf.compat.v1.train.summary_iterator(event_path):
                if not getattr(event, "summary", None) or not event.summary.value:
                    continue
                for v in event.summary.value:
                    if _has_image_bytes(v):
                        seen.add((v.tag, event.step))
                count += 1
                if count >= max_events:
                    break
        except Exception as e:
            return None, str(e)
    elif HAS_TB:
        try:
            for event in event_file_loader.EventFileLoader(event_path).Load():
                if not getattr(event, "summary", None) or not event.summary.value:
                    continue
                for v in event.summary.value:
                    if _has_image_bytes(v):
                        seen.add((v.tag, event.step))
                count += 1
                if count >= max_events:
                    break
        except Exception as e:
            return None, str(e)
    return list(seen), None


def get_image_from_event_file(event_path, tag="top_down_map", target_step=None):
    """
    从单个 event 文件中读取图像。
    target_step=None: 取该 tag 下 step 最大的那一帧（默认，方便）。
    target_step=int: 只取该 step 的图像。
    返回 (step, encoded_bytes) 或 None。
    """
    if HAS_TF:
        best_step, best_encoded = None, None
        try:
            for event in tf.compat.v1.train.summary_iterator(event_path):
                encoded = _extract_image_from_event(event, tag)
                if encoded is None:
                    continue
                if target_step is not None:
                    if event.step == target_step:
                        return event.step, encoded
                else:
                    if best_step is None or event.step > best_step:
                        best_step, best_encoded = event.step, encoded
        except Exception:
            pass
        if target_step is None and best_step is not None:
            return best_step, best_encoded
        return None

    if HAS_TB:
        best_step, best_encoded = None, None
        try:
            for event in event_file_loader.EventFileLoader(event_path).Load():
                encoded = _extract_image_from_event(event, tag)
                if encoded is None:
                    continue
                if target_step is not None:
                    if event.step == target_step:
                        return event.step, encoded
                else:
                    if best_step is None or event.step > best_step:
                        best_step, best_encoded = event.step, encoded
        except Exception:
            pass
        if target_step is None and best_step is not None:
            return best_step, best_encoded
        return None

    return None


def main():
    parser = argparse.ArgumentParser(description="从 TensorBoard 导出 top_down_map 图像（每 run 一张）。")
    parser.add_argument("--logdir", type=str, default="runs/habitat_visualization", help="TensorBoard log 根目录")
    parser.add_argument("--out", type=str, default="exported_topdown", help="输出目录")
    parser.add_argument("--tag", type=str, default="top_down_map", help="TensorBoard 中的 image tag")
    parser.add_argument("--step", type=str, default=None,
                        help="可选：指定 step 数字则导出该 step；不指定则导出最后一帧")
    parser.add_argument("--debug", action="store_true", help="列出 event 文件里的 image tag，便于排查")
    args = parser.parse_args()

    if not HAS_TF and not HAS_TB:
        print("需要安装 tensorflow 或 tensorboard: pip install tensorflow 或 pip install tensorboard", file=sys.stderr)
        sys.exit(1)

    logdir = os.path.abspath(args.logdir)
    outdir = os.path.abspath(args.out)
    tag = args.tag
    target_step = None
    if args.step is not None:
        try:
            target_step = int(args.step.strip())
        except ValueError:
            print("--step 需为数字", file=sys.stderr)
            sys.exit(1)

    if not os.path.isdir(logdir):
        print(f"logdir 不存在: {logdir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    run_dirs = sorted([d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))])
    if not run_dirs:
        print(f"未找到 run 子目录: {logdir}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        run_path = os.path.join(logdir, run_dirs[0])
        event_files = sorted(glob.glob(os.path.join(run_path, "events.out.tfevents.*")))
        if not event_files:
            print(f"[debug] 第一个 run {run_dirs[0]} 下没有 event 文件", file=sys.stderr)
            sys.exit(1)
        tags_list, err = _list_tags_from_event_file(event_files[0], max_events=100)
        if err:
            print(f"[debug] 读取 event 失败: {err}", file=sys.stderr)
            sys.exit(1)
        if not tags_list:
            print("[debug] 未发现任何 image 类 tag。请确认 logdir 是否正确、是否为 PyTorch SummaryWriter 写的。", file=sys.stderr)
            sys.exit(1)
        print("[debug] 前 100 个事件中出现的 image tag (tag, step):")
        for t, s in sorted(tags_list, key=lambda x: (x[0], x[1]))[:50]:
            print(f"  tag={t!r}  step={s}")
        if len(tags_list) > 50:
            print(f"  ... 共 {len(tags_list)} 条")
        sys.exit(0)

    saved = 0
    for run_name in run_dirs:
        run_path = os.path.join(logdir, run_name)
        event_files = sorted(glob.glob(os.path.join(run_path, "events.out.tfevents.*")))
        if not event_files:
            continue

        found = None
        for ef in event_files:
            result = get_image_from_event_file(ef, tag=tag, target_step=target_step)
            if result is not None:
                step_val, encoded_val = result
                if found is None or step_val > found[0]:
                    found = (step_val, encoded_val)
                if target_step is not None:
                    break

        if found is None:
            step_hint = target_step if target_step is not None else "最后一帧"
            print(f"[skip] {run_name}: 未找到 {tag} (step={step_hint})")
            continue

        actual_step, encoded = found
        out_name = f"{run_name}_step{actual_step}.png"
        out_path = os.path.join(outdir, out_name)

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(encoded))
            img.save(out_path)
        except Exception:
            if HAS_TF:
                arr = tf.io.decode_png(encoded).numpy()
                try:
                    import cv2
                    cv2.imwrite(out_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                except Exception:
                    from PIL import Image
                    Image.fromarray(arr).save(out_path)
            else:
                print(f"[error] {run_name}: 保存失败", file=sys.stderr)
                continue

        print(f"[saved] {out_path}")
        saved += 1

    print(f"共导出 {saved} 张图到 {outdir}")


if __name__ == "__main__":
    main()
