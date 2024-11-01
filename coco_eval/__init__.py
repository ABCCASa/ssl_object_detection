import torch
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
import sys
from tools.timer import Timer


@torch.inference_mode()
def evaluate(model, data_loader, device):
    print("[Evaluate Model]")
    eval_timer = Timer()
    eval_timer.start()

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco)

    progress = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
        progress += 1
        if progress % 20 == 0:
            print("\r", end="")
            print(f"eval[{progress}/{len(data_loader)}]", end="")
            sys.stdout.flush()
    print("\r", end="")
    sys.stdout.flush()

    eval_timer.stop()
    eval_time = eval_timer.get_total_time()
    print(f"[Evaluate Complete] {eval_time:.2f} s, {eval_time / len(data_loader.dataset):.5f} s/iter")

    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return coco_evaluator
