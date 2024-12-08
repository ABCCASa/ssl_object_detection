import coco_eval
import global_config
import plot
from model_log import ModelLog
import os.path
import shutil
import torch
from torch import Tensor
import torch.nn as nn
from tools.timer import Timer
from train_config import TrainConfig
from augmentation.reversible_augmentation import get_reversible_augmentation


def ema_update(m1: nn.Module, m2: nn.Module, beta: float):
    if not (0 <= beta <= 1):
        raise ValueError(f"decay is {beta}, decay value must between 0 and 1")
    with torch.no_grad():
        for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values()):
            p1.copy_(p1 * beta + (1 - beta) * p2)


def sum_loss(x) -> Tensor:
    res = None
    for i in x.values():
        if res is None:
            res = i
        else:
            res = res + i
    return res


def full_supervised_train_one_epoch(model: nn.Module, train_loader, valid_loader, optimizer, lr_scheduler, model_log: ModelLog, train_config: TrainConfig):
    model.train()
    train_timer = Timer()
    train_timer.start()
    batch_index = 0

    for images, targets in train_loader:
        early_complete = model_log.iter_num+1 >= train_config.SEMI_SUPERVISED_TRAIN_START

        images = list(image.to(global_config.DEVICE) for image in images)
        targets = [{k: v.to(global_config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum_loss(loss_dict)

        # loss log
        lr = optimizer.param_groups[0]['lr']
        if model_log.iter_num % global_config.TRAIN_STATE_PRINT_FREQ == 0 or batch_index == 0 or batch_index + 1 == len(train_loader):
            loss_log = f"[epoch: {model_log.epoch_num}, iter: {batch_index + 1}/{len(train_loader)} total_iter: {model_log.iter_num}]"
            loss_log += f" lr: {lr:.6f}, loss: {loss: .6f}"
            for k, v in loss_dict.items():
                loss_log += f", {k}: {v:.6f}"
            print(loss_log)

        # update learning rate
        loss = loss / global_config.ACCUMULATION_STEPS
        loss.backward()
        need_step = (batch_index+1) % global_config.ACCUMULATION_STEPS == 0 or batch_index+1 == len(train_loader) or early_complete
        if need_step:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        # update model logs
        model_log.one_iter({
            "lr": lr,
            "loss_dict": {k: v.item() for k, v in loss_dict.items()},
            "full_supervised": True,
            "step": need_step
        })

        # evaluate the model
        if model_log.iter_num % global_config.SUPERVISED_EVAL_FREQ == 0:
            train_timer.stop()
            evals = dict()
            print("\n[Evaluate Model]")
            evals["supervised"] = coco_eval.evaluate(model, valid_loader, device=global_config.DEVICE).coco_eval.stats
            model.train()
            model_log.update_eval(evals)
            model_log.plot_eval("runtime")
            train_timer.start()
        batch_index += 1

        if early_complete:
            break

    train_timer.stop()
    train_time = train_timer.get_total_time()
    print(f"[One epoch train complete] {train_time:.2f} s, {train_time / batch_index:.5f} s/iter")

    # update model log for one_epoch
    model_log.one_epoch()


def semi_supervised_train_one_epoch(
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader,
        valid_loader,
        optimizer,
        lr_scheduler,
        fusion_function,
        model_log: ModelLog,
        train_config: TrainConfig
):
    ema_beta = train_config.EMA_UPDATE_BETA
    unsupervised_weight = train_config.UNSUPERVISED_WEIGHT
    student_model.train()
    train_timer = Timer()
    train_timer.start()
    batch_index = 0
    for images, targets in train_loader:
        early_complete = model_log.iter_num + 1 >= train_config.SEMI_SUPERVISED_TRAIN_END
        is_supervised = targets[0]["supervised"]
        images = list(image.to(global_config.DEVICE) for image in images)
        targets = [{k: v.to(global_config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if is_supervised:
            loss_dict = ssl_labeled(student_model, images, targets)
        else:
            loss_dict = ssl_unlabeled(student_model, images, targets, fusion_function, train_config.FUSE_COUNT)

        loss = sum_loss(loss_dict)

        # training log
        lr = optimizer.param_groups[0]['lr']
        if model_log.iter_num % global_config.TRAIN_STATE_PRINT_FREQ == 0 or batch_index == 0 or batch_index + 1 == len(train_loader):
            loss_log = f"[epoch: {model_log.epoch_num+1}, iter: {batch_index + 1}/{len(train_loader)}, total_iter: {model_log.iter_num}]"
            loss_log += f" labeled: {is_supervised}, lr: {lr:.6f}, loss: {loss: .6f}, weight: {1 if is_supervised else unsupervised_weight: .6f}"
            for k, v in loss_dict.items():
                loss_log += f", {k}: {v:.6f}"
            print(loss_log)

        # model update
        loss = loss if is_supervised else loss * unsupervised_weight
        loss = loss / global_config.ACCUMULATION_STEPS
        loss.backward()
        need_step = (batch_index + 1) % global_config.ACCUMULATION_STEPS == 0 or batch_index + 1 == len(train_loader) or early_complete
        if need_step:
            optimizer.step()
            optimizer.zero_grad()
            ema_update(teacher_model, student_model, ema_beta)
            lr_scheduler.step()

        # update model logs
        model_log.one_iter({
            "lr": lr,
            "loss_dict": {k: v.item() for k, v in loss_dict.items()},
            "full_supervised": False,
            "labeled": is_supervised,
            "weight": 1 if is_supervised else unsupervised_weight,
            "step": need_step
        })

        # evaluate the model
        if model_log.iter_num % global_config.SEMI_SUPERVISED_EVAL_FREQ == 0:
            train_timer.stop()
            evals = {}
            print("\n[Evaluate Student Model]")
            evals["student"] = coco_eval.evaluate(student_model, valid_loader, device=global_config.DEVICE).coco_eval.stats
            student_model.train()
            if ema_beta < 1:
                print("[Evaluate Teacher Model]")
                evals["teacher"] = coco_eval.evaluate(teacher_model, valid_loader, device=global_config.DEVICE).coco_eval.stats
            model_log.update_eval(evals)
            model_log.plot_eval("runtime")
            train_timer.start()
        batch_index += 1
        if early_complete:
            break

    train_timer.stop()
    train_time = train_timer.get_total_time()
    print(f"[One epoch train complete] {train_time:.2f} s, {train_time / len(train_loader):.5f} s/iter")

    # update model log for one_epoch
    model_log.one_epoch()


def ssl_labeled(model, images, targets):
    model.set_ssl_mode(False)
    loss_dict = model(images, targets)
    return loss_dict


reversible_augmentation = get_reversible_augmentation()


def ssl_unlabeled(model, images, targets, fusion_function, max_fuse_count):
    model.set_ssl_mode(True)
    undos = []
    ids = []
    raw_images = []
    for i in range(len(images)):
        raw_images.append(images[i])
        intensity = min(targets[i]["fuse_count"] / max_fuse_count, 1)
        images[i], targets[i]["boxes"], undo = reversible_augmentation.apply(images[i], targets[i]["boxes"], intensity)
        undos.append(undo)
        ids.append(targets[i]["image_id"])

        plot.plot_data(images[i], targets[i], global_config.CLASSES, "runtime", f"{ids[i]}.png")
    detections, losses = model(images, targets)
    for id, target, undo, in zip(ids, detections, undos):
        target["boxes"] = reversible_augmentation.undo(target["boxes"], undo)
        fusion_function(id, target)
    model.set_ssl_mode(False)
    return losses


def save(student_model, teacher_model, train_config, optimizer, lr, model_log, save_folder, set_checkpoint):
    print("Model Saving...")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    train_config.save(os.path.join(save_folder, 'train_config.json'))
    file_name = os.path.join(save_folder, "model.pth")
    data = {
        "student_model": student_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr": lr.state_dict(),
        "log": model_log
    }
    if model_log.get_ssl_init():
        data["teacher_model"] = teacher_model.state_dict()

    torch.save(data, file_name)

    if set_checkpoint:
        checkpoint_folder = os.path.join(save_folder, "check_point")
        checkpoint_file = os.path.join(checkpoint_folder, f"{model_log.epoch_num}epochs, {model_log.iter_num}iters.pth")
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        shutil.copy(file_name, checkpoint_file)


def load(load_folder, student_model=None, teacher_model=None, optimizer=None, lr=None) -> (ModelLog, TrainConfig):
    file_dir = os.path.join(load_folder, "model.pth")
    if os.path.exists(file_dir):
        datas = torch.load(file_dir)
        log = datas["log"]

        if student_model is not None:
            student_model.load_state_dict(datas["student_model"])

        if teacher_model is not None and log.get_ssl_init():
            if "teacher_model" in datas.keys():
                teacher_model.load_state_dict(datas["teacher_model"])
            else:
                print("[Warning] log shows ssl is init but did not find teacher model data, use student model data to init")
                teacher_model.load_state_dict(datas["student_model"])

        if optimizer is not None:
            optimizer.load_state_dict(datas["optimizer"])

        if lr is not None:
            lr.load_state_dict(datas["lr"])
        return log, TrainConfig(os.path.join(load_folder, 'train_config.json'))
    else:
        return ModelLog(), TrainConfig()
