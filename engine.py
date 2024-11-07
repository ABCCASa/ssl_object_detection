import math
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
import json


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


def full_supervised_train_one_epoch(model: nn.Module, train_loader, valid_loader, optimizer, lr_scheduler, model_log: ModelLog):
    model.train()
    warm_up_lr = None
    if model_log.epoch_num == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_loader) - 1)
        warm_up_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    train_timer = Timer()
    train_timer.start()
    optimizer.zero_grad()
    for batch_index, (images, targets) in enumerate(train_loader):
        images = list(image.to(global_config.DEVICE) for image in images)
        targets = [{k: v.to(global_config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        raw_losses = sum_loss(loss_dict)
        accumulation_loss = raw_losses / global_config.GRADIENT_ACCUMULATION
        accumulation_loss.backward()

        need_step = (batch_index + 1) % global_config.GRADIENT_ACCUMULATION == 0 or batch_index + 1 == len(train_loader)
        if need_step:
            optimizer.step()
            optimizer.zero_grad()

            # update learning rate
            if warm_up_lr is not None:
                warm_up_lr.step()
            else:
                lr_scheduler.step()

        # training state
        training_state = {"lr": optimizer.param_groups[0]['lr'], "losses": raw_losses.item()}
        for k, v in loss_dict.items():
            training_state[k] = v.item()

        # log the train state
        if model_log.iter_num % global_config.TRAIN_STATE_PRINT_FREQ == 0 or batch_index == 0 or batch_index+1 == len(train_loader):
            loss_log = f"[epoch: {model_log.epoch_num}, iter: {batch_index+1}/{len(train_loader)}, total_iter: {model_log.iter_num}]"
            for k, v in training_state.items():
                loss_log += f" {k}: {v:.6f}"
            print(loss_log)

        # update model logs
        training_state["supervised"] = True
        training_state["step"] = need_step
        model_log.one_iter(training_state)

        # evaluate the model
        if model_log.iter_num % global_config.EVAL_FREQ == 0:
            train_timer.stop()
            evals = dict()
            evals["supervised"] = coco_eval.evaluate(model, valid_loader, device=global_config.DEVICE).coco_eval.stats
            model.train()
            model_log.update_eval(evals)
            train_timer.start()

    train_timer.stop()
    train_time = train_timer.get_total_time()
    print(f"[One epoch train complete] {train_time:.2f} s, {train_time/len(train_loader):.5f} s/iter")

    # update model log for one_epoch
    model_log.one_epoch()


def semi_supervised_train_one_epoch(
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader,
        valid_loader,
        optimizer,
        lr_scheduler,
        model_log: ModelLog,
        ema_beta,
        unsupervised_weight
        ):

    student_model.train()
    train_timer = Timer()
    train_timer.start()
    optimizer.zero_grad()
    for batch_index, (images, targets) in enumerate(train_loader):
        is_supervised = targets[0]["supervised"]
        images = list(image.to(global_config.DEVICE) for image in images)
        targets = [{k: v.to(global_config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict = student_model(images, targets)
        raw_losses = sum_loss(loss_dict)

        weighted_losses = raw_losses/global_config.GRADIENT_ACCUMULATION
        if not is_supervised:
            weighted_losses = weighted_losses * unsupervised_weight

        weighted_losses.backward()
        need_step = (batch_index + 1) % global_config.GRADIENT_ACCUMULATION == 0 or (batch_index + 1) == len(train_loader)
        if need_step:
            optimizer.step()
            optimizer.zero_grad()
            ema_update(teacher_model, student_model, ema_beta)

            lr_scheduler.step()

        # training state
        training_state = {"lr": optimizer.param_groups[0]['lr'], "losses": raw_losses.item()}
        for k, v in loss_dict.items():
            training_state[k] = v.item()

        # log the train state
        if model_log.iter_num % global_config.TRAIN_STATE_PRINT_FREQ == 0 or batch_index == 0 or batch_index + 1 == len(train_loader):
            loss_log = f"[epoch: {model_log.epoch_num}, iter: {batch_index+1}/{len(train_loader)}, total_iter: {model_log.iter_num}]"
            for k, v in training_state.items():
                loss_log += f" {k}: {v:.6f}"
            print(loss_log)

        training_state["supervised"] = False
        training_state["step"] = need_step
        model_log.one_iter(training_state)

        # evaluate the model
        if model_log.iter_num % global_config.EVAL_FREQ == 0:
            train_timer.stop()
            evals = {}
            print("Student Model:")
            evals["student"] = coco_eval.evaluate(student_model, valid_loader, device=global_config.DEVICE).coco_eval.stats
            student_model.train()
            if ema_beta < 1:
                print("\nTeacher Model:")
                evals["teacher"] = coco_eval.evaluate(teacher_model, valid_loader, device=global_config.DEVICE).coco_eval.stats

            model_log.update_eval(evals)
            train_timer.start()

    train_timer.stop()
    train_time = train_timer.get_total_time()
    print(f"[One epoch train complete] {train_time:.2f} s, {train_time/len(train_loader):.5f} s/iter")

    # update model log for one_epoch
    model_log.one_epoch()


def save(student_model, teacher_model, train_config, optimizer, lr, model_log, save_folder, checkpoint_freq=None):
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

    if checkpoint_freq is not None and model_log.epoch_num % checkpoint_freq == 0:
        checkpoint_folder = os.path.join(save_folder, "check_point")
        checkpoint_file = os.path.join(checkpoint_folder, f"{model_log.epoch_num}.pth")
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
