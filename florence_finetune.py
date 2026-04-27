"""Florence-2 LoRA fine-tune for GUI grounding (point prediction).

Loss = CE(answer tokens) + lambda_l2 * L2(expected (x,y), gt (x,y))
where the L2 term is a differentiable expectation over Florence's
1000 <loc_*> tokens at the two answer positions.

Designed for Colab/A100. The notebook does:
    !git pull
    from florence_finetune import (
        attach_lora, build_training_items, FlorencePointDataset,
        FlorencePointTrainer, train, evaluate, parse_point,
        get_loc_token_ids, standardize_sample_for_training,
    )
"""
from __future__ import annotations

import io
import json
import os
import queue
import random
import re
import shutil
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

LOC_TOKEN_RE = re.compile(r"<loc_(\d+)>")
TASK = "<REFERRING_EXPRESSION_COMPREHENSION>"
TARGET_RESOLUTION = 512


# ---------------------------------------------------------------------------
# Image / coord utilities (verbatim from MS3_Notebook_Tim_Qwen cell 50)
# ---------------------------------------------------------------------------

def standardize_sample_for_training(img: Image.Image, original_coords_999):
    """Letterbox to TARGET_RESOLUTION and remap [0,999] coords to the new canvas."""
    orig_w, orig_h = img.size
    scale = min(TARGET_RESOLUTION / orig_w, TARGET_RESOLUTION / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    pad_x = (TARGET_RESOLUTION - new_w) // 2
    pad_y = (TARGET_RESOLUTION - new_h) // 2
    canvas = Image.new("RGB", (TARGET_RESOLUTION, TARGET_RESOLUTION), (128, 128, 128))
    canvas.paste(resized, (pad_x, pad_y))

    adjusted = []
    for nx, ny in original_coords_999:
        px = nx / 999.0 * orig_w
        py = ny / 999.0 * orig_h
        new_px = px * scale + pad_x
        new_py = py * scale + pad_y
        adjusted.append((
            int(round(new_px / TARGET_RESOLUTION * 999)),
            int(round(new_py / TARGET_RESOLUTION * 999)),
        ))
    return canvas, adjusted


def decode_image(img_field):
    if isinstance(img_field, Image.Image):
        img = img_field
    elif isinstance(img_field, bytes):
        img = Image.open(io.BytesIO(img_field))
    elif isinstance(img_field, dict) and "bytes" in img_field:
        img = Image.open(io.BytesIO(img_field["bytes"]))
    else:
        raise ValueError(f"Unsupported image field type: {type(img_field)}")
    return img.convert("RGB") if img.mode != "RGB" else img


def parse_coordinates_xy(text: str):
    """Parse '(x, y)' from raw text. Used by build_training_items."""
    m = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


# ---------------------------------------------------------------------------
# Train-item builder (adapted from Qwen cell 62 — memory-safe column pulls)
# ---------------------------------------------------------------------------

def build_training_items(dataset, excluded_rows, cap=None, seed=456):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    convos_col = dataset["conversations"]
    widths_col = dataset["width"]
    heights_col = dataset["height"]

    items = []
    for ri in indices:
        if cap is not None and len(items) >= cap:
            break
        if ri in excluded_rows:
            continue
        try:
            convos = json.loads(convos_col[ri])
        except (json.JSONDecodeError, TypeError):
            continue
        for pi in range(0, len(convos) - 1, 2):
            h, g = convos[pi], convos[pi + 1]
            if h.get("from") != "human" or g.get("from") != "gpt":
                continue
            gt = parse_coordinates_xy(g.get("value", ""))
            if gt is None:
                continue
            items.append({
                "row_idx": ri,
                "width": widths_col[ri],
                "height": heights_col[ri],
                "human_text": h.get("value", ""),
                "gt_x": gt[0],
                "gt_y": gt[1],
            })
            break
    return items


# ---------------------------------------------------------------------------
# Loc token vocabulary
# ---------------------------------------------------------------------------

def get_loc_token_ids(processor, verbose=True):
    """Return tensor of token IDs for <loc_0>..<loc_999>. Asserts contiguous."""
    tok = processor.tokenizer
    ids = [tok.convert_tokens_to_ids(f"<loc_{i}>") for i in range(1000)]
    if any(i is None or i == tok.unk_token_id for i in ids):
        bad = [i for i, v in enumerate(ids) if v is None or v == tok.unk_token_id]
        raise RuntimeError(f"Loc tokens missing/unk in vocab. First few: {bad[:5]}")
    contiguous = all(ids[i + 1] - ids[i] == 1 for i in range(len(ids) - 1))
    if verbose:
        print(f"[florence] loc tokens: <loc_0>={ids[0]} .. <loc_999>={ids[-1]} "
              f"contiguous={contiguous}")
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# RE cleaning (from MS3_Notebook.ipynb cell 53)
# ---------------------------------------------------------------------------

def extract_element_description(text: str) -> str:
    text = (text or "").replace("<image>", "").strip()
    for marker in ["described as follows:", "element is:", "looking for:", "locate:"]:
        if marker in text.lower():
            idx = text.lower().index(marker) + len(marker)
            text = text[idx:].strip()
            break
    text = re.sub(
        r"^(click\s+(on\s+)?|tap\s+(on\s+)?|press\s+|select\s+|find\s+|locate\s+|identify\s+)",
        "", text, flags=re.IGNORECASE,
    )
    text = re.split(r"\s+(at|in|near|located|within)\s+", text)[0]
    return text.strip() or "the element"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FlorencePointDataset(Dataset):
    """Encoder-decoder fine-tune samples for Florence-2.

    Returns:
      input_ids       — encoder input (prompt only, via the processor)
      attention_mask  — encoder mask
      pixel_values    — preprocessed image
      labels          — decoder target tokens for "<loc_X><loc_Y>"
      gt_x, gt_y      — remapped GT coords (for L2 loss)

    Florence-2 is BART-like (seq2seq): the model right-shifts `labels` into
    decoder inputs internally, so logits[:, t, :] predict labels[:, t].
    """

    def __init__(self, items, ds, processor, debug_first=True):
        self.items = items
        self.ds = ds
        self.processor = processor
        self._debug_done = not debug_first

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = decode_image(self.ds["image"][item["row_idx"]])
        image, adj = standardize_sample_for_training(image, [(item["gt_x"], item["gt_y"])])
        gx, gy = adj[0]

        cleaned = extract_element_description(item["human_text"])
        prompt_text = f"{TASK} {cleaned}"
        answer_text = f"<loc_{gx}><loc_{gy}>"

        # Encoder input: prompt + image, via the processor (handles Florence's
        # text/vision interleaving and any task-specific token additions).
        enc = self.processor(
            text=prompt_text, images=image, return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        pixel_values = enc["pixel_values"][0]
        attention_mask = enc.get("attention_mask")
        attention_mask = attention_mask[0] if attention_mask is not None else torch.ones_like(input_ids)

        # Decoder labels: tokenized answer with BOS/EOS. shift_tokens_right is
        # done by the model when `labels` is passed.
        labels = self.processor.tokenizer(
            answer_text, return_tensors="pt", add_special_tokens=True,
        ).input_ids[0]

        if not self._debug_done:
            print(f"[florence] sample0: input_ids={tuple(input_ids.shape)} "
                  f"pixel_values={tuple(pixel_values.shape)} "
                  f"labels={tuple(labels.shape)} answer='{answer_text}' "
                  f"label_tokens={labels.tolist()} gt=({gx},{gy})")
            self._debug_done = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "gt_x": torch.tensor(gx, dtype=torch.float32),
            "gt_y": torch.tensor(gy, dtype=torch.float32),
        }


def make_collate(pad_id: int):
    """Pad encoder input_ids and decoder labels to per-batch max length."""
    def collate(features):
        bsz = len(features)
        enc_max = max(f["input_ids"].shape[0] for f in features)
        dec_max = max(f["labels"].shape[0] for f in features)

        input_ids = torch.full((bsz, enc_max), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, enc_max), dtype=torch.long)
        labels = torch.full((bsz, dec_max), -100, dtype=torch.long)
        pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)
        gt_x = torch.stack([f["gt_x"] for f in features])
        gt_y = torch.stack([f["gt_y"] for f in features])

        for i, f in enumerate(features):
            T_enc = f["input_ids"].shape[0]
            T_dec = f["labels"].shape[0]
            input_ids[i, :T_enc] = f["input_ids"]
            attention_mask[i, :T_enc] = f["attention_mask"]
            labels[i, :T_dec] = f["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "gt_x": gt_x,
            "gt_y": gt_y,
        }
    return collate


# ---------------------------------------------------------------------------
# Combined-loss Trainer
# ---------------------------------------------------------------------------

class FlorencePointTrainer(Trainer):
    """Adds a differentiable L2 loss over the two <loc_*> answer positions."""

    def __init__(self, *args, loc_token_ids: torch.Tensor, lambda_l2: float = 0.01,
                 debug_first_n: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        # Register as buffer-like; moved to model device in compute_loss.
        self._loc_token_ids = loc_token_ids
        self.lambda_l2 = lambda_l2
        self._debug_left = debug_first_n

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        gt_x = inputs.pop("gt_x")
        gt_y = inputs.pop("gt_y")
        labels = inputs["labels"]  # [B, T_dec]; needed to find loc-token positions

        outputs = model(**inputs)
        loss_ce = outputs.loss
        logits = outputs.logits  # [B, T_dec, V] — logits[:, t, :] predicts labels[:, t]

        loc_ids = self._loc_token_ids.to(logits.device)
        loc_id_min = int(loc_ids.min().item())
        loc_id_max = int(loc_ids.max().item())

        # For each example, find the first two label positions whose token ID is
        # in the [<loc_0>..<loc_999>] range. These are the X and Y positions.
        is_loc = (labels >= loc_id_min) & (labels <= loc_id_max) & (labels != -100)
        # Index of the first and second loc token per row. If a row is malformed
        # (only 0 or 1 loc tokens), skip its L2 contribution.
        B, T = labels.shape
        device = logits.device
        l2_terms = []
        for b in range(B):
            positions = is_loc[b].nonzero(as_tuple=False).flatten()
            if positions.numel() < 2:
                continue
            x_pos = positions[0].item()
            y_pos = positions[1].item()
            x_logits = logits[b, x_pos, loc_ids]
            y_logits = logits[b, y_pos, loc_ids]
            x_probs = torch.softmax(x_logits.float(), dim=-1)
            y_probs = torch.softmax(y_logits.float(), dim=-1)
            coord_grid = torch.arange(1000, device=device, dtype=torch.float32)
            exp_x = (x_probs * coord_grid).sum()
            exp_y = (y_probs * coord_grid).sum()
            term = torch.sqrt(
                (exp_x - gt_x[b].to(device).float()) ** 2
                + (exp_y - gt_y[b].to(device).float()) ** 2
                + 1e-6
            )
            l2_terms.append(term)

        if l2_terms:
            loss_l2 = torch.stack(l2_terms).mean() / 999.0
        else:
            # No valid loc positions in this batch; pass a zero with grad so
            # the optimizer step is well-defined.
            loss_l2 = torch.zeros((), device=device, dtype=torch.float32)

        loss = loss_ce + self.lambda_l2 * loss_l2

        # Lightweight debug: print the first few steps so the user can spot
        # NaNs, format issues, or wildly imbalanced loss terms.
        if self._debug_left > 0:
            ex_x = float(exp_x.detach().item()) if l2_terms else float("nan")
            ex_y = float(exp_y.detach().item()) if l2_terms else float("nan")
            print(f"[florence] step debug: loss={loss.item():.4f} "
                  f"ce={loss_ce.item():.4f} l2_norm={loss_l2.item():.4f} "
                  f"exp_xy=({ex_x:.0f},{ex_y:.0f}) "
                  f"gt0=({gt_x[0].item():.0f},{gt_y[0].item():.0f})")
            self._debug_left -= 1

        # Surface both terms in Trainer's normal logging stream.
        try:
            self.log({"loss_ce": float(loss_ce.detach()),
                      "loss_l2_norm": float(loss_l2.detach())})
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# LoRA attach
# ---------------------------------------------------------------------------

def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                target_modules=None, verbose: bool = True):
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        # Florence-2 (BART-based language model + DaViT vision encoder) uses
        # `out_proj` rather than `o_proj`. We list both so LoRA finds at least
        # one match on each backbone variant.
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    if verbose:
        # Quick dump so we can see what PEFT will actually wrap.
        names = [n for n, _ in model.named_modules()
                 if any(t in n.split(".")[-1] for t in target_modules)]
        print(f"[florence] LoRA target_modules={target_modules} "
              f"(matched {len(names)} modules; first 5: {names[:5]})")

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


# ---------------------------------------------------------------------------
# Drive mirror (verbatim from Qwen cell 65)
# ---------------------------------------------------------------------------

class DriveMirrorCallback(TrainerCallback):
    def __init__(self, local_dir, drive_dir):
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.q = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            local_path, ckpt_name = job
            try:
                drive_path = os.path.join(self.drive_dir, ckpt_name)
                if os.path.isdir(drive_path):
                    shutil.rmtree(drive_path)
                shutil.copytree(local_path, drive_path)
                self._sync_deletions()
                print(f"[drive-mirror] synced {ckpt_name} to Drive", flush=True)
            except Exception as e:
                print(f"[drive-mirror] WARN syncing {ckpt_name}: {e}", flush=True)
            finally:
                self.q.task_done()

    def _sync_deletions(self):
        local_ckpts = {d for d in os.listdir(self.local_dir) if d.startswith("checkpoint-")}
        drive_ckpts = {d for d in os.listdir(self.drive_dir) if d.startswith("checkpoint-")}
        for stale in drive_ckpts - local_ckpts:
            shutil.rmtree(os.path.join(self.drive_dir, stale), ignore_errors=True)

    def on_save(self, args, state, control, **kwargs):
        ckpt_name = f"checkpoint-{state.global_step}"
        local_path = os.path.join(self.local_dir, ckpt_name)
        if os.path.isdir(local_path):
            self.q.put((local_path, ckpt_name))

    def on_train_end(self, args, state, control, **kwargs):
        print(f"[drive-mirror] waiting for {self.q.qsize()} pending sync(s)...", flush=True)
        self.q.join()
        print("[drive-mirror] all checkpoints synced to Drive.", flush=True)


# ---------------------------------------------------------------------------
# Train entry point
# ---------------------------------------------------------------------------

@dataclass
class TrainCfg:
    output_dir: str
    drive_dir: str | None = None
    smoke: bool = False
    lambda_l2: float = 0.01
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 4
    epochs: int = 1
    save_steps: int = 1000
    logging_steps: int = 50
    save_total_limit: int = 3
    smoke_max_train_samples: int = 100
    smoke_max_steps: int = 20


def train(model, processor, ds, train_items, cfg: TrainCfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.drive_dir:
        os.makedirs(cfg.drive_dir, exist_ok=True)

    if cfg.smoke:
        train_items = train_items[: cfg.smoke_max_train_samples]
        print(f"[florence] SMOKE mode: {len(train_items)} train items, "
              f"max_steps={cfg.smoke_max_steps}")

    # Sanity-check loc tokens once before building the dataset.
    loc_ids = get_loc_token_ids(processor, verbose=True)

    train_ds = FlorencePointDataset(train_items, ds, processor)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    collate = make_collate(pad_id)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    args_kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=bf16,
        fp16=fp16,
        save_strategy="steps",
        save_steps=cfg.save_steps if not cfg.smoke else max(1, cfg.smoke_max_steps // 2),
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps if not cfg.smoke else 2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )
    if cfg.smoke:
        args_kwargs["max_steps"] = cfg.smoke_max_steps
    else:
        args_kwargs["num_train_epochs"] = cfg.epochs

    training_args = TrainingArguments(**args_kwargs)

    callbacks = []
    if cfg.drive_dir and not cfg.smoke:
        callbacks.append(DriveMirrorCallback(cfg.output_dir, cfg.drive_dir))

    trainer = FlorencePointTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate,
        loc_token_ids=loc_ids,
        lambda_l2=cfg.lambda_l2,
        callbacks=callbacks,
    )

    print(f"[florence] starting training: smoke={cfg.smoke} "
          f"items={len(train_items)} bsz={cfg.batch_size} "
          f"accum={cfg.grad_accum} lambda_l2={cfg.lambda_l2}")

    def _has_ckpts(d):
        return os.path.isdir(d) and any(
            x.startswith("checkpoint-") for x in os.listdir(d)
        )

    resume = (not cfg.smoke) and _has_ckpts(cfg.output_dir)
    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(cfg.output_dir)
    if cfg.drive_dir and not cfg.smoke:
        for fname in os.listdir(cfg.output_dir):
            src = os.path.join(cfg.output_dir, fname)
            dst = os.path.join(cfg.drive_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"[florence] adapter saved -> {cfg.output_dir} (mirrored to {cfg.drive_dir})")
    else:
        print(f"[florence] adapter saved -> {cfg.output_dir}")
    return trainer


# ---------------------------------------------------------------------------
# Inference / parsing
# ---------------------------------------------------------------------------

def parse_point(raw_text: str):
    """Return (x, y) in [0, 999] from the first two <loc_*> tokens, or None."""
    if not raw_text:
        return None
    hits = LOC_TOKEN_RE.findall(raw_text)
    if len(hits) < 2:
        return None
    return int(hits[0]), int(hits[1])


def _normalize_eval_item(item, ds):
    """Accept both build_training_items shape (gt_x/gt_y/human_text/row_idx)
    and florence_samples shape (gt_point/referring_expression, image inline or
    via source_idx). Returns a dict with row_idx, gt_x, gt_y, human_text,
    width, height."""
    if "gt_x" in item and "gt_y" in item:
        return {
            "row_idx": item["row_idx"],
            "gt_x": item["gt_x"],
            "gt_y": item["gt_y"],
            "human_text": item.get("human_text") or item.get("referring_expression", ""),
            "width": item["width"],
            "height": item["height"],
        }
    # florence_samples shape
    gx, gy = item["gt_point"]
    return {
        "row_idx": item.get("source_idx"),
        "gt_x": gx,
        "gt_y": gy,
        "human_text": item.get("referring_expression", ""),
        "width": item["width"],
        "height": item["height"],
    }


def evaluate(model, processor, eval_items, ds, max_new_tokens=8, verbose_first_n=3):
    """Greedy point-prediction inference. Returns DataFrame matching compute_metrics.

    Accepts items in either the build_training_items shape (gt_x/gt_y) or the
    florence_samples shape (gt_point=(x,y))."""
    model.eval()
    device = next(model.parameters()).device
    rows = []
    t0 = time.time()
    for i, raw_item in enumerate(eval_items):
        item = _normalize_eval_item(raw_item, ds)
        try:
            image = decode_image(ds["image"][item["row_idx"]])
            image, _ = standardize_sample_for_training(
                image, [(item["gt_x"], item["gt_y"])]
            )
            cleaned = extract_element_description(item["human_text"])
            prompt = f"{TASK} {cleaned}"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            pixel_values = inputs["pixel_values"]
            if device.type == "cuda":
                # Match the model's actual dtype (cell 47 loads Florence in fp16,
                # but a user may load it in bf16). LoRA adapters can be fp32; we
                # need the dtype of a *base* (non-LoRA) float parameter so the
                # vision encoder's biases match.
                base_dtype = next(
                    p.dtype for n, p in model.named_parameters()
                    if "lora_" not in n and p.is_floating_point()
                )
                pixel_values = pixel_values.to(base_dtype)
            with torch.inference_mode():
                gen_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            raw = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            pred = parse_point(raw)
        except Exception as e:
            raw = f"[ERROR: {type(e).__name__}: {e}]"
            pred = None

        if i < verbose_first_n:
            print(f"[florence-eval {i}] gt=({item['gt_x']},{item['gt_y']}) "
                  f"pred={pred} raw={raw[:120]!r}")

        rows.append({
            "row_idx": item["row_idx"],
            "width": item["width"],
            "height": item["height"],
            "gt_x": item["gt_x"],
            "gt_y": item["gt_y"],
            "raw_output": raw,
            "pred_x": pred[0] if pred else None,
            "pred_y": pred[1] if pred else None,
        })
    elapsed = time.time() - t0
    print(f"[florence] eval done: {len(rows)} samples in {elapsed:.1f}s "
          f"({elapsed / max(len(rows), 1):.2f}s/sample)")
    return pd.DataFrame(rows)
