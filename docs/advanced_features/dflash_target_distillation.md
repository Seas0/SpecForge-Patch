# DFlash Target-Logit Distillation

DFlash target-logit distillation adds an output-side soft target loss to
`scripts/train_dflash.py`. The draft model still receives the normal DFlash
all-mask block input and still trains with the original token cross entropy. When
enabled, an extra loss teaches the draft output distribution to match the target
model distribution over the target model's high-probability candidate tokens.

This is different from input-side soft embedding. The distillation loss is
applied to the draft output logits. It does not change DFlash inference, and it
does not add tree expansion or extra verification branches.

## Parameters

Enable the loss with:

```bash
--target-distillation-weight 0.3 \
--target-distillation-top-k 5 \
--target-distillation-top-p 0.9 \
--target-distillation-temperature 1.5
```

The parameters are:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--target-distillation-weight` | `0.0` | Weight of the distillation loss. `0.0` disables the feature and preserves the original DFlash training behavior. |
| `--target-distillation-top-k` | `3` | Number of target top-k tokens considered for each supervised draft position. |
| `--target-distillation-top-p` | unset | Optional nucleus cutoff applied inside the selected target top-k set. At least the highest-probability token is always kept. |
| `--target-distillation-temperature` | `1.0` | Temperature used for both target probabilities and draft log-probabilities in the KD loss. |

The loss gathers target logits from position `p - 1` for the draft label at
position `p`, matching causal language model next-token alignment. It then takes
the target top-k candidates, optionally removes the low-probability tail with
top-p filtering, renormalizes the remaining target probabilities, and computes a
weighted soft cross entropy against the draft logits.

Target distillation currently requires:

```bash
--target-model-backend hf
```

The SGLang target backend returns hidden states for DFlash training, but this KD
loss also needs full target sequence logits.

## Starting from a Pretrained DFlash Checkpoint

Target-logit distillation can start from an existing DFlash draft checkpoint
instead of a freshly initialized draft. Use either spelling:

```bash
--init-draft-model-path <official-dflash-checkpoint-or-hf-repo>
```

or the shorter alias:

```bash
--ckpt-dir <official-dflash-checkpoint-or-hf-repo>
```

This loads the draft checkpoint config and weights, then starts a new training
run with a fresh optimizer and scheduler. It is not the same as `--resume`.
`--resume` only restores training state from `--output-dir/epoch_*` checkpoints.
If `--resume` finds a checkpoint in `--output-dir`, that checkpoint takes
precedence over `--init-draft-model-path`.

When an init checkpoint is provided, its config is used for the draft model to
keep tensor shapes aligned. This overrides `--draft-config-path`.

You can also use the DFlash example scripts with:

```bash
DFLASH_CKPT_DIR=<official-dflash-checkpoint-or-hf-repo> \
bash examples/run_qwen3_8b_dflash_online.sh
```

## Example

```bash
torchrun \
  --standalone \
  --nproc_per_node 8 \
  scripts/train_dflash.py \
  --target-model-path Qwen/Qwen3-8B \
  --draft-config-path configs/qwen3-8b-dflash.json \
  --init-draft-model-path <official-dflash-checkpoint-or-hf-repo> \
  --train-data-path cache/dataset/perfectblend_qwen3-8b_regen.jsonl \
  --output-dir outputs/qwen3-8b-dflash-kd-w03-k5-p09-t15 \
  --num-epochs 6 \
  --batch-size 4 \
  --learning-rate 6e-4 \
  --warmup-ratio 0.04 \
  --max-grad-norm 1.0 \
  --max-length 3072 \
  --chat-template qwen \
  --attention-backend flex_attention \
  --num-anchors 512 \
  --loss-decay-gamma 7.0 \
  --target-model-backend hf \
  --block-size 16 \
  --target-distillation-weight 0.3 \
  --target-distillation-top-k 5 \
  --target-distillation-top-p 0.9 \
  --target-distillation-temperature 1.5 \
  --log-interval 50 \
  --save-interval 1000
```

## Validation Plan

Training loss or top-k agreement alone is not enough to validate this method.
The final metric is greedy DFlash acceptance under the normal single draft path
and one target verification pass.

Use the same target model, tokenizer, prompt template, benchmark data,
generation temperature, block size, and max generation length for every row in
the comparison. If a run starts from an official DFlash checkpoint, every CE-only
and KD ablation row should start from the same checkpoint.

| Stage | What to compare | Pass condition |
| --- | --- | --- |
| Implementation checks | `python -m compileall`, unit tests for top-k/top-p KD, and a tiny one-step training smoke test. | No syntax errors, finite loss, checkpoint save works. |
| Pretrained-init control | Load the official DFlash checkpoint with `--init-draft-model-path` and evaluate it before KD training. | Initial greedy DFlash acceptance matches the expected official-checkpoint range. |
| Zero-weight control | Baseline DFlash training with `--target-distillation-weight 0.0`. | Behavior matches the original DFlash training path. |
| Small ablation | Same data subset and seed for CE-only vs KD settings. | KD improves greedy acceptance length on held-out prompts, not just training CE. |
| Full validation | Best small-run KD setting vs the CE-only checkpoint trained with the same budget. | Higher mean accepted length and no material tokens/sec regression. |
| Robustness | Held-out domains or datasets not used for selection. | Gains persist outside the tuning split. |

Recommended first ablation:

| Run | Weight | Top-k | Top-p | Temperature |
| --- | --- | --- | --- | --- |
| CE baseline | `0.0` | unused | unused | unused |
| KD light | `0.1` | `5` | unset | `1.5` |
| KD nucleus | `0.3` | `5` | `0.9` | `1.5` |
| KD strong | `1.0` | `5` | `0.9` | `2.0` |

If the KD runs reduce CE but do not improve greedy accepted length, treat the
method as a training-signal improvement only, not an acceptance-rate improvement.
If the best KD run only improves acceptance by using extra inference branches,
discard that result for DFlash validation.

## Reporting

Report at least:

- Training budget: dataset, number of samples, epochs, global steps, GPUs, and
  effective batch size.
- KD settings: weight, top-k, top-p, temperature, and target backend.
- Checkpoint identifier and the exact evaluation command.
- Whether the run started from scratch, from `--init-draft-model-path`, or from
  `--resume`.
- Mean accepted length, full-block acceptance rate if available, output
  tokens/sec, and target tokens/sec.
- CE-only baseline trained from the same starting point and with the same budget.
