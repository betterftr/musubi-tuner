# Full finetuning on deterministic parameter splice proxies

## Control args

`--ff 0.01` (how much params)
`--ffid 1` (where)

This allows full-finetuning to a targeted parameter space in the model, baking, without overwriting knowledge of other slices.

### How the Selection Algorithm v2.0 Works

The algorithm is designed to be deterministic, non-overlapping, and strategically intelligent. It is robust to any fraction (`--ff`) by satisfying two non-negotiable goals: **40/40 block coverage** and **meaningful training impact**. It achieves this through a priority-aware, single-pass selection from a specially constructed candidate pool that ensures high-priority parameters are spread evenly across all layers.

1.  **Zero Overlap (Partitioning):**
    *   First, all model parameters are deterministically sorted by name to create a consistent global order.
    *   This list is then "dealt" into a fixed number of separate pools, much like dealing cards. The `--ffid` parameter selects one of these pools. This partitioning is the cornerstone that guarantees parameter independence: `ffid=1` and `ffid=2` will **never** train the same parameter.

**2.  Strategic Pool Construction & Selection:**
After partitioning, the algorithm builds a single, balanced `candidate_pool` for the `ffid` designed to distribute selections evenly. **This replaces the previous two-phase logic with a more elegant single-pass approach.**

*   **Step 1: Group & Rank:** All parameters in the partition are first grouped by their transformer block (0-39). Within each block, they are sorted by importance (e.g., major weights > biases > norms) and then by size.
*   **Step 2: Interleaved Draft:** The `candidate_pool` is constructed using a round-robin "draft." It takes the #1 ranked parameter from Block 0, then #1 from Block 1, ..., up to Block 39. It then circles back to take the #2 ranked parameter from each block in sequence, and so on.
*   **Step 3: Fill the Budget:** The algorithm iterates through this perfectly interleaved pool, selecting parameters one by one until the target fraction (`--ff`) is met or slightly exceeded to guarantee meaningful training with at least 1 high-impact matrix.
---

**Final Selection Properties:**

*   **Full Coverage:** The selection always includes parameters from all 40 transformer blocks to promote stable training.
*   **Disjoint Sets:** The partitioning method ensures that parameter sets for different `ffid`s are mutually exclusive, preventing training overlap.
*   **Fraction Robustness:** The interleaved pool construction guarantees that for any fraction, the selection includes a mix of both small, full-coverage parameters and the highest-impact parameters available within the budget, distributed evenly.
*   **Even Distribution:** The candidate pool is constructed using a round-robin draft across all 40 blocks, causing the selection process to distribute high-impact parameters evenly throughout the model, rather than clustering them in early layers.
*   **Budget Management:** The target fraction is treated as a soft constraint. The algorithm adheres to it by selecting parameters from the prioritized pool until the cumulative size meets or slightly exceeds the target budget.

---

```
Selection algorithm in action for --ff 0.001 --ffid 1:

INFO:__main__:Model loaded. Found 1,095 tensors with 14,288,491,584 total parameters.
INFO:musubi_tuner.utils.selective_finetune:Target: 14,288,491 params (0.001000) for set 1
INFO:musubi_tuner.utils.selective_finetune:Partitioned pool size: 137 tensors
INFO:musubi_tuner.utils.selective_finetune:Created a balanced candidate pool of 137 tensors.
INFO:musubi_tuner.utils.selective_finetune:Final selection: 44 tensors, 26,434,560 params
INFO:__main__:
   --- Detailed Analysis for ffid 1 ---
INFO:__main__:   Coverage: ✅ Perfect 40/40 blocks.
INFO:__main__:   Selection Rate Report (target is 0.10%):
WARNING:__main__: - Block 00: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight, cross_attn.o.bias] [LOW]
WARNING:__main__: - Block 01: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.k.bias, self_attn.v.bias] [LOW]
WARNING:__main__: - Block 02: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight, norm3.weight] [LOW]
WARNING:__main__: - Block 03: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [cross_attn.o.weight, cross_attn.q.bias] [HIGH]
WARNING:__main__: - Block 04: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 05: 0.00% selected (5,120 / 351,394,304 params) | Selected: [self_attn.o.bias] [LOW]
WARNING:__main__: - Block 06: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 07: 0.00% selected (5,120 / 351,394,304 params) | Selected: [ffn.2.bias] [LOW]
WARNING:__main__: - Block 08: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 09: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.q.bias] [LOW]
WARNING:__main__: - Block 10: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 11: 0.00% selected (5,120 / 351,394,304 params) | Selected: [self_attn.o.bias] [LOW]
WARNING:__main__: - Block 12: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 13: 0.00% selected (5,120 / 351,394,304 params) | Selected: [ffn.2.bias] [LOW]
WARNING:__main__: - Block 14: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 15: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.q.bias] [LOW]
WARNING:__main__: - Block 16: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 17: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.k.bias] [LOW]
WARNING:__main__: - Block 18: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 19: 0.00% selected (5,120 / 351,394,304 params) | Selected: [self_attn.o.bias] [LOW]
WARNING:__main__: - Block 20: 0.00% selected (5,120 / 351,394,304 params) | Selected: [ffn.2.bias] [LOW]
WARNING:__main__: - Block 21: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 22: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.q.bias] [LOW]
WARNING:__main__: - Block 23: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 24: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.k.bias] [LOW]
WARNING:__main__: - Block 25: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 26: 0.00% selected (5,120 / 351,394,304 params) | Selected: [self_attn.o.bias] [LOW]
WARNING:__main__: - Block 27: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 28: 0.00% selected (5,120 / 351,394,304 params) | Selected: [ffn.2.bias] [LOW]
WARNING:__main__: - Block 29: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 30: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 31: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.k.bias] [LOW]
WARNING:__main__: - Block 32: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 33: 0.00% selected (5,120 / 351,394,304 params) | Selected: [self_attn.o.bias] [LOW]
WARNING:__main__: - Block 34: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 35: 0.00% selected (5,120 / 351,394,304 params) | Selected: [ffn.2.bias] [LOW]
WARNING:__main__: - Block 36: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 37: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.q.bias] [LOW]
WARNING:__main__: - Block 38: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.norm_q.weight] [LOW]
WARNING:__main__: - Block 39: 0.00% selected (5,120 / 351,394,304 params) | Selected: [cross_attn.k.bias] [LOW]

For --ff 0.01 --ffid 2:

INFO:__main__:Model loaded. Found 1,095 tensors with 14,288,491,584 total parameters.
INFO:musubi_tuner.utils.selective_finetune:Target: 142,884,915 params (0.010000) for set 2
INFO:musubi_tuner.utils.selective_finetune:Partitioned pool size: 137 tensors
INFO:musubi_tuner.utils.selective_finetune:Created a balanced candidate pool of 137 tensors.
INFO:musubi_tuner.utils.selective_finetune:Final selection: 81 tensors, 157,798,400 params
INFO:__main__:
   --- Detailed Analysis for ffid 2 ---
INFO:__main__:   Coverage: ✅ Perfect 40/40 blocks.
INFO:__main__:   Selection Rate Report (target is 1.00%):
WARNING:__main__: - Block 00: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.q.bias, cross_attn.q.weight, self_attn.norm_k.weight] [HIGH]
WARNING:__main__: - Block 01: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.o.bias, norm3.bias] [LOW]
WARNING:__main__: - Block 02: 0.00% selected (10,240 / 351,394,304 params) | Selected: [self_attn.norm_k.weight, self_attn.o.bias] [LOW]
WARNING:__main__: - Block 03: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [cross_attn.q.weight, cross_attn.v.bias] [HIGH]
WARNING:__main__: - Block 04: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.k.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 05: 0.01% selected (35,840 / 351,394,304 params) | Selected: [modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 06: 0.00% selected (10,240 / 351,394,304 params) | Selected: [self_attn.norm_k.weight, self_attn.o.bias] [LOW]
WARNING:__main__: - Block 07: 0.00% selected (10,240 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.bias] [LOW]
WARNING:__main__: - Block 08: 0.00% selected (10,240 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 09: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [cross_attn.q.weight, cross_attn.v.bias] [HIGH]
WARNING:__main__: - Block 10: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.k.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 11: 0.01% selected (35,840 / 351,394,304 params) | Selected: [modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 12: 0.00% selected (10,240 / 351,394,304 params) | Selected: [self_attn.norm_k.weight, self_attn.o.bias] [LOW]
WARNING:__main__: - Block 13: 0.00% selected (10,240 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.bias] [LOW]
WARNING:__main__: - Block 14: 0.00% selected (10,240 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 15: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [cross_attn.q.weight, cross_attn.v.bias] [HIGH]
WARNING:__main__: - Block 16: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.q.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 17: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.o.bias, norm3.bias] [LOW]
WARNING:__main__: - Block 18: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.k.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 19: 0.01% selected (35,840 / 351,394,304 params) | Selected: [modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 20: 0.00% selected (10,240 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.bias] [LOW]
WARNING:__main__: - Block 21: 0.00% selected (10,240 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 22: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [cross_attn.q.weight, cross_attn.v.bias] [HIGH]
WARNING:__main__: - Block 23: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.q.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 24: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.o.bias, norm3.bias] [LOW]
WARNING:__main__: - Block 25: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.k.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 26: 0.01% selected (35,840 / 351,394,304 params) | Selected: [modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 27: 0.00% selected (10,240 / 351,394,304 params) | Selected: [self_attn.norm_k.weight, self_attn.o.bias] [LOW]
WARNING:__main__: - Block 28: 0.00% selected (10,240 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.bias] [LOW]
WARNING:__main__: - Block 29: 0.00% selected (10,240 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 30: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.q.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 31: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.o.bias, norm3.bias] [LOW]
WARNING:__main__: - Block 32: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.k.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 33: 0.01% selected (35,840 / 351,394,304 params) | Selected: [modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 34: 0.00% selected (10,240 / 351,394,304 params) | Selected: [self_attn.norm_k.weight, self_attn.o.bias] [LOW]
WARNING:__main__: - Block 35: 0.00% selected (10,240 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.bias] [LOW]
WARNING:__main__: - Block 36: 0.00% selected (10,240 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 37: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [cross_attn.q.weight, cross_attn.v.bias] [HIGH]
WARNING:__main__: - Block 38: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.q.bias, self_attn.norm_k.weight] [LOW]
WARNING:__main__: - Block 39: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.o.bias, norm3.bias] [LOW]

For --ff 0.05 --ffid 4:

INFO:__main__:Model loaded. Found 1,095 tensors with 14,288,491,584 total parameters.
INFO:musubi_tuner.utils.selective_finetune:Target: 714,424,579 params (0.050000) for set 4
INFO:musubi_tuner.utils.selective_finetune:Partitioned pool size: 137 tensors
INFO:musubi_tuner.utils.selective_finetune:Created a balanced candidate pool of 137 tensors.
INFO:musubi_tuner.utils.selective_finetune:Final selection: 113 tensors, 734,609,920 params
INFO:__main__:
   --- Detailed Analysis for ffid 4 ---
INFO:__main__:   Coverage: ✅ Perfect 40/40 blocks.
INFO:__main__:   Selection Rate Report (target is 5.00%):
WARNING:__main__: - Block 00: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 01: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.v.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 02: 0.01% selected (24,064 / 351,394,304 params) | Selected: [cross_attn.k.bias, ffn.0.bias, self_attn.v.bias] [LOW]
WARNING:__main__: - Block 03: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, self_attn.k.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 04: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.q.bias, norm3.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 05: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.o.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 06: 0.01% selected (24,064 / 351,394,304 params) | Selected: [cross_attn.k.bias, ffn.0.bias, self_attn.v.bias] [LOW]
WARNING:__main__: - Block 07: 0.01% selected (40,960 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 08: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.weight, self_attn.o.bias] [HIGH]
WARNING:__main__: - Block 09: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, self_attn.k.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 10: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.q.bias, norm3.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 11: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.o.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 12: 0.01% selected (24,064 / 351,394,304 params) | Selected: [cross_attn.k.bias, ffn.0.bias, self_attn.v.bias] [LOW]
WARNING:__main__: - Block 13: 0.01% selected (40,960 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 14: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.weight, self_attn.o.bias] [HIGH]
WARNING:__main__: - Block 15: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, self_attn.k.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 16: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 17: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.v.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 18: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.q.bias, norm3.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 19: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.o.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 20: 0.01% selected (40,960 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 21: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.weight, self_attn.o.bias] [HIGH]
WARNING:__main__: - Block 22: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, self_attn.k.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 23: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 24: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.v.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 25: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.q.bias, norm3.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 26: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.o.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 27: 0.01% selected (24,064 / 351,394,304 params) | Selected: [cross_attn.k.bias, ffn.0.bias, self_attn.v.bias] [LOW]
WARNING:__main__: - Block 28: 0.01% selected (40,960 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 29: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.weight, self_attn.o.bias] [HIGH]
WARNING:__main__: - Block 30: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 31: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.v.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 32: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.q.bias, norm3.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 33: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.o.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 34: 0.01% selected (24,064 / 351,394,304 params) | Selected: [cross_attn.k.bias, ffn.0.bias, self_attn.v.bias] [LOW]
WARNING:__main__: - Block 35: 0.01% selected (40,960 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, modulation, self_attn.q.bias] [LOW]
WARNING:__main__: - Block 36: 7.46% selected (26,224,640 / 351,394,304 params) | Selected: [norm3.bias, self_attn.k.weight, self_attn.o.bias] [HIGH]
WARNING:__main__: - Block 37: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, self_attn.k.bias] [LOW]
WARNING:__main__: - Block 38: 7.46% selected (26,219,520 / 351,394,304 params) | Selected: [ffn.2.bias, self_attn.k.weight] [HIGH]
WARNING:__main__: - Block 39: 0.00% selected (10,240 / 351,394,304 params) | Selected: [cross_attn.norm_k.weight, cross_attn.v.bias] [LOW]

🎯 OVERLAP ANALYSIS (Parameter Contamination)

INFO:__main__:   ✅ ffid 1 vs ffid 2: OK! (0 overlap)
INFO:__main__:   ✅ ffid 1 vs ffid 3: OK! (0 overlap)
INFO:__main__:   ✅ ffid 1 vs ffid 4: OK! (0 overlap)
INFO:__main__:   ✅ ffid 1 vs ffid 5: OK! (0 overlap)
INFO:__main__:   ✅ ffid 2 vs ffid 3: OK! (0 overlap)
INFO:__main__:   ✅ ffid 2 vs ffid 4: OK! (0 overlap)
INFO:__main__:   ✅ ffid 2 vs ffid 5: OK! (0 overlap)
INFO:__main__:   ✅ ffid 3 vs ffid 4: OK! (0 overlap)
INFO:__main__:   ✅ ffid 3 vs ffid 5: OK! (0 overlap)
INFO:__main__:   ✅ ffid 4 vs ffid 5: OK! (0 overlap)

If anyone wants to tweak the selection algorithm (select_parameters_deterministic) to their liking, here is the test file I used: https://drive.google.com/file/d/1HjaVJXEHySwhKVCZmOYIGAEyzbKIHSYv/view?usp=sharing
```

## Notes

*   **Note:** Only tested for Wan 2.2 (trained the low_noise weights)
*   **Note2:** Doesnt work with bnb's adamw8bit, will result in an error (tested this with block swapping, maybe when everything is on GPU it's compatible):
    ```
    File "\site-packages\bitsandbytes\optim\optimizer.py", line 500, in update_step
    p.grad = p.grad.contiguous()
    ^^^^^^
    RuntimeError: attempting to assign a gradient with device type 'cuda' to a tensor with device type 'cpu'. Please
    ensure that the gradient and the tensor are on the same device
    ```
*   **Note3:** Tried to keep main repo file changes minimal, implemented core logic to: `utils\selective_finetune.py` and debugging file (there was a lot of need for it for end to end gradient flow debugging): `utils\training_debug`
*   **Note4:** Working with `--blocks_to_swap`, `--mixed_precision`, `--gradient_checkpointing` args
*   **Note5:** Code was 95% written by Claude Code

---

## Results:

### Baseline DiT sample for prompt character1

```
python wan_generate_video.py --task t2v-14B --video_length 1 --infer_steps 30 --prompt "character1 sitting in a modern restaurant at night, wearing elegant dress" --save_path C:/train/saves/test/ --output_type both --dit "C:/train/ckpts/wan22_t2v_14B_low_noise_bf16.safetensors" --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --attn_mode sdpa --seed 1234 --blocks_to_swap 24 --video_size 720 1280 --flow_shift 12 --seed 1234 --infer_steps 30 --video_length 1 --guidance_scale 3
```

**Sample:**

![baseline](https://github.com/user-attachments/assets/6d83e544-fb6c-48fc-884b-b7ce5db1b3af)


### Trained character1 with --ff 0.001 --ffid 1 for 50 epochs
>ideal starting point should be 0.01+, but wanted to test directly how training would go, only on mostly 1 important matrix for limit testing

**Full training command:**
```
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py --task t2v-A14B --dit C:/train/ckpts/wan22_t2v_14B_low_noise_bf16.safetensors --dataset_config C:/train/hunyuan_dataset.toml --sdpa --mixed_precision bf16 --optimizer_type adamW --learning_rate 1e-4 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --timestep_sampling shift --discrete_flow_shift 3.0 --max_train_epochs 1000 --save_every_n_epochs 5 --save_state --seed 1234 --output_dir C:/train/saves --output_name character1_512 --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --sample_prompts "C:/train/sample/random_prompt.txt" --sample_every_n_epochs 5 --logging_dir "C:/train/logs" --lr_warmup_steps "50" --lr_scheduler "linear" --preserve_distribution_shape --min_timestep "0" --max_timestep "1000" --preserve_distribution_shape --ff 0.01 --ffid 1 --blocks_to_swap 24
```

### character1.safetensors sample for prompt character1:

```
python wan_generate_video.py --task t2v-14B --video_length 1 --infer_steps 30 --prompt "character1 sitting in a modern restaurant at night, wearing elegant dress" --save_path C:/train/saves/test/ --output_type both --dit "C:/train/saves/character1_512-000040.safetensors" --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --attn_mode sdpa --seed 1234 --blocks_to_swap 24 --video_size 720 1280 --flow_shift 12 --seed 1234 --infer_steps 30 --video_length 1 --guidance_scale 3
```

**Sample:**




### Trained character2 with --ff 0.001 --ffid 2 for 50 epochs with "character1.safetensors" loaded in as baseline

**Full training command:**
```
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py --task t2v-A14B --dit C:/train/saves/character1_512-000040.safetensors --dataset_config C:/train/hunyuan_dataset.toml --sdpa --mixed_precision bf16 --optimizer_type adamW --learning_rate 1e-4 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --timestep_sampling shift --discrete_flow_shift 3.0 --max_train_epochs 1000 --save_every_n_epochs 5 --save_state --seed 1234 --output_dir C:/train/saves --output_name character1_2_512 --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --sample_prompts "C:/train/sample/random_prompt.txt" --sample_every_n_epochs 5 --logging_dir "C:/train/logs" --lr_warmup_steps "50" --lr_scheduler "linear" --preserve_distribution_shape --min_timestep "0" --max_timestep "1000" --preserve_distribution_shape --ff 0.001 --ffid 2 --blocks_to_swap 24
```

### character1_2.safetensors sample for prompt character2:

```
python wan_generate_video.py --task t2v-14B --video_length 1 --infer_steps 30 --prompt "character2 sitting in a modern restaurant at night, wearing elegant dress" --save_path C:/train/saves/test/ --output_type both --dit "C:/train/saves/character1_2_512-000040.safetensors" --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --attn_mode sdpa --seed 1234 --blocks_to_swap 24 --video_size 720 1280 --flow_shift 12 --seed 1234 --infer_steps 30 --video_length 1 --guidance_scale 3
```

**Sample:**

### character1_2.safetensors sample for prompt character1, to verify knowledge of her is also intact:

```
python wan_generate_video.py --task t2v-14B --video_length 1 --infer_steps 30 --prompt "character1 sitting in a modern restaurant at night, wearing elegant dress" --save_path C:/train/saves/test/ --output_type both --dit "C:/train/saves/character1_2_512-000040.safetensors" --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --attn_mode sdpa --seed 1234 --blocks_to_swap 24 --video_size 720 1280 --flow_shift 12 --seed 1234 --infer_steps 30 --video_length 1 --guidance_scale 3
```

**Sample:**

---

## Vram usage during training with --ff 0.001 --blocks_to_swap 24:

**Resources:**

![vram](https://github.com/user-attachments/assets/0f13884a-6dbd-499b-9b58-b89a3bed8eaf)


## Convergence in tensorboard --ff 0.001 (blue) vs 0.0001 (orange) (this was with algo v1.0 ):

**Tensorboard:**

![tensorboard](https://github.com/user-attachments/assets/80b5fa78-cc1f-4199-8c70-edbb70e41dff)


---

### Scaling and VRAM Estimates (`~14.3B` Model) - Gemini 2.5 Pro calcs

The table below provides VRAM estimates for training with the AdamW optimizer. It assumes `bfloat16` precision and includes a ~1.5 GB base VRAM overhead for the model weights and CUDA context.

| `--ff` Value | Trainable Parameters | % of Model | Est. GPU VRAM | Use Case                                               |
| :----------- | :------------------- | :--------- | :------------ | :----------------------------------------------------- |
| `0.001`      | ~26M                 | ~0.19%     | **~1.8 GB**   | Baseline for a simple character                        |
| `0.01`       | ~158M                | ~1.10%     | **~3.4 GB**   | Multiple characters and/or concepts                    |
| `0.05`       | ~734M                | ~5.14%     | **~10.3 GB**  | Deeper style transfer or highly complex concepts/characters. |
| `0.10`       | ~1.47B               | ~10.3%     | **~19.1 GB**  | Very heavy fine-tuning on a specific domain.           |
| `1.0`        | ~14.3B               | 100%       | **~173+ GB**  | Full fine-tuning (not using this script's features).   |

### VRAM Calculation Breakdown:

The required VRAM for training is estimated using the following formula for each trainable parameter in `bfloat16` with the AdamW optimizer:
*   **Parameter:** 2 bytes (`bfloat16`)
*   **Gradient:** 2 bytes (`bfloat16`)
*   **Optimizer States:** 8 bytes (2x `float32` states for momentum and variance)
*   **Total per Parameter:** `2 + 2 + 8 = 12 bytes`

*   **Example for `ff=0.01`:**
    *   **Base Overhead:** ~1.5 GB
    *   **Trainable Tensors:** `158M params * 12 bytes/param ≈ 1.9 GB`.
    *   **Total:** `1.5 GB + 1.9 GB ≈ 3.4 GB`.

*   **Example for `ff=0.05`:**
    *   **Base Overhead:** ~1.5 GB
    *   **Trainable Tensors:** `734M params * 12 bytes/param ≈ 8.8 GB`.
    *   **Total:** `1.5 GB + 8.8 GB ≈ 10.3 GB`.

*   **Full Fine-tuning (`ff=1.0`) with AdamW:**
    *   **Parameters, Gradients, Optimizer:** `14.3B params * 12 bytes/param ≈ 171.6 GB`.
    *   **Total:** `1.5 GB + 171.6 GB ≈ 173.1 GB`.

---

It was a fun journey with Claude Code, it took 5 complete restarts/redoing from scratch of trying to get to the correct approach for this to work, through 3 days. Was trying different methods like tensor slicing (hell), delta training like in loras, boolean masks, which all had different kinds of and more and more issues the deeper we went down this rabbit hole especially with block swapping (Claude Code was stuck in "fixing" loops no matter how I prompted it, and I dont have the technical background to assist that deeply, so had to redo), before ending up and settling with parameter proxies with hooks.

Then after the main framework has been settled, came a 1.5 day long back and forth debugging spree (darn 5 hour limits and context windows).

The deterministic selection algorithm is actually the work of prompting Claude 4 Sonnet, Gemini 2.5 Pro, Kimi v2, Chatgpt5 for what they think would be the best approach for this and then put all of these suggestions into CLAUDE.md. Claude ended up with what it thought would be "the best of all worlds" kind of approach taking a little bit from here and there. Then there was some refining and fixing that led to v2.0, mostly the work of ChatGPT5 and Grok4 (thought for 18 minutes) in turns and the final spread was Gemini 2.5 Pro.

It's actually pretty awesome that someone can just have an idea, and have it realized with these coding models, which will only get better from here on out. A glimpse into the future for sure. **_"What a time to be alive"_**

---

History:


V1.0's Catastrophic interference problem, hidden behind the illusion of fast convergence (among other issues)

The Bias Contamination Issue (was filling it's budget with "easy" small params, like biases)

V1.0 Selection Pattern:

>ffid 1: trains 40 blocks × bias parameters → character1 biases

>ffid 2: trains 40 blocks × bias parameters → character2 biases

```
# Problem: Biases are shared across all attention patterns
attention_output = softmax(Q@K.T + shared_bias) @ V
#                                 ↑ contaminated!
```
Why Bias Focused Filling Caused Character Mixing:

1. Shared Attention Bias: All attention heads use the same bias terms
    - Character1 training shifts attention towards character1 features
    - Character2 training shifts the SAME bias terms towards character2 features
    - Result: Attention becomes confused between both characters
2. Activation Distribution Pollution:
    - FFN biases control activation thresholds globally
    - Training character1 shifts activation patterns
    - Training character2 shifts the SAME patterns again
    - Result: Mixed activation patterns for both characters
3. Fast Convergence = Fast Contamination:
    - Bias parameters converge quickly → immediate character mixing (especially when training overlapping ones :|)
    - No parameter isolation between characters

So what V1.0 actually selected:
- ✅ Easy targets: cross_attn.k.bias (5,120 params)
- ✅ Easy targets: norm.weight (5,120 params)
- ✅ Easy targets: ffn.2.bias (5,120 params)
- ❌ Avoided: cross_attn.q.weight (26,214,400 params) - "too big!"
- ❌ Avoided: ffn.0.weight (70,778,880 params) - "too big!"


I believe there is still room for improvement with the algorithm but I'm happy so far. Sadly without tensor slicing (way above me), I think there is no way to achieve exact target param count for --ff fractions and perfect even distribution.
