# Full finetuning on deterministic parameter slices that are spread across evenly among blocks

## Control args

`--ff 0.0001` (how much params)
`--ffid 1` (where)

This allows full-finetuning to a targeted parameter space in the model, baking, without overwriting knowledge of other slices.

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
*   **Note5:** Code was written by Claude Code

---

## Results:

### Baseline DiT sample for prompt character1

```
python wan_generate_video.py --task t2v-14B --video_length 1 --infer_steps 30 --prompt "character1 sitting in a modern restaurant at night, wearing elegant dress" --save_path C:/train/saves/test/ --output_type both --dit "C:/train/ckpts/wan22_t2v_14B_low_noise_bf16.safetensors" --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --attn_mode sdpa --seed 1234 --blocks_to_swap 24 --video_size 720 1280 --flow_shift 12 --seed 1234 --infer_steps 30 --video_length 1 --guidance_scale 3
```

**Sample:**

![baseline](https://github.com/user-attachments/assets/6d83e544-fb6c-48fc-884b-b7ce5db1b3af)


### Trained character1 with --ff 0.001 --ffid 1 for 45 epochs

**Full training command:**
```
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py --task t2v-A14B --dit C:/train/ckpts/wan22_t2v_14B_low_noise_bf16.safetensors --dataset_config C:/train/hunyuan_dataset.toml --sdpa --mixed_precision bf16 --optimizer_type adamW --learning_rate 1e-4 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --timestep_sampling shift --discrete_flow_shift 3.0 --max_train_epochs 1000 --save_every_n_epochs 5 --save_state --seed 1234 --output_dir C:/train/saves --output_name character1_512 --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --sample_prompts "C:/train/sample/random_prompt.txt" --sample_every_n_epochs 5 --logging_dir "C:/train/logs" --lr_warmup_steps "50" --lr_scheduler "linear" --preserve_distribution_shape --min_timestep "0" --max_timestep "1000" --preserve_distribution_shape --ff 0.001 --ffid 1 --blocks_to_swap 24
```

### character1.safetensors sample for prompt character1:

```
python wan_generate_video.py --task t2v-14B --video_length 1 --infer_steps 30 --prompt "character1 sitting in a modern restaurant at night, wearing elegant dress" --save_path C:/train/saves/test/ --output_type both --dit "C:/train/saves/character1_512-000040.safetensors" --vae C:/train/ckpts/wan_2.1_vae.safetensors --t5 C:/train/ckpts/models_t5_umt5-xxl-enc-bf16.pth --attn_mode sdpa --seed 1234 --blocks_to_swap 24 --video_size 720 1280 --flow_shift 12 --seed 1234 --infer_steps 30 --video_length 1 --guidance_scale 3
```

**Sample:**




### Trained character2 with --ff 0.001 --ffid 2 for 40 epochs with "character1.safetensors" loaded in as baseline

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


## Convergence in tensorboard --ff 0.001 (blue) vs 0.0001 (orange):

**Tensorboard:**

![tensorboard](https://github.com/user-attachments/assets/80b5fa78-cc1f-4199-8c70-edbb70e41dff)


---

### Scaling Options (~14.9B Model, Full `bfloat16` Precision)

The `--ff` argument controls how many parameters are trained on the GPU. The full **27.8 GB model remains in CPU RAM (as per block swapping)**. The VRAM estimates below assume **all GPU tensors (parameters, gradients, and optimizer states) are in `bfloat16`**, plus a base overhead of ~1.5 GB.

| `--ff` Value | Trainable Parameters | % of Model | Est. GPU VRAM | Use Case                               |
| :----------- | :------------------- | :--------- | :------------ | :------------------------------------- |
| `0.00001`    | ~149k                | 0.001%     | **~1.5 GB**   | Very simple concept/object             |
| `0.0001`     | ~1.49M               | 0.01%      | **~1.6 GB**   | Simple characters                      |
| `0.0005`     | ~7.45M               | 0.05%      | **~1.7 GB**   | More complex characters/styles         |
| `0.001`      | ~14.9M               | 0.1%       | **~1.8 GB**   | Multiple concepts, more detailed style - Used in my tests |
| `1.0`        | ~14.9B               | 100%       | **~115+ GB**  | Full fine-tuning (does not use this script's memory-saving features) |

### Calculation Breakdown:

*   **Low `ff` Values (`ff=0.001`):**
    *   **Base Cost:** ~1.5 GB (CUDA context, activations, etc.)
    *   **Trainable Params:** `14.9M params * 8 bytes/param (all bf16) ≈ 119 MB`.
    *   **Total:** `1.5 GB + 119 MB ≈ 1.62 GB`. The **~1.8 GB** estimate provides a safe buffer.

*   **Full Fine-tuning (`ff=1.0`) with full `bf16`:**
    *   **Model:** 27.8 GB (`bfloat16`)
    *   **Gradients:** 27.8 GB (`bfloat16`)
    *   **Optimizer States:** `14.9B params * 4 bytes/param (2x bf16) ≈ 59.6 GB`.
    *   **Total:** `27.8 + 27.8 + 59.6 ≈ 115.2 GB`. The **~115+ GB** estimation for this scenario.

---

It was a fun journey with Claude Code, it took 5 complete restarts/redoing from scratch of trying to get to the correct approach for this to work, through 3 days. Was trying different methods like tensor slicing, delta training like in loras, boolean masks, which all had different kinds of and more and more issues the deeper we went down this rabbit hole especially with block swapping (Claude Code was stuck in "fixing" loops no matter how I prompted it, and I dont have the technical background to assist that deeply, so had to redo), before ending up and settling with parameter proxies with hooks. 

The deterministic selection algorithm is actually the work of prompting Claude 4 Sonnet, Gemini 2.5 Pro, Kimi v2, Chatgpt5 (total fail) for what they think would be the best approach for this and then put all of these suggestions into CLAUDE.md. Claude ended up with what it thought would be "the best of all worlds" kind of approach taking a little bit from here and there.

Then after the main framework has been settled, came a 1.5 day long back and forth debugging spree (darn 5 hour limits and context windows).

It's actually pretty awesome that someone can just have an idea, and have it realized with these coding models, which will only get better from here on out. A glimpse into the future for sure. **"What a time to be alive"**

