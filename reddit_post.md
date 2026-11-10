Getting incredible results with a single RTX 3090ti and 128gb of ram. Deepseek-V4.1-Flash-1017 (35 tok/s decode) and Qwen3.8-Flash-Next (130 tok/s decode)
Resources
I'm sharing this here because this is working way better than it used to a few months ago with older models. I see people here posting slower speeds with much better hardware than mine and hopefully this information helps.

I used gemini-flash to help me setup everything and I asked it to write up the sequence of steps to get me here - happy to provide more details if there's anything missing.

Since I use gentoo some tools/commands might be unfamiliar to you.

My 6 year old hardware:
OS & CPU Gentoo Linux running on an AMD Ryzen Threadripper 3990X (64 cores, 128 threads).

System Memory: 128 GB DDR4 RAM.

GPU: 1 NVIDIA GeForce RTX 3090 Ti (24 GB GDDR6X VRAM, PCIe Gen 4 x16).

Primary Storage: 4 TB SSD NVMe.

Models Used, Download Sources, and Measured Speeds
Model	Architecture	Quantization / Files	Download Source	Real Measured Speed
DeepSeek-V4.1-Flash-1017	552B MoE (40 layers, 384 experts/layer, ~39B active)	XQ / GSQ V2 + rank-1017-085.bin + FP4 KV + Engram NVMe tables	IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored	~35 tok/s decode (with MTP/DSpark active)
Qwen3.8-Flash-Next	125B MoE + 51B N-gram (GDN + QSA hybrid, ~3.5B active)	pack-qfn45 (4-bit intermediate, DeltaNet linear KV)	IndrasMirror/Qwen3.8-Flash-Next-Swift-1.5-XQ-Uncensored	~138–140 tok/s decode
IMPORTANT: NVIDIA GPU Unlock

Consumer Ampere cards (RTX 3000 series: GA102, GA104) feature artificial firmware throttles on FP16/BF16 matrix instruction issue rates compared to their enterprise workstation counterparts. The open-source gpu-unlock kernel module lifts this throttle prior to driver attachment.

1. Compile and Install the Kernel Module

As root:

git clone https://github.com/Zagitta/gpu-unlock.git /usr/src/gpu-unlock
cd /usr/src/gpu-unlock
make -j$(nproc)
make install
2. Pass the Unlock Flag to the Kernel Command Line

Add gpu_unlock.targets=all to your bootloader kernel parameters (e.g. in /etc/default/grub under GRUB_CMDLINE_LINUX_DEFAULT, or your direct EFISTUB / systemd-boot entry):

gpu_unlock.targets=all
Regenerate your boot configuration and rebuild your initramfs so the module is included and loads before the NVIDIA kernel modules:

# If using GRUB:
grub-mkconfig -o /boot/grub/grub.cfg

# If using dracut:
dracut -f --regenerate-all
3. NVIDIA Driver Options & Persistence

Configure NVIDIA kernel driver parameters in /etc/modprobe.d/nvidia-perf.conf (as root):

cat << 'EOF' > /etc/modprobe.d/nvidia-perf.conf
options nvidia NVreg_PreserveVideoMemoryAllocations=1
options nvidia NVreg_EnableStreamMemOPs=1
EOF
After rebooting, verify that the unlock executed:

dmesg | grep -i gpu_unlock
Enable persistence mode and set the GPU to its maximum factory power ceiling:

nvidia-smi -pm 1
nvidia-smi -pl $(nvidia-smi -q -d POWER | grep "Max Power Limit" | head -n1 | awk '{print $5}' | cut -d. -f1)
Step 2: System Dependencies & Tooling

Install build dependencies and runtime libraries:

emerge sys-devel/gcc:14 


# If GCC 14 is not selected, set it to this version
gcc-config -l 
# then replace <N> with the number from  gcc-config -l): 
# gcc-config <N> && source /etc/profile

# this won't compile with GCC > 14
emerge dev-util/nvidia-cuda-toolkit

# for compleness, just in case

emerge --ask --verbose \
    dev-build/cmake \
    dev-build/ninja \
    dev-vcs/git \
    net-misc/curl \
    app-misc/jq \
    sys-libs/liburing \
    app-arch/snappy \
    dev-lang/python
Prepare the directory structure under your user environment:

mkdir -p "$HOME/.local/bin"
mkdir -p "$HOME/.local/llm-tools"
mkdir -p "$HOME/.quetza-data/conductor"
mkdir -p "$HOME/models/deepseek-v41"
mkdir -p "$HOME/models/qwen-flash"

# Dedicated Python environment for tool scripts & Hugging Face downloads
python3 -m venv "$HOME/.local/llm-tools/venv_hf"
source "$HOME/.local/llm-tools/venv_hf/bin/activate"
pip install --upgrade pip
pip install "huggingface_hub[cli]" torch safetensors numpy
Step 3: Build the Xolotl Inference Engine

Xolotl integrates the multi-tier memory management, NVMe Engram streaming, and DSpark speculative decoding logic needed to run large MoE models on a single GPU.

cd "$HOME/.local/llm-tools"
git clone https://github.com/Indras-Mirror/Xolotl.git
cd Xolotl

# Fetch pinned llama.cpp dependencies and apply patches
bash tools/xolotl/fetch-llama.sh

# Configure and compile for Ampere (sm_86)
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_GGML_CUDA=ON \
    -DSTRATA_MMQ_KQUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
    -DSTRATA_DSV41_FQ_OP=ON \
    -DSTRATA_DSV41_HC_MIX=ON

ninja -C build

# Link the build directory so wrapper scripts resolve ds41_generate
ln -sfn "$PWD/build" "$PWD/build-xolotl"
Step 4: Model Downloads & Engram Extraction

1. DeepSeek-V4.1-Flash (552B MoE)

Bash

source "$HOME/.local/llm-tools/venv_hf/bin/activate"
cd "$HOME/models/deepseek-v41"

# Download model weights, draft sidecar, and expert rank index
hf download IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored \
    --include "v2/*" "tokenizer/*" "V41-MTP-Vision-1017*" \
    --local-dir .

# Download original FP8 Engram table shards
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    model-00047-of-00048.safetensors \
    model-00048-of-00048.safetensors \
    model.safetensors.index.json config.json \
    --local-dir ./FP8-engram

# Extract Engram runtime binary tables onto NVMe
cd "$HOME/.local/llm-tools/Xolotl"
python3 tools/v41/extract_engram.py \
    --input "$HOME/models/deepseek-v41/FP8-engram" \
    --output "$HOME/models/deepseek-v41/engram_rt"
2. Qwen3.8-Flash-Next (125B MoE)

cd "$HOME/models/qwen-flash"
hf download Qwen/Qwen3.8-Flash-Next-GGUF \
    --include "pack-qfn45/*" \
    --local-dir .
Step 5: MTP Speculative Decoding Setup

MTP (Multi-Token Prediction) on DeepSeek-V4.1-Flash is driven by the DSpark sidecar engine:

Draft Engine: Uses V41-MTP-Vision-1017.Q2_K.gguf as the proposal head.

Expert Pruning: Restricts verification sweeps to the top 64 active draft experts (dspark_expert_counts.txt:64), preventing PCIe bus saturation.

On-the-fly Re-quantization (STRATA_DENSE_REQUANT=q6_K): Re-quantizes the dense attention and output heads from Q8_0 to Q6_K at launch. This reclaims 1.6 GiB of VRAM, fitting the speculative draft cache alongside active experts.

PCIe Batching (--tier-batch-pcie 0.25): Batches DMA transfers between the 80 GiB pinned host RAM arena and the GPU.

1. Expand Process Memory Boundary in the Engine Wrapper

Allow the systemd user scope to hold the full 80 GiB arena plus runtime buffers:

Bash

sed -i 's/MemoryMax=84G/MemoryMax=110G/g' "$HOME/.local/llm-tools/Xolotl/tools/v41/serve/ds41-engine.sh"
2. Generate Base Configuration: $HOME/models/deepseek-v41/strata-v41.json

cat << EOF > "$HOME/models/deepseek-v41/strata-v41.json"
{
  "exe": "$HOME/.local/llm-tools/Xolotl/tools/v41/serve/ds41-engine.sh",
  "args": [
    "-m",
    "$HOME/models/deepseek-v41/v2/V41-XQ-1017.gguf",
    "--backend",
    "cuda",
    "--experts",
    "gpu",
    "--engram",
    "$HOME/models/deepseek-v41/engram_rt",
    "--rank",
    "$HOME/models/deepseek-v41/v2/rank-1017-085.bin",
    "--ctx",
    "131072",
    "--stop",
    "1",
    "--arena-adapt",
    "--kv-fp4",
    "--route-bias",
    "0.5",
    "--route-bias-file",
    "0.3",
    "--arena-gib",
    "80",
    "--prompt-batch",
    "12288",
    "--tier-batch-pcie",
    "0.25"
  ],
  "cwd": "$HOME/.local/llm-tools/Xolotl",
  "tokenizer": "$HOME/models/deepseek-v41/tokenizer",
  "model_name": "deepseek-v4.1",
  "aliases": "deepseek-v41,deepseek/deepseek-v4.1-flash",
  "log": "$HOME/models/deepseek-v41/xolotl-v41.log",
  "port": 8000,
  "gpu": 0,
  "sampling": {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
  },
  "api_monitor": true,
  "fit_max_tokens": true,
  "reasoning_budget_tokens": 4096,
  "env": {
    "STRATA_BUILD_DIR": "$HOME/.local/llm-tools/Xolotl/build",
    "STRATA_SERVE_DSPARK": "$HOME/models/deepseek-v41/V41-MTP-Vision-1017.Q2_K.gguf",
    "STRATA_ATTN_F16_GATHER": "1",
    "STRATA_NO_SCOPE": "1",
    "STRATA_SERVE_PASS_BIAS": "0",
    "STRATA_DENSE_REQUANT": "q6_K",
    "STRATA_SERVE_LOOKUP": "14",
    "STRATA_PRESTAGE_GIB": "1.5"
  }
}
EOF
Step 6: Parameterized Launch Scripts

1. DeepSeek-V4.1 Launcher: $HOME/.local/bin/start-deepseek-custom.sh

cat << 'EOF' > "$HOME/.local/bin/start-deepseek-custom.sh"
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$HOME/models/deepseek-v41"
XOLOTL_DIR="$HOME/.local/llm-tools/Xolotl"
BASE_CONFIG="$MODEL_DIR/strata-v41.json"
RUN_CONFIG="$MODEL_DIR/strata-v41-run.json"

CTX="131072"
PORT="8000"
TEMP="0.1"
THINK_BUDGET="4096"
ARENA_GIB="80"
PASS_BIAS="0"
PROMPT_BATCH="12288"

usage() {
  cat << USAGE
Usage: $(basename "$0") [options]

Options:
  -c, --ctx INT           Context length in tokens (default: 131072; up to 1048576)
  -p, --port INT          Port to bind (default: 8000)
  -t, --temp FLOAT        Sampling temperature (default: 0.1)
  -b, --thinking INT      Reasoning budget tokens (default: 4096)
  -a, --arena INT         Pinned RAM expert arena in GiB (default: 80)
  --bias 0|1              Routing bias: 0=exact, 1=fast biased (default: 0)
  --batch INT             Prompt batch size (default: 12288; auto-clamps to 8192 for >=512k)
  -h, --help              Show help message
USAGE
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--ctx)       CTX="$2"; shift 2 ;;
    -p|--port)      PORT="$2"; shift 2 ;;
    -t|--temp)      TEMP="$2"; shift 2 ;;
    -b|--thinking)  THINK_BUDGET="$2"; shift 2 ;;
    -a|--arena)     ARENA_GIB="$2"; shift 2 ;;
    --bias)         PASS_BIAS="$2"; shift 2 ;;
    --batch)        PROMPT_BATCH="$2"; shift 2 ;;
    -h|--help)      usage ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [ "$CTX" -ge 524288 ] && [ "$PROMPT_BATCH" -gt 8192 ]; then
  echo "[-] Large context specified ($CTX tokens). Clamping --prompt-batch to 8192 to prevent graph OOM."
  PROMPT_BATCH=8192
fi

python3 - << PY
import json

with open("$BASE_CONFIG", "r") as f:
    cfg = json.load(f)

cfg["port"] = int("$PORT")
cfg["max_context"] = int("$CTX")
cfg["reasoning_budget_tokens"] = int("$THINK_BUDGET")
cfg["sampling"]["temperature"] = float("$TEMP")

args = cfg.get("args", [])
def set_arg(flag, val):
    for i in range(len(args) - 1):
        if args[i] == flag:
            args[i+1] = str(val)
            return
    args.extend([flag, str(val)])

set_arg("--ctx", "$CTX")
set_arg("--arena-gib", "$ARENA_GIB")
set_arg("--prompt-batch", "$PROMPT_BATCH")
cfg["args"] = args

env = cfg.setdefault("env", {})
env["STRATA_SERVE_PASS_BIAS"] = "$PASS_BIAS"
env["STRATA_PRESTAGE_GIB"] = "1.0" if int("$CTX") >= 524288 else "1.5"

with open("$RUN_CONFIG", "w") as f:
    json.dump(cfg, f, indent=2)
PY

source "$HOME/.local/llm-tools/venv_hf/bin/activate"
cd "$XOLOTL_DIR"

echo "==> Launching DeepSeek-V4.1-Flash (552B MoE)..."
echo "    Context        : $CTX tokens"
echo "    Port           : $PORT"
echo "    Arena RAM      : $ARENA_GIB GiB"
echo "    Prompt Batch   : $PROMPT_BATCH"

exec python3 tools/v41/serve/serve_ds41.py \
  --engine strata \
  --config "$RUN_CONFIG" \
  --port "$PORT" \
  --host 0.0.0.0
EOF
chmod +x "$HOME/.local/bin/start-deepseek-custom.sh"
2. Qwen3.8-Flash-Next Launcher: $HOME/.local/bin/start-qwen.sh

cat << 'EOF' > "$HOME/.local/bin/start-qwen.sh"
#!/usr/bin/env bash
set -euo pipefail

XOLOTL_DIR="$HOME/.local/llm-tools/Xolotl"
MODEL_DIR="$HOME/models/qwen-flash/pack-qfn45"
PORT="8099"

source "$HOME/.local/llm-tools/venv_hf/bin/activate"
cd "$XOLOTL_DIR"

echo "==> Launching Qwen3.8-Flash-Next (125B MoE)..."
exec python3 -m serve.server \
  --engine strata \
  --config "$MODEL_DIR/strata-qwen.json" \
  --port "$PORT" \
  --host 0.0.0.0
EOF
chmod +x "$HOME/.local/bin/start-qwen.sh"
Step 7: Run

# Start DeepSeek-V4.1-Flash on :8000 (default 128k context, 1M context work just fine too)
$HOME/.local/bin/start-deepseek-custom.sh

# Or start Qwen3.8-Flash-Next on :8099
$HOME/.local/bin/start-qwen.sh