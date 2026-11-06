# Running disensa on INDACO

INDACO is the University of Milan HPC cluster. There, models are served by
[vLLM](https://docs.vllm.ai) on a GPU node instead of Ollama, and sweeps run as SLURM job
arrays. **Untested until INDACO access works**: everything marked *CHECK WITH INDACO* is a
placeholder.

## To check with INDACO

- [ ] the GPU partition name, and the `--gres` syntax for one GPU
- [ ] GPU memory: an 8B model in 16-bit needs ~16 GB for the weights plus room for the KV
      cache; a 70B model needs several GPUs (`vllm serve ... --tensor-parallel-size N`)
- [ ] the time limit of the partition (`--time`)
- [ ] the scratch filesystem path (`$SCRATCH`) and its quota
- [ ] module names (CUDA, Python), and whether compute nodes can download from Hugging Face
      (if not, download the weights on a login node into `$HF_HOME` first)

## Model strings

vLLM speaks the OpenAI request format, but in a disensa config a vLLM model is written
`hosted_vllm/<model>` with `api_base: http://localhost:8000/v1`, **not** `openai/<model>`:
LiteLLM's `openai/` connector refuses to send a request without an OpenAI API key, so every
call would fail (disensa stops such a config). `<model>` is the name given to `vllm serve`.
See `configs/indaco_example.yaml`.

```yaml
models:
  - name: hosted_vllm/meta-llama/Llama-3.1-8B-Instruct
    api_base: http://localhost:8000/v1
max_concurrency: 48   # vLLM batches requests: raise this to ~32–64
```

`max_concurrency` is the number of requests disensa keeps in flight. Ollama on a Mac queues
them anyway, so 4 is plenty there; vLLM batches them on the GPU, so 32–64 keeps it busy.

## Submitting a sweep

```bash
mkdir -p logs
disensa sweep configs/sweeps/topics_core.yaml --dry-run      # how many runs and LLM calls
sbatch --array=1-4 scripts/indaco/sweep.slurm configs/sweeps/topics_core.yaml
```

Task *i* of the array runs shard *i/N* of the sweep (1-based; `--array=0-3` would be wrong),
each with its own vLLM server and its own cache folder under `$SCRATCH/disensa-cache/`.
Runs that already finished are skipped, so resubmitting the same array after a timeout
continues where it stopped. All tasks write into `runs/<sweep name>/`.

## Getting the runs back to the Mac

```bash
# on the Mac, from the disensa folder
rsync -av <user>@<indaco login host>:<path to disensa>/runs/topics_core runs/
disensa app
```

The lab lists every folder in `runs/`, including sweep folders, so the copied runs appear
under *Previous runs*, grouped by sweep.

## Never mix Ollama and vLLM runs in one comparison

The same model name is not the same model: Ollama usually serves 4-bit quantized weights,
vLLM the full ones, and the two servers differ in other details too. Every run records what
served it in `meta.json` (`serving`: Ollama's digest and quantization, or the model ids
vLLM reported), so the two can always be told apart. Compare Ollama runs with Ollama runs
and vLLM runs with vLLM runs.
