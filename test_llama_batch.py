from vllm import LLM, SamplingParams

def run(batch_size):
    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        dtype="float16",
        max_model_len=2048,
        max_num_seqs=batch_size,
        block_size=1024,
        num_gpu_blocks_override=5,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
    )

    sampling_params = SamplingParams(temperature=0.0)

    # Batch for batch inference
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The largest mammal in the world is",
    ] * ((batch_size+1) // 2)

    print(f"Number of prompts (batch size): {len(prompts)}")
    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        print(f"[{i}] {output.outputs[0].text[:100]}")

if __name__ == "__main__":
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    for batch_size in batch_sizes:
        print(f"Running batch size: {batch_size}")
        run(batch_size)
        print("-" * 100)