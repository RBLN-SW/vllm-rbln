from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="float16",
    max_model_len=2048,
    max_num_seqs=1,
    block_size=1024,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
)
print(llm.generate(["Hello"], SamplingParams(temperature=0.0))[0].outputs[0].text)

