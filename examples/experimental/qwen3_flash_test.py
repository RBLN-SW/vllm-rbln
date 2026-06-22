# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E501

from vllm import LLM, SamplingParams

prompts = [
    """
Rebellions, SK Telecom, and DOCOMO Innovations Partner to Accelerate Next-Gen AI Infrastructure
News
Apr 24, 2025
[Seoul, April 24, 2025] Rebellions today announced the signing of a strategic MOU with SK Telecom and DOCOMO Innovations (DII), a subsidiary of Japan’s leading telecom provider NTT DOCOMO. The agreement lays the foundation for joint development and validation of cutting-edge AI acceleration technologies.

Under this collaboration, the three companies will focus on evaluating Rebellions’ ATOM-based NPU servers within SK Telecom’s NPU farm. Building on the success of this initiative, the partners plan to expand validation efforts to include a broader range of Rebellions’ product portfolio.

Rebellions will bring its high-performance AI chips, proven system reliability, and software optimization expertise to the table. SK Telecom will provide its NPU farm and AI infrastructure capabilities, while DII will conduct technical evaluations and help facilitate discussions on potential paths to commercialization. This partnership is expected to combine the core strengths of each company to help accelerate the growth of the global AI hardware ecosystem.

Rebellions recently established a Japanese subsidiary, further accelerating its global expansion. This agreement marks a significant step in the company’s ambition to become a key player in the global AI infrastructure market.

“Rebellions is committed to proving the real-world stability and performance of our technology in live infrastructure environments,” said Jinwook Oh, CTO of Rebellions. “Collaboration between a hardware provider, an infrastructure leader, and actual users is crucial – and together with SK Telecom and DII, we’re building meaningful progress toward the future of AI.”

Yoshikazu Akinaga, CEO of DOCOMO Innovations, added: “At DOCOMO Innovations, we are dedicated to driving forward innovation in practical AI solutions by working closely with global technology leaders. This partnership—bringing together Rebellions’ advanced semiconductor technology and SK Telecom’s infrastructure capabilities—will allow us to explore and assess the potential of scalable, sustainable AI systems, while maintaining a technology-agnostic approach to ensure optimal solutions for future applications.”

Sangmin Lee, Vice President of Growth Business Development Office at SK Telecom, commented: “SK Telecom delivers world-class, AI-optimized cloud services. This collaboration provides an opportunity to demonstrate our NPU cloud technology, which integrates a wide range of AI data center solutions. We are committed to contributing to the success of next-generation AI infrastructure services—powered not just by GPUs, but also by NPUs.”
""",
    """
Rebellions Partners on Strategic Collaboration Initiative to Advance Global AI Data Center Ecosystem
News
Mar 04, 2025
[Seoul, March 4, 2025] Rebellions, a pioneering AI chip company, today announced a strategic partnership with Penguin Solutions and SK Telecom at the Mobile World Congress (MWC) 2025 in Barcelona, Spain, taking a significant step toward building a global AI data center ecosystem.

The collaboration aims to establish technical foundations and business capabilities for large-scale data center operations. By combining the unique strengths and experiences of the Rebellions, Penguin Solutions and SK Telecom, the companies will pursue strategic development around AI inference and software stack delivery in the AI data center sector.

Within the planned collaboration, Rebellions brings its portfolio of energy-efficient AI accelerators optimized for Generative AI workloads. Penguin Solutions, a premiere AI infrastructure expert with more than 85,000 deployed GPUs under management, brings deep AI infrastructure expertise to the partnership. SK Telecom, actively accelerating its AI infrastructure business with key software elements and major investments in AI infra companies around the world, completes the strategic alliance.

The three companies will collaborate on multiple fronts, including:

Developing AI infrastructure solutions and creating testing environments for enterprise clients
Joint development of AI data center management solutions by integrating Rebellions’ AI accelerators while supporting both GPU and NPU environments
Leveraging each party’s technical expertise for software development specialized in AI data center infrastructure
“Since ‘DeepSeek’, efficient operations have emerged as a key concept of an AI business, making energy efficiency and cost of ownership critical evaluation criteria for customers,” said Sunghyun Park, CEO of Rebellions. “This partnership represents a crucial first step in establishing an efficient AI data center ecosystem by bringing together companies with diverse technological expertise.”

Mark Seamans, Penguin Solutions’ Vice President of Global Marketing, stated “With this partnership, our deep expertise in HPC and AI cluster management will now extend beyond GPU infrastructure to NPU environments. We’re committed to providing state-of-the-art AI infrastructure that meets the diverse needs of customers, solves for complexity, and accelerates business outcomes in the rapidly-growing global AI market.”

Through this strategic partnership, Rebellions, Penguin Solutions, and SK Telecom aim to identify and expand new business opportunities in the global AI data center market, positioning themselves at the forefront of AI infrastructure innovation.
""",
]

qwen3_0_6_model_id = "Qwen/Qwen3-0.6B"
qwen3_1_7_model_id = "Qwen/Qwen3-1.7B"
qwen3_4_model_id = "Qwen/Qwen3-4B"
qwen3_8_model_id = "Qwen/Qwen3-8B"
qwen3_30_moe_model_id = "Qwen/Qwen3-30B-A3B"
qwen1_5_moe_model_id = "Qwen/Qwen1.5-MoE-A2.7B"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
llm = LLM(
    model=qwen3_4_model_id,
    max_model_len=16 * 128,
    block_size=128,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
    max_num_seqs=5,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
