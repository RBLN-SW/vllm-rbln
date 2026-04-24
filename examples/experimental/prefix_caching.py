# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict
from time import perf_counter

from vllm import LLM, EngineArgs, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.metrics.reader import Counter, Metric

# Common prefix.
prefix = """
Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
    While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door.
“’Tis some visitor,” I muttered, “tapping at my chamber door—
            Only this and nothing more.”

    Ah, distinctly I remember it was in the bleak December;
And each separate dying ember wrought its ghost upon the floor.
    Eagerly I wished the morrow;—vainly I had sought to borrow
    From my books surcease of sorrow—sorrow for the lost Lenore—
For the rare and radiant maiden whom the angels name Lenore—
            Nameless here for evermore.

    And the silken, sad, uncertain rustling of each purple curtain
Thrilled me—filled me with fantastic terrors never felt before;
    So that now, to still the beating of my heart, I stood repeating
    “’Tis some visitor entreating entrance at my chamber door—
Some late visitor entreating entrance at my chamber door;—
            This it is and nothing more.”

    Presently my soul grew stronger; hesitating then no longer,
“Sir,” said I, “or Madam, truly your forgiveness I implore;
    But the fact is I was napping, and so gently you came rapping,
    And so faintly you came tapping, tapping at my chamber door,
That I scarce was sure I heard you”—here I opened wide the door;—
            Darkness there and nothing more.

    Deep into that darkness peering, long I stood there wondering, fearing,
Doubting, dreaming dreams no mortal ever dared to dream before;
    But the silence was unbroken, and the stillness gave no token,
    And the only word there spoken was the whispered word, “Lenore?”
This I whispered, and an echo murmured back the word, “Lenore!”—
            Merely this and nothing more.

    Back into the chamber turning, all my soul within me burning,
Soon again I heard a tapping somewhat louder than before.
    “Surely,” said I, “surely that is something at my window lattice;
      Let me see, then, what thereat is, and this mystery explore—
Let my heart be still a moment and this mystery explore;—
            ’Tis the wind and nothing more!”

    Open here I flung the shutter, when, with many a flirt and flutter,
In there stepped a stately Raven of the saintly days of yore;
    Not the least obeisance made he; not a minute stopped or stayed he;
    But, with mien of lord or lady, perched above my chamber door—
Perched upon a bust of Pallas just above my chamber door—
            Perched, and sat, and nothing more.

Then this ebony bird beguiling my sad fancy into smiling,
By the grave and stern decorum of the countenance it wore,
“Though thy crest be shorn and shaven, thou,” I said, “art sure no craven,
Ghastly grim and ancient Raven wandering from the Nightly shore—
Tell me what thy lordly name is on the Night’s Plutonian shore!”
            Quoth the Raven “Nevermore.”

    Much I marvelled this ungainly fowl to hear discourse so plainly,
Though its answer little meaning—little relevancy bore;
    For we cannot help agreeing that no living human being
    Ever yet was blessed with seeing bird above his chamber door—
Bird or beast upon the sculptured bust above his chamber door,
            With such name as “Nevermore.”

    But the Raven, sitting lonely on the placid bust, spoke only
That one word, as if his soul in that one word he did outpour.
    Nothing farther then he uttered—not a feather then he fluttered—
    Till I scarcely more than muttered “Other friends have flown before—
On the morrow he will leave me, as my Hopes have flown before.”
            Then the bird said “Nevermore.”

    Startled at the stillness broken by reply so aptly spoken,
“Doubtless,” said I, “what it utters is its only stock and store
    Caught from some unhappy master whom unmerciful Disaster
    Followed fast and followed faster till his songs one burden bore—
Till the dirges of his Hope that melancholy burden bore
            Of ‘Never—nevermore’.”

    But the Raven still beguiling all my fancy into smiling,
Straight I wheeled a cushioned seat in front of bird, and bust and door;
    Then, upon the velvet sinking, I betook myself to linking
    Fancy unto fancy, thinking what this ominous bird of yore—
What this grim, ungainly, ghastly, gaunt, and ominous bird of yore
            Meant in croaking “Nevermore.”

    This I sat engaged in guessing, but no syllable expressing
To the fowl whose fiery eyes now burned into my bosom’s core;
    This and more I sat divining, with my head at ease reclining
    On the cushion’s velvet lining that the lamp-light gloated o’er,
But whose velvet-violet lining with the lamp-light gloating o’er,
            She shall press, ah, nevermore!

    Then, methought, the air grew denser, perfumed from an unseen censer
Swung by Seraphim whose foot-falls tinkled on the tufted floor.
    “Wretch,” I cried, “thy God hath lent thee—by these angels he hath sent thee
    Respite—respite and nepenthe from thy memories of Lenore;
Quaff, oh quaff this kind nepenthe and forget this lost Lenore!”
            Quoth the Raven “Nevermore.”

    “Prophet!” said I, “thing of evil!—prophet still, if bird or devil!—
Whether Tempter sent, or whether tempest tossed thee here ashore,
    Desolate yet all undaunted, on this desert land enchanted—
    On this home by Horror haunted—tell me truly, I implore—
Is there—is there balm in Gilead?—tell me—tell me, I implore!”
            Quoth the Raven “Nevermore.”

    “Prophet!” said I, “thing of evil!—prophet still, if bird or devil!
By that Heaven that bends above us—by that God we both adore—
    Tell this soul with sorrow laden if, within the distant Aidenn,
    It shall clasp a sainted maiden whom the angels name Lenore—
Clasp a rare and radiant maiden whom the angels name Lenore.”
            Quoth the Raven “Nevermore.”

    “Be that word our sign of parting, bird or fiend!” I shrieked, upstarting—
“Get thee back into the tempest and the Night’s Plutonian shore!
    Leave no black plume as a token of that lie thy soul hath spoken!
    Leave my loneliness unbroken!—quit the bust above my door!
Take thy beak from out my heart, and take thy form from off my door!”
            Quoth the Raven “Nevermore.”

    And the Raven, never flitting, still is sitting, still is sitting
On the pallid bust of Pallas just above my chamber door;
    And his eyes have all the seeming of a demon’s that is dreaming,
    And the lamp-light o’er him streaming throws his shadow on the floor;
And my soul from out that shadow that lies floating on the floor
            Shall be lifted—nevermore!
"""

# Sample prompts.
prompts = [
    prefix
    + """
    Question: Who wrote this Poem and what's the title?
    Answer:
    """,
    prefix
    + """
    Question: What is the main theme of this poem?
    Answer:
    """,
    prefix
    + """
    Question: What is the function of the raven as a symbol, and how does its meaning shift during the poem?
    Answer:
    """,  # noqa: E501
    prefix
    + """
    Question: Who is Lenore and what role does she play in the poem?
    Answer:
    """,
]

sampling_params = SamplingParams(temperature=0.0)


def timed_generate(llm: LLM, prompts, sampling_params):
    start = perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_s = perf_counter() - start
    return outputs, elapsed_s


def get_counter_value(metrics: list[Metric], name: str) -> int:
    return sum(
        metric.value
        for metric in metrics
        if isinstance(metric, Counter) and metric.name == name
    )


def get_prompt_without_prefix(prompt):
    if prompt.startswith(prefix):
        return "... " + prompt[len(prefix) :].strip()
    return prompt


def main():
    args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_num_seqs=2,
        max_model_len=2048,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        num_gpu_blocks_override=32,
    )
    # Can't directly pass this value due to validation
    args.block_size = 1024  # type: ignore

    # Create an LLM without prefix caching as a baseline.
    args.enable_prefix_caching = False
    regular_llm = LLM(**asdict(args))

    print("Results without `enable_prefix_caching`")

    outputs, regular_generate_time_s = timed_generate(
        regular_llm, prompts, sampling_params
    )
    print(f"regular generate() execution time: {regular_generate_time_s:.3f} seconds")

    regular_generated_texts = []
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = get_prompt_without_prefix(output.prompt)
        generated_text = output.outputs[0].text
        regular_generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Destroy the LLM object and free up the GPU memory.
    del regular_llm
    try:
        cleanup_dist_env_and_memory()
    except RuntimeError as e:
        # ignore error when not using torch-rbln
        if "Cannot access accelerator device when none is available" not in str(e):
            raise

    # Create an LLM with prefix caching enabled.
    args.enable_prefix_caching = True
    prefix_cached_llm = LLM(**asdict(args))

    # Warmup so that the shared prompt's KV cache is computed.
    prefix_cached_llm.generate(prompts[0], sampling_params)

    metrics_before = prefix_cached_llm.get_metrics()
    prefix_hits_before = get_counter_value(metrics_before, "vllm:prefix_cache_hits")
    prefix_queries_before = get_counter_value(
        metrics_before, "vllm:prefix_cache_queries"
    )

    # Generate with prefix caching.
    outputs, cached_generate_time_s = timed_generate(
        prefix_cached_llm, prompts, sampling_params
    )
    print(f"cached generate() execution time: {cached_generate_time_s:.3f} seconds")

    metrics_after = prefix_cached_llm.get_metrics()
    prefix_hits_after = get_counter_value(metrics_after, "vllm:prefix_cache_hits")
    prefix_queries_after = get_counter_value(metrics_after, "vllm:prefix_cache_queries")
    prefix_hits = prefix_hits_after - prefix_hits_before
    prefix_queries = prefix_queries_after - prefix_queries_before

    print("Results with `enable_prefix_caching`")

    cached_generated_texts = []
    # Print the outputs. You should see the same outputs as before.
    print("-" * 50)
    for output in outputs:
        prompt = get_prompt_without_prefix(output.prompt)
        generated_text = output.outputs[0].text
        cached_generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Compare the results and display the speedup
    generated_same = all(
        [
            regular_generated_texts[i] == cached_generated_texts[i]
            for i in range(len(prompts))
        ]
    )
    print(f"Generated answers are the same: {generated_same}")

    print(f"Prefix cache queried tokens during cached generate(): {prefix_queries}")
    print(f"Prefix cache hit tokens during cached generate(): {prefix_hits}")
    if prefix_hits <= 0:
        raise AssertionError("Prefix cache was not hit during cached generate().")


if __name__ == "__main__":
    main()
