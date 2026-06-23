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

# A SHORT shared prefix (< W=512 tokens) for exercising Case 1 sub-block hits:
# the shared prefix lands within stable block 0, so a later request matches it at
# sub-block (128-token) granularity with num_computed <= W. ~5 stanzas ~= 300 tokens.
short_prefix = """
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
"""

# A MEDIUM shared prefix (> W=512 tokens, with a non-block-aligned remainder) for
# exercising Case 2 straddle sub-block hits: a later request full-block-matches block 0
# then sub-block-matches into block 1, so num_computed = W + k*sub_block (mid-block) and
# the rolling window straddles blocks 0 and 1. ~11 stanzas ~= 650-720 tokens.
medium_prefix = """
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
    # Single LLM per process (mode on|off) -> avoids the two-LLMs-in-one-process NPU
    # runtime conflict. Run twice and diff the printed token ids to check on==off.
    import argparse
    import json

    from vllm.transformers_utils.config import get_hf_text_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["on", "off"], required=True)
    parser.add_argument("--out", type=str, default=None, help="write token ids JSON here")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="wrap each prompt in the gemma chat template (proper instruct format)",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="use gemma's native sliding_window (512) + block_size 512 (no 1024 override)",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="use the short (<W) shared prefix to exercise Case 1 sub-block hits",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="use the medium (>W, non-aligned) shared prefix for Case 2 straddle hits",
    )
    cli = parser.parse_args()

    global prompts, prefix
    if cli.short or cli.medium:
        prefix = short_prefix if cli.short else medium_prefix
        prompts = [
            prefix + q
            for q in (
                "\nQuestion: Who wrote this Poem and what's the title?\nAnswer:\n",
                "\nQuestion: What is the main theme of this poem?\nAnswer:\n",
                "\nQuestion: What feeling does the opening create?\nAnswer:\n",
                "\nQuestion: Who is Lenore in the poem?\nAnswer:\n",
            )
        ]
    if cli.chat:
        prompts = [
            f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
            for p in prompts
        ]

    def _sw_override(hf_config):
        # gemma-3 keeps sliding_window in text_config; force it to the block size so the
        # scratch+stable spec's block_size == sliding_window assertion holds.
        get_hf_text_config(hf_config).update({"sliding_window": 1024})
        return hf_config

    args = EngineArgs(
        model="google/gemma-3-1b-it",
        max_num_seqs=2,
        max_model_len=2048,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        hf_overrides=None if cli.native else _sw_override,
        trust_remote_code=True,
        enable_prefix_caching=(cli.mode == "on"),
        num_gpu_blocks_override=512,  # cap pool (avoids OOM; plenty for these prompts)
    )
    # Can't directly pass this value due to validation
    args.block_size = 512 if cli.native else 1024  # type: ignore

    llm = LLM(**asdict(args))

    if cli.mode == "on":
        # Warmup so the shared prefix's KV cache is computed (and cached) first.
        llm.generate(prompts[0], sampling_params)

    metrics_before = llm.get_metrics()
    hits_before = get_counter_value(metrics_before, "vllm:prefix_cache_hits")

    outputs, elapsed = timed_generate(llm, prompts, sampling_params)
    print(f"[{cli.mode}] generate() time: {elapsed:.3f}s")

    hits = get_counter_value(llm.get_metrics(), "vllm:prefix_cache_hits") - hits_before

    token_ids = []
    print("-" * 50)
    for output in outputs:
        ids = list(output.outputs[0].token_ids)
        token_ids.append(ids)
        print(
            f"Prompt: {get_prompt_without_prefix(output.prompt)!r}\n"
            f"Generated: {output.outputs[0].text!r}"
        )
        print("-" * 50)

    print(f"[{cli.mode}] prefix_cache_hit_tokens (vllm metric) = {hits}")
    if cli.out:
        with open(cli.out, "w") as f:
            json.dump(token_ids, f)
        print(f"[{cli.mode}] wrote token ids -> {cli.out}")
    # NOTE: the vllm prefix_cache_hits metric reads 0 for the hybrid SWA groups even when
    # the scratch+stable seed copy fires (it accounts hits at the full-attn group); the
    # authoritative checks are the [SS-HIT] seed-copy logs (SS_DBG=1) + token-id match
    # against the --mode off run.


if __name__ == "__main__":
    main()
