from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    # HybridAttentionManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    # ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
)

from vllm.v1.core.kv_cache_utils import (
    # BlockHashList,
    # BlockHashWithGroupId,
    KVCacheBlock,
)
from collections.abc import Sequence
class HybridAttentionSpec(FullAttentionSpec):
    pass

class HybridAttentionManager(FullAttentionManager):
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        print("@@@@ num_tokens", num_tokens)
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )

spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    HybridAttentionSpec: HybridAttentionManager,
}


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int,
    max_model_len: int,
    **kwargs,
) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager