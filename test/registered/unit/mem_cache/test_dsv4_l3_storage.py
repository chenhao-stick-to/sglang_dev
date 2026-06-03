import inspect
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE, ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode
from sglang.srt.mem_cache.utils import get_hash_str


class _HostPool:
    def __init__(self, page_size: int, layout: str = "page_first", layer_num: int = 1):
        self.page_size = page_size
        self.layout = layout
        self.layer_num = layer_num


class _IOHostPool(_HostPool):
    def get_page_buffer_meta(self, host_indices):
        page_count = len(host_indices) // self.page_size
        return list(range(page_count)), [1] * page_count


def _make_dsv4_store(kv_page_size: int = 4, swa_page_size: int = 2):
    store = MooncakeStore.__new__(MooncakeStore)
    store.mem_pool_host = SimpleNamespace(page_size=kv_page_size)
    store._v1_kv_storage_enabled = False
    store.extra_backend_tag = None
    store.mha_suffix = "0"
    store.mla_suffix = ""
    store.registered_pools = {
        PoolName.DEEPSEEK_V4_C4: _HostPool(kv_page_size),
        PoolName.DEEPSEEK_V4_C4_INDEXER: _HostPool(kv_page_size),
        PoolName.DEEPSEEK_V4_C128: _HostPool(kv_page_size),
        PoolName.SWA: _HostPool(swa_page_size),
        PoolName.DEEPSEEK_V4_C4_STATE: _HostPool(swa_page_size),
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE: _HostPool(swa_page_size),
        PoolName.DEEPSEEK_V4_C128_STATE: _HostPool(swa_page_size),
    }
    return store


class TestDeepSeekV4MooncakeKeying(unittest.TestCase):
    def test_required_prefix_pool_keys_reuse_kv_page_hashes(self):
        store = _make_dsv4_store(kv_page_size=4)
        token_ids = list(range(1, 9))
        prior_hash = "parent-page-hash"
        transfer = PoolTransfer(name=PoolName.DEEPSEEK_V4_C4)
        extra_info = HiCacheStorageExtraInfo(
            extra_info={"token_ids": token_ids, "last_hash": prior_hash}
        )

        keys = store._dsv4_storage_page_keys(
            ["tree-page-0", "tree-page-1"], transfer, extra_info
        )

        self.assertEqual(keys, ["tree-page-0", "tree-page-1"])

    def test_swa_and_state_pool_keys_are_unchained_by_swa_page_size(self):
        store = _make_dsv4_store(kv_page_size=4, swa_page_size=2)
        token_ids = list(range(1, 9))
        extra_info = HiCacheStorageExtraInfo(
            extra_info={"token_ids": token_ids, "last_hash": "parent-page-hash"}
        )
        expected = [
            get_hash_str(token_ids[start : start + 2], None)
            for start in range(0, len(token_ids), 2)
        ]

        for pool_name in (
            PoolName.SWA,
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.DEEPSEEK_V4_C128_STATE,
        ):
            with self.subTest(pool_name=pool_name):
                keys = store._dsv4_storage_page_keys(
                    ["tree-page-0", "tree-page-1"],
                    PoolTransfer(name=pool_name),
                    extra_info,
                )
                self.assertEqual(keys, expected)

    def test_batch_exists_v2_uses_common_full_prefix_for_all_dsv4_pools(self):
        store = _make_dsv4_store(kv_page_size=4, swa_page_size=2)

        def batch_exist(keys):
            result = []
            for idx, key in enumerate(keys):
                if "deepseek_v4_c4_indexer" in key and idx == 3:
                    result.append(0)
                else:
                    result.append(1)
            return result

        store._batch_exist = batch_exist
        token_ids = list(range(16))
        transfers = [
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4_INDEXER),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C128),
            PoolTransfer(name=PoolName.SWA),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4_STATE),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C128_STATE),
        ]
        extra_info = HiCacheStorageExtraInfo(
            extra_info={"token_ids": token_ids, "sliding_window_size": 4}
        )

        result = store.batch_exists_v2(
            ["kv0", "kv1", "kv2", "kv3"], transfers, extra_info
        )

        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(
            result.required_pool_hit_pages[str(PoolName.DEEPSEEK_V4_C4_INDEXER)],
            3,
        )
        self.assertEqual(
            result.pool_token_ranges[str(PoolName.SWA)],
            (0, 12),
        )
        self.assertEqual(
            result.pool_token_ranges[str(PoolName.DEEPSEEK_V4_C4_STATE)],
            (0, 12),
        )

    def test_batch_exists_v2_checks_every_swa_page_in_common_prefix(self):
        store = _make_dsv4_store(kv_page_size=4, swa_page_size=2)
        token_ids = list(range(16))
        extra_info = HiCacheStorageExtraInfo(
            extra_info={"token_ids": token_ids, "sliding_window_size": 4}
        )
        window_transfer = PoolTransfer(name=PoolName.SWA)
        all_swa_keys = store._dsv4_storage_page_keys(
            ["kv0", "kv1", "kv2", "kv3"], window_transfer, extra_info
        )
        # KV page size 4 and SWA page size 2: indices 6 and 7 are both part of
        # KV page 4. Missing either subpage must roll the common prefix back.
        missing_non_sentinel_for_page4 = all_swa_keys[6]

        def batch_exist(keys):
            result = []
            for key in keys:
                if missing_non_sentinel_for_page4 in key:
                    result.append(0)
                else:
                    result.append(1)
            return result

        store._batch_exist = batch_exist
        transfers = [
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4_INDEXER),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C128),
            window_transfer,
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4_STATE),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C128_STATE),
        ]

        result = store.batch_exists_v2(
            ["kv0", "kv1", "kv2", "kv3"], transfers, extra_info
        )

        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(result.pool_token_ranges[str(PoolName.SWA)], (0, 12))
        self.assertEqual(
            result.pool_token_ranges[str(PoolName.DEEPSEEK_V4_C4_STATE)],
            (0, 12),
        )

    def test_batch_get_v2_groups_required_prefix_pools_by_kv_page(self):
        store = _make_dsv4_store(kv_page_size=4, swa_page_size=2)
        for pool_name in (
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
        ):
            store.registered_pools[pool_name] = _IOHostPool(4)

        def get_batch(keys, ptrs, sizes):
            self.assertEqual(len(keys), 3)
            page_idx = ptrs[0] % 4
            if page_idx == 3 and any("deepseek_v4_c128" in key for key in keys):
                return [-1 if "deepseek_v4_c128" in key else 1 for key in keys]
            return [1] * len(keys)

        store._get_batch_zero_copy_impl = get_batch
        transfers = [
            PoolTransfer(
                name=pool_name,
                host_indices=torch.arange(16),
                keys=["kv0", "kv1", "kv2", "kv3"],
            )
            for pool_name in (
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
            )
        ]
        extra_info = HiCacheStorageExtraInfo(extra_info={"token_ids": list(range(16))})

        results = store.batch_get_v2(transfers, extra_info)

        expected = [True, True, True, False]
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4], expected)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4_INDEXER], expected)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C128], expected)

    def test_batch_set_v2_groups_required_prefix_pools_by_kv_page(self):
        store = _make_dsv4_store(kv_page_size=4, swa_page_size=2)
        for pool_name in (
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
        ):
            store.registered_pools[pool_name] = _IOHostPool(4)
        store._batch_exist = lambda keys: [0] * len(keys)

        def put_batch(keys, ptrs, sizes):
            self.assertEqual(len(keys), 3)
            page_idx = ptrs[0] % 4
            if page_idx == 2 and any("deepseek_v4_c4_indexer" in key for key in keys):
                return [
                    -1 if "deepseek_v4_c4_indexer" in key else 0 for key in keys
                ]
            return [0] * len(keys)

        store._put_batch_zero_copy_impl = put_batch
        transfers = [
            PoolTransfer(
                name=pool_name,
                host_indices=torch.arange(16),
                keys=["kv0", "kv1", "kv2", "kv3"],
            )
            for pool_name in (
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
            )
        ]
        extra_info = HiCacheStorageExtraInfo(extra_info={"token_ids": list(range(16))})

        results = store.batch_set_v2(transfers, extra_info)

        expected = [True, True, False, False]
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4], expected)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4_INDEXER], expected)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C128], expected)

    def test_batch_get_v2_groups_all_dsv4_pools_by_kv_page(self):
        store = _make_dsv4_store(kv_page_size=4, swa_page_size=2)
        for pool_name in (
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
            PoolName.SWA,
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.DEEPSEEK_V4_C128_STATE,
        ):
            store.registered_pools[pool_name] = _IOHostPool(
                4 if pool_name in (
                    PoolName.DEEPSEEK_V4_C4,
                    PoolName.DEEPSEEK_V4_C4_INDEXER,
                    PoolName.DEEPSEEK_V4_C128,
                ) else 2
            )

        token_ids = list(range(16))
        get_calls = []

        def get_batch(keys, ptrs, sizes):
            # 3 required pages + 4 window pools * 2 swa pages per KV page.
            self.assertEqual(len(keys), 11)
            get_calls.append(keys)
            page_idx = ptrs[0] % 4
            if page_idx == 3:
                return [
                    -1 if "deepseek_v4_c128_state" in key else 1 for key in keys
                ]
            return [1] * len(keys)

        store._get_batch_zero_copy_impl = get_batch
        transfers = [
            PoolTransfer(
                name=pool_name,
                host_indices=torch.arange(16),
                keys=["kv0", "kv1", "kv2", "kv3"],
            )
            for pool_name in (
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                PoolName.DEEPSEEK_V4_C128_STATE,
            )
        ]
        extra_info = HiCacheStorageExtraInfo(extra_info={"token_ids": token_ids})

        results = store.batch_get_v2(transfers, extra_info)

        self.assertEqual(len(get_calls), 4)
        expected_kv = [True, True, True, False]
        expected_swa = [True, True, True, True, True, True, False, False]
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4], expected_kv)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4_INDEXER], expected_kv)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C128], expected_kv)
        self.assertEqual(results[PoolName.SWA], expected_swa)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4_STATE], expected_swa)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4_INDEXER_STATE], expected_swa)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C128_STATE], expected_swa)

    def test_batch_set_v2_groups_all_dsv4_pools_by_kv_page(self):
        store = _make_dsv4_store(kv_page_size=4, swa_page_size=2)
        for pool_name in (
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
            PoolName.SWA,
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.DEEPSEEK_V4_C128_STATE,
        ):
            store.registered_pools[pool_name] = _IOHostPool(
                4
                if pool_name
                in (
                    PoolName.DEEPSEEK_V4_C4,
                    PoolName.DEEPSEEK_V4_C4_INDEXER,
                    PoolName.DEEPSEEK_V4_C128,
                )
                else 2
            )
        store._batch_exist = lambda keys: [0] * len(keys)

        def put_batch(keys, ptrs, sizes):
            self.assertEqual(len(keys), 11)
            page_idx = ptrs[0] % 4
            if page_idx == 2:
                return [
                    -1 if "deepseek_v4_c4_state" in key else 0 for key in keys
                ]
            return [0] * len(keys)

        store._put_batch_zero_copy_impl = put_batch
        transfers = [
            PoolTransfer(
                name=pool_name,
                host_indices=torch.arange(16),
                keys=["kv0", "kv1", "kv2", "kv3"],
            )
            for pool_name in (
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                PoolName.DEEPSEEK_V4_C128_STATE,
            )
        ]
        extra_info = HiCacheStorageExtraInfo(extra_info={"token_ids": list(range(16))})

        results = store.batch_set_v2(transfers, extra_info)

        expected_kv = [True, True, False, False]
        expected_swa = [True, True, True, True, False, False, False, False]
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4], expected_kv)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4_INDEXER], expected_kv)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C128], expected_kv)
        self.assertEqual(results[PoolName.SWA], expected_swa)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4_STATE], expected_swa)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C4_INDEXER_STATE], expected_swa)
        self.assertEqual(results[PoolName.DEEPSEEK_V4_C128_STATE], expected_swa)


class TestUnifiedRadixStorageBackup(unittest.TestCase):
    def test_write_backup_skips_swa_tombstone_node(self):
        root = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        node = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        node.parent = root
        node.key = RadixKey([1, 2, 3, 4], None)
        node.component_data[BASE_COMPONENT_TYPE].value = torch.arange(4)
        node.component_data[ComponentType.SWA].value = None

        class CaptureController:
            write_policy = "write_through"

            def __init__(self):
                self.write_called = False

            def write(self, *args, **kwargs):
                self.write_called = True
                return torch.arange(4)

        controller = CaptureController()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = controller
        cache.root_node = root

        backed_up = cache.write_backup(node)

        self.assertEqual(backed_up, 0)
        self.assertFalse(controller.write_called)
        self.assertIsNone(node.component_data[BASE_COMPONENT_TYPE].host_value)

    def test_write_backup_storage_passes_parent_hash_to_controller(self):
        parent = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        parent.key = RadixKey([1, 2, 3, 4], None)
        parent.hash_value = ["parent-hash"]

        node = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        node.parent = parent
        node.key = RadixKey([5, 6, 7, 8], None)
        node.hash_value = ["child-hash"]
        node.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(4)

        class CaptureController:
            def write_storage(
                self,
                host_indices,
                token_ids,
                hash_value=None,
                prefix_keys=None,
                last_hash=None,
                extra_pools=None,
            ):
                self.last_hash = last_hash
                self.token_ids = token_ids
                self.extra_pools = extra_pools
                return 123

        controller = CaptureController()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = controller
        cache.hicache_storage_pass_prefix_keys = False
        cache.ongoing_backup = {}
        cache._collect_storage_backup_pool_transfers = lambda _node: []

        cache.write_backup_storage(node)

        self.assertEqual(controller.last_hash, "parent-hash")
        self.assertEqual(controller.token_ids, [5, 6, 7, 8])

    def test_write_backup_storage_rejects_extra_pools_without_hybrid_controller(self):
        node = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        node.key = RadixKey([5, 6, 7, 8], None)
        node.hash_value = ["child-hash"]
        node.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(4)

        class PlainController:
            def write_storage(self, *args, **kwargs):
                return 123

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = PlainController()
        cache.hicache_storage_pass_prefix_keys = False
        cache.ongoing_backup = {}
        cache._collect_storage_backup_pool_transfers = lambda _node: [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.arange(4),
                keys=["child-hash"],
            )
        ]

        with self.assertRaisesRegex(RuntimeError, "extra storage pools"):
            cache.write_backup_storage(node)

    def test_write_backup_storage_skips_partial_dsv4_extra_pools(self):
        node = UnifiedTreeNode((ComponentType.FULL, ComponentType.SWA))
        node.key = RadixKey([5, 6, 7, 8], None)
        node.hash_value = ["child-hash"]
        node.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(4)

        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.DEEPSEEK_V4_C4: SimpleNamespace(host_pool=_HostPool(4)),
                PoolName.SWA: SimpleNamespace(host_pool=_HostPool(2)),
            }
        )
        controller.write_storage_called = False

        def write_storage(*args, **kwargs):
            controller.write_storage_called = True
            return 123

        controller.write_storage = write_storage
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = controller
        cache.hicache_storage_pass_prefix_keys = False
        cache.ongoing_backup = {}
        cache._collect_storage_backup_pool_transfers = lambda _node: [
            PoolTransfer(
                name=PoolName.DEEPSEEK_V4_C4,
                host_indices=torch.arange(4),
                keys=["child-hash"],
            )
        ]

        cache.write_backup_storage(node)

        self.assertFalse(controller.write_storage_called)
        self.assertEqual(cache.ongoing_backup, {})


class TestUnifiedRadixPrefetch(unittest.TestCase):
    def test_prefetch_truncates_key_to_reduced_host_allocation(self):
        class MemPoolHost:
            def __init__(self):
                self.alloc_calls = []

            def alloc(self, n):
                self.alloc_calls.append(n)
                if n == 8:
                    return torch.arange(8)
                return None

            def available_size(self):
                return 8

        class CaptureController:
            def __init__(self):
                self.mem_pool_host = MemPoolHost()
                self.prefetch_tokens_occupied = 0

            def prefetch_rate_limited(self):
                return False

            def prefetch(
                self,
                req_id,
                host_indices,
                new_input_tokens,
                last_hash=None,
                prefix_keys=None,
                extra_pools=None,
            ):
                self.prefetch_key_len = len(new_input_tokens)
                self.host_indices_len = len(host_indices)
                return SimpleNamespace(host_indices=host_indices, pool_transfers=[])

        controller = CaptureController()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.enable_storage = True
        cache.cache_controller = controller
        cache.page_size = 4
        cache.prefetch_threshold = 4
        cache.is_eagle = False
        cache.ongoing_prefetch = {}
        cache.evict_host = lambda _n: 0
        cache._collect_storage_prefetch_pool_transfers = lambda _node: []

        last_host_node = UnifiedTreeNode((ComponentType.FULL,))
        last_host_node.key = RadixKey([], None)

        cache.prefetch_from_storage(
            "req-1", last_host_node, list(range(12)), last_hash=None
        )

        self.assertEqual(controller.prefetch_key_len, 8)
        self.assertEqual(controller.host_indices_len, 8)
        self.assertEqual(controller.prefetch_tokens_occupied, 8)

    def test_prefetch_returns_if_reduced_host_allocation_fails(self):
        class MemPoolHost:
            def alloc(self, n):
                return None

            def available_size(self):
                return 8

        class CaptureController:
            def __init__(self):
                self.mem_pool_host = MemPoolHost()
                self.prefetch_tokens_occupied = 0
                self.prefetch_called = False

            def prefetch_rate_limited(self):
                return False

            def prefetch(self, *args, **kwargs):
                self.prefetch_called = True
                return SimpleNamespace(host_indices=None, pool_transfers=[])

        controller = CaptureController()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.enable_storage = True
        cache.cache_controller = controller
        cache.page_size = 4
        cache.prefetch_threshold = 4
        cache.is_eagle = False
        cache.ongoing_prefetch = {}
        cache.evict_host = lambda _n: 0
        cache._collect_storage_prefetch_pool_transfers = lambda _node: []

        last_host_node = UnifiedTreeNode((ComponentType.FULL,))
        last_host_node.key = RadixKey([], None)

        cache.prefetch_from_storage(
            "req-oom", last_host_node, list(range(12)), last_hash=None
        )

        self.assertFalse(controller.prefetch_called)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)
        self.assertNotIn("req-oom", cache.ongoing_prefetch)


class TestHybridControllerPrefetchPools(unittest.TestCase):
    def test_prepare_storage_prefetch_uses_hit_query_token_range(self):
        class AllocatingHostPool:
            page_size = 2

            def __init__(self):
                self.allocations = []

            def alloc(self, n):
                self.allocations.append(n)
                return torch.arange(n)

            def free(self, indices):
                pass

        swa_pool = AllocatingHostPool()
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 4
        controller.sliding_window_size = 8
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.DEEPSEEK_V4_C4: SimpleNamespace(host_pool=_HostPool(4)),
                PoolName.SWA: SimpleNamespace(
                    host_pool=swa_pool,
                    host_evict_fn=None,
                ),
            }
        )

        operation = SimpleNamespace(
            hash_value=["h0", "h1", "h2"],
            pool_transfers=[PoolTransfer(name=PoolName.SWA)],
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=3,
                extra_pool_hit_pages={},
                pool_token_ranges={str(PoolName.SWA): (4, 8)},
            ),
        )

        self.assertTrue(controller._prepare_storage_prefetch_pool_transfers(operation))

        self.assertEqual(swa_pool.allocations, [4])
        self.assertEqual(len(operation.pool_transfers[0].host_indices), 4)
        self.assertEqual(operation.pool_transfers[0].keys, ["h0", "h1", "h2"])
        self.assertEqual(
            operation.pool_storage_result.pool_token_ranges[str(PoolName.SWA)],
            (4, 8),
        )

    def test_prepare_storage_prefetch_falls_back_to_full_prefix_range(self):
        class AllocatingHostPool:
            page_size = 2

            def __init__(self):
                self.allocations = []

            def alloc(self, n):
                self.allocations.append(n)
                return torch.arange(n)

            def free(self, indices):
                pass

        swa_pool = AllocatingHostPool()
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 4
        controller.sliding_window_size = 4
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.DEEPSEEK_V4_C4: SimpleNamespace(host_pool=_HostPool(4)),
                PoolName.SWA: SimpleNamespace(
                    host_pool=swa_pool,
                    host_evict_fn=None,
                ),
            }
        )

        operation = SimpleNamespace(
            hash_value=["h0", "h1", "h2"],
            pool_transfers=[PoolTransfer(name=PoolName.SWA)],
            pool_storage_result=PoolTransferResult(
                kv_hit_pages=3,
                extra_pool_hit_pages={},
                pool_token_ranges={},
            ),
        )

        self.assertTrue(controller._prepare_storage_prefetch_pool_transfers(operation))

        self.assertEqual(swa_pool.allocations, [12])
        self.assertEqual(
            operation.pool_storage_result.pool_token_ranges[str(PoolName.SWA)],
            (0, 12),
        )

    def test_completed_tokens_use_window_full_prefix(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 4
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.DEEPSEEK_V4_C4: SimpleNamespace(host_pool=_HostPool(4)),
                PoolName.SWA: SimpleNamespace(host_pool=_HostPool(2)),
            }
        )
        controller._anchor_has_storage_payload = lambda: False
        operation = SimpleNamespace(
            hash_value=["h0", "h1", "h2", "h3"],
            token_ids=list(range(16)),
            pool_transfers=[
                PoolTransfer(name=PoolName.DEEPSEEK_V4_C4),
                PoolTransfer(name=PoolName.SWA),
            ],
        )
        results = {
            PoolName.DEEPSEEK_V4_C4: [True, True, True, True],
            PoolName.SWA: [True, True, True, True, True, True, False, False],
        }

        completed = controller._completed_tokens_from_v2_results(
            operation,
            results,
            require_swa_window=True,
        )

        self.assertEqual(completed, 12)

    def test_backup_completed_tokens_use_all_dsv4_pool_prefixes(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 4
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.DEEPSEEK_V4_C4: SimpleNamespace(host_pool=_HostPool(4)),
                PoolName.SWA: SimpleNamespace(host_pool=_HostPool(2)),
            }
        )
        controller._anchor_has_storage_payload = lambda: False
        operation = SimpleNamespace(
            hash_value=["h0", "h1", "h2", "h3"],
            token_ids=list(range(16)),
            pool_transfers=[
                PoolTransfer(name=PoolName.DEEPSEEK_V4_C4),
                PoolTransfer(name=PoolName.SWA),
            ],
        )
        results = {
            PoolName.DEEPSEEK_V4_C4: [True, True, True, True],
            PoolName.SWA: [True, True, True, True, False, False, False, False],
        }

        completed = controller._completed_tokens_from_v2_results(
            operation,
            results,
            require_swa_window=False,
        )

        self.assertEqual(completed, 8)


class TestBaseHiCacheControllerSignatures(unittest.TestCase):
    def test_base_controller_storage_methods_accept_extra_pools(self):
        self.assertIn("extra_pools", inspect.signature(HiCacheController.prefetch).parameters)
        self.assertIn(
            "extra_pools", inspect.signature(HiCacheController.write_storage).parameters
        )
        self.assertIn(
            "last_hash", inspect.signature(HiCacheController.write_storage).parameters
        )


if __name__ == "__main__":
    unittest.main()
