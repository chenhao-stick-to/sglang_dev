from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    expand_page_keys_for_host_pool,
    pool_page_boundary_to_kv_pages,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()

_DSV4_REQUIRED_PREFIX_POOL_NAMES = frozenset(
    {
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
    }
)
_DSV4_SWA_WINDOW_POOL_NAMES = frozenset(
    {
        PoolName.SWA,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        PoolName.DEEPSEEK_V4_C128_STATE,
    }
)


class CacheOperation(BaseCacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[list[PoolTransfer]]:
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {}
        for op in ops:
            for t in op.pool_transfers or []:
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=ts[0].name,
                host_indices=cat_or_none(t.host_indices for t in ts),
                device_indices=cat_or_none(t.device_indices for t in ts),
                keys=[k for t in ts if t.keys for k in t.keys] or None,
                hit_policy=ts[0].hit_policy,
                indices_from_pool=ts[0].indices_from_pool,
            )
            for ts in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
    ):
        startup_storage_backend = storage_backend
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            pp_rank=pp_rank,
            pp_size=pp_size,
            enable_storage_metrics=enable_storage_metrics,
        )
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    def reset(self):
        super().reset()
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=True,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None

        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        self.write_queue.clear()
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                self.io_backend,
                pool_transfers=resolved_pool_transfers,
            )
            finish_event.record()
            self._record_transfer_indices_on_stream(
                self.write_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        need_load_kv = host_indices.numel() > 0

        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None

        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=False,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                full_allocator.free(device_indices)
            return None

        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                    pool_transfers=resolved_pool_transfers,
                )
                producer_event.complete(i)
            self._record_transfer_indices_on_stream(
                self.load_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def _record_transfer_indices_on_stream(
        self,
        stream: torch.Stream,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)
        for transfer in pool_transfers or []:
            if transfer.host_indices is not None and transfer.host_indices.is_cuda:
                transfer.host_indices.record_stream(stream)
            if transfer.device_indices is not None and transfer.device_indices.is_cuda:
                transfer.device_indices.record_stream(stream)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        last_hash: Optional[str] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            last_hash=last_hash,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _is_deepseek_v4_storage_pool(self, pool_name: PoolName) -> bool:
        entry_map = getattr(self.mem_pool_host, "entry_map", {})
        return pool_name in entry_map and (
            pool_name in _DSV4_REQUIRED_PREFIX_POOL_NAMES
            or pool_name in _DSV4_SWA_WINDOW_POOL_NAMES
        )

    def _is_deepseek_v4_storage_enabled(self) -> bool:
        entry_map = getattr(self.mem_pool_host, "entry_map", {})
        return any(name in entry_map for name in _DSV4_REQUIRED_PREFIX_POOL_NAMES)

    def _storage_extra_info(self, operation) -> HiCacheStorageExtraInfo:
        extra = {
            "token_ids": list(operation.token_ids),
            "last_hash": operation.last_hash,
            "page_size": self.page_size,
            "sliding_window_size": getattr(self, "sliding_window_size", None),
        }
        pool_token_ranges = getattr(
            operation.pool_storage_result, "pool_token_ranges", None
        )
        if pool_token_ranges:
            extra["pool_token_ranges"] = pool_token_ranges
        return HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None,
            extra_info=extra,
        )

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        hash_value = []
        for start in range(0, len(operation.token_ids), self.page_size):
            last_hash = self.get_hash_str(
                operation.token_ids[start : start + self.page_size], last_hash
            )
            hash_value.append(last_hash)

        extra_info = self._storage_extra_info(operation)
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_from_hit_result(hit_result)

        if kv_hit_pages > 0 and operation.pool_transfers:
            self._sync_trailing_keys(operation.pool_transfers, hash_value, kv_hit_pages)

        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _anchor_has_storage_payload(self) -> bool:
        mem_pool_host = self.mem_pool_host
        if hasattr(mem_pool_host, "anchor_has_storage_payload"):
            return mem_pool_host.anchor_has_storage_payload()
        return (
            getattr(mem_pool_host, "kv_buffer", None) is not None
            and mem_pool_host.get_ksize_per_token() > 0
        )

    def _host_pool_page_size(self, pool_name: PoolName) -> int:
        entry = getattr(self.mem_pool_host, "entry_map", {}).get(pool_name)
        if entry is not None:
            return entry.host_pool.page_size
        return self.page_size

    def _page_keys_for_host_pool(
        self, page_keys: list[str], pool_name: PoolName
    ) -> list[str]:
        return expand_page_keys_for_host_pool(
            page_keys,
            self.page_size,
            self._host_pool_page_size(pool_name),
        )

    def _dsv4_swa_window_token_range(self, total_tokens: int) -> tuple[int, int]:
        sliding_window_size = getattr(self, "sliding_window_size", None)
        if total_tokens <= 0:
            return 0, 0
        window_tokens = (
            min(total_tokens, sliding_window_size)
            if sliding_window_size and sliding_window_size > 0
            else total_tokens
        )
        start = total_tokens - window_tokens
        start -= start % self.page_size
        return start, total_tokens

    def _prepare_storage_prefetch_pool_transfers(self, operation) -> bool:
        if not operation.pool_transfers:
            return True
        hit_tokens = len(operation.hash_value) * self.page_size
        if hit_tokens <= 0:
            return True

        pool_token_ranges = (
            getattr(operation.pool_storage_result, "pool_token_ranges", None) or {}
        )
        newly_allocated: list[tuple[Callable, torch.Tensor, PoolTransfer]] = []

        def rollback() -> None:
            for free_fn, indices, transfer in newly_allocated:
                free_fn(indices)
                transfer.host_indices = None

        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is not None:
                continue
            if transfer.host_indices is not None:
                continue
            if transfer.name not in _DSV4_SWA_WINDOW_POOL_NAMES:
                continue
            if not self._is_deepseek_v4_storage_enabled():
                continue

            entry = self.mem_pool_host.entry_map.get(transfer.name)
            if entry is None:
                continue
            token_range = pool_token_ranges.get(str(transfer.name))
            if token_range is None:
                token_range = self._dsv4_swa_window_token_range(hit_tokens)
            pool_token_ranges[str(transfer.name)] = token_range
            need_size = token_range[1] - token_range[0]
            if need_size <= 0:
                continue
            indices = entry.host_pool.alloc(need_size)
            if indices is None and entry.host_evict_fn:
                entry.host_evict_fn(need_size)
                indices = entry.host_pool.alloc(need_size)
            if indices is None:
                rollback()
                return False
            transfer.host_indices = indices
            transfer.keys = operation.hash_value
            newly_allocated.append((entry.host_pool.free, indices, transfer))

        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            source = next(
                (
                    t
                    for t in operation.pool_transfers
                    if t.indices_from_pool is None and t.name == transfer.indices_from_pool
                ),
                None,
            )
            if source is None:
                continue
            transfer.host_indices = source.host_indices
            transfer.keys = source.keys or operation.hash_value
            if str(source.name) in pool_token_ranges:
                pool_token_ranges[str(transfer.name)] = pool_token_ranges[
                    str(source.name)
                ]

        operation.pool_storage_result.pool_token_ranges = pool_token_ranges
        return True

    def _prefix_success_pages(
        self, results: dict[str, list[bool]], transfer: PoolTransfer
    ) -> int:
        page_results = results.get(transfer.name, [])
        boundary = 0
        for ok in page_results:
            if not ok:
                break
            boundary += 1
        return pool_page_boundary_to_kv_pages(
            boundary,
            self.page_size,
            self._host_pool_page_size(transfer.name),
        )

    def _completed_tokens_from_v2_results(
        self,
        operation,
        results: dict[str, list[bool]],
        *,
        require_swa_window: bool,
    ) -> int:
        hit_pages = len(operation.hash_value) if operation.hash_value else (
            len(operation.token_ids) // self.page_size
        )
        required_pages = hit_pages
        saw_required = False
        for transfer in operation.pool_transfers or []:
            if transfer.name in _DSV4_REQUIRED_PREFIX_POOL_NAMES:
                saw_required = True
                required_pages = min(
                    required_pages, self._prefix_success_pages(results, transfer)
                )

        if require_swa_window:
            for transfer in operation.pool_transfers or []:
                if transfer.name in _DSV4_SWA_WINDOW_POOL_NAMES and transfer.name in results:
                    if not all(results.get(transfer.name, [])):
                        return 0

        if not saw_required and not self._anchor_has_storage_payload():
            required_pages = hit_pages
        return required_pages * self.page_size

    def _page_transfer(self, operation):
        # Transfer extra pools
        if operation.pool_transfers and not operation.is_terminated():
            if not self._prepare_storage_prefetch_pool_transfers(operation):
                operation.mark_terminate()
                return
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(
                operation.pool_transfers, self._storage_extra_info(operation)
            )
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        # Transfer kv pools
        if self._anchor_has_storage_payload():
            super()._page_transfer(operation)
        elif not operation.is_terminated():
            operation.completed_tokens = self._completed_tokens_from_v2_results(
                operation,
                results if operation.pool_transfers else {},
                require_swa_window=True,
            )

    def _page_backup(self, operation):
        # Backup extra pools
        if operation.pool_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(
                operation.pool_transfers, self._storage_extra_info(operation)
            )
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        # Backup kv pools
        if self._anchor_has_storage_payload():
            super()._page_backup(operation)
        else:
            operation.completed_tokens = self._completed_tokens_from_v2_results(
                operation,
                results if operation.pool_transfers else {},
                require_swa_window=False,
            )

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        sources: dict[PoolName, tuple[torch.Tensor, Optional[list[str]]]] = {
            PoolName.KV: (operation.host_indices, operation.hash_value),
        }
        for transfer in operation.pool_transfers or []:
            if transfer.indices_from_pool is not None:
                continue
            if transfer.host_indices is None:
                continue
            keys = (
                transfer.keys
                if transfer.keys is not None
                else operation.hash_value
            )
            sources[transfer.name] = (transfer.host_indices, keys)

        for transfer in operation.pool_transfers or []:
            if transfer.indices_from_pool is None:
                continue
            src_name = transfer.indices_from_pool
            if src_name not in sources:
                continue
            host_indices, keys = sources[src_name]
            transfer.host_indices = host_indices
            if keys is None:
                keys = operation.hash_value
            if self._is_deepseek_v4_storage_pool(transfer.name):
                transfer.keys = keys
            else:
                transfer.keys = self._page_keys_for_host_pool(keys, transfer.name)

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        """
        for transfer in pool_transfers:
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Auto-alloc host or device indices for PoolTransfers where they are None."""
        if not extra_pools:
            return None
        # (pool, free_fn, indices) for atomic rollback on failure.
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                if alloc_host:
                    prev_pool.host_indices = None
                else:
                    prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if alloc_host:
                if pool.host_indices is not None or pool.device_indices is None:
                    continue
                alloc_fn = entry.host_pool.alloc
                free_fn = entry.host_pool.free
                evict_fn = entry.host_evict_fn
                size = len(pool.device_indices)
            else:
                if pool.device_indices is not None or pool.host_indices is None:
                    continue
                # device_alloc_fn / device_free_fn override entry.device_pool's
                # methods for pools whose device_pool is a raw KV pool (layout)
                # rather than an allocator (e.g. SWA).
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                free_fn = entry.device_free_fn or entry.device_pool.free
                evict_fn = entry.device_evict_fn
                size = len(pool.host_indices)
            indices = alloc_fn(size)
            if indices is None and evict_fn:
                evict_fn(size)
                indices = alloc_fn(size)
            if indices is None:
                # Atomic rollback: free everything we successfully allocated.
                rollback_allocated()
                return None
            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # Assign indices to deferred pools from their source.
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices
        return extra_pools
