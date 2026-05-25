"""Debug logging for UnifiedRadixCache HiCache (L2 host, L3 storage).

Enable:
  export SGLANG_UNIFIED_RADIX_HICACHE_DEBUG=1

Optional (reduce noise):
  export SGLANG_UNIFIED_RADIX_HICACHE_DEBUG_MATCH=1   # log every match_prefix
  export SGLANG_UNIFIED_RADIX_HICACHE_DEBUG_REQ=<req_id>  # only this request
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache_components import BASE_COMPONENT_TYPE, ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

logger = logging.getLogger(__name__)
_LOG_PREFIX = "[UnifiedRadixHiCache]"


def hicache_debug_enabled() -> bool:
    return envs.SGLANG_UNIFIED_RADIX_HICACHE_DEBUG.get()


def hicache_match_debug_enabled() -> bool:
    return hicache_debug_enabled() and envs.SGLANG_UNIFIED_RADIX_HICACHE_DEBUG_MATCH.get()


def _req_filter_passes(req_id: Optional[str]) -> bool:
    filt = envs.SGLANG_UNIFIED_RADIX_HICACHE_DEBUG_REQ.get()
    if not filt:
        return True
    return req_id is not None and filt in req_id


def _should_log(*, req_id: Optional[str] = None, match: bool = False) -> bool:
    if match:
        if not hicache_match_debug_enabled():
            return False
    elif not hicache_debug_enabled():
        return False
    return _req_filter_passes(req_id)


def _log_on_rank0(msg: str) -> None:
    import torch

    from sglang.srt.distributed import get_tensor_model_parallel_rank

    try:
        if torch.distributed.is_initialized():
            if get_tensor_model_parallel_rank() != 0:
                return
        logger.info("%s %s", _LOG_PREFIX, msg)
    except Exception:
        logger.info("%s %s", _LOG_PREFIX, msg)


def format_node_state(node: UnifiedTreeNode) -> str:
    """Single-node residency tags: D=device, H=host(L2), evicted, backuped, hash."""
    parts: list[str] = []
    key_len = len(node.key) if node.key else 0
    parts.append(f"id={node.id}")
    parts.append(f"key_len={key_len}")

    full_cd = node.component_data[BASE_COMPONENT_TYPE]
    if full_cd.value is not None:
        parts.append(f"D={len(full_cd.value)}")
    elif node.evicted:
        parts.append("D=evicted")
    else:
        parts.append("D=none")

    if full_cd.host_value is not None:
        parts.append(f"H={len(full_cd.host_value)}")
    if node.backuped:
        parts.append("L2_backuped")

    if ComponentType.SWA in node.tree_components:
        swa_cd = node.component_data[ComponentType.SWA]
        if swa_cd.value is not None:
            parts.append(f"SWA_D={len(swa_cd.value)}")
        elif swa_cd.host_value is not None:
            parts.append(f"SWA_H={len(swa_cd.host_value)}")
        elif node.parent is not None and swa_cd.value is None:
            parts.append("SWA_tombstone")

    if node.hash_value:
        tail = node.get_last_hash_value()
        if tail is not None:
            parts.append(f"hash_tail={tail[:16]}")

    parts.append(f"hit={node.hit_count}")
    parts.append(f"lock={full_cd.lock_ref}")
    return " ".join(parts)


def format_path_to_root(node: UnifiedTreeNode, *, anchor: str = "leaf") -> str:
    """Full radix path from root to ``node`` (inclusive)."""
    chain: list[UnifiedTreeNode] = []
    cur: Optional[UnifiedTreeNode] = node
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    chain.reverse()
    lines = [f"path_to_{anchor} ({len(chain)} nodes):"]
    depth = 0
    for n in chain:
        indent = "  " * depth
        lines.append(f"{indent}-> {format_node_state(n)}")
        depth += 1
    return "\n".join(lines)


def log_hicache_event(
    layer: str,
    op: str,
    *,
    node: Optional[UnifiedTreeNode] = None,
    req_id: Optional[str] = None,
    match: bool = False,
    extra: Optional[dict[str, Any]] = None,
    also_log_path: bool = True,
    path_anchor: Optional[UnifiedTreeNode] = None,
) -> None:
    """Log one HiCache event. ``layer`` is L2 or L3."""
    if not _should_log(req_id=req_id, match=match):
        return

    bits = [f"[{layer}]", op]
    if req_id:
        bits.append(f"req={req_id}")
    if node is not None:
        bits.append(format_node_state(node))
    if extra:
        for k, v in extra.items():
            bits.append(f"{k}={v}")

    _log_on_rank0(" ".join(bits))
    path_node = path_anchor if path_anchor is not None else node
    if also_log_path and path_node is not None:
        _log_on_rank0(format_path_to_root(path_node, anchor=op))
