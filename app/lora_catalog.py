"""Inspect only bounded safetensors metadata, never loading weights to RAM/GPU."""
import json
import struct
from pathlib import Path


def describe_lora(path: Path, root: Path) -> dict:
    item = {'name': path.relative_to(root).as_posix(), 'size_mb': round(path.stat().st_size / 1024**2, 1), 'family': 'unknown', 'compatible_models': [], 'evidence': '缺少明确的基础架构元数据；不按文件名推断兼容性。'}
    try:
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if not 2 <= length <= min(path.stat().st_size - 8, 4 * 1024**2):
                return item
            metadata = json.loads(stream.read(length)).get('__metadata__', {})
        architecture = str(metadata.get('modelspec.architecture', ''))
        base = str(metadata.get('ss_base_model_version', ''))
        # These are declared architecture matches, not visual quality guarantees.
        if architecture.startswith('stable-diffusion-xl-v1-base/') or base == 'sdxl_base_v1-0':
            item.update(family='sdxl', compatible_models=['illustrious'])
        elif architecture.startswith('stable-diffusion-v1/') or base == 'sd_v1':
            item.update(family='sd1')
        elif architecture.lower().startswith('newbie/') or base.lower() == 'newbie':
            item.update(family='newbie', compatible_models=['newbie'])
        if architecture or base:
            item['evidence'] = f'文件元数据：{architecture or base}。基础架构匹配不等于角色、画风或质量已验证。'
    except (OSError, ValueError, TypeError, AttributeError, struct.error):
        item['evidence'] = '无法读取有效的架构元数据，未确认兼容性。'
    return item


def lora_catalog(root: Path) -> list[dict]:
    return [describe_lora(path, root) for path in sorted(root.rglob('*.safetensors'))] if root.exists() else []
