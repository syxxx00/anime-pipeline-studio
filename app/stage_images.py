"""Image-stage evidence, kept separate from experiment outcomes and reviews."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path


class StageImages:
    def __init__(self, output_root: Path, engine_root: Path):
        self.output_root = output_root.resolve()
        self.root = self.output_root / 'stages'
        self.cache_root = (engine_root / 'output').resolve()

    @staticmethod
    def _key(value: str) -> str:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', value):
            raise ValueError('无效的阶段或任务标识')
        return value

    @staticmethod
    def _inside(root: Path, path: Path) -> Path:
        path = path.resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('阶段文件超出允许目录')
        return path

    def _directory(self, job_id: str) -> Path:
        return self._inside(self.output_root, self.root / self._key(job_id))

    def retain(self, job_id: str, stage_id: str, label: str, model: str,
               workflow: dict, image_info: dict, payload: bytes, *, evidence='retained', warning='') -> dict:
        directory = self._directory(job_id)
        digest = hashlib.sha256(payload).hexdigest()
        artifact_id = f'{self._key(stage_id)}-{digest[:16]}'
        image_path = self._inside(directory, directory / f'{artifact_id}.png')
        record_path = self._inside(directory, directory / f'{artifact_id}.json')
        if record_path.exists():
            record = self._read(record_path, job_id)
            self.read_bytes(record)
            return record
        directory.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(payload)
        record = dict(id=artifact_id, job_id=job_id, stage_id=stage_id, label=label, model=model,
                      file_path=str(image_path), sha256=digest, evidence=evidence, warning=warning,
                      workflow=workflow, engine_output=image_info,
                      created_at=datetime.now(UTC).isoformat())
        temporary = self._inside(directory, record_path.with_suffix('.tmp'))
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(record_path)
        return record

    def _read(self, path: Path, job_id: str) -> dict:
        directory = self._directory(job_id)
        record = json.loads(self._inside(directory, path).read_text(encoding='utf-8'))
        self._key(record['id'])
        if record['job_id'] != job_id or path.stem != record['id']:
            raise ValueError('阶段记录身份不一致')
        record['file_path'] = str(self._inside(directory, Path(record['file_path'])))
        return record

    def retained(self, job_id: str) -> list[dict]:
        directory = self._directory(job_id)
        return [self._read(path, job_id) for path in sorted(directory.glob('*.json'))]

    def read_bytes(self, record: dict) -> bytes:
        root = self.cache_root if record['evidence'] == 'cache_match' else self.output_root
        payload = self._inside(root, Path(record['file_path'])).read_bytes()
        if record.get('sha256') and hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('阶段文件校验失败，未使用已改变的图片')
        return payload

    def collect(self, job, saved_evidence: dict | None = None) -> list[dict]:
        records = self.retained(job.id)
        recorded_stages = {record['stage_id'] for record in records}
        retained_cache_files = {record.get('engine_output', {}).get('source_file') for record in records}
        # Only a saved graph can name cache candidates. Never use a reconstructed
        # graph or guess the latest file; multiple matches are all shown.
        if saved_evidence and saved_evidence['source'] != 'reconstructed':
            stages = saved_evidence['workflow'].get('stages', {})
            for position, (stage_id, graph) in enumerate(stages.items(), start=1):
                if stage_id in recorded_stages:
                    continue
                for node in graph.values():
                    if not isinstance(node, dict) or node.get('class_type') != 'SaveImage':
                        continue
                    prefix = str(node.get('inputs', {}).get('filename_prefix', '')).replace('\\', '/')
                    name = Path(prefix).name
                    if name != job.id and not name.startswith(job.id + '_'):
                        continue
                    try:
                        candidate = self._inside(self.cache_root, self.cache_root / prefix)
                    except ValueError:
                        continue
                    if not candidate.parent.is_dir():
                        continue
                    for path in sorted(candidate.parent.iterdir()):
                        # Comfy SaveImage appends a counter, not another stage
                        # suffix. A final prefix "job" must not match "job_source".
                        if not re.fullmatch(re.escape(candidate.name) + r'_[0-9]+_?\.png', path.name, re.IGNORECASE):
                            continue
                        path = self._inside(self.cache_root, path)
                        if str(path) in retained_cache_files:
                            continue
                        token = hashlib.sha256(f'{stage_id}:{path}'.encode()).hexdigest()[:20]
                        records.append(dict(id=f'cache-{token}', job_id=job.id, stage_id=stage_id,
                                            label=f'历史阶段 {position} · {stage_id}', model='',
                                            file_path=str(path), evidence='cache_match', workflow=graph,
                                            warning='按保存工作流的输出前缀匹配引擎缓存，未记录原始文件哈希；多个匹配均列出，不能确定哪个是该次执行的输出。'))
        for index, filename in enumerate(job.files):
            path = self._inside(self.output_root, Path(filename))
            if path.is_file():
                records.append(dict(id=f'final-{index + 1}', job_id=job.id, stage_id='delivered',
                                    label=f'最终交付 · {index + 1}', model='', file_path=str(path),
                                    evidence='delivered', warning='', workflow=None,
                                    artwork_id=f'{job.id}-{index + 1:02d}'))
        return records

    @staticmethod
    def public(record: dict) -> dict:
        result = {key: value for key, value in record.items() if key not in {'workflow', 'engine_output'}}
        result['image_url'] = f"/api/jobs/{record['job_id']}/stages/{record['id']}/image"
        graph = record.get('workflow') or {}
        result['sampling'] = [dict(node_id=key, **{name: node['inputs'].get(name) for name in
                             ('seed', 'steps', 'cfg', 'sampler_name', 'scheduler', 'denoise')})
                              for key, node in graph.items() if node.get('class_type') == 'KSampler']
        result['model_files'] = [str(value) for node in graph.values() for key, value in node.get('inputs', {}).items()
                                 if key in {'unet_name', 'ckpt_name'}]
        return result
