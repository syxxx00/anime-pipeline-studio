"""Finite owned dispatch checkpoints, not a replayable generation queue.

The full submitted graph and request are kept without reconstruction. Atomic
replacement does not establish engine completion, ownership or image quality;
those facts must come from the saved, attributable live consumer receipts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


class EngineReceiptPersistenceError(RuntimeError):
    pass


def receipt_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_exact_inspection(dispatch: dict, proof: dict) -> None:
    """Validate this explicitly read original task, not generic idle state."""
    if (not isinstance(proof, dict) or proof.get('ownership_verified') is not True
            or proof.get('cancelled') is not False
            or proof.get('prompt_id') != dispatch.get('prompt_id')
            or proof.get('client_id') != dispatch.get('client_id')
            or proof.get('workflow_sha256') != dispatch.get('workflow_sha256')
            or proof.get('instance_id') != dispatch.get('backend_instance_id')
            or not dispatch.get('prompt_id') or not dispatch.get('client_id')
            or not dispatch.get('backend_instance_id')
            or proof.get('state') not in {'running', 'pending', 'terminal'}):
        raise EngineReceiptPersistenceError('原引擎检查回执与实例／ID／owner／graph 绑定不符。')
    history = proof.get('history')
    if proof['state'] != 'terminal':
        if history is not None:
            raise EngineReceiptPersistenceError('运行／待处理回执不能冒充终止 history。')
        return
    validate_terminal_history(dispatch, history)


def validate_terminal_history(dispatch: dict, history: dict) -> None:
    """Check a saved raw history against its original submission binding."""
    prompt = history.get('prompt') if isinstance(history, dict) else None
    if (not isinstance(history, dict) or not isinstance(history.get('outputs'), dict)
            or 'status' not in history or not isinstance(prompt, (list, tuple)) or len(prompt) < 4
            or prompt[1] != dispatch['prompt_id'] or not isinstance(prompt[2], dict)
            or not isinstance(prompt[3], dict) or prompt[3].get('client_id') != dispatch['client_id']
            or receipt_digest(prompt[2]) != dispatch['workflow_sha256']):
        raise EngineReceiptPersistenceError('原任务完整终止 history／实际 graph 缺失或不符。')


class EngineReceiptStore:
    VERSION = 1
    TERMINAL = {'not_submitted', 'rejected_before_queue', 'history_returned', 'cancelled_or_already_terminal'}

    def __init__(self, root: Path):
        self.root = Path(root)
        # Construction/loading must not create or modify runtime files.

    def path(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job_id):
            raise EngineReceiptPersistenceError('引擎回执 job_id 无效，未读写目录外文件。')
        target = self.root / (job_id + '.json')
        if target.resolve().parent != self.root.resolve():
            raise EngineReceiptPersistenceError('引擎回执路径越出指定目录。')
        return target

    def validate(self, record: dict, *, job_id: str) -> dict:
        if (not isinstance(record, dict) or record.get('version') != self.VERSION
                or record.get('job_id') != job_id or not isinstance(record.get('request'), dict)
                or not isinstance(record.get('created_at'), str)
                or not isinstance(record.get('backend_url'), str) or not record['backend_url']
                or not isinstance(record.get('dispatches'), list) or not record['dispatches']):
            raise EngineReceiptPersistenceError('引擎回执文件格式不完整，未推断或迁移历史状态。')
        self.path(job_id)
        unhashed = {key: value for key, value in record.items() if key != 'record_sha256'}
        if (record.get('record_sha256') != receipt_digest(unhashed)
                or record.get('request_sha256') != receipt_digest(record['request'])):
            raise EngineReceiptPersistenceError('引擎回执原件或原请求 SHA 校验失败。')
        for dispatch in record['dispatches']:
            if (not isinstance(dispatch, dict) or not isinstance(dispatch.get('workflow'), dict)
                    or dispatch.get('workflow_sha256') != receipt_digest(dispatch['workflow'])
                    or dispatch.get('owner_request_sha256') != record['request_sha256']
                    or type(dispatch.get('completed')) is not bool
                    or type(dispatch.get('submission_started')) is not bool
                    or type(dispatch.get('backend_dispatch_started')) is not bool
                    or not isinstance(dispatch.get('status'), str) or not dispatch['status']
                    or not isinstance(dispatch.get('stage'), str)
                    or (dispatch.get('client_id') is not None and
                        (not isinstance(dispatch['client_id'], str) or not dispatch['client_id']))
                    or (dispatch.get('prompt_id') is not None and
                        (not isinstance(dispatch['prompt_id'], str) or not dispatch['prompt_id']))):
                raise EngineReceiptPersistenceError('引擎阶段回执／实际工作流绑定不完整。')
            instance = dispatch.get('backend_instance_id')
            if instance is not None:
                try:
                    valid_instance = isinstance(instance, str) and str(uuid.UUID(instance)) == instance
                except ValueError:
                    valid_instance = False
                if not valid_instance:
                    raise EngineReceiptPersistenceError('后端 instance 原件不是规范 UUID，未升级历史身份。')
            inspections = dispatch.get('inspection_receipts', [])
            if not isinstance(inspections, list):
                raise EngineReceiptPersistenceError('显式核对回执原件列表无效。')
            for proof in inspections:
                validate_exact_inspection(dispatch, proof)
            if dispatch['completed']:
                status = dispatch['status']
                if status not in self.TERMINAL:
                    raise EngineReceiptPersistenceError('没有可识别的阶段终止依据，未认证 completed。')
                if status == 'not_submitted' and (dispatch['prompt_id'] is not None
                        or dispatch['submission_started'] or dispatch['backend_dispatch_started']):
                    raise EngineReceiptPersistenceError('未派发终止依据与实际提交标记冲突。')
                if status == 'history_returned' and (not dispatch['prompt_id']
                        or not isinstance(dispatch.get('terminal_history'), dict)):
                    raise EngineReceiptPersistenceError('原任务终止 history 原件缺失。')
                if status == 'history_returned' and inspections and (
                        inspections[-1]['state'] != 'terminal'
                        or inspections[-1]['history'] != dispatch['terminal_history']):
                    raise EngineReceiptPersistenceError('显式核对终止证据与保存的原 history 不一致。')
                if status == 'rejected_before_queue' and (dispatch['prompt_id'] is not None
                        or not isinstance(dispatch.get('rejection_response'), dict)):
                    raise EngineReceiptPersistenceError('标准入队前拒收原件缺失或存在冲突回执。')
                if status == 'cancelled_or_already_terminal':
                    proof = dispatch.get('cancellation_receipt')
                    if (not dispatch['prompt_id'] or not isinstance(proof, dict)
                            or proof.get('prompt_id') != dispatch['prompt_id']
                            or proof.get('absent_from_running_and_pending') is not True):
                        raise EngineReceiptPersistenceError('确切取消／退出队列回执缺失。')
        return record

    def read(self, job_id: str) -> dict:
        try:
            record = json.loads(self.path(job_id).read_text(encoding='utf-8'))
            return self.validate(record, job_id=job_id)
        except EngineReceiptPersistenceError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise EngineReceiptPersistenceError(f'引擎回执无法完整读取：{job_id}：{exc}') from exc

    def write(self, *, job_id: str, created_at: str, backend_url: str,
              request: dict, dispatches: list[dict], workflows: list[dict],
              cancel_requested: bool) -> None:
        temporary = None
        try:
            target = self.path(job_id)
            if len(dispatches) != len(workflows):
                raise EngineReceiptPersistenceError('实际阶段工作流数量与回执不一致，未重建缺失图。')
            request = deepcopy(request)
            record = {'version': self.VERSION, 'job_id': job_id, 'created_at': created_at,
                'backend_url': backend_url, 'request': request, 'request_sha256': receipt_digest(request),
                'cancel_requested': cancel_requested,
                'dispatches': [deepcopy(item) | {'workflow': deepcopy(graph)}
                               for item, graph in zip(dispatches, workflows)]}
            record['record_sha256'] = receipt_digest(record)
            self.validate(record, job_id=job_id)
            if target.exists():
                previous = self.read(job_id)
                if any(previous[key] != record[key] for key in
                       ('created_at', 'backend_url', 'request_sha256')):
                    raise EngineReceiptPersistenceError('原任务请求／后端身份变化，未静默改写派发原件。')
                if len(previous['dispatches']) > len(record['dispatches']):
                    raise EngineReceiptPersistenceError('原阶段回执不能被截短。')
                for old, new in zip(previous['dispatches'], record['dispatches']):
                    if any(old.get(key) != new.get(key) for key in
                           ('workflow_sha256', 'owner_request_sha256', 'client_id', 'stage')):
                        raise EngineReceiptPersistenceError('原阶段 graph／owner 身份发生变化。')
                    if old.get('prompt_id') is not None and old['prompt_id'] != new.get('prompt_id'):
                        raise EngineReceiptPersistenceError('已知 prompt_id 不能被替换或清空。')
                    old_instance, new_instance = old.get('backend_instance_id'), new.get('backend_instance_id')
                    if old_instance != new_instance:
                        first_POST_start = (old_instance is None and not old['backend_dispatch_started']
                            and not old['completed'] and old.get('prompt_id') is None
                            and new['backend_dispatch_started'] and new_instance is not None)
                        if not first_POST_start:
                            raise EngineReceiptPersistenceError('原后端 instance 不可替换或在 POST-start 后回填。')
                    if old['completed'] and old != new:
                        raise EngineReceiptPersistenceError('已持久终止的阶段不能复活或重写。')
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / (job_id + '.' + uuid.uuid4().hex + '.tmp')
            data = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
            with temporary.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except EngineReceiptPersistenceError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise EngineReceiptPersistenceError(f'引擎派发回执未确认持久保存：{exc}') from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)  # Only this exclusive tmp.
                except OSError:
                    pass

    def load(self) -> tuple[list[dict], list[str]]:
        records, errors = [], []
        try:
            paths = sorted(self.root.glob('*.json')) if self.root.exists() else []
        except OSError as exc:
            return [], [f'{self.root}: {exc}']
        for path in paths:
            try:
                records.append(self.read(path.stem))
            except EngineReceiptPersistenceError as exc:
                errors.append(f'{path}: {exc}')
        return records, errors
