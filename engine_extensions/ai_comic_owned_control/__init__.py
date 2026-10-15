"""Project-owned queue control; interruption acknowledgement is not termination.

The inspected ComfyUI has one sequential worker and a PromptQueue RLock shared
by get/task_done. Its executor can clear an interrupt after claiming a prompt,
so running receipts deliberately never claim cancellation. A client must
recheck/re-request its exact prompt with a deadline. client_id is an ownership
binding to queue data, not a new authentication mechanism.

Exact recovery binds the native accepted graph, not a silently normalized host
proposal. Core validation/replacements can change its SHA and must fail closed.
Instance/owner/graph/ID matching does not distinguish an intentionally identical
same-instance ID reuse after its old history has been deleted.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import heapq
import json
import re
import threading
import uuid

from aiohttp import web
import nodes
from server import PromptServer


NODE_CLASS_MAPPINGS = {}
PROTOCOL = "ai-comic-owned-control-v1"
INSTANCE_ID = str(uuid.uuid4())
_RLOCK_TYPE = type(threading.RLock())


def _compatible(queue) -> bool:
    return (isinstance(getattr(queue, "mutex", None), _RLOCK_TYPE)
            and isinstance(getattr(queue, "queue", None), list)
            and isinstance(getattr(queue, "currently_running", None), dict)
            and isinstance(getattr(queue, "history", None), dict)
            and callable(getattr(getattr(queue, "server", None), "queue_updated", None)))


def _receipt(prompt_id, client_id, state, *, verified=False, cancelled=False):
    return {"prompt_id": prompt_id, "client_id": client_id,
            "ownership_verified": verified, "cancelled": cancelled, "state": state}


def _owner(entry, prompt_id):
    if (not isinstance(entry, (tuple, list)) or len(entry) < 4
            or entry[1] != prompt_id or not isinstance(entry[3], dict)
            or not isinstance(entry[3].get("client_id"), str)
            or not entry[3]["client_id"]):
        raise ValueError("Queue ownership is not verifiable")
    return entry[3]["client_id"]


def _workflow_digest(workflow):
    if not isinstance(workflow, dict):
        raise ValueError("The complete graph must be a JSON object")
    return hashlib.sha256(json.dumps(workflow, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _control(queue, prompt_id, client_id, *, binding=None, inspect=False):
    """No await or task handoff between binding verification and the action."""
    actual_sha = None

    def respond(state, *, verified=False, cancelled=False, history=None):
        receipt = _receipt(prompt_id, client_id, state, verified=verified, cancelled=cancelled)
        if binding is not None:
            receipt.update(instance_id=INSTANCE_ID, workflow_sha256=actual_sha)
        if inspect:
            receipt["history"] = history
        return receipt

    if not _compatible(queue):
        return respond("unsupported_queue"), 503
    with queue.mutex:
        if binding is not None and binding["instance_id"] != INSTANCE_ID:
            return respond("instance_mismatch"), 409
        running = list(queue.currently_running.values())
        if len(running) > 1:
            return respond("ambiguous_queue"), 409
        try:
            entries = running + queue.queue
            if any(not isinstance(entry, (tuple, list)) or len(entry) < 4
                   or not isinstance(entry[1], str) for entry in entries):
                raise ValueError("Malformed queue entry")
            matching = [entry for entry in entries if entry[1] == prompt_id]
            history = queue.history.get(prompt_id)
            if len(matching) > 1 or (matching and prompt_id in queue.history):
                return respond("ambiguous_queue"), 409
            if not matching:
                if history is None:
                    return respond("not_found"), 404
                if not isinstance(history, dict):
                    raise ValueError("Malformed history")
            entry = matching[0] if matching else history.get("prompt")
            if _owner(entry, prompt_id) != client_id:
                return respond("foreign"), 403
            if binding is not None:
                actual_sha = _workflow_digest(entry[2])
                if actual_sha != binding["workflow_sha256"]:
                    return respond("workflow_mismatch"), 409
            if inspect:
                if not matching and (not isinstance(history.get("outputs"), dict) or "status" not in history):
                    raise ValueError("Incomplete terminal history")
                state = ("running" if running and entry is running[0] else "pending") if matching else "terminal"
                return respond(state, verified=True, history=deepcopy(history) if not matching else None), 200
            if not matching:
                return respond("already_finished", verified=True), 200
            if running and entry is running[0]:
                nodes.interrupt_processing()
                return respond("interrupt_requested", verified=True), 200
            index = next(i for i, pending in enumerate(queue.queue) if pending is entry)
            queue.queue.pop(index)
            heapq.heapify(queue.queue)
            queue.server.queue_updated()
            return respond("pending_removed", verified=True, cancelled=True), 200
        except (ValueError, TypeError):
            return respond("ambiguous_queue"), 409


def _cancel(queue, prompt_id, client_id):
    return _control(queue, prompt_id, client_id)


@PromptServer.instance.routes.get("/ai-comic/owned-control/capabilities")
async def owned_control_capabilities(request):
    supported = _compatible(PromptServer.instance.prompt_queue)
    return web.json_response({"protocol": PROTOCOL, "instance_id": INSTANCE_ID, "owner_checks": supported,
                              "atomic_queue_lock": supported,
                              "atomic_graph_owner_checks": supported, "prompt_inspection": supported,
                              "pending_exact_removal": supported,
                              "interrupt_acknowledgement_is_termination": False})


@PromptServer.instance.routes.post("/ai-comic/prompts/{prompt_id}/cancel")
async def cancel_owned_prompt(request):
    return await _handle(request, inspect=False)


@PromptServer.instance.routes.post("/ai-comic/prompts/{prompt_id}/inspect")
async def inspect_owned_prompt(request):
    return await _handle(request, inspect=True)


async def _handle(request, *, inspect):
    prompt_id = request.match_info.get("prompt_id")
    client_id = None
    if (not isinstance(prompt_id, str)
            or re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", prompt_id) is None
            or (request.content_length is not None and request.content_length > 1024)):
        return web.json_response(_receipt(prompt_id, client_id, "invalid_request"), status=400)
    try:
        body = await request.json()
        exact_fields = {"client_id", "workflow_sha256", "instance_id"}
        if (not isinstance(body, dict) or set(body) not in
                ([exact_fields] if inspect else [{"client_id"}, exact_fields])):
            raise ValueError("Only the explicit ownership contract is accepted")
        client_id = body["client_id"]
        if not isinstance(client_id, str) or not 1 <= len(client_id) <= 160 or not client_id.strip():
            raise ValueError("client_id must be a bounded nonempty string")
        exact = set(body) == exact_fields
        if exact and (not isinstance(body["workflow_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", body["workflow_sha256"]) is None
                or not isinstance(body["instance_id"], str) or not 1 <= len(body["instance_id"]) <= 160):
            raise ValueError("Exact recovery requires the complete graph SHA and engine instance")
    except (ValueError, TypeError):
        return web.json_response(_receipt(prompt_id, client_id, "invalid_request"), status=400)
    receipt, status = _control(PromptServer.instance.prompt_queue, prompt_id, client_id,
                               binding=body if exact else None, inspect=inspect)
    return web.json_response(receipt, status=status)
