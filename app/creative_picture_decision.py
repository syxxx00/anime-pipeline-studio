"""One complete picture and its bounded method, without a second author call.

Legacy method models retain their exact schemas. These new text consumers bind
an authored frame to the method proposed in the same return; they neither
dispatch work nor certify pixels, artistic value or user approval.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .creative_reference import ReferenceNextChoice
from .creative_starting_choice import StartingModelChoice
from .prose_picture import PROSE_PICTURE_SYSTEM, ProsePictureChoice


INITIAL_PROSE_DECISION_PROFILE = 'local_creative_initial_prose_decision_6703d7894'
NEXT_PROSE_DECISION_PROFILE = 'local_creative_next_prose_decision_6703d7894'
REFERENCE_PROSE_DECISION_PROFILE = 'local_creative_reference_prose_decision_6703d7894'


class NextChoice(BaseModel):
    # Exact legacy declaration, including title, member order and defaults.
    model_config = {'extra': 'forbid'}
    operation: Literal['keep', 'regenerate', 'handoff', 'sketch', 'local_edit', 'stop']
    target: Literal['newbie', 'illustrious', 'anima', 'z_image_turbo']
    objective: str = Field(min_length=3, max_length=1200)
    reason: str = Field(min_length=3, max_length=1500)
    preserve: str = Field(max_length=1200)
    denoise: float = Field(default=.25, gt=0, le=1)


class InitialProseDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    picture: ProsePictureChoice
    starting_choice: StartingModelChoice | None


def _same_return_picture(operation: str, picture: ProsePictureChoice | None) -> None:
    if operation == 'sketch':
        raise ValueError('联合完整描述没有可执行的结构草案，不开放sketch。')
    needs_picture = operation in {'regenerate', 'handoff'}
    if needs_picture != (picture is not None):
        raise ValueError('regenerate/handoff必须采用同次完整picture；keep/stop/local_edit必须picture=null。')


class NextProseDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    choice: NextChoice
    picture: ProsePictureChoice | None

    @model_validator(mode='after')
    def picture_matches_operation(self):
        _same_return_picture(self.choice.operation, self.picture)
        return self


class ReferenceProseDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    choice: ReferenceNextChoice
    picture: ProsePictureChoice | None

    @model_validator(mode='after')
    def picture_matches_operation(self):
        _same_return_picture(self.choice.operation, self.picture)
        return self


_COMMON_DECISION_SYSTEM = """Form the artistic purpose, complete picture and feasible
method together for ONE current frame, using the full original, attributed
corrections, selected identity/style conditions, applicable research and actual
capabilities. A method is a means to the picture, not a permanent model role.
Research conditions and raw reviews can inform new line/color/space choices;
explain their current use rather than copying an old subject, scene or pose.
Separate what the user wants painted from context they explain to the system.
Context may inform your choices; it is not permission to assert biography, a
former name, past ownership of an object or another depicted event. Use the
bound subject name, not an invented translation or alias. Historical requests
and captions describe past attempts, not what their pixels achieved. Do not
infer a model's superiority or another model's failure from available filenames,
sampling compatibility, unsupported anecdotes or an unreviewed old request.
User requirements, replaceable author choices, old declarations and fallible
pixel observations have different authority. Text describing a requested gaze,
contact, carving, identity or emotion is not evidence that those pixels exist.
Only claim to have seen an image actually supplied to this consumer. Keep
material uncertainty. Do not require an all-purpose check, invent a quality
score, self-certify an image or claim user approval. Make a feasible choice for
this operation while preserving the original purpose and relevant local value.
The host owns exact identity tags, full recipes, parameter compatibility,
serialization, task receipts and execution; never author workflow code or
silently remove a selected LoRA, parameter lock or other explicit condition.
Independent candidate counts belong to the host collection, not this frame's
count or an objective to create several compositions at once. The same return's
picture is the adopted draft; no second author will fill in vague placeholders.
Return only the supplied complete JSON."""


INITIAL_PROSE_DECISION_SYSTEM = _COMMON_DECISION_SYSTEM + """
Author picture as one complete drawable frame, and choose an INITIAL target only
within the supplied compatible starts. If the host says the start is explicit
or there is only one compatible start, starting_choice must be null: the host
uses that fixed configuration. Otherwise return the compatible target, a
task-based reason and material failure risks in starting_choice. Do not choose
a follow-up chain or alter seed, frame, sampling, knowledge or style selections.
""" + PROSE_PICTURE_SYSTEM


NEXT_PROSE_DECISION_SYSTEM = _COMMON_DECISION_SYSTEM + """
Choose ONE currently available operation in choice. For regenerate or handoff,
First relate the actual observations and current feedback to this operation's
purpose. Use reason to explain what visible value matters now, which connected
picture choices need reconsideration, and the consequences of another feasible
method. The same return's picture must realise those choices. Unfulfilled goals
remain goals: do not place them in preserve as if the source already shows them.
Retaining identity or useful colors is different from inheriting a failed pose
or arrangement. Consider whether changing framing, visible faces, distances,
orientation or light together can serve the intended relationship; adding
texture alone cannot establish a missing interaction. Do not prescribe a
universal pose, palette or model. These are artistic responsibilities, not a
keyword checklist or a claim that the observations are always correct.
picture MUST be a complete new ProsePictureChoice in this same return, consistent
with the chosen objective, preservation, target and actual source conditions.
Regenerate creates a new composition and need not copy the source's rejected
layout. Handoff consumes the entire source image and can inherit its mistakes
or damage style/layout; denoise is not a preservation percentage. For keep,
stop or local_edit, picture MUST be null. A local edit uses the bound source
scene and its local objective; the host must locate the original pixels and
execute the existing protected local consumer. Do not supply coordinates or a
mask, or assume a small eye edit can fix scene-wide attention. Sketch is not an
available operation for this joint prose consumer. A keep/stop choice creates
no image and does not certify quality or imply an unmet request was fulfilled.
The picture-writing instructions below apply only when picture is required;
for keep/stop/local_edit keep picture=null rather than authoring an unused frame.
""" + PROSE_PICTURE_SYSTEM


REFERENCE_PROSE_DECISION_SYSTEM = _COMMON_DECISION_SYSTEM + """
Choose the exact bound parent, child or neither reference and ONE next operation
in choice, together with the complete picture required by that operation.
Reason must connect actual reported changes and current purpose, distinguish
visible value from still-unfulfilled requirements, reconsider related picture
choices together, and explain the cost of another feasible action. Do not call
denser detail, a new texture or a requested relationship an achieved improvement
without actual support. A goal does not become preservation evidence.
Read the visible delta within its actual contract and comparison_material.kind:
full_original_pair covers full unchanged originals; protected_local_crop_pair
covers only the declared mask-bbox regions, not whole-image relations or quality.
Neither scope certifies purpose success. Missing full observations stay missing; never substitute
another version's reading. Use capabilities bound to the selected reference.
Neither permits only regenerate or stop; it adopts neither image's pixels or
old composition. Regenerate/handoff require the full same-return picture.
Keep/stop/local_edit require picture=null; local_edit retains the selected
source scene with an explicit local objective. Sketch is not available here.
A parent/child choice is an execution reference, not user selection or approval.
Whole-image handoff inherits complete content and may preserve the wrong
structure; denoise is not a preservation percentage. Never supply a mask or
coordinates. State why this complete choice serves the current purpose, not
an assumed need for realism, more props or denser texture. The picture-writing
instructions below apply only when picture is required; keep/stop/local_edit
must still return picture=null rather than an unused new frame.
""" + PROSE_PICTURE_SYSTEM
