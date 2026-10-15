"""Optional relational authoring rounds, not a permanent model/component split.

Both authors receive the current original and choices. The second may refuse
the first; the host joins unchanged returns, never repairs artistic conflicts.
The final native consumer still receives every complete visual field.
"""
from copy import deepcopy
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .picture_event import CanonicalPictureChoice, _VisualSpec


CORE_FIELDS = ('count', 'expression', 'body_action', 'left_hand', 'right_hand',
               'objects', 'relations', 'framing')
FINISH_FIELDS = ('background', 'lighting', 'style', 'exclusions')


def _visual_fields(title, fields):
    return create_model(title, __config__=ConfigDict(extra='forbid', strict=True),
        **{key: (_VisualSpec.model_fields[key].annotation,
                 deepcopy(_VisualSpec.model_fields[key])) for key in fields})


CoreVisualChoice = _visual_fields('CoreVisualChoice', CORE_FIELDS)
FinishVisualChoice = _visual_fields('FinishVisualChoice', FINISH_FIELDS)


class RelationalCore(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    visual_spec: CoreVisualChoice
    character_position: str = Field(min_length=1, max_length=1500, pattern=r'\S')
    material_reading: str = Field(min_length=1, max_length=1200, pattern=r'\S')
    visual_logic: str = Field(min_length=1, max_length=1500, pattern=r'\S')
    uncertainties: list[str] = Field(max_length=6)

    @model_validator(mode='after')
    def declared_entities(self):
        # Only structural validation. Empty finish fields are not adopted as
        # art choices; a separate complete finish must precede compilation.
        CanonicalPictureChoice.model_validate({
            'visual_spec': {**self.visual_spec.model_dump(mode='json'),
                            **{key: [] for key in FINISH_FIELDS}},
            'character_position': self.character_position,
            'visual_logic': self.visual_logic, 'uncertainties': self.uncertainties})
        return self


class RelationalFinish(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    visual_spec: FinishVisualChoice
    visual_logic: str = Field(min_length=1, max_length=900, pattern=r'\S')
    core_conflicts: list[str] = Field(max_length=8)
    uncertainties: list[str] = Field(max_length=6)


CORE_SYSTEM = """Develop one drawable attention/support event, not a whole picture inventory.
The full current original, identity, frame and style selections still govern.
Choose expression, ONE bodily pose with compatible visible hands, the independent
attention/support objects, their spatial relationships, subject position and view together.
Other environment, lighting and style choices will be developed in a later round.
An actual supplied research image is support only for its explicit use_for; identify
visible relationships you can borrow, changes you choose and what pixels cannot establish
in material_reading. With no image, say there was no pixel observation. Historical prompts,
reviews and requested features are not things you saw. Do not invent identity, costume,
official lore or approval. Use character and declared obj_1..obj_8 references only.
Do not turn eyes, hair or costume into extra objects or fill all slots. Empty arrays
are permitted; do not describe hands outside the chosen frame as visible.
Choose concrete forms and support, not just 'beautiful', 'sad' or 'finely carved'.
Unstated art choices are replaceable author proposals, not user requirements.
Looking does not mean touching or working; an interest is not a current action.
All visual fields and positions must be English; explanations may be Chinese.
Return only this supplied core schema, with no native prompt, second position map
or environment inventory. A subsequent round can find this choice needs revision;
this is not certification of meaning, identity or artistic compatibility.
"""

FINISH_SYSTEM = """Develop the environment, shared light/color and drawing choices for
this exact proposed relational core, while checking the whole current original.
You have the core author's material_reading, not your own pixel observation.
Your changes must work with the viewing direction, attention object, body support
and framing together. Full user selections/recipes govern; do not duplicate supplied
artist strings or invent identity/costume. Do not add independent action/attention
objects or silently change the core. Background and light can have concrete forms
without becoming extra object slots. Unspecified features are not automatically forbidden.
The core is not permanently frozen or approved: if it needs a different pose, view,
object form or position to make this picture work, state the exact needed revision in
core_conflicts and do not pretend merely adding background/lighting resolves it.
The host will retain both returns and stop this attempt rather than repair the core.
Empty arrays are allowed; do not fill them with synonyms. All visual fields must be
English; visual_logic and uncertainty may be Chinese. Return only the supplied finish
schema. Empty core_conflicts is your unverified judgment, never quality approval.
"""


def merge_relational_picture(core, finish):
    core = RelationalCore.model_validate(core.model_dump(mode='json') if isinstance(core, RelationalCore) else deepcopy(core))
    finish = RelationalFinish.model_validate(finish.model_dump(mode='json') if isinstance(finish, RelationalFinish) else deepcopy(finish))
    if finish.core_conflicts:
        raise ValueError('后继表现选择要求修订核心关系；两份原回应保留，没有补画或修答案：' + '; '.join(finish.core_conflicts))
    return CanonicalPictureChoice.model_validate({
        'visual_spec': {**core.visual_spec.model_dump(mode='json'), **finish.visual_spec.model_dump(mode='json')},
        'character_position': core.character_position,
        'visual_logic': core.visual_logic + '\n\n' + finish.visual_logic,
        'uncertainties': [*core.uncertainties, *finish.uncertainties]})


def relational_receipt(core, finish, merged):
    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode('utf-8')).hexdigest()
    return {'method': 'relational_steps', 'core_sha256': digest(core),
        'finish_sha256': digest(finish), 'merged_sha256': digest(merged),
        'merge': 'exact_disjoint_visual_fields_and_both_original_explanations',
        'semantic_repair_applied': False, 'artistic_compatibility_verified': False,
        'material_reading_is_unverified_author_interpretation': True}
