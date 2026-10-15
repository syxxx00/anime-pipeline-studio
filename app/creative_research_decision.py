"""An optional research read or a complete same-return initial picture.

The catalogue offers attributed research identities, not scene conditions.
This model neither fetches material nor approves an image or dispatches work;
the host checks requested identities and permits at most one material round.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .creative_picture_decision import INITIAL_PROSE_DECISION_SYSTEM
from .creative_starting_choice import StartingModelChoice
from .prose_picture import ProsePictureChoice


RESEARCH_PROSE_DECISION_PROFILE = 'local_creative_research_prose_decision_6703d7894'


class ResearchQuery(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    artwork_id: str = Field(min_length=1, max_length=80)
    purpose: str = Field(min_length=12, max_length=800)


class ResearchProseDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    picture: ProsePictureChoice | None
    starting_choice: StartingModelChoice | None
    research_queries: list[ResearchQuery] = Field(min_length=0, max_length=2)

    @model_validator(mode='after')
    def compose_or_read(self) -> 'ResearchProseDecision':
        if self.picture is not None:
            if self.research_queries:
                raise ValueError('compose必须返回完整picture且research_queries为空；不混合未读取材料与最终稿。')
        elif self.starting_choice is not None or not self.research_queries:
            raise ValueError('read必须picture=null、starting_choice=null且research_queries非空；没有伪造最终稿。')
        return self


RESEARCH_PROSE_DECISION_SYSTEM = INITIAL_PROSE_DECISION_SYSTEM + """
This consumer has an optional material-read return instead of always composing.
The research catalogue is an index of attributed examples, NOT current scene
conditions, already-read images, approval or a template. If no read is needed,
return the complete picture and compatible initial choice now, with
research_queries=[]. If a specific current artistic or realization question
needs evidence, request one or two catalogue artwork_id values and explain each
purpose; return picture=null and starting_choice=null. Do not claim to have read
pixels or full conditions not supplied. The host validates IDs and may provide
the complete requested material for ONE round, then the existing initial
consumer must compose the complete final picture; no mandatory two-call path.
Background explanations are context, not independently established biography,
depicted past events, mandatory carving subjects or extra present actions.
Return only the complete ResearchProseDecision JSON, not a draft plus a read.
"""
