"""The verbatim request remains separate from interpretations and step prompts."""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CreationGoal(BaseModel):
    id: str = Field(pattern=r'^[a-zA-Z][a-zA-Z0-9_.-]*$')
    description: str = Field(min_length=1, max_length=2000)
    priority: Literal['required', 'preferred', 'context', 'proposal'] = 'required'
    source_quote: str = Field(default='', max_length=4000)
    observable: str = Field(default='', max_length=2000)


class CreationIntent(BaseModel):
    original_text: str = Field(min_length=1, max_length=12000)
    original_sha256: str = ''
    goals: list[CreationGoal] = Field(default_factory=list, max_length=60)
    interpretation_notes: list[str] = Field(default_factory=list, max_length=30)
    research_refs: list[str] = Field(default_factory=list, max_length=30)
    interpretation_method: str = Field(default='reviewed', max_length=200)

    @model_validator(mode='after')
    def verify_original(self):
        digest = hashlib.sha256(self.original_text.encode('utf-8')).hexdigest()
        if self.original_sha256 and self.original_sha256 != digest:
            raise ValueError('原句校验失败：不能用改写后的描述替换原始输入')
        self.original_sha256 = digest
        ids = [goal.id for goal in self.goals]
        if len(ids) != len(set(ids)):
            raise ValueError('任务目标编号重复')
        for goal in self.goals:
            if goal.source_quote and goal.source_quote not in self.original_text:
                raise ValueError(f'目标 {goal.id} 引用的原句片段不在原文中')
        return self
