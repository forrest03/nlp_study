from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SkillStep(BaseModel):
    step_num: int
    title: str
    description: str
    command: Optional[str] = None
    expected_output: Optional[str] = None


class SkillInfo(BaseModel):
    name: str
    description: str
    path: str
    steps: list[SkillStep] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    skill_name: Optional[str] = None
    messages: list[ChatMessage] = Field(default_factory=list)
    user_input: str
    context: dict = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    status: str = "completed"
    duration_ms: int = 0
    summary: str = ""
    comparison: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
    next_step: Optional[int] = None
    action_suggestion: Optional[str] = None
    skill_info: Optional[SkillInfo] = None
    skill_used: Optional[str] = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    total_time_ms: int = 0
    token_usage: dict = Field(default_factory=dict)


class ExecuteRequest(BaseModel):
    skill_name: Optional[str] = None
    user_input: str = ""
    parameters: dict = Field(default_factory=dict)


class StepResult(BaseModel):
    step_num: int
    title: str
    status: str
    output: str = ""
    next_step: Optional[int] = None


class ExecuteResponse(BaseModel):
    skill_name: str
    current_step: int
    total_steps: int
    results: list[StepResult] = Field(default_factory=list)
    status: str
    final_output: Optional[str] = None


class SkillListResponse(BaseModel):
    skills: list[SkillInfo]


class SkillDetailResponse(BaseModel):
    skill: SkillInfo


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
