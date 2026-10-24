"""Pydantic schemas for the HTTP API.

Why: the dashboard, the tests and any client need a stable, typed contract. The models mirror the
pipeline dataclasses one to one so a response is built by validating ``analysis_to_dict`` output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WordOut(BaseModel):
    text: str
    start: float
    end: float
    probability: float = 1.0
    prefix_space: bool = True


class SegmentOut(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str
    words: list[WordOut] = Field(default_factory=list)


class TranscriptOut(BaseModel):
    duration: float
    language: str
    backend: str
    model: str
    speakers: list[str]
    segments: list[SegmentOut]


class RegionOut(BaseModel):
    start: float
    end: float


class SpeakersOut(BaseModel):
    count: int
    changes: list[float]
    labels: list[int]


class KeywordOut(BaseModel):
    phrase: str
    score: float
    count: int
    timestamps: list[float]


class BulletOut(BaseModel):
    text: str
    start: float
    speaker: str
    score: float


class SectionOut(BaseModel):
    title: str
    start: float
    end: float
    bullets: list[BulletOut]


class NotesOut(BaseModel):
    title_terms: list[str]
    summary: list[BulletOut]
    sections: list[SectionOut]
    n_sentences: int
    compression: float


class ActionItemOut(BaseModel):
    text: str
    start: float
    end: float
    speaker: str
    owner: str
    due: str | None
    score: float
    cues: list[str]


class JobSummary(BaseModel):
    id: str
    title: str
    created_at: str
    duration: float
    backend: str
    model: str
    n_speakers: int
    n_actions: int


class JobOut(JobSummary):
    speech_ratio: float
    real_time_factor: float
    timings_s: dict[str, float]
    transcript: TranscriptOut
    regions: list[RegionOut]
    speakers: SpeakersOut
    keywords: list[KeywordOut]
    notes: NotesOut
    notes_markdown: str
    actions: list[ActionItemOut]


class SearchHitOut(BaseModel):
    sentence_index: int
    score: float
    start: float
    end: float
    text: str
    speaker: str


class SearchOut(BaseModel):
    query: str
    hits: list[SearchHitOut]


class HealthOut(BaseModel):
    status: str
    version: str
    backend: str
    model: str
    auth_required: bool
    jobs: int
    demo_available: bool


class ErrorOut(BaseModel):
    detail: str
    request_id: str | None = None
