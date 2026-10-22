
from pydantic import BaseModel, ConfigDict, Field


class StructuredLogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_id: int = Field(gt=0)
    date: str = Field(min_length=1, max_length=6)
    time: str = Field(min_length=1, max_length=6)
    pid: str = Field(min_length=1)
    level: str = Field(min_length=1)
    component: str = Field(min_length=1)
    content: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_template: str = Field(min_length=1)
    block_ids: list[str] = Field(min_length=1)


class IngestionBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1, max_length=100)
    records: list[StructuredLogRecord] = Field(
        min_length=1,
        max_length=500,
    )


class IngestionBatchResponse(BaseModel):
    status: str
    batch_id: str
    records_received: int
    unique_line_ids: int
    unique_block_ids: int
