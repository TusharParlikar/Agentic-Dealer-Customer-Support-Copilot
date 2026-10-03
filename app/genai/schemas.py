from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class OfficeRequirements(BaseModel):
    headcount: int = Field(description="Number of people who need a desk")
    meeting_rooms: list[int] = Field(default=[], description="Seats in each meeting room, one number per room")
    budget_inr: int | None = Field(default=None, description="Total budget in rupees, if given")
    preferences: list[str] = Field(default=[], description="e.g. premium, budget, ergonomic, sit-stand")
    storage: bool = Field(default=True, description="Include drawer cabinets")


class IssueType(str, Enum):
    MECHANISM_FAILURE = "MECHANISM_FAILURE"
    STRUCTURAL_DAMAGE = "STRUCTURAL_DAMAGE"
    UPHOLSTERY = "UPHOLSTERY"
    IGNITION_FAILURE = "IGNITION_FAILURE"
    GAS_ODOR = "GAS_ODOR"
    VENTING_SMOKE = "VENTING_SMOKE"
    GLASS_DAMAGE = "GLASS_DAMAGE"
    ELECTRICAL = "ELECTRICAL"
    DELIVERY = "DELIVERY"
    MISSING_PARTS = "MISSING_PARTS"
    OTHER = "OTHER"


Urgency = Literal["LOW", "MEDIUM", "HIGH"]


class TicketFields(BaseModel):
    product: str | None = Field(default=None, description="Product name or line the complaint is about")
    issue_type: IssueType
    urgency: Urgency
    summary: str = Field(description="One-sentence summary of the problem")
