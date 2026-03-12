from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.triage_agent import TriageGraphManager


graph_manager = TriageGraphManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await graph_manager.initialize()
    yield
    await graph_manager.shutdown()


app = FastAPI(
    title="Phase 2 HR Interview Task",
    version="0.1.0",
    lifespan=lifespan,
)


class RunRequest(BaseModel):
    thread_id: str = Field(..., description="Durable LangGraph thread id")
    message: str = Field(..., description="User message for the current turn")


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Durable LangGraph thread id")


class ApprovalRequest(BaseModel):
    thread_id: str = Field(..., description="Durable LangGraph thread id")
    approved: bool = Field(..., description="Human approval decision")


def _format_run_response(result: dict) -> dict:
    values = result.get("values", {})
    policy_hits = values.get("policy_hits", [])
    interrupts = result.get("interrupts", [])

    response = {
        "thread_id": result.get("thread_id"),
        "status": values.get("status"),
        "issue_type": values.get("issue_type"),
        "order_id": values.get("order_id"),
        "policy_supported": values.get("policy_supported"),
        "citations": [
            {
                "label": hit.get("citation_label"),
                "snippet": hit.get("snippet"),
            }
            for hit in policy_hits
        ],
    }

    if interrupts:
        first_interrupt = interrupts[0]
        if isinstance(first_interrupt, dict):
            interrupt_value = first_interrupt.get("value", {})
        else:
            interrupt_value = getattr(first_interrupt, "value", {}) or {}
        response["awaiting_approval"] = True
        response["approval_request"] = {
            "recommendation": interrupt_value.get("recommendation"),
            "refund_preview": interrupt_value.get("refund_preview"),
            "policy_citations": interrupt_value.get("policy_citations"),
        }
    else:
        response["awaiting_approval"] = False

    if values.get("refund_committed"):
        response["refund_committed"] = values.get("refund_committed")

    if values.get("refund_preview"):
        response["refund_preview"] = values.get("refund_preview")

    if values.get("approval_decision"):
        response["approval_decision"] = values.get("approval_decision")

    messages = values.get("messages", [])
    if messages:
        response["final_reply"] = messages[-1].get("content")

    return response


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs/start")
async def start_run(request: RunRequest) -> dict:
    result = await graph_manager.run_turn(
        thread_id=request.thread_id,
        message=request.message,
    )
    return _format_run_response(result)


@app.post("/runs/resume")
async def resume_run(request: ResumeRequest) -> dict:
    result = await graph_manager.resume_thread(thread_id=request.thread_id)
    return _format_run_response(result)


@app.post("/runs/approve")
async def approve_run(request: ApprovalRequest) -> dict:
    result = await graph_manager.submit_approval(
        thread_id=request.thread_id,
        approved=request.approved,
    )
    return _format_run_response(result)
