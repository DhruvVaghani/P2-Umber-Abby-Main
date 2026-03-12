from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import traceable
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from app.payments import PaymentsClient


load_dotenv()


def append_messages(
    current: list[dict[str, str]],
    incoming: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [*current, *incoming]


class Citation(TypedDict):
    doc_id: str
    file_name: str
    section: str
    snippet: str
    citation_label: str


class CandidateChunk(TypedDict):
    doc_id: str
    file_name: str
    section: str
    snippet: str
    citation_label: str


class RetrievalPlan(BaseModel):
    strategy: str = Field(description="Retriever strategy to use")
    query: str = Field(description="Search query for vector retrieval")
    top_k: int = Field(description="Number of candidates to fetch")


class SelectedCitation(BaseModel):
    doc_id: str
    rationale: str = Field(description="Why this chunk supports the policy decision")


class CitationSelection(BaseModel):
    selected_doc_ids: list[str]
    rationale: str
    citations: list[SelectedCitation]


class FinalReply(BaseModel):
    message: str


class TriageState(TypedDict):
    thread_id: str
    user_message: str
    messages: Annotated[list[dict[str, str]], append_messages]
    issue_type: NotRequired[str]
    order_id: NotRequired[str]
    retrieval_strategy: NotRequired[str]
    retrieval_query: NotRequired[str]
    retrieval_k: NotRequired[int]
    policy_hits: NotRequired[list[Citation]]
    policy_supported: NotRequired[bool]
    recommendation: NotRequired[str]
    needs_approval: NotRequired[bool]
    refund_preview: NotRequired[dict[str, Any]]
    refund_committed: NotRequired[dict[str, Any]]
    approval_decision: NotRequired[str]
    status: NotRequired[
        Literal[
            "received",
            "classified",
            "policy_reviewed",
            "remedy_proposed",
            "awaiting_approval",
            "refund_committed",
            "approval_rejected",
            "completed",
        ]
    ]
    metadata: NotRequired[dict[str, Any]]


@dataclass
class TriageGraphManager:
    persist_directory: str = "mock_data/chroma"
    collection_name: str = "policy_chunks"
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/langgraph_phase2",
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def __post_init__(self) -> None:
        self.chroma_client: chromadb.ClientAPI | None = None
        self.policy_collection: Collection | None = None
        self.checkpointer_cm = None
        self.checkpointer: AsyncPostgresSaver | None = None
        self.graph = None
        self.payments = PaymentsClient()
        self.openai_client: OpenAI | None = None

    async def initialize(self) -> None:
        if self.graph is not None:
            return

        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.policy_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI()
        self.checkpointer_cm = AsyncPostgresSaver.from_conn_string(self.database_url)
        self.checkpointer = await self.checkpointer_cm.__aenter__()
        await self.checkpointer.setup()
        self.graph = self._build_graph()

    async def shutdown(self) -> None:
        if self.checkpointer_cm is not None:
            await self.checkpointer_cm.__aexit__(None, None, None)
            self.checkpointer_cm = None
            self.checkpointer = None
        self.graph = None

    def _build_graph(self):
        workflow = StateGraph(TriageState)
        workflow.add_node("intake", self.intake)
        workflow.add_node("issue_classifier", self.issue_classifier)
        workflow.add_node("kb_orchestrator", self.kb_orchestrator)
        workflow.add_node("policy_evaluator", self.policy_evaluator)
        workflow.add_node("propose_remedy", self.propose_remedy)
        workflow.add_node("finalize_reply", self.finalize_reply)

        workflow.add_edge(START, "intake")
        workflow.add_edge("intake", "issue_classifier")
        workflow.add_edge("issue_classifier", "kb_orchestrator")
        workflow.add_edge("kb_orchestrator", "policy_evaluator")
        workflow.add_edge("policy_evaluator", "propose_remedy")
        workflow.add_edge("propose_remedy", "finalize_reply")
        workflow.add_edge("finalize_reply", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _thread_config(self, thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    async def _snapshot_response(self, thread_id: str) -> dict[str, Any]:
        snapshot = await self.graph.aget_state(self._thread_config(thread_id))
        values = dict(snapshot.values) if snapshot and snapshot.values else {}
        interrupts: list[Any] = []
        if snapshot and getattr(snapshot, "tasks", None):
            for task in snapshot.tasks:
                task_interrupts = getattr(task, "interrupts", None)
                if task_interrupts:
                    interrupts.extend(task_interrupts)

        return {
            "thread_id": thread_id,
            "values": values,
            "next": list(snapshot.next) if snapshot and snapshot.next else [],
            "interrupts": interrupts,
        }

    async def run_turn(self, thread_id: str, message: str) -> dict[str, Any]:
        if self.graph is None:
            await self.initialize()

        await self.graph.ainvoke(
            {
                "thread_id": thread_id,
                "user_message": message,
            },
            config=self._thread_config(thread_id),
        )
        return await self._snapshot_response(thread_id)

    async def resume_thread(self, thread_id: str) -> dict[str, Any]:
        if self.graph is None:
            await self.initialize()

        return await self._snapshot_response(thread_id)

    async def submit_approval(self, thread_id: str, approved: bool) -> dict[str, Any]:
        if self.graph is None:
            await self.initialize()

        decision = "approved" if approved else "rejected"
        await self.graph.ainvoke(
            Command(resume={"approved": approved, "decision": decision}),
            config=self._thread_config(thread_id),
        )
        return await self._snapshot_response(thread_id)

    def intake(self, state: TriageState) -> TriageState:
        return {
            "messages": [
                {"role": "user", "content": state["user_message"]},
            ],
            "status": "received",
            "metadata": {"source": "api"},
        }

    @traceable(name="issue_classifier", run_type="chain")
    def issue_classifier(self, state: TriageState) -> TriageState:
        message = state["user_message"].lower()
        issue_type = "general_support"
        order_id = self._extract_order_id(state["user_message"])

        if "charged twice" in message or "duplicate charge" in message:
            issue_type = "duplicate_charge"
        elif "late" in message or "delayed" in message:
            issue_type = "late_delivery"
        elif "stopped working" in message or "warranty" in message:
            issue_type = "warranty"
        elif "missing" in message:
            issue_type = "missing_item"
        elif "refund" in message:
            issue_type = "refund_request"

        return {
            "issue_type": issue_type,
            "order_id": order_id,
            "status": "classified",
        }

    @traceable(name="policy_evaluator", run_type="chain")
    def policy_evaluator(self, state: TriageState) -> TriageState:
        policy_hits = state.get("policy_hits", [])
        metadata = state.get("metadata", {})

        if not policy_hits:
            return {
                "recommendation": "No policy support found. Escalate to a human reviewer.",
                "policy_supported": False,
                "status": "policy_reviewed",
                "metadata": {
                    **metadata,
                    "policy_evaluator": {"has_citations": False},
                },
            }

        return {
            "policy_hits": policy_hits,
            "policy_supported": True,
            "status": "policy_reviewed",
            "metadata": {
                **metadata,
                "policy_evaluator": {"has_citations": True},
            },
        }

    @traceable(name="kb_orchestrator", run_type="retriever")
    def kb_orchestrator(self, state: TriageState) -> TriageState:
        strategy, query, top_k = self._plan_retrieval(state)
        candidates = self._run_top_k_retrieval(query, top_k)
        citations = self._select_citations_with_llm(state, candidates)

        return {
            "retrieval_strategy": strategy,
            "retrieval_query": query,
            "retrieval_k": top_k,
            "policy_hits": citations,
            "metadata": {
                **state.get("metadata", {}),
                "kb": {
                    "strategy": strategy,
                    "query": query,
                    "top_k": top_k,
                    "candidate_doc_ids": [candidate["doc_id"] for candidate in candidates],
                    "selected_doc_ids": [citation["doc_id"] for citation in citations],
                },
            },
        }

    @traceable(name="propose_remedy", run_type="tool")
    def propose_remedy(self, state: TriageState) -> TriageState:
        issue_type = state["issue_type"]
        recommendation_map = {
            "duplicate_charge": "Prepare a full refund preview for the duplicate charge.",
            "late_delivery": "Prepare a partial refund preview for the shipping delay.",
            "warranty": "Prepare a replacement proposal under warranty policy.",
            "missing_item": "Prepare a partial refund preview for the missing item.",
            "refund_request": "Validate eligibility and prepare a refund preview.",
            "general_support": "Escalate to support with policy-backed context.",
        }
        recommendation = recommendation_map[issue_type]

        if not state.get("policy_supported", False):
            return {
                "recommendation": state.get(
                    "recommendation",
                    "No policy support found. Escalate to a human reviewer.",
                ),
                "needs_approval": False,
                "status": "remedy_proposed",
            }

        if issue_type == "general_support":
            return {
                "recommendation": recommendation,
                "needs_approval": False,
                "status": "remedy_proposed",
            }

        preview = state.get("refund_preview") or self.payments.refund_preview(
            thread_id=state["thread_id"],
            issue_type=issue_type,
            recommendation=recommendation,
            order_id=state.get("order_id"),
        )
        approval = interrupt(
            {
                "kind": "refund_approval",
                "thread_id": state["thread_id"],
                "issue_type": issue_type,
                "recommendation": recommendation,
                "refund_preview": preview,
                "policy_citations": [hit["doc_id"] for hit in state.get("policy_hits", [])],
            }
        )

        if not approval.get("approved", False):
            return {
                "recommendation": "Approval rejected. Escalate to a human reviewer.",
                "needs_approval": True,
                "refund_preview": preview,
                "approval_decision": approval.get("decision", "rejected"),
                "status": "approval_rejected",
            }

        committed = self.payments.refund_commit(preview)

        return {
            "recommendation": recommendation,
            "needs_approval": True,
            "refund_preview": preview,
            "refund_committed": committed,
            "approval_decision": approval.get("decision", "approved"),
            "status": "refund_committed",
        }

    @traceable(name="finalize_reply", run_type="chain")
    def finalize_reply(self, state: TriageState) -> TriageState:
        citations = state.get("policy_hits", [])
        preview = state.get("refund_preview")
        commit = state.get("refund_committed")
        if commit:
            action_text = (
                f"{state['recommendation']} Commit id: {commit['commit_id']} "
                f"for preview {commit['preview_id']}."
            )
        elif preview:
            action_text = (
                f"{state['recommendation']} Preview id: {preview['preview_id']} "
                f"for amount {preview['amount']} {preview['currency']}."
            )
        else:
            action_text = state["recommendation"]

        citation_lines = [
            f"{citation['citation_label']}: {citation['snippet']}"
            for citation in citations
        ]
        final_message = self._generate_final_reply(
            user_message=state["user_message"],
            recommendation=action_text,
            citations=citation_lines,
            approved=state.get("approval_decision"),
        )

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": final_message,
                }
            ],
            "status": "completed",
        }

    def _lookup_policy_hits(self, issue_type: str) -> list[Citation]:
        return self._run_top_k_retrieval(issue_type.replace("_", " "), 3)

    @traceable(name="run_top_k_retrieval", run_type="retriever")
    def _run_top_k_retrieval(self, query: str, top_k: int) -> list[CandidateChunk]:
        if self.policy_collection is None:
            return []

        results = self.policy_collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        citations: list[CandidateChunk] = []
        for doc_id, metadata, snippet in zip(ids, metadatas, documents):
            citations.append(
                {
                    "doc_id": doc_id,
                    "file_name": str(metadata.get("file_name", "unknown"))
                    if metadata
                    else "unknown",
                    "section": str(metadata.get("section", "unknown"))
                    if metadata
                    else "unknown",
                    "snippet": snippet,
                    "citation_label": self._format_citation_label(
                        str(metadata.get("file_name", "unknown")) if metadata else "unknown",
                        str(metadata.get("section", "unknown")) if metadata else "unknown",
                    ),
                }
            )
        return citations

    def _plan_retrieval(self, state: TriageState) -> tuple[str, str, int]:
        llm_plan = self._plan_retrieval_with_llm(state)
        if llm_plan is not None:
            return llm_plan.strategy, llm_plan.query, max(2, min(llm_plan.top_k, 6))

        strategy = self._choose_retrieval_strategy(state)
        query = self._build_retrieval_query(state, strategy)
        top_k = self._determine_top_k(strategy)
        return strategy, query, top_k

    @traceable(name="plan_retrieval_with_llm", run_type="llm")
    def _plan_retrieval_with_llm(self, state: TriageState) -> RetrievalPlan | None:
        if self.openai_client is None:
            return None

        prompt = (
            "You are planning retrieval for a customer-support policy agent. "
            "Choose one strategy from: policy_keyword, issue_plus_message, message_only. "
            "Return the best search query and a top_k value between 2 and 6."
        )
        try:
            response = self.openai_client.responses.parse(
                model=self.openai_model,
                input=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Issue type: {state['issue_type']}\n"
                            f"Order id: {state.get('order_id')}\n"
                            f"User message: {state['user_message']}"
                        ),
                    },
                ],
                text_format=RetrievalPlan,
            )
            return response.output_parsed
        except Exception:
            return None

    @traceable(name="select_citations_with_llm", run_type="chain")
    def _select_citations_with_llm(
        self,
        state: TriageState,
        candidates: list[CandidateChunk],
    ) -> list[Citation]:
        if not candidates:
            return []

        llm_selection = self._rerank_candidates_with_llm(state, candidates)
        if llm_selection is not None:
            candidate_map = {candidate["doc_id"]: candidate for candidate in candidates}
            structured: list[Citation] = []
            for selected in llm_selection.citations:
                candidate = candidate_map.get(selected.doc_id)
                if candidate is None:
                    continue
                structured.append(
                    {
                        "doc_id": candidate["doc_id"],
                        "file_name": candidate["file_name"],
                        "section": candidate["section"],
                        "snippet": candidate["snippet"],
                        "citation_label": candidate["citation_label"],
                    }
                )
            if structured:
                return structured

        return [
            {
                "doc_id": candidate["doc_id"],
                "file_name": candidate["file_name"],
                "section": candidate["section"],
                "snippet": candidate["snippet"],
                "citation_label": candidate["citation_label"],
            }
            for candidate in candidates[: min(3, len(candidates))]
        ]

    @traceable(name="rerank_candidates_with_llm", run_type="llm")
    def _rerank_candidates_with_llm(
        self,
        state: TriageState,
        candidates: list[CandidateChunk],
    ) -> CitationSelection | None:
        if self.openai_client is None:
            return None

        candidate_text = "\n\n".join(
            (
                f"doc_id: {candidate['doc_id']}\n"
                f"file_name: {candidate['file_name']}\n"
                f"section: {candidate['section']}\n"
                f"snippet: {candidate['snippet']}"
            )
            for candidate in candidates
        )
        prompt = (
            "You are selecting policy citations for a support agent. "
            "Choose only from the provided candidate doc_ids. "
            "Return the best 1 to 3 citations that support the action."
        )
        try:
            response = self.openai_client.responses.parse(
                model=self.openai_model,
                input=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Issue type: {state['issue_type']}\n"
                            f"User message: {state['user_message']}\n\n"
                            f"Candidate chunks:\n{candidate_text}"
                        ),
                    },
                ],
                text_format=CitationSelection,
            )
            return response.output_parsed
        except Exception:
            return None

    def _choose_retrieval_strategy(self, state: TriageState) -> str:
        issue_type = state["issue_type"]
        if issue_type in {"duplicate_charge", "refund_request", "missing_item"}:
            return "policy_keyword"
        if issue_type in {"late_delivery", "warranty"}:
            return "issue_plus_message"
        return "message_only"

    def _build_retrieval_query(self, state: TriageState, strategy: str) -> str:
        issue_type = state["issue_type"].replace("_", " ")
        user_message = state["user_message"]
        order_id = state.get("order_id")

        if strategy == "policy_keyword":
            return issue_type
        if strategy == "issue_plus_message":
            parts = [issue_type, user_message]
            if order_id:
                parts.append(order_id)
            return " ".join(parts)
        return user_message

    def _determine_top_k(self, strategy: str) -> int:
        if strategy == "issue_plus_message":
            return 4
        return 3

    def _format_citation_label(self, file_name: str, section: str) -> str:
        return f"{file_name} | {section}"

    @traceable(name="generate_final_reply", run_type="llm")
    def _generate_final_reply(
        self,
        *,
        user_message: str,
        recommendation: str,
        citations: list[str],
        approved: str | None,
    ) -> str:
        if self.openai_client is None:
            citations_text = "; ".join(citations) if citations else "No policy citations available."
            return f"{recommendation} Policy support: {citations_text}"

        prompt = (
            "You are a support agent writing the final customer-facing response. "
            "Use only the provided recommendation and citations. "
            "Keep it concise, mention approval status when relevant, and cite policies as file | section."
        )
        approval_text = approved or "not_applicable"
        try:
            response = self.openai_client.responses.parse(
                model=self.openai_model,
                input=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"User message: {user_message}\n"
                            f"Recommendation: {recommendation}\n"
                            f"Approval status: {approval_text}\n"
                            f"Citations:\n" + "\n".join(citations)
                        ),
                    },
                ],
                text_format=FinalReply,
            )
            if response.output_parsed and response.output_parsed.message:
                return response.output_parsed.message
        except Exception:
            pass

        citations_text = "; ".join(citations) if citations else "No policy citations available."
        return f"{recommendation} Policy support: {citations_text}"

    def _extract_order_id(self, message: str) -> str | None:
        match = re.search(r"\bORD\d+\b", message.upper())
        if match:
            return match.group(0)
        return None
