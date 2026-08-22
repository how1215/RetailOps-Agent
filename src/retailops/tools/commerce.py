import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from retailops.agent.policies import PolicyIndex
from retailops.domain.models import AuditEvent, Customer, IdempotencyRecord, Order


class ToolError(Exception):
    """Safe domain failure that can be shown to the model."""


class GetOrderArgs(BaseModel):
    order_id: str = Field(description="Exact order ID supplied by the customer")


class SearchPolicyArgs(BaseModel):
    query: str = Field(description="Concise policy question")


class CancelOrderArgs(BaseModel):
    order_id: str
    reason: str = Field(min_length=3, max_length=300)


class UpdateAddressArgs(BaseModel):
    order_id: str
    new_address: str = Field(min_length=8, max_length=500)


class CreateReturnArgs(BaseModel):
    order_id: str
    reason: str = Field(min_length=3, max_length=500)


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    customer_id: str
    idempotency_key: str | None = None


ToolHandler = Callable[[BaseModel, ToolContext, Session], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler
    requires_approval: bool = False

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class CommerceTools:
    def __init__(self, session_factory: sessionmaker[Session], policies: PolicyIndex) -> None:
        self.session_factory = session_factory
        self.policies = policies
        self._tools = {
            spec.name: spec
            for spec in [
                ToolSpec(
                    "get_order",
                    "Get an authenticated customer's order.",
                    GetOrderArgs,
                    self._get_order,
                ),
                ToolSpec(
                    "search_policy",
                    "Search commerce policies.",
                    SearchPolicyArgs,
                    self._search_policy,
                ),
                ToolSpec(
                    "cancel_order",
                    "Cancel a processing order after explicit approval.",
                    CancelOrderArgs,
                    self._cancel_order,
                    True,
                ),
                ToolSpec(
                    "update_shipping_address",
                    "Change a processing order's address after explicit approval.",
                    UpdateAddressArgs,
                    self._update_address,
                    True,
                ),
                ToolSpec(
                    "create_return",
                    "Create a return for an eligible delivered order after explicit approval.",
                    CreateReturnArgs,
                    self._create_return,
                    True,
                ),
            ]
        }

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.openai_schema() for spec in self._tools.values()]

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]

    def assert_customer_exists(self, customer_id: str) -> None:
        with self.session_factory() as session:
            if session.get(Customer, customer_id) is None:
                raise ToolError("Authenticated customer does not exist.")

    def validate(self, name: str, args: dict[str, Any]) -> BaseModel:
        return self.get(name).args_model.model_validate(args)

    def execute(
        self, name: str, args: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        spec = self.get(name)
        parsed = spec.args_model.model_validate(args)
        with self.session_factory() as session:
            if spec.requires_approval:
                if not context.idempotency_key:
                    raise ToolError("A mutation requires an approved idempotency key.")
                existing = session.scalar(
                    select(IdempotencyRecord).where(
                        IdempotencyRecord.key == context.idempotency_key,
                        IdempotencyRecord.action == name,
                    )
                )
                if existing:
                    return json.loads(existing.result_json)

            result = spec.handler(parsed, context, session)
            if spec.requires_approval:
                session.add(
                    IdempotencyRecord(
                        key=context.idempotency_key or "",
                        action=name,
                        result_json=json.dumps(result),
                    )
                )
            session.commit()
            return result

    @staticmethod
    def _owned_order(session: Session, order_id: str, customer_id: str) -> Order:
        order = session.scalar(
            select(Order).where(Order.id == order_id, Order.customer_id == customer_id)
        )
        if order is None:
            raise ToolError("Order not found for the authenticated customer.")
        return order

    def _get_order(self, args: BaseModel, context: ToolContext, session: Session) -> dict[str, Any]:
        assert isinstance(args, GetOrderArgs)
        order = self._owned_order(session, args.order_id, context.customer_id)
        return self._serialize_order(order)

    def _search_policy(
        self, args: BaseModel, context: ToolContext, session: Session
    ) -> dict[str, Any]:
        assert isinstance(args, SearchPolicyArgs)
        return {
            "matches": [
                {"source": chunk.source, "text": chunk.text, "score": chunk.score}
                for chunk in self.policies.search(args.query)
            ]
        }

    def _cancel_order(
        self, args: BaseModel, context: ToolContext, session: Session
    ) -> dict[str, Any]:
        assert isinstance(args, CancelOrderArgs)
        order = self._owned_order(session, args.order_id, context.customer_id)
        if order.status != "processing":
            raise ToolError(f"Only processing orders can be cancelled; status is {order.status}.")
        order.status = "cancelled"
        self._audit(session, context, "cancel_order", order.id, args.model_dump())
        return {"status": "cancelled", "order_id": order.id}

    def _update_address(
        self, args: BaseModel, context: ToolContext, session: Session
    ) -> dict[str, Any]:
        assert isinstance(args, UpdateAddressArgs)
        order = self._owned_order(session, args.order_id, context.customer_id)
        if order.status != "processing":
            raise ToolError(f"Only processing orders can be updated; status is {order.status}.")
        order.shipping_address = args.new_address
        self._audit(session, context, "update_shipping_address", order.id, args.model_dump())
        return {"status": "updated", "order_id": order.id, "shipping_address": args.new_address}

    def _create_return(
        self, args: BaseModel, context: ToolContext, session: Session
    ) -> dict[str, Any]:
        assert isinstance(args, CreateReturnArgs)
        order = self._owned_order(session, args.order_id, context.customer_id)
        if order.status != "delivered" or order.delivered_at is None:
            raise ToolError("Only delivered orders are eligible for return.")
        delivered_at = order.delivered_at
        if delivered_at.tzinfo is None:
            delivered_at = delivered_at.replace(tzinfo=UTC)
        if datetime.now(UTC) - delivered_at > timedelta(days=30):
            raise ToolError("The 30-day return window has expired.")
        if order.return_requested_at is not None:
            raise ToolError("A return already exists for this order.")
        order.return_requested_at = datetime.now(UTC)
        self._audit(session, context, "create_return", order.id, args.model_dump())
        return {"status": "return_requested", "order_id": order.id}

    @staticmethod
    def _audit(
        session: Session,
        context: ToolContext,
        action: str,
        resource_id: str,
        payload: dict[str, Any],
    ) -> None:
        session.add(
            AuditEvent(
                session_id=context.session_id,
                customer_id=context.customer_id,
                action=action,
                resource_id=resource_id,
                payload_json=json.dumps(payload),
            )
        )

    @staticmethod
    def _serialize_order(order: Order) -> dict[str, Any]:
        return {
            "id": order.id,
            "status": order.status,
            "item_name": order.item_name,
            "amount_cents": order.amount_cents,
            "shipping_address": order.shipping_address,
            "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
            "return_requested_at": (
                order.return_requested_at.isoformat() if order.return_requested_at else None
            ),
        }
