from sqlalchemy import func, select

from retailops.domain.models import AuditEvent, Order
from retailops.tools.commerce import CommerceTools, ToolContext, ToolError


def test_order_lookup_is_scoped_to_authenticated_customer(tools: CommerceTools) -> None:
    context = ToolContext(session_id="ses_test", customer_id="cus_001")

    try:
        tools.execute("get_order", {"order_id": "ord_private"}, context)
    except ToolError as exc:
        assert "authenticated customer" in str(exc)
    else:
        raise AssertionError("Cross-customer lookup must fail")


def test_mutation_requires_idempotency_key(tools: CommerceTools) -> None:
    context = ToolContext(session_id="ses_test", customer_id="cus_001")

    try:
        tools.execute(
            "cancel_order",
            {"order_id": "ord_processing", "reason": "Ordered by mistake"},
            context,
        )
    except ToolError as exc:
        assert "idempotency" in str(exc)
    else:
        raise AssertionError("Unapproved mutation must fail")


def test_approved_mutation_is_idempotent(
    tools: CommerceTools, database
) -> None:
    context = ToolContext(
        session_id="ses_test",
        customer_id="cus_001",
        idempotency_key="act_same",
    )
    args = {"order_id": "ord_processing", "reason": "Ordered by mistake"}

    first = tools.execute("cancel_order", args, context)
    second = tools.execute("cancel_order", args, context)

    assert first == second == {"status": "cancelled", "order_id": "ord_processing"}
    with database.session_factory() as session:
        order = session.get(Order, "ord_processing")
        count = session.scalar(select(func.count()).select_from(AuditEvent))
        assert order is not None and order.status == "cancelled"
        assert count == 1
