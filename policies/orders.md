# Order management policy

Customers may cancel an order only while its status is `processing`. Shipped, delivered,
cancelled, and returned orders cannot be cancelled. Cancellation is a state-changing operation
and requires explicit confirmation immediately before execution.

A shipping address may be changed only while an order is `processing`. The agent must show the
new address and receive explicit confirmation before applying the change.

The agent must verify that an order belongs to the authenticated customer. Never reveal another
customer's order, address, or payment information.
