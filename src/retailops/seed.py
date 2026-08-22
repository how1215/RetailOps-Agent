from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from retailops.db import Database
from retailops.domain.models import Customer, Order


def seed_database(database: Database) -> None:
    database.create_schema()
    with database.session_factory() as session:
        if session.scalar(select(Customer.id).limit(1)) is not None:
            return

        customer = Customer(id="cus_001", name="Alex Chen", email="alex@example.com")
        second_customer = Customer(id="cus_002", name="Morgan Lin", email="morgan@example.com")
        session.add_all(
            [
                customer,
                second_customer,
                Order(
                    id="ord_processing",
                    customer=customer,
                    status="processing",
                    item_name="Mechanical Keyboard",
                    amount_cents=329900,
                    shipping_address="100 Demo Road, Taipei",
                ),
                Order(
                    id="ord_shipped",
                    customer=customer,
                    status="shipped",
                    item_name="USB-C Dock",
                    amount_cents=219900,
                    shipping_address="100 Demo Road, Taipei",
                ),
                Order(
                    id="ord_delivered",
                    customer=customer,
                    status="delivered",
                    item_name="Webcam",
                    amount_cents=189900,
                    shipping_address="100 Demo Road, Taipei",
                    delivered_at=datetime.now(UTC) - timedelta(days=7),
                ),
                Order(
                    id="ord_private",
                    customer=second_customer,
                    status="processing",
                    item_name="Private Item",
                    amount_cents=99900,
                    shipping_address="Private address",
                ),
            ]
        )
        session.commit()
