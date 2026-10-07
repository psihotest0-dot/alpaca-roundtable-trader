with open("src/trading_system/execution/engine.py", "r") as f:
    content = f.read()

# First replace the method call to mock dry-run early in submit_order!
idx = content.find("            # Check for existing")
if idx != -1:
    replace_submit = """            if self.is_dry_run:
                logger.info(f"[DRY-RUN] Would submit order {client_order_id} for {intent.symbol}")
                broker_order = BrokerOrder(
                    broker_id=f"dry_run_{client_order_id}",
                    client_order_id=client_order_id,
                    symbol=str(intent.symbol),
                    side=intent.side,
                    type=intent.type,
                    status=OrderStatus.SUBMITTED,
                    quantity=intent.quantity,
                    filled_quantity=Decimal("0"),
                    limit_price=intent.limit_price,
                    stop_price=intent.stop_price,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    updated_at=datetime.datetime.now(datetime.timezone.utc),
                )
                try:
                    self.order_repo.save_broker_order(broker_order)
                except Exception as e:
                    logger.error(f"Failed to persist dry-run broker order {client_order_id}: {e}")
                    return None
                return broker_order

            # Check for existing"""
    content = content[:idx] + replace_submit + content[idx + len("            # Check for existing"):]

with open("src/trading_system/execution/engine.py", "w") as f:
    f.write(content)
