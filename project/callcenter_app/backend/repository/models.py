from backend.repository.db_init import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, func, String, ForeignKey
from datetime import datetime


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)

    def __str__(self):
        return f"<Customer id={self.id} name={self.name} phone={self.phone}>"


# 상담 기록 저장
class CallHistory(Base):
    __tablename__ = "call_history"
    call_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    transcript: Mapped[str]
    summary: Mapped[str]
    category: Mapped[str]
    sentiment: Mapped[str]
    customer_issue: Mapped[str]
    resolution: Mapped[str]
    # created_at: Mapped[datetime] = mapped_column(
    #     server_default=func.now(), nullable=False
    # )
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(), nullable=False)
