from sqlalchemy.orm import Session
from backend.repository.db_init import SessionLocal
from backend.repository.models import Customer

# 강제 회원가입

db = SessionLocal()

DEFALT_CUSTOMERS = [
    Customer(name="홍길동", phone="010-1111-1111"),
    Customer(name="김철수", phone="010-2222-2222"),
    Customer(name="이영희", phone="010-3333-3333"),
]


def seed_customers(db: Session):
    """
    customer 에이블에 기본(연습용) 회원 데이터 삽입
    (중복 실행 방지: 있으면 실행 안하게)
    """
    existing = db.query(Customer).first()
    if existing:
        print("[Seed] customer 테이블에 이미 데이터가 있습니다.")
        return

    db.add_all(DEFALT_CUSTOMERS)
    db.commit()
    db.close()
    print(f"[Seed] 기본회원 {len(DEFALT_CUSTOMERS)}명 삽입 완료")
