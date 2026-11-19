"""Persistence layer for measurements and anomalies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, UniqueConstraint, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models.schemas import AnomalyPoint, TimePoint
from app.settings import Settings

try:
    from sqlalchemy.orm import DeclarativeBase
    class Base(DeclarativeBase):
        """Declarative base model for SQLAlchemy metadata."""
except ImportError:
    from sqlalchemy.orm import declarative_base
    Base = declarative_base()

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc


class Measurement(Base):
    """SQL model for time-series measurements."""

    __tablename__ = "measurements"
    __table_args__ = (UniqueConstraint("service_id", "ts", name="uq_measurement_service_ts"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    service_id = Column(String(128), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    y = Column(Float, nullable=False)


class AnomalyRecord(Base):
    """SQL model for detected anomalies."""

    __tablename__ = "anomalies"
    __table_args__ = (UniqueConstraint("service_id", "ts", name="uq_anomaly_service_ts"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    service_id = Column(String(128), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    y = Column(Float, nullable=False)
    y_hat = Column(Float, nullable=True)
    residual = Column(Float, nullable=True)
    q_alpha = Column(Float, nullable=True)
    is_anomaly = Column(Boolean, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def init_db(settings: Settings) -> None:
    """Initialise the database engine and create tables.

    Args:
        settings: Application settings providing the database URL.
    """

    global engine, SessionLocal
    engine = create_engine(settings.effective_db_url, future=True, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session, autoflush=False)
    Base.metadata.create_all(engine)


def _ensure_initialized() -> None:
    if engine is None or SessionLocal is None:
        raise RuntimeError("Database engine is not initialised. Call init_db first.")


def _normalise_ts(ts: datetime) -> datetime:
    """Ensure timestamps are stored as UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _sqlite_chunk_size(row_length: int) -> int:
    """Return maximum rows per statement for SQLite based on variable limit."""

    max_sql_params = 999  # default SQLite param limit
    return max(1, max_sql_params // row_length)


def _sqlite_insert_in_chunks(
    session: Session,
    rows: list[dict],
    statement_factory: Callable[[list[dict]], any],
) -> int:
    """Execute SQLite insert statements in chunks to avoid variable limits."""

    if not rows:
        return 0
    chunk_size = _sqlite_chunk_size(len(rows[0]))
    inserted = 0
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        result = session.execute(statement_factory(chunk))
        inserted += result.rowcount or 0
    return inserted


def write_measurements(service_id: str, points: Iterable[TimePoint]) -> int:
    """Persist measurement points for a given service.

    Args:
        service_id: Service identifier belonging to the time-series.
        points: Collection of time-series points to persist.

    Returns:
        Number of points inserted (ignoring duplicates).
    """

    _ensure_initialized()
    assert engine and SessionLocal  # for type-checkers
    data = [{"service_id": service_id, "ts": _normalise_ts(point.ts), "y": float(point.y)} for point in points]
    if not data:
        return 0

    inserted = 0
    with SessionLocal.begin() as session:
        dialect = engine.dialect.name
        if dialect == "postgresql":
            stmt = pg_insert(Measurement).values(data)
            stmt = stmt.on_conflict_do_nothing(index_elements=["service_id", "ts"])
            result = session.execute(stmt)
            inserted = result.rowcount or 0
        elif dialect == "sqlite":

            def stmt_factory(chunk: list[dict]) -> Any:
                stmt = sqlite_insert(Measurement).values(chunk)
                return stmt.on_conflict_do_nothing(index_elements=["service_id", "ts"])

            inserted = _sqlite_insert_in_chunks(session, data, stmt_factory)
        else:
            for row in data:
                measurement = Measurement(**row)
                try:
                    session.add(measurement)
                    session.flush()
                    inserted += 1
                except IntegrityError:
                    session.rollback()
    return inserted


def write_anomalies(service_id: str, points: Iterable[AnomalyPoint]) -> int:
    """Persist anomaly detections for a given service.

    Args:
        service_id: Service identifier associated with anomalies.
        points: Iterable of anomaly records to store.

    Returns:
        Number of anomaly entries inserted (duplicates ignored).
    """

    _ensure_initialized()
    assert engine and SessionLocal
    data = []
    now = datetime.now(UTC)
    for point in points:
        data.append(
            {
                "service_id": service_id,
                "ts": _normalise_ts(point.ts),
                "y": float(point.y),
                "y_hat": float(point.y_hat) if point.y_hat is not None else None,
                "residual": float(point.residual) if point.residual is not None else None,
                "q_alpha": float(point.q_alpha) if point.q_alpha is not None else None,
                "is_anomaly": point.is_anomaly,
                "created_at": now,
            }
        )

    if not data:
        return 0

    inserted = 0
    with SessionLocal.begin() as session:
        dialect = engine.dialect.name
        if dialect == "postgresql":
            stmt = pg_insert(AnomalyRecord).values(data)
            stmt = stmt.on_conflict_do_nothing(index_elements=["service_id", "ts"])
            result = session.execute(stmt)
            inserted = result.rowcount or 0
        elif dialect == "sqlite":

            def stmt_factory(chunk: list[dict]) -> Any:
                stmt = sqlite_insert(AnomalyRecord).values(chunk)
                return stmt.on_conflict_do_nothing(index_elements=["service_id", "ts"])

            inserted = _sqlite_insert_in_chunks(session, data, stmt_factory)
        else:
            for row in data:
                record = AnomalyRecord(**row)
                try:
                    session.add(record)
                    session.flush()
                    inserted += 1
                except IntegrityError:
                    session.rollback()
    return inserted


def query_anomalies(service_id: str, since: datetime | None, limit: int = 100) -> list[AnomalyPoint]:
    """Retrieve persisted anomalies for a service.

    Args:
        service_id: Identifier of the service to query.
        since: Optional lower bound for timestamps.
        limit: Maximum number of rows to return.

    Returns:
        A list of anomaly points ordered chronologically.
    """

    _ensure_initialized()
    assert engine and SessionLocal
    with SessionLocal() as session:
        stmt = select(AnomalyRecord).where(AnomalyRecord.service_id == service_id)
        if since is not None:
            stmt = stmt.where(AnomalyRecord.ts >= _normalise_ts(since))
        stmt = stmt.order_by(AnomalyRecord.ts.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
    results: list[AnomalyPoint] = []
    for row in rows:
        results.append(
            AnomalyPoint(
                ts=row.ts,
                y=row.y,
                y_hat=row.y_hat,
                residual=row.residual,
                q_alpha=row.q_alpha,
                is_anomaly=row.is_anomaly,
            )
        )
    return list(reversed(results))


def query_measurements(service_id: str, since: datetime | None, limit: int = 1000) -> list[TimePoint]:
    """Retrieve measurements for a service ordered chronologically."""

    _ensure_initialized()
    assert engine and SessionLocal
    with SessionLocal() as session:
        stmt = select(Measurement).where(Measurement.service_id == service_id)
        if since is not None:
            stmt = stmt.where(Measurement.ts >= _normalise_ts(since))
        stmt = stmt.order_by(Measurement.ts.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
    points = [TimePoint(ts=row.ts, y=row.y) for row in reversed(rows)]
    return points


def count_measurements() -> int:
    """Return the total number of persisted measurements."""

    _ensure_initialized()
    assert SessionLocal
    with SessionLocal() as session:
        stmt = select(func.count()).select_from(Measurement)
        return int(session.execute(stmt).scalar_one())


def count_anomalies() -> int:
    """Return the total number of persisted anomalies."""

    _ensure_initialized()
    assert SessionLocal
    with SessionLocal() as session:
        stmt = select(func.count()).select_from(AnomalyRecord)
        return int(session.execute(stmt).scalar_one())


def list_services_with_anomalies(limit: int | None = None) -> list[str]:
    """Return distinct service identifiers that have stored anomalies."""

    _ensure_initialized()
    assert SessionLocal
    with SessionLocal() as session:
        stmt = select(AnomalyRecord.service_id).distinct().order_by(AnomalyRecord.service_id)
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).scalars().all()
    return list(rows)
