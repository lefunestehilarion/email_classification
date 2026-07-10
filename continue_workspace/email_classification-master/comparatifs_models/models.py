"""Modèles SQLAlchemy (SQLite) : emails, labels (many-to-many), et feedback utilisateur."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, ForeignKey, Integer, String, Table, Text, create_engine, DateTime, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


email_labels = Table(
    "email_labels",
    Base.metadata,
    Column("email_id", ForeignKey("emails.id", ondelete="CASCADE"), primary_key=True),
    Column("label_id", ForeignKey("labels.id", ondelete="CASCADE"), primary_key=True),
)


class Email(Base):
    """Un mail ; les libellés passent par ``labels`` (objets ``Label``)."""

    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    split: Mapped[str] = mapped_column(String(64), index=True)
    subject: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)

    labels: Mapped[list["Label"]] = relationship(
        secondary=email_labels,
        back_populates="emails",
    )
    feedback_entries: Mapped[list["Feedback"]] = relationship(back_populates="email")


class Label(Base):
    """Un libellé unique (ex. « Business ») ; partagé par plusieurs mails."""

    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    emails: Mapped[list["Email"]] = relationship(
        secondary=email_labels,
        back_populates="labels",
    )


class Feedback(Base):
    """Retour utilisateur sur une prédiction zero-shot.

    label_predit / label_correct sont stockés en texte (nom du label), pas
    en clé étrangère stricte, pour rester robustes même si un label est
    renommé ou si le libellé prédit n'existe plus dans la table `labels`.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), index=True)
    label_predit: Mapped[str] = mapped_column(String(256))
    label_correct: Mapped[str] = mapped_column(String(256))
    score_predit: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())
    used_in_training: Mapped[int] = mapped_column(Integer, default=0)  # 0/1, évite de réutiliser un feedback déjà appris

    email: Mapped["Email"] = relationship(back_populates="feedback_entries")


def sqlite_engine(db_path: str | Path):
    path = Path(db_path).resolve()
    return create_engine(f"sqlite:///{path}", echo=False)
