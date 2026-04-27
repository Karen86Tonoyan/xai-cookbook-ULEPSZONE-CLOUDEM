from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal


TemporalPartition = Literal["yesterday", "today", "tomorrow"]
Intent = Literal["recall", "analyze", "execute", "plan", "predict", "reflect"]
Domain = Literal["code", "data", "creative", "ops", "research", "conversation", "unknown"]


TEMPORAL_SIGNALS: dict[TemporalPartition, list[str]] = {
    "yesterday": [
        "yesterday",
        "last week",
        "previously",
        "history",
        "recall",
        "remember",
        "past",
        "earlier",
        "retrospective",
        "lessons",
        "archive",
        "wczoraj",
        "poprzednio",
        "historia",
        "wczesniej",
        "pamietaj",
        "bylo",
        "archiwum",
    ],
    "today": [
        "now",
        "current",
        "today",
        "right now",
        "fix",
        "debug",
        "run",
        "execute",
        "build",
        "check",
        "implement",
        "status",
        "teraz",
        "dzis",
        "zrob",
        "sprawdz",
        "uruchom",
        "napraw",
    ],
    "tomorrow": [
        "tomorrow",
        "next week",
        "plan",
        "roadmap",
        "future",
        "forecast",
        "predict",
        "what if",
        "strategy",
        "vision",
        "idea",
        "jutro",
        "przyszlosc",
        "strategia",
        "prognoza",
        "co jesli",
        "wizja",
    ],
}

INTENT_SIGNALS: dict[Intent, list[str]] = {
    "recall": ["remember", "what was", "history", "log", "wczoraj", "historia"],
    "analyze": ["analyze", "why", "explain", "compare", "analizuj"],
    "execute": ["run", "do", "build", "fix", "write", "implement", "zrob", "uruchom"],
    "plan": ["plan", "schedule", "roadmap", "goal", "strategy", "zaplanuj"],
    "predict": ["predict", "forecast", "what if", "estimate", "prognoza"],
    "reflect": ["review", "lessons", "retrospective", "podsumuj"],
}

DOMAIN_SIGNALS: dict[Domain, list[str]] = {
    "code": ["code", "function", "bug", "class", "typescript", "python", "kod"],
    "data": ["data", "database", "query", "metric", "chart", "dane", "baza"],
    "creative": ["design", "idea", "story", "art", "pomysl", "projekt"],
    "ops": ["deploy", "server", "pipeline", "docker", "serwer"],
    "research": ["research", "study", "paper", "learn", "badanie"],
    "conversation": ["hello", "hi", "thanks", "okay", "czesc", "dzieki"],
    "unknown": [],
}


@dataclass
class StudioLabel:
    partition: TemporalPartition
    intent: Intent
    domain: Domain
    confidence: float
    signals: list[str]
    labeled_at: str


@dataclass
class PartitionConfig:
    partition: TemporalPartition
    system_prompt: str
    temperature: float
    max_tokens: int
    memory_window: int
    description: str


@dataclass
class RoutedRequest:
    label: StudioLabel
    partition: PartitionConfig
    messages: list[dict[str, str]]
    model_profile: dict[str, str | int | float]


PARTITIONS: dict[TemporalPartition, PartitionConfig] = {
    "yesterday": PartitionConfig(
        partition="yesterday",
        description="History, memory recall, retrospectives.",
        system_prompt=(
            "You are operating in YESTERDAY mode. Prioritize historical context, "
            "past decisions, lessons learned, and careful recall."
        ),
        temperature=0.3,
        max_tokens=2048,
        memory_window=20,
    ),
    "today": PartitionConfig(
        partition="today",
        description="Active execution, debugging, and present-state analysis.",
        system_prompt=(
            "You are operating in TODAY mode. Focus on the active task, inspect current "
            "state, and provide direct executable next steps."
        ),
        temperature=0.5,
        max_tokens=4096,
        memory_window=10,
    ),
    "tomorrow": PartitionConfig(
        partition="tomorrow",
        description="Planning, forecasting, and strategy.",
        system_prompt=(
            "You are operating in TOMORROW mode. Think ahead, build roadmaps, "
            "compare options, and surface risks."
        ),
        temperature=0.8,
        max_tokens=3072,
        memory_window=5,
    ),
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def score_signals(text: str, signal_map: dict[str, list[str]]) -> dict[str, int]:
    lowered = normalize(text)
    return {
        key: sum(1 for signal in signals if signal in lowered)
        for key, signals in signal_map.items()
    }


def pick_top(scores: dict[str, int], default: str) -> tuple[str, float]:
    total = sum(scores.values())
    if total == 0:
        return default, 0.33
    winner, top_score = max(scores.items(), key=lambda item: item[1])
    return winner, round(min(top_score / total, 1.0), 2)


class TemporalTagger:
    def __init__(self) -> None:
        self.history: list[StudioLabel] = []

    def label(self, prompt: str) -> StudioLabel:
        temporal_scores = score_signals(prompt, TEMPORAL_SIGNALS)
        intent_scores = score_signals(prompt, INTENT_SIGNALS)
        domain_scores = score_signals(prompt, DOMAIN_SIGNALS)

        partition, confidence = pick_top(temporal_scores, "today")
        if confidence < 0.4 and self.history:
            partition = self.history[-1].partition
            confidence = max(confidence, 0.3)

        intent, _ = pick_top(intent_scores, "execute")
        domain, _ = pick_top(domain_scores, "unknown")
        signals = [
            signal
            for signal in TEMPORAL_SIGNALS[partition]  # type: ignore[index]
            if signal in normalize(prompt)
        ][:5]

        label = StudioLabel(
            partition=partition,  # type: ignore[arg-type]
            intent=intent,  # type: ignore[arg-type]
            domain=domain,  # type: ignore[arg-type]
            confidence=confidence,
            signals=signals,
            labeled_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )
        self.history.append(label)
        self.history = self.history[-50:]
        return label


class PartitionRouter:
    def route(
        self,
        prompt: str,
        label: StudioLabel,
        conversation_history: list[dict[str, str]] | None = None,
        model_name: str = "gpt-oss:120b",
    ) -> RoutedRequest:
        partition = PARTITIONS[label.partition]
        history = conversation_history or []
        trimmed_history = history[-partition.memory_window * 2 :]

        messages = [
            {"role": "system", "content": partition.system_prompt},
            *trimmed_history,
            {"role": "user", "content": prompt},
        ]

        model_profile = {
            "model": model_name,
            "temperature": partition.temperature,
            "num_predict": partition.max_tokens,
            "memory_window": partition.memory_window,
        }

        return RoutedRequest(
            label=label,
            partition=partition,
            messages=messages,
            model_profile=model_profile,
        )


def route_prompt(prompt: str, history: list[dict[str, str]] | None = None) -> RoutedRequest:
    tagger = TemporalTagger()
    label = tagger.label(prompt)
    return PartitionRouter().route(prompt, label, history)


if __name__ == "__main__":
    examples = [
        "Remember what we changed yesterday in the Ollama setup.",
        "Sprawdz teraz build i napraw blad w pipeline.",
        "Build a roadmap for releasing this next week.",
    ]
    for example in examples:
        routed = route_prompt(example)
        print(json.dumps(asdict(routed), indent=2, ensure_ascii=False))
