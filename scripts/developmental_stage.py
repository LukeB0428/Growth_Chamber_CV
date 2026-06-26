"""
developmental_stage.py — Unified phenological stage tracker

Reads existing signals already computed by the pipeline (bolting_flag,
germination_flag, canopy_cover_%, canopy_height_max_mm) and CSV history
to assign a BBCH-coded developmental stage each day.

Stage definitions live in config/species/{species}.json — zero code changes
needed to support a new species' stage ladder.

Architecture:
    BaseStageDetector (ABC)
      ├── ArabidopsisStageDetector  (calibrated on EE496 trial data)
      ├── BarleyStageDetector       (cover/height skeleton — extend Q4 2026)
      └── BrassicaStageDetector     (cover/bolt skeleton — extend Q1 2027)

Main interface:
    stage = detect_stage(current_metrics, chamber_id, pot_label, csv_path, cfg)
    # → DevelopmentalStage(stage_id, stage_name, bbch_code, confidence)
"""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DevelopmentalStage:
    stage_id:   int
    stage_name: str
    bbch_code:  int
    confidence: float   # 0.0 – 1.0


class BaseStageDetector(ABC):
    def __init__(self, cfg: dict):
        ds = cfg.get('developmental_stage', {})
        self.stages     = ds.get('stages', [])
        self.thresholds = ds.get('thresholds', {})

    @abstractmethod
    def detect(self, current_metrics: dict, history_rows: list) -> DevelopmentalStage:
        ...

    def _stage(self, name: str, confidence: float = 0.8) -> DevelopmentalStage:
        for s in self.stages:
            if s['name'] == name:
                return DevelopmentalStage(s['id'], s['name'], s['bbch'], confidence)
        return DevelopmentalStage(0, 'unknown', 0, 0.0)


class ArabidopsisStageDetector(BaseStageDetector):
    """
    Ladder: dormant → germination → seedling → vegetative →
            bolting → reproductive → senescence

    Signals (all already in pot_metrics.csv):
      canopy_cover_%   — primary growth proxy
      bolting_flag     — 1 when bolting_detection.py fires
      germination_flag — 1 when first green pixels cross threshold
    """

    def detect(self, current_metrics: dict, history_rows: list) -> DevelopmentalStage:
        cover = float(current_metrics.get('canopy_cover_%')      or 0.0)
        bolt  = int(current_metrics.get('bolting_flag')          or 0)
        germ  = int(current_metrics.get('germination_flag')      or 0)
        t     = self.thresholds

        hist_covers = [float(r.get('canopy_cover_%') or 0.0) for r in history_rows]
        all_covers  = hist_covers + [cover]

        # 1. Senescence — sustained drop from peak (checked first to handle post-reproductive decline)
        min_days = int(t.get('senescence_min_days', 5))
        if len(all_covers) >= min_days:
            peak = max(all_covers)
            if peak > 10.0:
                drop_threshold = peak * (1.0 - float(t.get('senescence_cover_drop_pct', 30.0)) / 100.0)
                if all(v < drop_threshold for v in all_covers[-min_days:]):
                    return self._stage('senescence', confidence=0.9)

        # 2. Bolting / reproductive — once bolting has ever fired, never revert to vegetative
        prev_bolt_days = sum(1 for r in history_rows if int(r.get('bolting_flag') or 0) == 1)
        ever_bolted    = prev_bolt_days > 0

        if bolt or ever_bolted:
            total_bolt_days = prev_bolt_days + (1 if bolt else 0)
            repro_threshold = int(t.get('reproductive_days_after_bolt', 7))
            if total_bolt_days >= repro_threshold:
                return self._stage('reproductive', confidence=0.95)
            conf = round(min(0.7 + 0.05 * total_bolt_days, 0.9), 2)
            return self._stage('bolting', confidence=conf)

        # 3. Rosette growth — cover thresholds
        if cover >= float(t.get('vegetative_cover_pct', 5.0)):
            return self._stage('vegetative', confidence=0.9)
        if cover >= float(t.get('seedling_cover_pct', 1.0)):
            return self._stage('seedling', confidence=0.8)
        if cover >= float(t.get('germination_cover_pct', 0.5)) or germ:
            return self._stage('germination', confidence=0.8)

        return self._stage('dormant', confidence=0.95)


class BarleyStageDetector(BaseStageDetector):
    """
    Skeleton for Hordeum vulgare — calibration target Q4 2026 (Grace's trial).

    Stages beyond tillering require:
      - stem_extension / heading: canopy_height_max_mm vs heading_depth_height_mm
      - grain_fill / maturity:    colour shift (golden hue) — needs colour_stage.py

    For now: cover ladder up to tillering, then height proxy for heading.
    Confidence is deliberately low (≤ 0.6) to flag these as estimates.
    """

    def detect(self, current_metrics: dict, history_rows: list) -> DevelopmentalStage:
        cover  = float(current_metrics.get('canopy_cover_%')       or 0.0)
        height = float(current_metrics.get('canopy_height_max_mm') or 0.0)
        t      = self.thresholds

        heading_h = float(t.get('heading_depth_height_mm', 60.0))
        if height > heading_h:
            return self._stage('heading', confidence=0.5)
        if cover >= float(t.get('tillering_cover_pct', 8.0)):
            return self._stage('tillering', confidence=0.6)
        if cover >= float(t.get('seedling_cover_pct', 2.0)):
            return self._stage('seedling', confidence=0.7)
        if cover >= float(t.get('germination_cover_pct', 0.5)):
            return self._stage('germination', confidence=0.7)
        return self._stage('dormant', confidence=0.9)


class BrassicaStageDetector(BaseStageDetector):
    """
    Brassica napus / rapa — full BBCH lifecycle tracker.

    Ladder: dormant → germination → seedling → vegetative → bolting →
            flowering → pod_fill → senescence

    Signals (per day; history from pot_metrics.csv):
      canopy_cover_%   — growth proxy (germination → vegetative; senescence drop)
      bolting_flag     — bolting onset (BBCH 51), before petals open
      flower_area_pct  — REAL yellow-flower coverage (flower_metrics.py); drives
                         flowering (BBCH 61) and, once it declines, pod_fill (71)
      germination_flag — first green pixels

    Phenology is monotonic: `ever_*` history flags stop the stage reverting (e.g.
    a low-flower day mid-flowering does not drop back to bolting). pod_fill is an
    RGB heuristic (flowered, then flowering sustained-declined → siliques forming),
    so its confidence is low until the pod model validates it.
    """

    def detect(self, current_metrics: dict, history_rows: list) -> DevelopmentalStage:
        t = self.thresholds
        cover = float(current_metrics.get('canopy_cover_%') or 0.0)
        bolt  = int(current_metrics.get('bolting_flag') or 0)
        germ  = int(current_metrics.get('germination_flag') or 0)
        flower = float(current_metrics.get('flower_area_pct',
                                           current_metrics.get('flowering_pct')) or 0.0)

        flower_th = float(t.get('flowering_yellow_pixel_pct', 2.0))
        hist_cover  = [float(r.get('canopy_cover_%') or 0.0) for r in history_rows]
        hist_flower = [float(r.get('flower_area_pct') or 0.0) for r in history_rows]
        ever_bolted  = bolt == 1 or any(int(r.get('bolting_flag') or 0) == 1 for r in history_rows)
        ever_flowered = flower >= flower_th or any(f >= flower_th for f in hist_flower)

        # 1. Senescence — sustained canopy drop from peak (post-reproductive decline).
        min_days = int(t.get('senescence_min_days', 5))
        all_cover = hist_cover + [cover]
        if len(all_cover) >= min_days:
            peak = max(all_cover)
            if peak > float(t.get('vegetative_cover_pct', 12.0)):
                drop = peak * (1.0 - float(t.get('senescence_cover_drop_pct', 35.0)) / 100.0)
                if all(v < drop for v in all_cover[-min_days:]):
                    return self._stage('senescence', confidence=0.85)

        # 2. Reproductive ladder (monotonic via ever_* flags).
        if ever_flowered:
            pod_min = int(t.get('pod_fill_min_days', 3))
            recent = hist_flower[-(pod_min - 1):] if pod_min > 1 else []
            declined = flower < flower_th and all(f < flower_th for f in recent) \
                and len(recent) >= pod_min - 1
            if declined:                                    # flowers shed -> siliques
                return self._stage('pod_fill', confidence=0.5)
            return self._stage('flowering', confidence=0.8)
        if bolt or ever_bolted:
            return self._stage('bolting', confidence=0.7)

        # 3. Vegetative ladder (cover thresholds).
        if cover >= float(t.get('vegetative_cover_pct', 12.0)):
            return self._stage('vegetative', confidence=0.8)
        if cover >= float(t.get('seedling_cover_pct', 1.5)):
            return self._stage('seedling', confidence=0.75)
        if cover >= float(t.get('germination_cover_pct', 0.5)) or germ:
            return self._stage('germination', confidence=0.75)
        return self._stage('dormant', confidence=0.9)


# ─────────────────────────────────────────────
# FACTORY + PUBLIC API
# ─────────────────────────────────────────────

_DETECTORS = {
    'arabidopsis': ArabidopsisStageDetector,
    'barley':      BarleyStageDetector,
    'brassica':    BrassicaStageDetector,
}


def get_detector(species_name: str, cfg: dict) -> BaseStageDetector:
    cls = _DETECTORS.get(species_name, ArabidopsisStageDetector)
    return cls(cfg)


def _load_history(chamber_id: str, pot_label: str, csv_path: str) -> list:
    rows = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('chamber') == chamber_id and row.get('pot_label') == pot_label:
                rows.append(row)
    return rows


def detect_stage(
    current_metrics: dict,
    chamber_id:      str,
    pot_label:       str,
    csv_path:        str,
    cfg:             dict,
) -> DevelopmentalStage:
    """
    Return the current phenological stage for one pot.

    Args:
        current_metrics : dict — today's metrics (needs canopy_cover_%, bolting_flag,
                          germination_flag, and optionally canopy_height_max_mm)
        chamber_id      : 'enriched' or 'control'
        pot_label       : 'P1' – 'P8'
        csv_path        : path to pot_metrics.csv (for history)
        cfg             : full species config dict

    Returns:
        DevelopmentalStage(stage_id, stage_name, bbch_code, confidence)
    """
    species  = cfg.get('name', 'arabidopsis')
    detector = get_detector(species, cfg)
    history  = _load_history(chamber_id, pot_label, csv_path)
    return detector.detect(current_metrics, history)
