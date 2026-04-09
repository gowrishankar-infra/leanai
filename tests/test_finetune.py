"""
Tests for LeanAI Continuous Fine-Tuning System.
"""

import os
import json
import shutil
import tempfile
import pytest

from training.finetune_pipeline import TrainingDataPipeline, TrainingExample, DatasetStats
from training.adapter_manager import AdapterManager, AdapterInfo
from training.finetune_runner import FineTuneRunner, TrainingConfig, TrainingRun


# ══════════════════════════════════════════════════════════════════
# Training Data Pipeline Tests
# ══════════════════════════════════════════════════════════════════

class TestTrainingExample:
    def test_to_sharegpt(self):
        e = TrainingExample(instruction="Q", response="A", quality_score=0.9)
        sg = e.to_sharegpt()
        assert sg["conversations"][0]["from"] == "human"
        assert sg["conversations"][1]["value"] == "A"

    def test_to_alpaca(self):
        e = TrainingExample(instruction="Q", response="A", quality_score=0.9)
        a = e.to_alpaca()
        assert a["instruction"] == "Q"
        assert a["output"] == "A"

    def test_to_chatml(self):
        e = TrainingExample(instruction="Q", response="A", quality_score=0.9)
        c = e.to_chatml()
        assert "<|im_start|>user" in c
        assert "<|im_start|>assistant" in c

    def test_content_hash_deterministic(self):
        e1 = TrainingExample(instruction="hello", response="world", quality_score=0.5)
        e2 = TrainingExample(instruction="hello", response="world", quality_score=0.9)
        assert e1.content_hash == e2.content_hash

    def test_content_hash_different(self):
        e1 = TrainingExample(instruction="hello", response="world", quality_score=0.5)
        e2 = TrainingExample(instruction="goodbye", response="world", quality_score=0.5)
        assert e1.content_hash != e2.content_hash


class TestTrainingDataPipeline:
    @pytest.fixture
    def pipeline(self):
        d = tempfile.mkdtemp()
        p = TrainingDataPipeline(data_dir=d)
        yield p
        shutil.rmtree(d)

    def test_add_example(self, pipeline):
        added = pipeline.add_example(
            instruction="write a sort function in Python",
            response="def sort(arr): return sorted(arr)",
            quality_score=0.8,
        )
        assert added is True
        assert pipeline.count == 1

    def test_deduplicate(self, pipeline):
        pipeline.add_example("write sort function", "def sort(): pass something here", 0.8)
        pipeline.add_example("write sort function", "def sort(): pass something here", 0.9)
        assert pipeline.count == 1

    def test_skip_too_short(self, pipeline):
        added = pipeline.add_example("hi", "ok", 0.8)
        assert added is False

    def test_verified_boost(self, pipeline):
        pipeline.add_example(
            "write a test for authentication module",
            "def test_auth(): assert authenticate('user', 'pass')",
            quality_score=0.5, verified=True,
        )
        assert pipeline._examples[0].quality_score >= 0.85

    def test_curate(self, pipeline):
        for i in range(10):
            pipeline.add_example(
                f"question {i} about programming topics",
                f"answer {i} with detailed explanation here",
                quality_score=0.9 if i < 5 else 0.3,
            )
        curated = pipeline.curate(min_quality=0.7, min_examples=3)
        assert len(curated) == 5  # only high quality

    def test_get_stats(self, pipeline):
        pipeline.add_example("question about coding patterns", "answer with code example here", 0.9, verified=True)
        pipeline.add_example("another question here", "another answer with detail", 0.5)
        stats = pipeline.get_stats()
        assert stats.total_pairs == 2
        assert stats.verified_code == 1
        assert stats.high_quality == 1

    def test_export_sharegpt(self, pipeline):
        pipeline.add_example("question about functions", "answer about functions here", 0.9)
        path = pipeline.export_sharegpt(min_quality=0.5)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.loads(f.readline())
        assert "conversations" in data

    def test_export_alpaca(self, pipeline):
        pipeline.add_example("question about classes in python", "answer about classes here", 0.9)
        path = pipeline.export_alpaca(min_quality=0.5)
        assert os.path.exists(path)

    def test_persistence(self):
        d = tempfile.mkdtemp()
        p1 = TrainingDataPipeline(data_dir=d)
        p1.add_example("persistent question about AI", "persistent answer about AI", 0.8)
        p1.save()

        p2 = TrainingDataPipeline(data_dir=d)
        assert p2.count == 1
        shutil.rmtree(d)

    def test_stats_summary(self, pipeline):
        pipeline.add_example("test question here about code", "test answer here about code", 0.8)
        s = pipeline.get_stats().summary()
        assert "1 total" in s


# ══════════════════════════════════════════════════════════════════
# Adapter Manager Tests
# ══════════════════════════════════════════════════════════════════

class TestAdapterInfo:
    def test_summary(self):
        a = AdapterInfo(name="work", path="/tmp", created=0, training_examples=100, quality_score=0.85)
        s = a.summary()
        assert "work" in s
        assert "100" in s

    def test_to_dict(self):
        a = AdapterInfo(name="test", path="/tmp", created=1.0)
        d = a.to_dict()
        assert d["name"] == "test"

    def test_from_dict(self):
        a = AdapterInfo.from_dict({"name": "x", "path": "/p", "created": 1.0})
        assert a.name == "x"


class TestAdapterManager:
    @pytest.fixture
    def mgr(self):
        d = tempfile.mkdtemp()
        m = AdapterManager(data_dir=d)
        yield m
        shutil.rmtree(d)

    def test_create(self, mgr):
        a = mgr.create("work", description="Work adapter")
        assert a.name == "work"
        assert "work" in mgr._adapters

    def test_set_active(self, mgr):
        mgr.create("test")
        assert mgr.set_active("test") is True
        assert mgr.get_active().name == "test"

    def test_deactivate(self, mgr):
        mgr.create("test")
        mgr.set_active("test")
        mgr.deactivate()
        assert mgr.get_active() is None

    def test_list_adapters_empty(self, mgr):
        s = mgr.list_adapters()
        assert "No adapters" in s

    def test_list_adapters(self, mgr):
        mgr.create("work")
        s = mgr.list_adapters()
        assert "work" in s

    def test_delete(self, mgr):
        mgr.create("temp")
        assert mgr.delete("temp") is True
        assert "temp" not in mgr._adapters

    def test_persistence(self):
        d = tempfile.mkdtemp()
        m1 = AdapterManager(data_dir=d)
        m1.create("persistent", description="test")

        m2 = AdapterManager(data_dir=d)
        assert "persistent" in m2._adapters
        shutil.rmtree(d)

    def test_stats(self, mgr):
        mgr.create("a")
        s = mgr.stats()
        assert s["total_adapters"] == 1


# ══════════════════════════════════════════════════════════════════
# Fine-Tune Runner Tests
# ══════════════════════════════════════════════════════════════════

class TestTrainingConfig:
    def test_defaults(self):
        c = TrainingConfig()
        assert c.epochs == 3
        assert c.lora_rank == 16
        assert c.min_examples == 50

    def test_custom(self):
        c = TrainingConfig(epochs=5, lora_rank=32)
        assert c.epochs == 5
        assert c.lora_rank == 32


class TestTrainingRun:
    def test_summary(self):
        r = TrainingRun(run_id="abc123", started=0, status="completed", examples_used=100, duration_minutes=30)
        s = r.summary()
        assert "completed" in s
        assert "100" in s


class TestFineTuneRunner:
    @pytest.fixture
    def runner(self):
        d = tempfile.mkdtemp()
        r = FineTuneRunner(data_dir=d)
        yield r
        shutil.rmtree(d)

    def test_check_readiness_no_data(self, runner):
        report = runner.check_readiness()
        assert "0 examples" in report or "0" in report

    def test_check_readiness_with_data(self, runner):
        for i in range(60):
            runner.pipeline.add_example(
                f"question {i} about coding topic {i}",
                f"answer {i} with detailed response here",
                quality_score=0.9,
            )
        report = runner.check_readiness()
        assert "60" in report

    def test_train_not_enough_data(self, runner):
        run = runner.train(TrainingConfig(min_examples=50))
        assert run.status == "failed"
        assert "Not enough" in run.error

    def test_train_exports_data(self, runner):
        for i in range(60):
            runner.pipeline.add_example(
                f"coding question number {i} here",
                f"detailed coding answer number {i} here",
                quality_score=0.9,
            )
        run = runner.train(TrainingConfig(min_examples=50, adapter_name="test"))
        assert run.status == "completed"
        assert run.examples_used >= 50

    def test_list_runs_empty(self, runner):
        s = runner.list_runs()
        assert "No training" in s

    def test_list_runs(self, runner):
        runner._runs.append(TrainingRun(
            run_id="test1", started=0, status="completed",
            examples_used=100, duration_minutes=30,
        ))
        s = runner.list_runs()
        assert "completed" in s

    def test_stats(self, runner):
        s = runner.stats()
        assert "total_runs" in s
        assert "data" in s
