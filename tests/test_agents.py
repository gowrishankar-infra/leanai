"""
Tests for LeanAI Phase 4d — Agentic Multi-Step Planner + Pipeline
"""

import os
import json
import shutil
import tempfile
import pytest
from agents.planner import (
    TaskPlanner,
    Plan,
    PlanStep,
    StepType,
    StepStatus,
)
from agents.pipeline import (
    AgenticPipeline,
    PipelineConfig,
    PipelineResult,
)


# ── Test fixtures ─────────────────────────────────────────────────

@pytest.fixture
def planner():
    return TaskPlanner()


@pytest.fixture
def workspace(tmp_path):
    ws = str(tmp_path / "test_project")
    os.makedirs(ws, exist_ok=True)
    return ws


@pytest.fixture
def sample_plan_json():
    return json.dumps([
        {
            "id": "step_00",
            "title": "Create project structure",
            "description": "Create the main directory",
            "step_type": "create_dir",
            "target_file": "src",
            "dependencies": [],
        },
        {
            "id": "step_01",
            "title": "Write main module",
            "description": "Create the main application file",
            "step_type": "write_code",
            "target_file": "src/app.py",
            "dependencies": ["step_00"],
        },
        {
            "id": "step_02",
            "title": "Write tests",
            "description": "Create test file",
            "step_type": "run_test",
            "target_file": "tests/test_app.py",
            "dependencies": ["step_01"],
        },
        {
            "id": "step_03",
            "title": "Final verification",
            "description": "Run all tests",
            "step_type": "verify",
            "dependencies": ["step_02"],
            "verification": "python -c \"print(42)\"",
        },
    ])


# ── Planner tests ─────────────────────────────────────────────────

class TestTaskPlanner:
    def test_parse_valid_json(self, planner, sample_plan_json):
        steps = planner.parse_plan_response(sample_plan_json)
        assert len(steps) == 4
        assert steps[0].id == "step_00"
        assert steps[0].step_type == StepType.CREATE_DIR

    def test_parse_with_markdown_fences(self, planner, sample_plan_json):
        wrapped = f"```json\n{sample_plan_json}\n```"
        steps = planner.parse_plan_response(wrapped)
        assert len(steps) == 4

    def test_parse_with_preamble(self, planner, sample_plan_json):
        wrapped = f"Here is the plan:\n{sample_plan_json}\nDone."
        steps = planner.parse_plan_response(wrapped)
        assert len(steps) == 4

    def test_parse_preserves_dependencies(self, planner, sample_plan_json):
        steps = planner.parse_plan_response(sample_plan_json)
        assert steps[1].dependencies == ["step_00"]
        assert steps[2].dependencies == ["step_01"]

    def test_parse_auto_adds_dependencies(self, planner):
        data = json.dumps([
            {"id": "s0", "title": "A", "description": "a", "step_type": "create_dir"},
            {"id": "s1", "title": "B", "description": "b", "step_type": "write_code"},
        ])
        steps = planner.parse_plan_response(data)
        # s1 should auto-depend on s0 since no deps specified
        assert steps[1].dependencies == ["s0"]

    def test_parse_invalid_json_raises(self, planner):
        with pytest.raises(ValueError):
            planner.parse_plan_response("this is not json at all")

    def test_parse_no_array_raises(self, planner):
        with pytest.raises(ValueError):
            planner.parse_plan_response('{"not": "an array"}')

    def test_parse_trailing_commas(self, planner):
        data = """[
            {"id": "s0", "title": "A", "description": "a", "step_type": "create_dir",},
        ]"""
        steps = planner.parse_plan_response(data)
        assert len(steps) == 1

    def test_build_plan_prompt(self, planner):
        system, user = planner.build_plan_prompt("Build a calculator app")
        assert "json" in system.lower()
        assert "Build a calculator app" in user

    def test_build_code_prompt(self, planner):
        plan = Plan(task="Build API", workspace="/tmp/test")
        plan.created_files = ["main.py"]
        plan.file_contents = {"main.py": "def hello(): pass"}
        step = PlanStep(
            id="s1", title="Write models", description="Create data models",
            step_type=StepType.WRITE_CODE, target_file="models.py"
        )
        system, user = planner.build_code_prompt(plan, step)
        assert "programmer" in system.lower()
        assert "main.py" in user  # file contents section
        assert "models.py" in user

    def test_build_fix_prompt(self, planner):
        plan = Plan(task="Build API", workspace="/tmp/test")
        plan.file_contents = {"app.py": "import flask"}
        step = PlanStep(
            id="s1", title="Fix code", description="Fix broken code",
            step_type=StepType.WRITE_CODE, target_file="app.py",
            code="import flask\napp = flask()", error="TypeError: module not callable"
        )
        system, user = planner.build_fix_prompt(plan, step)
        assert "fix" in system.lower()
        assert "TypeError" in user
        assert "import flask" in user

    def test_create_fallback_plan(self, planner):
        plan = planner.create_fallback_plan("Build something", "/tmp/ws")
        assert plan.task == "Build something"
        assert plan.total_steps == 4
        assert plan.steps[0].step_type == StepType.CREATE_DIR
        assert plan.steps[-1].step_type == StepType.VERIFY


# ── Plan dataclass tests ──────────────────────────────────────────

class TestPlan:
    def test_progress_empty(self):
        plan = Plan(task="test")
        assert plan.progress_pct == 0.0

    def test_progress_partial(self):
        plan = Plan(task="test", steps=[
            PlanStep(id="s0", title="A", description="a", step_type=StepType.CREATE_DIR, status=StepStatus.PASSED),
            PlanStep(id="s1", title="B", description="b", step_type=StepType.WRITE_CODE, status=StepStatus.PENDING),
        ])
        assert plan.progress_pct == 50.0

    def test_progress_full(self):
        plan = Plan(task="test", steps=[
            PlanStep(id="s0", title="A", description="a", step_type=StepType.CREATE_DIR, status=StepStatus.PASSED),
            PlanStep(id="s1", title="B", description="b", step_type=StepType.WRITE_CODE, status=StepStatus.PASSED),
        ])
        assert plan.progress_pct == 100.0

    def test_next_step(self):
        plan = Plan(task="test", steps=[
            PlanStep(id="s0", title="A", description="a", step_type=StepType.CREATE_DIR, status=StepStatus.PASSED),
            PlanStep(id="s1", title="B", description="b", step_type=StepType.WRITE_CODE, status=StepStatus.PENDING),
        ])
        nxt = plan.next_step()
        assert nxt.id == "s1"

    def test_next_step_none(self):
        plan = Plan(task="test", steps=[
            PlanStep(id="s0", title="A", description="a", step_type=StepType.CREATE_DIR, status=StepStatus.PASSED),
        ])
        assert plan.next_step() is None

    def test_deps_met(self):
        plan = Plan(task="test", steps=[
            PlanStep(id="s0", title="A", description="a", step_type=StepType.CREATE_DIR, status=StepStatus.PASSED),
            PlanStep(id="s1", title="B", description="b", step_type=StepType.WRITE_CODE, dependencies=["s0"]),
        ])
        assert plan.deps_met(plan.steps[1]) is True

    def test_deps_not_met(self):
        plan = Plan(task="test", steps=[
            PlanStep(id="s0", title="A", description="a", step_type=StepType.CREATE_DIR, status=StepStatus.PENDING),
            PlanStep(id="s1", title="B", description="b", step_type=StepType.WRITE_CODE, dependencies=["s0"]),
        ])
        assert plan.deps_met(plan.steps[1]) is False

    def test_get_step(self):
        plan = Plan(task="test", steps=[
            PlanStep(id="s0", title="A", description="a", step_type=StepType.CREATE_DIR),
        ])
        assert plan.get_step("s0").title == "A"
        assert plan.get_step("nonexistent") is None

    def test_summary_output(self):
        plan = Plan(task="Build thing", steps=[
            PlanStep(id="s0", title="Setup", description="do setup", step_type=StepType.CREATE_DIR, status=StepStatus.PASSED),
            PlanStep(id="s1", title="Code", description="write code", step_type=StepType.WRITE_CODE, status=StepStatus.FAILED, error="SyntaxError"),
        ])
        s = plan.summary()
        assert "Build thing" in s
        assert "●" in s  # passed icon
        assert "✗" in s  # failed icon
        assert "SyntaxError" in s

    def test_to_dict(self):
        plan = Plan(task="test", steps=[
            PlanStep(id="s0", title="A", description="a", step_type=StepType.CREATE_DIR),
        ])
        d = plan.to_dict()
        assert d["task"] == "test"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["step_type"] == "create_dir"


# ── PlanStep tests ────────────────────────────────────────────────

class TestPlanStep:
    def test_to_dict(self):
        step = PlanStep(id="s0", title="A", description="a", step_type=StepType.WRITE_CODE)
        d = step.to_dict()
        assert d["step_type"] == "write_code"
        assert d["status"] == "pending"

    def test_from_dict(self):
        d = {"id": "s0", "title": "A", "description": "a", "step_type": "create_dir", "status": "passed"}
        step = PlanStep.from_dict(d)
        assert step.step_type == StepType.CREATE_DIR
        assert step.status == StepStatus.PASSED

    def test_default_max_attempts(self):
        step = PlanStep(id="s0", title="A", description="a", step_type=StepType.WRITE_CODE)
        assert step.max_attempts == 3


# ── Pipeline tests ────────────────────────────────────────────────

class TestAgenticPipeline:
    def _mock_model(self, responses: list):
        """Create a model function that returns responses in order."""
        idx = [0]
        def fn(system: str, user: str) -> str:
            if idx[0] < len(responses):
                r = responses[idx[0]]
                idx[0] += 1
                return r
            return "# empty"
        return fn

    def test_create_dir_step(self, workspace):
        """Test that create_dir steps work."""
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Create src", description="Create source dir",
                     step_type=StepType.CREATE_DIR, target_file="src"),
        ])
        pipeline = AgenticPipeline(config=PipelineConfig(auto_fix=False))
        result = pipeline.execute_with_plan(plan)
        assert result.success
        assert os.path.isdir(os.path.join(workspace, "src"))

    def test_create_file_step(self, workspace):
        """Test that write_code steps create files."""
        code = 'def hello():\n    return "Hello, World!"\n\nprint(hello())'
        model_fn = self._mock_model([code])
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Write main", description="Write main.py",
                     step_type=StepType.WRITE_CODE, target_file="main.py"),
        ])
        pipeline = AgenticPipeline(model_fn=model_fn, config=PipelineConfig(auto_fix=False))
        result = pipeline.execute_with_plan(plan)
        assert os.path.exists(os.path.join(workspace, "main.py"))
        assert result.plan.steps[0].status == StepStatus.PASSED

    def test_failed_step_retries(self, workspace):
        """Test that failed steps trigger auto-fix."""
        bad_code = 'def add(a, b)\n    return a + b'  # missing colon — SyntaxError
        good_code = 'def add(a, b):\n    return a + b\n\nprint(add(2, 3))'
        model_fn = self._mock_model([bad_code, good_code])
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Write code", description="Write add function",
                     step_type=StepType.WRITE_CODE, target_file="add.py"),
        ])
        pipeline = AgenticPipeline(model_fn=model_fn, config=PipelineConfig(auto_fix=True))
        result = pipeline.execute_with_plan(plan)
        assert result.plan.steps[0].attempts >= 1

    def test_skipped_step_on_dep_failure(self, workspace):
        """Test that steps are skipped when dependencies fail."""
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="A", description="a",
                     step_type=StepType.WRITE_CODE, target_file="a.py",
                     status=StepStatus.FAILED),
            PlanStep(id="s1", title="B", description="b",
                     step_type=StepType.WRITE_CODE, target_file="b.py",
                     dependencies=["s0"]),
        ])
        model_fn = self._mock_model(['print("hello")'])
        pipeline = AgenticPipeline(model_fn=model_fn, config=PipelineConfig(auto_fix=False))
        result = pipeline.execute_with_plan(plan)
        assert result.plan.steps[1].status == StepStatus.SKIPPED

    def test_verify_step_runs_command(self, workspace):
        """Test that verify steps run verification commands."""
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Verify", description="Check",
                     step_type=StepType.VERIFY,
                     verification='python -c "print(42)"'),
        ])
        pipeline = AgenticPipeline(config=PipelineConfig(auto_fix=False))
        result = pipeline.execute_with_plan(plan)
        assert result.success
        assert "42" in result.plan.steps[0].output

    def test_run_command_step(self, workspace):
        """Test run_command step type."""
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Run echo", description="echo hello",
                     step_type=StepType.RUN_COMMAND,
                     verification="python -c \"print('command works')\""),
        ])
        pipeline = AgenticPipeline(config=PipelineConfig(auto_fix=False))
        result = pipeline.execute_with_plan(plan)
        assert result.success

    def test_pipeline_result_summary(self, workspace):
        """Test that PipelineResult.summary() produces readable output."""
        plan = Plan(task="Build calculator", workspace=workspace, steps=[
            PlanStep(id="s0", title="Setup", description="d", step_type=StepType.CREATE_DIR, status=StepStatus.PASSED),
            PlanStep(id="s1", title="Code", description="d", step_type=StepType.WRITE_CODE, status=StepStatus.PASSED),
        ])
        result = PipelineResult(
            plan=plan, success=True, total_time=12.5, model_calls=3,
            files_created=["main.py"], errors=[]
        )
        s = result.summary()
        assert "SUCCESS" in s
        assert "12.5s" in s
        assert "main.py" in s

    def test_full_pipeline_with_mock_model(self, workspace):
        """Test full pipeline execution with mocked model."""
        plan_json = json.dumps([
            {"id": "s0", "title": "Create dir", "description": "Setup", "step_type": "create_dir", "target_file": "src"},
            {"id": "s1", "title": "Write code", "description": "Main file", "step_type": "write_code",
             "target_file": "src/calc.py", "dependencies": ["s0"]},
            {"id": "s2", "title": "Verify", "description": "Check", "step_type": "verify",
             "dependencies": ["s1"], "verification": "python -c \"print(42)\""},
        ])
        main_code = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n\nprint(add(1, 2))'
        model_fn = self._mock_model([plan_json, main_code])
        pipeline = AgenticPipeline(
            model_fn=model_fn,
            config=PipelineConfig(workspace_root=workspace, auto_fix=False),
        )
        result = pipeline.execute("Build a calculator", workspace=os.path.join(workspace, "calc_project"))
        assert result.plan.total_steps == 3
        assert result.plan.steps[0].status == StepStatus.PASSED  # create_dir
        assert result.model_calls >= 2  # plan + code gen

    def test_on_step_callbacks(self, workspace):
        """Test that step callbacks are called."""
        started = []
        ended = []
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Setup", description="d",
                     step_type=StepType.CREATE_DIR, target_file="src"),
        ])
        pipeline = AgenticPipeline(
            config=PipelineConfig(auto_fix=False),
            on_step_start=lambda s, p: started.append(s.id),
            on_step_end=lambda s, p: ended.append(s.id),
        )
        pipeline.execute_with_plan(plan)
        assert "s0" in started
        assert "s0" in ended

    def test_model_call_counting(self, workspace):
        """Test that model calls are tracked."""
        code = 'print("hello")'
        model_fn = self._mock_model([code])
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Write", description="d",
                     step_type=StepType.CREATE_FILE, target_file="hello.py"),
        ])
        pipeline = AgenticPipeline(model_fn=model_fn, config=PipelineConfig(auto_fix=False))
        result = pipeline.execute_with_plan(plan)
        assert result.model_calls == 1

    def test_clean_code_strips_fences(self):
        """Test that markdown fences are stripped from model output."""
        pipeline = AgenticPipeline()
        raw = '```python\ndef hello():\n    return "hi"\n```'
        clean = pipeline._clean_code(raw)
        assert "```" not in clean
        assert 'def hello():' in clean

    def test_clean_code_strips_json_fences(self):
        pipeline = AgenticPipeline()
        raw = '```json\n[{"id": "s0"}]\n```'
        clean = pipeline._clean_code(raw)
        assert "```" not in clean

    def test_workspace_auto_created(self):
        """Test that workspace directory is auto-created."""
        tmp = tempfile.mkdtemp()
        ws = os.path.join(tmp, "new_project")
        plan = Plan(task="test", workspace=ws, steps=[
            PlanStep(id="s0", title="Dir", description="d",
                     step_type=StepType.CREATE_DIR, target_file="."),
        ])
        pipeline = AgenticPipeline(config=PipelineConfig(create_workspace=True, auto_fix=False))
        pipeline.execute_with_plan(plan)
        assert os.path.isdir(ws)
        shutil.rmtree(tmp)

    def test_multi_step_with_nested_dirs(self, workspace):
        """Test creating files in nested directories."""
        code = '# config file\nDEBUG = True'
        model_fn = self._mock_model([code])
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Create dirs", description="d",
                     step_type=StepType.CREATE_DIR, target_file="src/config"),
            PlanStep(id="s1", title="Write config", description="d",
                     step_type=StepType.CREATE_FILE, target_file="src/config/settings.py",
                     dependencies=["s0"]),
        ])
        pipeline = AgenticPipeline(model_fn=model_fn, config=PipelineConfig(auto_fix=False))
        result = pipeline.execute_with_plan(plan)
        assert os.path.exists(os.path.join(workspace, "src", "config", "settings.py"))


# ── PipelineConfig tests ─────────────────────────────────────────

class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.max_retries == 3
        assert c.step_timeout == 60
        assert c.auto_fix is True

    def test_custom(self):
        c = PipelineConfig(max_retries=5, verbose=True)
        assert c.max_retries == 5
        assert c.verbose is True


class TestTestPromptAndSourceFinding:
    def test_build_test_prompt_includes_source(self):
        planner = TaskPlanner()
        plan = Plan(task="Build calc", workspace="/tmp/test")
        step = PlanStep(id="s2", title="Write tests", description="Test calc",
                        step_type=StepType.RUN_TEST, target_file="test_calc.py")
        source_code = "def add(a, b):\n    return a + b"
        system, user = planner.build_test_prompt(plan, step, "calc.py", source_code)
        assert "test" in system.lower()
        assert "def add" in user
        assert "calc.py" in user
        assert "test_calc.py" in user

    def test_find_source_for_test(self, workspace):
        pipeline = AgenticPipeline()
        plan = Plan(task="test", workspace=workspace,
                    created_files=["calculator.py"],
                    file_contents={"calculator.py": "def add(a,b): return a+b"})
        step = PlanStep(id="s2", title="Tests", description="test",
                        step_type=StepType.RUN_TEST, target_file="test_calculator.py")
        fname, content = pipeline._find_source_for_test(plan, step)
        assert fname == "calculator.py"
        assert "def add" in content

    def test_find_source_fallback(self, workspace):
        pipeline = AgenticPipeline()
        plan = Plan(task="test", workspace=workspace,
                    created_files=["app.py"],
                    file_contents={"app.py": "def run(): pass"})
        step = PlanStep(id="s2", title="Tests", description="test",
                        step_type=StepType.RUN_TEST, target_file="test_main.py")
        fname, content = pipeline._find_source_for_test(plan, step)
        # Falls back to most recent non-test .py file
        assert fname == "app.py"
        assert "def run" in content

    def test_file_contents_stored_after_create(self, workspace):
        code = 'def hello():\n    return "hi"\nprint(hello())'
        model_fn = lambda s, u: code
        plan = Plan(task="test", workspace=workspace, steps=[
            PlanStep(id="s0", title="Write", description="d",
                     step_type=StepType.CREATE_FILE, target_file="mod.py"),
        ])
        pipeline = AgenticPipeline(model_fn=model_fn, config=PipelineConfig(auto_fix=False))
        pipeline.execute_with_plan(plan)
        assert "mod.py" in plan.file_contents
        assert "def hello" in plan.file_contents["mod.py"]

    def test_fix_prompt_includes_related_files(self):
        planner = TaskPlanner()
        plan = Plan(task="Build API", workspace="/tmp",
                    file_contents={
                        "calc.py": "def add(a,b): return a+b",
                        "test_calc.py": "from calc import ad\ndef test(): ad(1,2)"
                    })
        step = PlanStep(id="s2", title="Fix test", description="fix",
                        step_type=StepType.RUN_TEST, target_file="test_calc.py",
                        code="from calc import ad", error="ImportError: cannot import name 'ad'")
        system, user = planner.build_fix_prompt(plan, step)
        # Should include calc.py contents in the fix prompt
        assert "def add" in user
        assert "ImportError" in user
