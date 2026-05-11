import json
import os
import re
from typing import Dict, List, Union, Any, Optional

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel

from evaluator.scorers.base import LLMScorer, ScorerResult
from evaluator.scorers.behavioral_metrics import TrajectoryMetrics, compute_behavioral_metrics
from evaluator.trajectory import Trajectory
from .prompts.prompt_enhanced import (
    REASONING_QUALITY_ENHANCED_SYSTEM_PROMPT,
    REASONING_QUALITY_ENHANCED_USER_PROMPT,
)

load_dotenv()


class ReasoningQualityScore(BaseModel):
    strategic_backtracking: Optional[float]
    task_decomposition: float
    observation_reading: float
    self_verification: float

    @property
    def overall(self) -> float:
        scores = [self.task_decomposition, self.observation_reading, self.self_verification]
        if self.strategic_backtracking is not None:
            scores.append(self.strategic_backtracking)
        return sum(scores) / len(scores)


class ReasoningQualityScorer(LLMScorer):
    def __init__(self,
                 weight: float = 1.0,
                 name: str = "ReasoningQuality",
                 description: str = "Evaluates the quality of the agent's reasoning process",
                 model: str = "gemini-2.5-pro-exp-03-25",
                 max_tokens: int = 2048,
                 temperature: float = 0.0,
                 **kwargs) -> None:
        super().__init__(weight, name, description, model, max_tokens, temperature)

    def generate_prompt(self, trajectory: Trajectory, bm: TrajectoryMetrics) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": REASONING_QUALITY_ENHANCED_SYSTEM_PROMPT},
            {"role": "user", "content": REASONING_QUALITY_ENHANCED_USER_PROMPT.format(
                objective=trajectory.objective,
                total_steps=bm.total_steps,
                backtrack_count=bm.backtrack_count,
                backtrack_score=f"{bm.backtrack_score:.1%}",
                exploration_ratio=f"{bm.exploration_ratio:.1%}",
                loop_detected=bm.loop_detected,
                explicit_undo_count=bm.explicit_undo_count,
                quadrant=bm.quadrant,
                health_tag=bm.health_tag,
                edit_total=bm.edit_total,
                verification_rate=f"{bm.verification_rate:.1%}",
                continue_edit_rate=f"{bm.continue_edit_rate:.1%}",
                max_edit_chain=bm.max_edit_chain,
                steps=self._format_steps(trajectory),
            )}
        ]

    MAX_ACTION_CHARS = 300
    MAX_REASONING_CHARS = 500
    MAX_STEPS = 60

    def _format_steps(self, trajectory: Trajectory) -> str:
        step_text = ""
        if not hasattr(trajectory, 'actions') or not trajectory.actions:
            return "No steps recorded."

        actions = trajectory.actions
        n = len(actions)
        if n > self.MAX_STEPS:
            head = int(self.MAX_STEPS * 2 / 3)
            tail = self.MAX_STEPS - head
            skipped = n - self.MAX_STEPS
            step_numbers = list(range(1, head + 1)) + list(range(n - tail + 1, n + 1))
            actions = actions[:head] + actions[-tail:]
            step_text += f"[{skipped} steps omitted for brevity]\n"
        else:
            step_numbers = list(range(1, n + 1))

        for step_num, action in zip(step_numbers, actions):
            step_text += f"\n--- Step {step_num} ---\n"
            if hasattr(action, 'action') and action.action:
                a = str(action.action)
                if len(a) > self.MAX_ACTION_CHARS:
                    a = a[:self.MAX_ACTION_CHARS] + "...[truncated]"
                step_text += f"Action: {a}\n"
            if hasattr(action, 'reasoning') and action.reasoning:
                r = str(action.reasoning)
                if len(r) > self.MAX_REASONING_CHARS:
                    r = r[:self.MAX_REASONING_CHARS] + "...[truncated]"
                step_text += f"Reasoning: {r}\n"
        return step_text

    def _post_process(
        self,
        backtrack_llm: Optional[float],
        self_verif_llm: float,
        bm: TrajectoryMetrics,
    ) -> tuple[Optional[float], float]:
        """Apply rule-based constraints on top of raw LLM scores.

        Rules are evaluated top-down; the first matching branch wins.

        strategic_backtracking
        ----------------------
        1. health_tag == "dead_loop"
              -> cap at 0.25.  LLM cannot know from text alone that the agent
                cycled dozens of times; behavioral hash detection is authoritative.

        2. health_tag == "healthy" + bt==0, loop==0, undo==0, exploration>=0.8
              -> None (N/A).  Dimension genuinely does not apply: clean linear path.

        3. health_tag == "blind_edit" + bt==0
              -> rule-based score: min(1 - continue_edit_rate, 0.40).

        4. health_tag == "blind_edit" + bt >= 2
              -> cap at 0.50.

        5. All other cases -> LLM score as-is.

        self_verification  (rule-primary blend)
        ----------------------------------------
        final = 0.7 x rule_score  +  0.3 x llm_score
          rule_score = min(1.0, verification_rate x 1.25)   when edit_total >= 3
          rule_score = 0.5  (neutral)                        when edit_total < 3
        Blind-edit -> final capped at 0.50 afterward
        """
        # -- strategic_backtracking --
        bt = backtrack_llm

        if bm.health_tag == "dead_loop":
            bt = min(bt, 0.25) if bt is not None else 0.25

        elif (bm.health_tag == "healthy"
              and bm.backtrack_count == 0
              and bm.loop_detected == 0
              and bm.explicit_undo_count == 0
              and bm.exploration_ratio >= 0.8):
            bt = None

        elif bm.health_tag == "blind_edit":
            if bm.backtrack_count == 0:
                bt = min(max(0.0, 1.0 - bm.continue_edit_rate), 0.40)
            elif bm.backtrack_count >= 2:
                bt = min(bt, 0.50) if bt is not None else 0.50

        # -- self_verification --
        if bm.edit_total >= 3:
            rule_score = min(1.0, bm.verification_rate * 1.25)
        else:
            rule_score = 0.5

        sv = 0.7 * rule_score + 0.3 * self_verif_llm

        if bm.health_tag == "blind_edit":
            sv = min(sv, 0.50)

        return bt, sv

    def parse_response(self, response: str, bm: TrajectoryMetrics) -> ScorerResult:
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            json_str = json_match.group(1) if json_match else response
            evaluation = json.loads(json_str)

            bt_raw = evaluation["backtrack_and_explore"]["score"]
            bt_llm = None if bt_raw == "N/A" else float(bt_raw) / 4.0

            task_decomposition = evaluation["task_decomposition"]["score"] / 4.0
            observation_reading = evaluation["observation_reading"]["score"] / 4.0
            self_verif_llm = evaluation["self_verification"]["score"] / 4.0

            bt_final, sv_final = self._post_process(bt_llm, self_verif_llm, bm)

            scores = [task_decomposition, observation_reading, sv_final]
            if bt_final is not None:
                scores.append(bt_final)
            overall_score = sum(scores) / len(scores)

            return ScorerResult(
                score=overall_score,
                name=self.name,
                description=self.description,
                details={
                    "reasoning_quality": {
                        "strategic_backtracking": bt_final,
                        "task_decomposition": task_decomposition,
                        "observation_reading": observation_reading,
                        "self_verification": sv_final,
                        "raw_scores": {
                            "strategic_backtracking": bt_raw,
                            "task_decomposition": evaluation["task_decomposition"]["score"],
                            "observation_reading": evaluation["observation_reading"]["score"],
                            "self_verification": evaluation["self_verification"]["score"],
                        },
                        "llm_scores": {
                            "strategic_backtracking": bt_llm,
                            "self_verification": self_verif_llm,
                        },
                    },
                    "justifications": {
                        "strategic_backtracking": evaluation["backtrack_and_explore"]["justification"],
                        "task_decomposition": evaluation["task_decomposition"]["justification"],
                        "observation_reading": evaluation["observation_reading"]["justification"],
                        "self_verification": evaluation["self_verification"]["justification"],
                    },
                    "behavioral_metrics": bm.to_dict(),
                },
                confidence=0.9
            )
        except Exception as e:
            return ScorerResult(
                score=0.0,
                name=self.name,
                description=self.description,
                details={"error": f"Failed to parse response: {str(e)}", "raw_response": response},
                confidence=0.0
            )

    def dry_run(self, trajectory: Trajectory) -> ScorerResult:
        bm = compute_behavioral_metrics(trajectory)
        prompt = self.generate_prompt(trajectory, bm)
        token_count = self.count_tokens(prompt)
        return ScorerResult(
            score=0.0,
            name=self.name,
            description=self.description,
            details={"dry_run": True, "token_count": token_count,
                     "behavioral_metrics": bm.to_dict()},
            confidence=0.0,
            weight=self.weight
        )

    def score(self, trajectory: Trajectory) -> ScorerResult:
        bm = compute_behavioral_metrics(trajectory)
        prompt = self.generate_prompt(trajectory, bm)

        api_key = os.getenv('LLM_API_KEY')

        response = completion(
            api_key=api_key,
            model=f"{self.model}",
            messages=prompt,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )

        try:
            response_text = response.choices[0].message.content
            return self.parse_response(response_text, bm)
        except Exception as e:
            return ScorerResult(
                score=0.0,
                name=self.name,
                description=self.description,
                details={"error": f"Failed to get LLM response: {str(e)}"},
                confidence=0.0
            )
