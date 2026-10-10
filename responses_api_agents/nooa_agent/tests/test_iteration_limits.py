# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native sessions with the pinned CodeAct runtime and scripted model replies."""

import asyncio
from json import dumps as json_dumps
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError

from environment_servers.nooa_single_agent_turn.tests.test_app import _environment, _paths, _request, _Response
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.rollout_collection import _episode_record
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.runner import NOOARunFailure
from responses_api_agents.nooa_agent.tests.test_app import close, open_session, request
from responses_api_agents.nooa_agent.tests.test_config import agent_config
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


class IterationLimitedAgent(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1, max_retries=1)))
    async def analyze(self, text: str) -> str:
        """Work on the task using execute_python."""
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
    async def followup(self) -> str:
        """Ask the model for more information."""
        ...


class TextOnlyLimitedAgent(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1, max_consecutive_text_only=1)))
    async def analyze(self, text: str) -> str:
        """Work on the task using execute_python."""
        ...


async def invoke_text(
    agent: IterationLimitedAgent | TextOnlyLimitedAgent, request: NeMoGymResponseCreateParamsNonStreaming
) -> str:
    return await agent.analyze(str(request.input))


def iteration_agent(*, scenario: str = "iterations") -> tuple[NOOAAgent, list[object]]:
    client = MagicMock(spec=ServerClient)
    model_calls: list[object] = []

    async def post(*, server_name: str, url_path: str, json: object, **_: object) -> FakeHTTPResponse:
        assert server_name == "policy"
        assert url_path.endswith("/v1/responses")
        model_calls.append(json)
        if len(model_calls) > 1:
            raise ConnectionError("unrecovered model outage")
        code = "print('partial progress')"
        if scenario == "retries":
            code = ""
        elif scenario == "model_outage":
            code = "try:\n    await self.followup()\nexcept Exception:\n    pass\nprint('partial progress')"
        output = [
            {"type": "reasoning", "id": "reasoning-1", "summary": [{"type": "summary_text", "text": "plan so far"}]},
            NeMoGymResponseOutputMessageForTraining(
                id="message-1",
                prompt_token_ids=[1],
                generation_token_ids=[2],
                generation_log_probs=[-0.1],
                content=[NeMoGymResponseOutputText(annotations=[], text="working on the task", logprobs=[])],
            ),
        ]
        if scenario != "text_only":
            output.append(
                NeMoGymResponseFunctionToolCall(
                    id="function-1",
                    call_id="call-1",
                    name="execute_python",
                    arguments=json_dumps({"code": code}),
                )
            )
        return FakeHTTPResponse(model_response(*output))

    client.post = AsyncMock(side_effect=post)
    agent_class = TextOnlyLimitedAgent if scenario == "text_only" else IterationLimitedAgent
    instance = NOOAAgent(
        config=agent_config(
            nooa={"agent_class": f"{__name__}:{agent_class.__name__}", "invocation_adapter": f"{__name__}:invoke_text"}
        ),
        server_client=client,
    )
    return instance, model_calls


def assert_partial_evidence(response: NeMoGymResponse) -> None:
    assert response.status == "failed"
    assert "max_iterations=1" in response.error.message
    assert any(item.type == "reasoning" for item in response.output)
    assert any(item.type == "message" and item.content[0].text == "working on the task" for item in response.output)
    assert any(item.type == "function_call" and item.call_id == "call-1" for item in response.output)
    assert any(
        item.type == "function_call_output" and item.call_id == "call-1" and "partial progress" in item.output
        for item in response.output
    )


async def test_real_codeact_native_response_retains_partial_evidence_without_verification() -> None:
    instance, model_calls = iteration_agent()
    req = await open_session(instance)
    try:
        response = await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert_partial_evidence(response)
        assert "_ng_failure_class" not in response.model_dump(mode="json")
        assert len(model_calls) == 1
    finally:
        receipt = await close(instance)
    assert "max_iterations_exceeded" in {gap.code for gap in receipt.agent_observations.gaps}


async def test_real_codeact_native_environment_verifies_partial_response_once() -> None:
    instance, model_calls = iteration_agent()
    environment, client = _environment()
    original = client.post.side_effect
    req = request(capture=True)
    verified: list[dict[str, object]] = []

    async def post(*, server_name: str, url_path: str, **kwargs) -> _Response:
        body = kwargs["json"]
        if server_name != "agent":
            response = await original(server_name=server_name, url_path=url_path, **kwargs)
            if url_path == "/verify":
                verified.append(body)
                response.body["reward"] = 0.75
            return response
        if url_path == "/v1/agent_sessions":
            response = await instance.seed_agent_session(req, AgentSeedSessionRequest.model_validate(body))
        elif url_path.endswith("/v1/responses"):
            response = await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming.model_validate(body))
        else:
            assert url_path in {"/v1/agent_sessions/finish", "/v1/agent_sessions/close"}
            method = instance.finish_agent_session if url_path.endswith("/finish") else instance.close_agent_session
            response = await method(req, AgentCloseSessionRequest.model_validate(body))
        return _Response(response.model_dump(mode="json"), cookies={"agent": "seed"})

    client.post.side_effect = post
    result = await environment.run_request(_request())
    assert result.failure is None
    assert result.result.reward == 0.75
    assert_partial_evidence(result.result.response)
    assert len(model_calls) == 1
    assert len(verified) == _paths(client).count("/verify") == 1
    assert verified[0]["response"] == result.result.response.model_dump(mode="json")
    record = _episode_record(result.model_dump(mode="json"))
    assert "_ng_failure_class" not in record
    assert record["reward"] == 0.75
    assert "max_iterations_exceeded" in {gap["code"] for gap in record["ng_agent_observations"]["gaps"]}
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]


@pytest.mark.parametrize("scenario", ["retries", "text_only", "model_outage"])
async def test_other_codeact_limits_and_unrecovered_outage_fail_native_activation(scenario: str) -> None:
    instance, model_calls = iteration_agent(scenario=scenario)
    req = await open_session(instance)
    try:
        with pytest.raises((NOOARunFailure, HTTPException)):
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert len(model_calls) == (2 if scenario == "model_outage" else 1)
        assert instance._require_agent_session("session").result.episode.response.output
    finally:
        receipt = await close(instance)
    gaps = {gap.code for gap in receipt.agent_observations.gaps}
    assert "infrastructure_error" in gaps
    assert "max_iterations_exceeded" not in gaps


async def test_matching_generation_error_message_is_not_a_gradable_native_limit() -> None:
    instance, model_calls = iteration_agent()

    async def invoke_error(agent: object, request: object) -> object:
        raise GenerationError("Generation failed after 1 iterations (max_iterations=1). Unable to complete `analyze`.")

    instance.runner._invocation_adapter = invoke_error
    req = await open_session(instance)
    try:
        with pytest.raises(NOOARunFailure):
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert model_calls == []
    finally:
        receipt = await close(instance)
    assert "infrastructure_error" in {gap.code for gap in receipt.agent_observations.gaps}


async def test_cancellation_after_iteration_limit_stays_cancelled_in_native_session() -> None:
    instance, _ = iteration_agent()

    async def cancel_after_limit(
        agent: IterationLimitedAgent, request: NeMoGymResponseCreateParamsNonStreaming
    ) -> object:
        try:
            await invoke_text(agent, request)
        except GenerationError:
            raise asyncio.CancelledError

    instance.runner._invocation_adapter = cancel_after_limit
    req = await open_session(instance)
    try:
        with pytest.raises(asyncio.CancelledError):
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert instance._require_agent_session("session").result.episode.response.output
    finally:
        receipt = await close(instance)
    gaps = {gap.code for gap in receipt.agent_observations.gaps}
    assert "cancelled" in gaps
    assert "max_iterations_exceeded" not in gaps
