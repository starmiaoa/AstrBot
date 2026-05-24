import json
from pathlib import Path

import httpx
import pytest
from openai.types.chat.chat_completion import ChatCompletion

import astrbot.core.provider.sources.openai_source as openai_source_module
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.provider.sources.vertex_ai_openai import (
    VertexAIOpenAIAuth,
    build_vertex_ai_openai_base_url,
    resolve_vertex_ai_project_id,
)


def _vertex_config(**overrides):
    config = {
        "id": "vertex-test",
        "provider": "google-vertex-ai",
        "type": "openai_chat_completion",
        "provider_type": "chat_completion",
        "model": "google/gemini-3.5-flash",
        "key": [],
        "api_base": "",
        "timeout": 120,
        "proxy": "",
        "custom_headers": {},
        "vertex_ai_auth_type": "service_account",
        "vertex_ai_project_id": "demo-project",
        "vertex_ai_location": "global",
        "vertex_ai_credentials_path": "",
        "vertex_ai_credentials_json": "",
    }
    config.update(overrides)
    return config


def test_vertex_ai_base_url_uses_global_default_location_from_service_account_json():
    config = _vertex_config(
        vertex_ai_project_id="",
        vertex_ai_location="",
        vertex_ai_credentials_json=json.dumps({"project_id": "json-project"}),
    )

    assert build_vertex_ai_openai_base_url(config) == (
        "https://aiplatform.googleapis.com/v1/projects/json-project/"
        "locations/global/endpoints/openapi"
    )


def test_vertex_ai_base_url_uses_regional_host():
    config = _vertex_config(
        vertex_ai_project_id="regional-project",
        vertex_ai_location="us-central1",
    )

    assert build_vertex_ai_openai_base_url(config) == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/regional-project/"
        "locations/us-central1/endpoints/openapi"
    )


def test_vertex_ai_base_url_treats_template_base_as_generated_default():
    config = _vertex_config(
        api_base="https://aiplatform.googleapis.com/v1",
        vertex_ai_project_id="regional-project",
        vertex_ai_location="us-central1",
    )

    assert build_vertex_ai_openai_base_url(config) == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/regional-project/"
        "locations/us-central1/endpoints/openapi"
    )


def test_vertex_ai_base_url_preserves_complete_openai_endpoint():
    config = _vertex_config(
        api_base=(
            "https://proxy.example.com/v1/projects/demo-project/"
            "locations/global/endpoints/openapi/"
        ),
    )

    assert build_vertex_ai_openai_base_url(config) == (
        "https://proxy.example.com/v1/projects/demo-project/"
        "locations/global/endpoints/openapi"
    )


def test_vertex_ai_base_url_appends_project_path_to_custom_host():
    config = _vertex_config(
        api_base="https://private.googleapis.com",
        vertex_ai_project_id="custom-project",
        vertex_ai_location="europe-west4",
    )

    assert build_vertex_ai_openai_base_url(config) == (
        "https://private.googleapis.com/v1/projects/custom-project/"
        "locations/europe-west4/endpoints/openapi"
    )


def test_vertex_ai_base_url_appends_endpoint_to_custom_location_path():
    config = _vertex_config(
        api_base="https://proxy.example.com/v1/projects/custom-project/locations/asia-east1",
        vertex_ai_project_id="ignored-project",
        vertex_ai_location="ignored-location",
    )

    assert build_vertex_ai_openai_base_url(config) == (
        "https://proxy.example.com/v1/projects/custom-project/"
        "locations/asia-east1/endpoints/openapi"
    )


def test_vertex_ai_project_id_can_be_loaded_from_service_account_file(tmp_path):
    credentials_path = tmp_path / "service-account.json"
    credentials_path.write_text(
        json.dumps({"project_id": "file-project"}),
        encoding="utf-8",
    )

    assert (
        resolve_vertex_ai_project_id(
            _vertex_config(
                vertex_ai_project_id="",
                vertex_ai_credentials_path=str(credentials_path),
            )
        )
        == "file-project"
    )


def test_vertex_ai_auth_refreshes_oauth_token(monkeypatch):
    class FakeCredentials:
        valid = False
        token = None

        def __init__(self):
            self.refresh_count = 0

        def refresh(self, request):
            assert request == "refresh-request"
            self.refresh_count += 1
            self.valid = True
            self.token = "ya29.test-token"

    credentials = FakeCredentials()
    auth = VertexAIOpenAIAuth(_vertex_config())

    monkeypatch.setattr(auth, "_get_credentials", lambda: credentials)
    monkeypatch.setattr(auth, "_make_refresh_request", lambda: "refresh-request")

    assert auth.refresh_client_api_key() == "ya29.test-token"
    assert auth.refresh_client_api_key() == "ya29.test-token"
    assert credentials.refresh_count == 1


def test_vertex_ai_auth_type_is_case_insensitive():
    auth = VertexAIOpenAIAuth(_vertex_config(vertex_ai_auth_type=" ADC "))

    assert auth.auth_type == "adc"


def test_vertex_ai_rejects_api_key_auth_for_openai_endpoint():
    with pytest.raises(ValueError, match="API key auth"):
        VertexAIOpenAIAuth(_vertex_config(vertex_ai_auth_type="api_key"))


def test_vertex_ai_adc_auth_can_discover_project_before_building_base_url(monkeypatch):
    monkeypatch.setattr(
        VertexAIOpenAIAuth,
        "_load_adc_credentials",
        staticmethod(lambda scopes: (object(), "adc-project")),
    )

    auth = VertexAIOpenAIAuth(
        _vertex_config(vertex_ai_auth_type="adc", vertex_ai_project_id="")
    )

    assert auth.project_id == "adc-project"
    assert auth.base_url == (
        "https://aiplatform.googleapis.com/v1/projects/adc-project/"
        "locations/global/endpoints/openapi"
    )


@pytest.mark.asyncio
async def test_vertex_ai_provider_sets_vertex_headers(monkeypatch):
    captured = {}

    def fake_create_proxy_client(
        provider_label,
        proxy=None,
        headers=None,
        verify=None,
        httpx_module=None,
    ):
        client = httpx.AsyncClient()
        captured["client"] = client
        captured["httpx_module"] = httpx_module
        return client

    monkeypatch.setattr(
        openai_source_module,
        "create_proxy_client",
        fake_create_proxy_client,
    )

    provider = ProviderOpenAIOfficial(
        _vertex_config(),
        provider_settings={},
    )
    try:
        assert str(provider.client.base_url).rstrip("/") == (
            "https://aiplatform.googleapis.com/v1/projects/demo-project/"
            "locations/global/endpoints/openapi"
        )
        assert provider.chosen_api_key == "vertex-ai-oauth"
        assert provider.custom_headers["x-goog-user-project"] == "demo-project"
        assert provider.vertex_ai_auth.default_query() == {}
        assert captured["client"].event_hooks["request"] == []
    finally:
        await provider.terminate()


@pytest.mark.asyncio
async def test_vertex_ai_oauth_token_refreshes_before_query(monkeypatch):
    provider = ProviderOpenAIOfficial(_vertex_config(), provider_settings={})
    captured = {}

    async def fake_create(**kwargs):
        captured["client_api_key"] = provider.client.api_key
        captured["kwargs"] = kwargs
        return ChatCompletion.model_validate(
            {
                "id": "chatcmpl-vertex-test",
                "object": "chat.completion",
                "created": 0,
                "model": "google/gemini-3.5-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    monkeypatch.setattr(
        provider.vertex_ai_auth,
        "refresh_client_api_key",
        lambda: "ya29.query-token",
    )
    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

    try:
        response = await provider._query(
            payloads={
                "model": "google/gemini-3.5-flash",
                "messages": [{"role": "user", "content": "ping"}],
            },
            tools=None,
        )

        assert captured["client_api_key"] == "ya29.query-token"
        assert captured["kwargs"]["messages"] == [
            {"role": "user", "content": "ping"}
        ]
        assert response.completion_text == "pong"
    finally:
        await provider.terminate()


def test_vertex_ai_config_template_defaults_to_service_account_and_global():
    provider_metadata = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]
    template = provider_metadata["config_template"]["Google Vertex AI"]
    assert template["provider"] == "google-vertex-ai"
    assert template["type"] == "openai_chat_completion"
    assert template["vertex_ai_auth_type"] == "service_account"
    assert template["vertex_ai_location"] == "global"
    assert template["api_base"] == "https://aiplatform.googleapis.com/v1"

    items = provider_metadata["items"]
    assert items["vertex_ai_auth_type"]["options"] == [
        "service_account",
        "adc",
    ]
    assert items["vertex_ai_credentials_path"]["condition"] == {
        "provider": "google-vertex-ai",
        "vertex_ai_auth_type": "service_account",
    }


def test_vertex_ai_config_metadata_i18n_keys_exist_for_all_locales():
    locales_dir = (
        Path(__file__).resolve().parents[1]
        / "dashboard"
        / "src"
        / "i18n"
        / "locales"
    )
    expected_keys = {
        "google_vertex_ai",
        "vertex_ai_auth_type",
        "vertex_ai_project_id",
        "vertex_ai_location",
        "vertex_ai_credentials_path",
        "vertex_ai_credentials_json",
    }

    for locale in ("zh-CN", "en-US", "ru-RU"):
        metadata_path = locales_dir / locale / "features" / "config-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        provider_translations = metadata["provider_group"]["provider"]

        assert expected_keys <= provider_translations.keys()
