import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


GOOGLE_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
VERTEX_AI_DEFAULT_LOCATION = "global"
VERTEX_AI_DEFAULT_API_BASE = "https://aiplatform.googleapis.com"
VERTEX_AI_OPENAI_API_VERSION = "v1"
VERTEX_AI_OPENAI_ENDPOINT_SUFFIX = "endpoints/openapi"
VERTEX_AI_SERVICE_ACCOUNT_AUTH = "service_account"
VERTEX_AI_ADC_AUTH = "adc"


def is_vertex_ai_openai_config(provider_config: dict[str, Any]) -> bool:
    """Return True when an OpenAI-compatible source targets Vertex AI."""

    return any(
        key in provider_config
        for key in (
            "vertex_ai_auth_type",
            "vertex_ai_project_id",
            "vertex_ai_location",
            "vertex_ai_credentials_path",
            "vertex_ai_credentials_json",
        )
    )


def normalize_vertex_ai_location(location: Any) -> str:
    normalized = str(location or "").strip()
    return normalized or VERTEX_AI_DEFAULT_LOCATION


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _base_url_has_openai_endpoint(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.path.rstrip("/").endswith(f"/{VERTEX_AI_OPENAI_ENDPOINT_SUFFIX}")


def _base_url_has_vertex_location(base_url: str) -> bool:
    parsed = urlparse(base_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 5:
        return False
    return parts[-4] == "projects" and parts[-2] == "locations"


def _append_openai_api_version(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith(f"/{VERTEX_AI_OPENAI_API_VERSION}"):
        return base_url
    return f"{base_url}/{VERTEX_AI_OPENAI_API_VERSION}"


def _default_vertex_ai_api_base(location: str) -> str:
    if location == VERTEX_AI_DEFAULT_LOCATION:
        return VERTEX_AI_DEFAULT_API_BASE
    return f"https://{location}-aiplatform.googleapis.com"


def _is_default_vertex_ai_api_base(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return (
        parsed.scheme.lower() == "https"
        and parsed.netloc.lower() == "aiplatform.googleapis.com"
        and parsed.path.strip("/") in {"", VERTEX_AI_OPENAI_API_VERSION}
    )


def _read_service_account_info(provider_config: dict[str, Any]) -> dict[str, Any]:
    credentials_json = str(provider_config.get("vertex_ai_credentials_json") or "")
    if credentials_json.strip():
        return json.loads(credentials_json)

    credentials_path = str(provider_config.get("vertex_ai_credentials_path") or "")
    if credentials_path.strip():
        return json.loads(Path(credentials_path).expanduser().read_text(encoding="utf-8"))

    return {}


def resolve_vertex_ai_project_id(provider_config: dict[str, Any]) -> str:
    project_id = str(provider_config.get("vertex_ai_project_id") or "").strip()
    if project_id:
        return project_id

    try:
        info = _read_service_account_info(provider_config)
    except Exception:
        return ""
    return str(info.get("project_id") or "").strip()


def build_vertex_ai_openai_base_url(provider_config: dict[str, Any]) -> str:
    project_id = resolve_vertex_ai_project_id(provider_config)
    if not project_id:
        raise ValueError("Vertex AI project id is required.")

    location = normalize_vertex_ai_location(provider_config.get("vertex_ai_location"))
    configured_base_url = _normalize_base_url(str(provider_config.get("api_base") or ""))

    if configured_base_url:
        if _base_url_has_openai_endpoint(configured_base_url):
            return configured_base_url
        if _base_url_has_vertex_location(configured_base_url):
            return f"{configured_base_url}/{VERTEX_AI_OPENAI_ENDPOINT_SUFFIX}"
        if _is_default_vertex_ai_api_base(configured_base_url):
            api_base = (
                f"{_default_vertex_ai_api_base(location)}/"
                f"{VERTEX_AI_OPENAI_API_VERSION}"
            )
        else:
            api_base = _append_openai_api_version(configured_base_url)
    else:
        api_base = f"{_default_vertex_ai_api_base(location)}/{VERTEX_AI_OPENAI_API_VERSION}"

    return (
        f"{api_base}/projects/{project_id}/locations/{location}/"
        f"{VERTEX_AI_OPENAI_ENDPOINT_SUFFIX}"
    )


class VertexAIOpenAIAuth:
    def __init__(self, provider_config: dict[str, Any]) -> None:
        self.auth_type = str(
            provider_config.get("vertex_ai_auth_type") or VERTEX_AI_SERVICE_ACCOUNT_AUTH
        ).strip().lower()
        self.project_id = resolve_vertex_ai_project_id(provider_config)
        self.location = normalize_vertex_ai_location(
            provider_config.get("vertex_ai_location")
        )
        self._credentials: Any | None = None

        if self.auth_type not in {
            VERTEX_AI_SERVICE_ACCOUNT_AUTH,
            VERTEX_AI_ADC_AUTH,
        }:
            raise ValueError(
                "Vertex AI OpenAI-compatible endpoint supports service_account "
                "or adc auth. API key auth is only available on the native "
                "Vertex AI Gemini API."
            )

        if self.auth_type == VERTEX_AI_ADC_AUTH and not self.project_id:
            credentials, project_id = self._load_adc_credentials(
                [GOOGLE_CLOUD_PLATFORM_SCOPE]
            )
            self._credentials = credentials
            if project_id:
                self.project_id = str(project_id)

        self.provider_config = {**provider_config, "vertex_ai_project_id": self.project_id}
        self.base_url = build_vertex_ai_openai_base_url(self.provider_config)

    @staticmethod
    def _load_adc_credentials(scopes: list[str]) -> tuple[Any, str | None]:
        try:
            from google.auth import default as google_auth_default
        except ImportError as exc:
            raise RuntimeError("google-auth is required for Vertex AI ADC auth.") from exc

        return google_auth_default(scopes=scopes)

    def initial_client_api_key(self) -> str:
        return "vertex-ai-oauth"

    def default_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.project_id:
            headers["x-goog-user-project"] = self.project_id
        return headers

    def default_query(self) -> dict[str, str]:
        return {}

    def refresh_client_api_key(self) -> str:
        credentials = self._get_credentials()
        if not getattr(credentials, "valid", False) or not getattr(
            credentials, "token", None
        ):
            credentials.refresh(self._make_refresh_request())
        token = getattr(credentials, "token", None)
        if not token:
            raise RuntimeError("Failed to refresh Vertex AI access token.")
        return str(token)

    def _get_credentials(self) -> Any:
        if self._credentials is not None:
            return self._credentials

        try:
            from google.oauth2 import service_account
        except ImportError as exc:
            raise RuntimeError(
                "google-auth is required for Vertex AI service account or ADC auth."
            ) from exc

        scopes = [GOOGLE_CLOUD_PLATFORM_SCOPE]
        credentials_json = str(
            self.provider_config.get("vertex_ai_credentials_json") or ""
        )
        credentials_path = str(
            self.provider_config.get("vertex_ai_credentials_path") or ""
        )

        if self.auth_type == VERTEX_AI_ADC_AUTH:
            credentials, project_id = self._load_adc_credentials(scopes)
            if not self.project_id and project_id:
                self.project_id = str(project_id)
        elif credentials_json.strip():
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(credentials_json),
                scopes=scopes,
            )
        elif credentials_path.strip():
            credentials = service_account.Credentials.from_service_account_file(
                str(Path(credentials_path).expanduser()),
                scopes=scopes,
            )
        else:
            credentials, project_id = self._load_adc_credentials(scopes)
            if not self.project_id and project_id:
                self.project_id = str(project_id)

        self._credentials = credentials
        return credentials

    def _make_refresh_request(self) -> Any:
        try:
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise RuntimeError(
                "google-auth requests transport is required for Vertex AI auth."
            ) from exc

        proxy = str(self.provider_config.get("proxy") or "").strip()
        if not proxy:
            return Request()

        try:
            import requests
        except ImportError:
            return Request()

        session = requests.Session()
        session.proxies.update({"http": proxy, "https": proxy})
        session.trust_env = False
        return Request(session=session)
