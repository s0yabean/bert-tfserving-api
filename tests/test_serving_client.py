"""TFServingClient against a stubbed requests.Session (no network)."""
import pytest
import requests

from bert_api.serving import ServingError, ServingUnavailable, TFServingClient
from bert_api.tokenizer import Features


class FakeResponse:
    def __init__(self, status_code=200, payload=None, raises=False):
        self.status_code = status_code
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json, timeout))
        if self.exc:
            raise self.exc
        return self.response

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None, timeout))
        if self.exc:
            raise self.exc
        return self.response


FEAT = Features(input_ids=[2, 5, 3], attention_mask=[1, 1, 1], token_type_ids=[0, 0, 0])


def make(session):
    return TFServingClient("http://tfserving:8501", "text_classifier", 1.5, 7.0, session=session)


def test_url_timeout_and_body():
    s = FakeSession(FakeResponse(payload={"predictions": [[0.1, 0.9]]}))
    assert make(s).predict([FEAT]) == [[0.1, 0.9]]
    method, url, body, timeout = s.calls[0]
    assert url == "http://tfserving:8501/v1/models/text_classifier:predict"
    assert timeout == (1.5, 7.0)  # 2021 code had no timeout at all
    assert body["instances"][0]["input_ids"] == [2, 5, 3]  # ints, not floats


def test_batch_is_padded_to_longest_text():
    short = Features(input_ids=[2, 3], attention_mask=[1, 1], token_type_ids=[0, 0])
    s = FakeSession(FakeResponse(payload={"predictions": [[0.5, 0.5], [0.5, 0.5]]}))
    make(s).predict([FEAT, short])
    inst = s.calls[0][2]["instances"]
    assert inst[0]["input_ids"] == [2, 5, 3]
    assert inst[1]["input_ids"] == [2, 3, 0]
    assert inst[1]["attention_mask"] == [1, 1, 0]


def test_named_output_dicts_are_unwrapped():
    s = FakeSession(FakeResponse(payload={"predictions": [{"probabilities": [0.3, 0.7]}]}))
    assert make(s).predict([FEAT]) == [[0.3, 0.7]]


@pytest.mark.parametrize("exc", [requests.ConnectionError(), requests.Timeout()])
def test_network_errors_are_unavailable(exc):
    with pytest.raises(ServingUnavailable):
        make(FakeSession(exc=exc)).predict([FEAT])


@pytest.mark.parametrize("resp", [
    FakeResponse(400, {"error": "Incompatible shapes"}),
    FakeResponse(500, raises=True),
    FakeResponse(200, {"oops": 1}),
    FakeResponse(200, raises=True),
    FakeResponse(200, {"predictions": [[0.1], [0.2]]}),  # count mismatch
])
def test_bad_responses_are_serving_errors(resp):
    with pytest.raises(ServingError):
        make(FakeSession(resp)).predict([FEAT])


def test_is_ready():
    ok = FakeResponse(payload={"model_version_status": [{"version": "1", "state": "AVAILABLE"}]})
    loading = FakeResponse(payload={"model_version_status": [{"version": "1", "state": "LOADING"}]})
    assert make(FakeSession(ok)).is_ready() is True
    assert make(FakeSession(loading)).is_ready() is False
    assert make(FakeSession(exc=requests.ConnectionError())).is_ready() is False
