import json
import logging

from bert_api.serving import ServingError, ServingUnavailable
from conftest import FakeClient


def test_predict_returns_200_with_label_and_scores(make_client):
    http, fake = make_client()
    r = http.post("/v1/predict", json={"text": "The service was great."})
    assert r.status_code == 200  # 2021 code returned 201
    body = r.get_json()
    assert body["model"] == "text_classifier"
    (res,) = body["results"]
    assert res["label"] == "label_2"
    assert set(res["scores"]) == {"label_0", "label_1", "label_2"}
    assert abs(sum(res["scores"].values()) - 1) < 1e-3
    assert len(fake.calls) == 1


def test_labels_file_is_used(make_client, tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["negative", "neutral", "positive"]))
    http, _ = make_client(labels_path=str(labels))
    res = http.post("/v1/predict", json={"text": "great"}).get_json()["results"][0]
    assert res["label"] == "positive"


def test_batch_of_texts(make_client):
    http, fake = make_client()
    r = http.post("/v1/predict", json={"texts": ["great", "slow service"]})
    assert r.status_code == 200
    assert len(r.get_json()["results"]) == 2
    assert len(fake.calls[0]) == 2


def test_long_text_is_truncated_to_max_seq_length(make_client):
    http, fake = make_client()
    r = http.post("/v1/predict", json={"text": "the service was great and slow " * 3})
    assert r.status_code == 200
    assert r.get_json()["results"][0]["num_tokens"] == 8
    assert fake.calls[0][0].input_ids[0] == 2 and fake.calls[0][0].input_ids[-1] == 3  # [CLS] ... [SEP]


def test_rejects_non_json(make_client):
    http, _ = make_client()
    r = http.post("/v1/predict", data="text=hello", content_type="application/x-www-form-urlencoded")
    assert r.status_code == 415


def test_rejects_malformed_json(make_client):
    http, _ = make_client()
    r = http.post("/v1/predict", data="{not json", content_type="application/json")
    assert r.status_code == 400
    assert r.get_json()["error"] == "invalid_json"


def test_double_encoded_json_is_rejected(make_client):
    # The 2021 handler did json.loads(request.get_json()), so clients had to send a JSON *string*.
    http, _ = make_client()
    r = http.post("/v1/predict", json=json.dumps({"text": "hello"}))
    assert r.status_code == 400


def test_missing_or_bad_text(make_client):
    http, fake = make_client()
    assert http.post("/v1/predict", json={}).status_code == 400
    assert http.post("/v1/predict", json={"text": ""}).status_code == 400
    assert http.post("/v1/predict", json={"text": "   "}).status_code == 400
    assert http.post("/v1/predict", json={"text": 42}).status_code == 400
    assert http.post("/v1/predict", json={"texts": []}).status_code == 400
    assert http.post("/v1/predict", json={"texts": ["ok", None]}).status_code == 400
    assert fake.calls == []


def test_too_long_and_too_many(make_client):
    http, _ = make_client()
    assert http.post("/v1/predict", json={"text": "a" * 101}).status_code == 413
    assert http.post("/v1/predict", json={"texts": ["a"] * 33}).status_code == 413


def test_tf_serving_down_is_503(make_client):
    http, _ = make_client(client=FakeClient(fail=ServingUnavailable("ConnectTimeout")))
    r = http.post("/v1/predict", json={"text": "great"})
    assert r.status_code == 503
    assert r.get_json()["error"] == "model_unavailable"


def test_tf_serving_bad_response_is_502(make_client):
    http, _ = make_client(client=FakeClient(fail=ServingError("TF-Serving returned 400: shape")))
    assert http.post("/v1/predict", json={"text": "great"}).status_code == 502


def test_health_and_readiness(make_client):
    http, _ = make_client()
    assert http.get("/healthz").status_code == 200
    assert http.get("/readyz").status_code == 200
    http, _ = make_client(client=FakeClient(ready=False))
    assert http.get("/readyz").status_code == 503


def test_unknown_route_and_method_return_json(make_client):
    http, _ = make_client()
    r = http.get("/v1/predict")
    assert r.status_code == 405 and r.get_json()["error"] == "method_not_allowed"
    assert http.get("/nope").status_code == 404


def test_cors_is_opt_in(make_client):
    http, _ = make_client()
    assert "Access-Control-Allow-Origin" not in http.get("/healthz").headers
    http, _ = make_client(cors_origin="https://example.com")
    assert http.get("/healthz").headers["Access-Control-Allow-Origin"] == "https://example.com"


def test_request_id_is_echoed(make_client):
    http, _ = make_client()
    assert http.get("/healthz", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"


def test_logs_never_contain_request_text(make_client, caplog):
    http, _ = make_client()
    secret = "my account number is 12345"
    with caplog.at_level(logging.DEBUG, logger="bert_api"):
        http.post("/v1/predict", json={"text": secret})
        http.post("/v1/predict", json={"text": secret * 10})  # rejected: too long
    assert caplog.records, "expected request log lines"
    assert all(secret not in r.getMessage() for r in caplog.records)
    assert any("n_chars=26" in r.getMessage() for r in caplog.records)


def test_tokenizer_and_client_are_built_once(settings, tokenizer, monkeypatch):
    """The 2021 bug: the tokenizer was loaded inside the request handler."""
    import bert_api

    calls = {"n": 0}

    def counting_loader(*args, **kwargs):
        calls["n"] += 1
        return tokenizer

    monkeypatch.setattr(bert_api, "load_tokenizer", counting_loader)
    app = bert_api.create_app(settings=settings, client=FakeClient())
    http = app.test_client()
    for _ in range(5):
        assert http.post("/v1/predict", json={"text": "great"}).status_code == 200
    assert calls["n"] == 1
