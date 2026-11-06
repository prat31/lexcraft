import threading
from http.server import ThreadingHTTPServer

import httpx

from lexcraft.backends.base import Backend, ModelDescription
from lexcraft.server import make_handler


class Echo(Backend):
    def describe(self) -> ModelDescription:
        return ModelDescription("fake", "fake")

    def generate(self, messages, max_tokens, temperature):  # type: ignore[no-untyped-def]
        return "<think></think>" + messages[-1]["content"].upper()


def test_openai_compatible_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Echo(), "lexcraft-test"))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    try:
        assert httpx.get(f"{base}/models").json()["data"][0]["id"] == "lexcraft-test"
        r = httpx.post(
            f"{base}/chat/completions", json={"messages": [{"role": "user", "content": "pay now"}]}
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "PAY NOW"
        assert httpx.post(f"{base}/chat/completions", content=b"not json").status_code == 400
    finally:
        httpd.shutdown()
