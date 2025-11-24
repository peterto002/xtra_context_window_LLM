from flask import Flask, render_template, request, jsonify

from infer_service import generate


def create_app():
    app = Flask(__name__)

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.post("/generate")
    def do_generate():
        data = request.get_json(force=True, silent=False)
        prompt = data.get("prompt_text", "What is the purpose of life?")
        tokens = int(data.get("max_new_tokens", 120))
        top_k = int(data.get("top_k", 3))
        temperature = float(data.get("temperature", 0.8))

        tokens = max(1, min(tokens, 1024))
        temperature = max(0.05, min(temperature, 2.5))

        text = generate(prompt, tokens, top_k, temperature)
        return jsonify({"text": text})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
