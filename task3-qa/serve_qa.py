import logging
import os
from pathlib import Path
from typing import Dict, Any

import typer
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


###############################################################################
# 1. Question-Answering Model Definition
###############################################################################
class QAModel(LightningModule):
    def __init__(self, pretrained: str, server_page: str):
        """
        :param pretrained: Path to the QA model or Hugging Face Hub ID.
        :param server_page: The HTML template file name inside the "templates" folder.
        """
        super().__init__()
        self.pretrained = pretrained
        self.server_page = server_page

        # 1) Load model (from_pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.pretrained)

        # 2) Initialize pipeline (for question-answering task)
        self.qa_pipeline = pipeline(
            task="question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def run_server(self, server: Flask, *args, **kwargs):
        """
        Run the Flask server.
        """
        QAModel.WebAPI.register(route_base='/', app=server, init_argument=self)
        server.run(*args, **kwargs)

    def infer_one(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate an answer using the model (including score, start, end positions).
        """
        if not question.strip():
            return {"question": question, "context": context, "answer": "(The question is empty.)"}
        if not context.strip():
            return {"question": question, "context": context, "answer": "(The context is empty.)"}

        result = self.qa_pipeline({"question": question, "context": context})

        return {
            "question": question,
            "context": context,
            "answer": result["answer"],
            "score": round(result["score"], 4),  # Round to 4 decimal places
            "start": result["start"],
            "end": result["end"]
        }

    ###########################################################################
    # 2. Web API Routes
    ###########################################################################
    class WebAPI(FlaskView):
        def __init__(self, model: "QAModel"):
            self.model = model

        @route('/')
        def index(self):
            """ Render the main page """
            return render_template(self.model.server_page)

        @route('/api', methods=['POST'])
        def api(self):
            """ Handle AJAX request (receive question-context input and return an answer) """
            data = request.json
            question = data.get("question", "")
            context = data.get("context", "")
            result = self.model.infer_one(question, context)
            return jsonify(result)


###############################################################################
# 3. serve() Function: Run Flask Server
###############################################################################
main = typer.Typer()


@main.command()
def serve(
        pretrained: str = typer.Option("output/korquad/train=KPF-BERT=dgx-a100/checkpoint-15723",
                                       help="Path to the QA model or Hugging Face model ID"),
        server_host: str = typer.Option("0.0.0.0"),
        server_port: int = typer.Option(9000),
        server_page: str = typer.Option("serve_qa.html", help="HTML template file inside the templates folder"),
        debug: bool = typer.Option(False),
):
    logging.basicConfig(level=logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1) Load model
    model = QAModel(pretrained=pretrained, server_page=server_page)

    # 2) Create Flask instance
    app = Flask(__name__, template_folder=Path("templates").resolve())

    # 3) Run the server
    model.run_server(app, host=server_host, port=server_port, debug=debug)


if __name__ == "__main__":
    main()
