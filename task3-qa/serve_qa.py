import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import typer
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route
from lightning import LightningModule

from chrisbase.io import paths
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

logger = logging.getLogger(__name__)


###############################################################################
# 1. Question-Answering Model Definition
###############################################################################
class QAModel(LightningModule):
    def __init__(self, pretrained: str, server_page: str, normalized: bool = True):
        """
        :param pretrained: Path to the QA model or Hugging Face Hub ID.
        :param server_page: The HTML template file name inside the "templates" folder.
        :param normalized: Whether to use softmax normalization for score calculation.
        """
        super().__init__()
        self.server_page = server_page
        self.normalized = normalized

        # 1) Load model (from_pretrained)
        logger.info(f"Loading model from {pretrained}")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained)
        self.model.eval()  # Set to evaluation mode

    def run_server(self, server: Flask, *args, **kwargs):
        """
        Run the Flask server.
        """
        QAModel.WebAPI.register(route_base='/', app=server, init_argument=self)
        server.run(*args, **kwargs)

    def infer_one(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate an answer using the BERT model
        """
        if not question.strip():
            return {"question": question, "context": context, "answer": "(The question is empty.)"}
        if not context.strip():
            return {"question": question, "context": context, "answer": "(The context is empty.)"}

        inputs = self.tokenizer.encode_plus(
            question, context, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

        predict_answer_tokens = inputs["input_ids"][0, start_index: end_index + 1]
        answer = self.tokenizer.decode(predict_answer_tokens)

        if self.normalized:
            start_probs = F.softmax(start_logits, dim=-1)
            end_probs = F.softmax(end_logits, dim=-1)
            score = (torch.max(start_probs) * torch.max(end_probs)).item()
        else:
            score = float(torch.max(start_logits) + torch.max(end_logits))

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "score": round(score, 4),
            "start": int(start_index),
            "end": int(end_index)
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
        # TODO: "output/korquad/train_qa-*/checkpoint-*" or "monologg/koelectra-base-v3-finetuned-korquad"
        pretrained: str = typer.Option("output/korquad/train_qa-*/checkpoint-*",
                                       help="Local pretrained model path or Hugging Face Hub ID"),
        server_host: str = typer.Option("0.0.0.0"),
        server_port: int = typer.Option(9164),
        server_page: str = typer.Option("serve_qa.html", help="HTML template file inside the templates folder"),
        normalized: bool = typer.Option(True, help="Use softmax normalization for score calculation"),
        debug: bool = typer.Option(False),
):
    logging.basicConfig(level=logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    checkpoint_paths = paths(pretrained)
    if checkpoint_paths and len(checkpoint_paths) > 0:
        pretrained = str(sorted(checkpoint_paths, key=os.path.getmtime)[-1])

    # 1) Load model
    model = QAModel(pretrained=pretrained, server_page=server_page, normalized=normalized)

    # 2) Create Flask instance
    app = Flask(__name__, template_folder=Path("templates").resolve())

    # 3) Run the server
    model.run_server(app, host=server_host, port=server_port, debug=debug)


if __name__ == "__main__":
    main()
