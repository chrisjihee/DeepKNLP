"""Generative QA serving lab entrypoint.

Step 3 focuses on how a fine-tuned seq2seq checkpoint is turned into an actual inference service.
Students complete the model loading and single-example answer generation blocks in this file.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import typer
from flask import Flask, request, jsonify, render_template
from flask_classful import FlaskView, route

from chrisbase.io import paths
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


###############################################################################
# 1. Question-Answering Model Definition
###############################################################################
class QAModel:
    def __init__(self, pretrained: str, server_page: str, num_beams: int = 5, max_length: int = 50):
        """
        :param pretrained: Path to the T5 model or Hugging Face Hub ID.
        :param server_page: The HTML template file name inside the "templates" folder.
        :param num_beams: Beam search width for text generation.
        :param max_length: Maximum length of the generated answer.
        """
        self.server_page = server_page
        self.num_beams = num_beams
        self.max_length = max_length

        # TODO Step 3-1:
        # Load the tokenizer and the fine-tuned seq2seq QA model from the checkpoint path.
        # logger.info(f"Loading model from {pretrained}")
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
        # self.model.eval()
        raise NotImplementedError(
            "TODO Step 3-1: load the tokenizer and seq2seq QA checkpoint here."
        )

    def run_server(self, server: Flask, *args, **kwargs):
        """
        Run the Flask server.
        """
        QAModel.WebAPI.register(route_base='/', app=server, init_argument=self)
        server.run(*args, **kwargs)

    def infer_one(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate an answer using the T5 model with score calculation.
        """
        if not question.strip():
            return {"question": question, "context": context, "answer": "(The question is empty.)"}
        if not context.strip():
            return {"question": question, "context": context, "answer": "(The context is empty.)"}

        # TODO Step 3-2:
        # Build the single-example generative QA inference flow in place.
        # input_text = f"question: {question} context: {context}"
        # inputs = self.tokenizer(...)
        # output_ids = self.model.generate(..., return_dict_in_generate=True, output_scores=True)
        # answer = self.tokenizer.decode(...)
        # token_probs = ...
        # score = ...
        raise NotImplementedError(
            "TODO Step 3-2: implement seq2seq QA generation and answer scoring here."
        )

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "score": round(score, 4)
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
        pretrained: str = typer.Option("output/korquad/train_qa_by-pkot5-*/checkpoint-*",
                                       help="Local pretrained model path or Hugging Face Hub ID"),
        server_host: str = typer.Option("0.0.0.0"),
        server_port: int = typer.Option(9164),
        server_page: str = typer.Option("serve_qa_seq2seq.html", help="HTML template file inside the templates folder"),
        num_beams: int = typer.Option(5, help="Beam search width for text generation"),
        max_length: int = typer.Option(50, help="Maximum answer length"),
        debug: bool = typer.Option(False),
):
    """
    Start the Flask-based QA server.

    :param pretrained: Path to the trained T5 model or Hugging Face model ID.
    :param server_host: The host address for the Flask server.
    :param server_port: The port number for the Flask server.
    :param server_page: HTML template name for the UI.
    :param num_beams: Number of beams used for beam search.
    :param max_length: Maximum length of generated answers.
    :param debug: Enable Flask debug mode.
    """
    logging.basicConfig(level=logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Find the latest checkpoint
    checkpoint_paths = paths(pretrained)
    if checkpoint_paths and len(checkpoint_paths) > 0:
        pretrained = str(sorted(checkpoint_paths, key=os.path.getmtime)[-1])

    # 1) Load the T5 model
    model = QAModel(pretrained=pretrained, server_page=server_page, num_beams=num_beams, max_length=max_length)

    # 2) Create Flask instance
    app = Flask(__name__, template_folder=Path("templates").resolve())

    # 3) Run the server
    model.run_server(app, host=server_host, port=server_port, debug=debug)


if __name__ == "__main__":
    main()
