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
# 1. Question-Answering 모델 클래스 정의
###############################################################################
class QAModel(LightningModule):
    def __init__(self, pretrained: str, server_page: str):
        """
        :param pretrained: QA 모델 경로 또는 HF Hub ID.
        :param server_page: templates 폴더 아래 사용할 HTML 템플릿 파일명.
        """
        super().__init__()
        self.pretrained = pretrained
        self.server_page = server_page

        # 1) 모델 로드 (from_pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.pretrained)
        # 2) 파이프라인 초기화 (question-answering 태스크)
        self.qa_pipeline = pipeline(
            task="question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def run_server(self, server: Flask, *args, **kwargs):
        """
        Flask 서버 실행
        """
        QAModel.WebAPI.register(route_base='/', app=server, init_argument=self)
        server.run(*args, **kwargs)

    def infer_one(self, question: str, context: str) -> Dict[str, Any]:
        """
        모델을 이용한 답변 생성 (score, start, end 추가)
        """
        if not question.strip():
            return {"question": question, "context": context, "answer": "(질문이 비어 있습니다.)"}
        if not context.strip():
            return {"question": question, "context": context, "answer": "(지문이 비어 있습니다.)"}

        result = self.qa_pipeline({"question": question, "context": context})

        return {
            "question": question,
            "context": context,
            "answer": result["answer"],
            "score": round(result["score"], 4),  # 소수점 4자리까지 반올림
            "start": result["start"],
            "end": result["end"]
        }

    ###########################################################################
    # 2. 웹 API 라우트 정의
    ###########################################################################
    class WebAPI(FlaskView):
        def __init__(self, model: "QAModel"):
            self.model = model

        @route('/')
        def index(self):
            """ 메인 페이지 렌더링 """
            return render_template(self.model.server_page)

        @route('/api', methods=['POST'])
        def api(self):
            """ AJAX 요청 처리 (질문-지문 입력받아 답변 반환) """
            data = request.json
            question = data.get("question", "")
            context = data.get("context", "")
            result = self.model.infer_one(question, context)
            return jsonify(result)


###############################################################################
# 3. serve() 함수: Flask 서버 실행
###############################################################################
main = typer.Typer()


@main.command()
def serve(
        pretrained: str = typer.Option("output/korquad/train=KPF-BERT=dgx-a100/checkpoint-15723",
                                       help="QA 모델 경로 혹은 HuggingFace model ID"),
        server_host: str = typer.Option("0.0.0.0"),
        server_port: int = typer.Option(9000),
        server_page: str = typer.Option("serve_qa.html", help="templates 폴더 내 템플릿 파일명"),
        debug: bool = typer.Option(False),
):
    logging.basicConfig(level=logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1) 모델 준비
    model = QAModel(pretrained=pretrained, server_page=server_page)

    # 2) Flask 인스턴스 생성
    app = Flask(__name__, template_folder=Path("templates").resolve())

    # 3) 모델에 정의된 run_server() 호출
    model.run_server(app, host=server_host, port=server_port, debug=debug)


if __name__ == "__main__":
    main()
