"""
MAIFS Web UI - Gradio 기반 웹 인터페이스
"""
import gradio as gr
from pathlib import Path
import numpy as np
from PIL import Image
import json

from src.maifs import MAIFS
from src.tools.base_tool import Verdict


def create_app() -> gr.Blocks:
    """Gradio 앱 생성"""

    # MAIFS 인스턴스 (전역)
    maifs = MAIFS(enable_debate=True)

    def analyze_image(
        image: np.ndarray,
        enable_debate: bool,
        algorithm: str
    ) -> tuple:
        """이미지 분석 실행"""
        if image is None:
            return "이미지를 업로드해주세요.", "", None, ""

        # 설정 업데이트
        maifs.enable_debate = enable_debate
        maifs.consensus_engine.default_algorithm = algorithm.lower()

        # 분석 실행
        result = maifs.analyze(image)

        # 결과 포맷팅
        verdict_emoji = {
            Verdict.AUTHENTIC: "✅",
            Verdict.MANIPULATED: "⚠️",
            Verdict.AI_GENERATED: "🤖",
            Verdict.UNCERTAIN: "❓"
        }

        verdict_text = f"{verdict_emoji.get(result.verdict, '')} {result.verdict.value.upper()}"
        confidence_text = f"{result.confidence:.1%}"

        # 판정 결과 HTML
        verdict_html = f"""
        <div style="text-align: center; padding: 20px;">
            <h2 style="font-size: 2em; margin: 0;">{verdict_text}</h2>
            <p style="font-size: 1.5em; color: #666;">신뢰도: {confidence_text}</p>
            <p style="font-size: 1em; color: #888;">
                처리 시간: {result.processing_time:.2f}초 |
                합의 알고리즘: {result.consensus_result.algorithm_used if result.consensus_result else 'N/A'}
            </p>
        </div>
        """

        # 에이전트별 결과
        agent_results = []
        for name, response in result.agent_responses.items():
            agent_results.append({
                "에이전트": response.agent_name.split("(")[0].strip(),
                "판정": response.verdict.value,
                "신뢰도": f"{response.confidence:.1%}",
                "처리시간": f"{response.processing_time:.2f}s"
            })

        # 조작 마스크 (있는 경우)
        mask_image = None
        for response in result.agent_responses.values():
            for tool_result in response.tool_results:
                if tool_result.manipulation_mask is not None:
                    mask = tool_result.manipulation_mask
                    mask_image = (mask * 255).astype(np.uint8)
                    break

        # JSON 결과
        json_output = result.to_json(indent=2)

        return verdict_html, result.detailed_report, mask_image, json_output

    # Gradio UI 구성
    with gr.Blocks(
        title="MAIFS - Multi-Agent Image Forensic System",
        theme=gr.themes.Soft()
    ) as app:

        gr.Markdown("""
        # 🔍 MAIFS - Multi-Agent Image Forensic System

        **다중 에이전트 기반 이미지 포렌식 시스템**

        4개의 전문가 에이전트가 이미지를 분석하여 원본/조작/AI생성 여부를 판별합니다.
        - 🔬 **주파수 분석**: FFT 기반 GAN 아티팩트 탐지
        - 📊 **노이즈 분석**: PRNU/SRM 센서 노이즈 패턴
        - 🔒 **워터마크 분석**: HiNet 기반 워터마크 탐지
        - 🖼️ **공간 분석**: ViT 기반 조작 영역 탐지
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="분석할 이미지",
                    type="numpy",
                    height=400
                )

                with gr.Row():
                    enable_debate = gr.Checkbox(
                        label="토론 활성화",
                        value=True,
                        info="의견 불일치 시 에이전트 간 토론 진행"
                    )
                    algorithm = gr.Dropdown(
                        choices=["DRWA", "RoT", "AVGA"],
                        value="DRWA",
                        label="합의 알고리즘"
                    )

                analyze_btn = gr.Button("🔍 분석 시작", variant="primary", size="lg")

            with gr.Column(scale=1):
                verdict_output = gr.HTML(label="판정 결과")

                with gr.Tabs():
                    with gr.Tab("📝 상세 보고서"):
                        report_output = gr.Textbox(
                            label="분석 보고서",
                            lines=15,
                            max_lines=30
                        )

                    with gr.Tab("🎭 조작 영역 마스크"):
                        mask_output = gr.Image(
                            label="조작 영역 (빨간색 = 조작 의심)",
                            height=300
                        )

                    with gr.Tab("📊 JSON 결과"):
                        json_output = gr.Code(
                            label="JSON 데이터",
                            language="json",
                            lines=15
                        )

        # 이벤트 연결
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, enable_debate, algorithm],
            outputs=[verdict_output, report_output, mask_output, json_output]
        )

        # 예제 이미지
        gr.Markdown("### 예제")
        gr.Examples(
            examples=[
                ["examples/authentic.jpg"],
                ["examples/manipulated.jpg"],
                ["examples/ai_generated.jpg"],
            ],
            inputs=[input_image],
            label="예제 이미지"
        )

        gr.Markdown("""
        ---
        **MAIFS v0.1.0** |
        [GitHub](https://github.com/jj8127/MAIFS) |
        [Documentation](https://maifs.readthedocs.io)
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
