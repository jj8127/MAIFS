#!/usr/bin/env python
"""
MAIFS - Multi-Agent Image Forensic System
메인 실행 파일
"""
import argparse
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.maifs import MAIFS, analyze_image


def main():
    parser = argparse.ArgumentParser(
        description="MAIFS - Multi-Agent Image Forensic System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py analyze image.jpg
  python main.py analyze image.jpg --no-debate
  python main.py analyze image.jpg --output report.json
  python main.py server --port 8080
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # analyze 명령어
    analyze_parser = subparsers.add_parser("analyze", help="이미지 분석")
    analyze_parser.add_argument("image", type=str, help="분석할 이미지 경로")
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="보고서 저장 경로 (.json 또는 .txt)"
    )
    analyze_parser.add_argument(
        "--no-debate",
        action="store_true",
        help="토론 비활성화"
    )
    analyze_parser.add_argument(
        "--algorithm",
        type=str,
        choices=["rot", "drwa", "avga"],
        default="drwa",
        help="합의 알고리즘 (기본: drwa)"
    )
    analyze_parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="연산 디바이스 (기본: cuda)"
    )

    # server 명령어
    server_parser = subparsers.add_parser("server", help="웹 서버 실행")
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="서버 포트 (기본: 7860)"
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="서버 호스트 (기본: 0.0.0.0)"
    )

    # version 명령어
    subparsers.add_parser("version", help="버전 정보")

    args = parser.parse_args()

    if args.command == "analyze":
        run_analysis(args)
    elif args.command == "server":
        run_server(args)
    elif args.command == "version":
        print(f"MAIFS v{MAIFS.VERSION}")
    else:
        parser.print_help()


def run_analysis(args):
    """이미지 분석 실행"""
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 - {image_path}")
        sys.exit(1)

    print(f"MAIFS v{MAIFS.VERSION}")
    print(f"분석 대상: {image_path}")
    print("-" * 40)

    # MAIFS 인스턴스 생성
    maifs = MAIFS(
        enable_debate=not args.no_debate,
        consensus_algorithm=args.algorithm,
        device=args.device
    )

    # 분석 실행
    result = maifs.analyze(
        image_path,
        save_report=Path(args.output) if args.output else None
    )

    # 결과 출력
    print()
    print("=" * 40)
    print(result.summary)
    print("=" * 40)
    print()
    print(result.get_verdict_explanation())
    print()
    print(f"처리 시간: {result.processing_time:.2f}초")

    if args.output:
        print(f"보고서 저장됨: {args.output}")


def run_server(args):
    """Gradio 웹 서버 실행"""
    try:
        from app import create_app
        app = create_app()
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=False
        )
    except ImportError:
        print("오류: Gradio가 설치되어 있지 않습니다.")
        print("설치: pip install gradio")
        sys.exit(1)


if __name__ == "__main__":
    main()
