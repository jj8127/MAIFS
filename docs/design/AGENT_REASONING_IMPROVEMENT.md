# Sub-Agent 추론 능력 개선 방안

**작성일**: 2026-01-23
**현재 문제**: Sub-agents가 의견을 낼 수 없음

---

## 🔍 현재 문제 분석

### Sub-agents의 현재 능력
```
Tool 실행 → 분석 결과 (판정, 신뢰도) → 규칙 기반 추론 텍스트 생성
```

현재 코드 흐름 (FrequencyAgent 예):
1. Tool 실행: `tool_result = self._tool(image)`
   - 결과: `verdict`, `confidence`, `evidence` 반환
2. 추론 생성: `reasoning = self.generate_reasoning([tool_result], context)`
   - 단순히 tool result를 텍스트로 포맷팅
   - 규칙: "격자 패턴이 많으면 AI_GENERATED"
3. 주요 논거 추출: `arguments = self._extract_arguments(tool_result)`
   - tool result에서 key points 추출

### 문제점

```python
# 현재 토론 시스템
rebuttal = responses[name_b].tool_results[0].explanation if responses[name_b].tool_results else "증거 없음"
```

- Sub-agent B가 A의 의견에 대해 **실제로 반박할 수 없음**
- 그냥 자신의 tool result 설명만 반환
- 진정한 "토론"이 아닌 "분석 결과 나열"

### 토론의 한계

현재 토론은:
```
A: "내 도구에서 X를 발견했으니 AI_GENERATED"
B: "내 도구에서 Y를 발견했으니 AUTHENTIC"

→ 서로의 의견에 대한 반박이 없음 (각자 자신의 도구 결과만 말함)
```

필요한 토론:
```
A: "X 패턴은 GAN 특성이기 때문에 AI_GENERATED"
B: "하지만 Y 패턴은 자연 이미지에서도 발견되므로 AUTHENTIC일 수 있다"
A: "그렇더라도 Z 증거 때문에 AI_GENERATED이 더 가능성이 높다"
```

---

## 💡 개선 방안 3가지

### Option 1: Sub-agents에 LLM 추가 (권장)

**구조**:
```python
class FrequencyAgent(BaseAgent):
    def __init__(self, llm_model: str = "claude-sonnet-4-20250514", api_key=None):
        super().__init__(...)
        self._tool = FrequencyAnalysisTool()
        self._llm = ClaudeClient(api_key=api_key, model=llm_model)  # 추가!

    def analyze(self, image: np.ndarray, context=None) -> AgentResponse:
        tool_result = self._tool(image)

        # LLM을 사용한 고급 추론
        reasoning = self._generate_llm_reasoning(tool_result)
        arguments = self._generate_llm_arguments(tool_result)

        return AgentResponse(...)

    def _generate_llm_reasoning(self, tool_result: ToolResult) -> str:
        """LLM을 사용한 고급 추론"""
        prompt = f"""
        당신은 주파수 분석 전문가입니다.

        분석 결과:
        - 판정: {tool_result.verdict.value}
        - 신뢰도: {tool_result.confidence}
        - 증거: {json.dumps(tool_result.evidence)}

        이 증거를 바탕으로 상세한 분석 보고서를 작성하세요.
        왜 이 판정이 맞는지 구체적으로 설명하세요.
        """
        return self._llm.generate(prompt)

    def respond_to_challenge(self, challenger_argument: str, my_evidence: Dict) -> str:
        """다른 에이전트의 도전에 대한 반박"""
        prompt = f"""
        당신은 주파수 분석 전문가입니다.

        다른 전문가가 이렇게 말했습니다:
        "{challenger_argument}"

        하지만 당신은 다음 증거를 가지고 있습니다:
        {json.dumps(my_evidence)}

        이 증거를 바탕으로 상대방의 주장에 대해 논리적으로 반박하세요.
        """
        return self._llm.generate(prompt)
```

**장점**:
- ✅ 진정한 토론 구현 가능
- ✅ 각 sub-agent가 자신의 분야에서 "전문가" 역할
- ✅ 더 설득력 있는 분석

**단점**:
- ❌ 모든 agent에 LLM 호출 필요 (비용 증가)
- ❌ API 대기 시간 증가

---

### Option 2: 규칙 기반 추론 강화

**구조**:
```python
class FrequencyAgent(BaseAgent):
    REASONING_RULES = {
        "grid_pattern_high": "GAN 특유의 규칙적 격자 패턴이 강하게 나타남. AI 생성 가능성 높음.",
        "grid_pattern_low": "격자 패턴이 약함. 자연 이미지일 가능성이 높음.",
        "high_frequency_abnormal": "비정상적인 고주파 분포. 조작 가능성 있음.",
    }

    COUNTERARGUMENT_RULES = {
        "grid_vs_authentic": {
            "my_evidence": "grid_pattern",
            "against": "authentic",
            "response": "격자 패턴은 자연 이미지에서는 거의 발견되지 않는 패턴입니다."
        },
        "frequency_vs_noise": {
            "my_evidence": "high_frequency",
            "against": "authentic_noise",
            "response": "고주파 분포의 비정상성은 PRNU 패턴과는 다른 분석입니다."
        }
    }

    def respond_to_challenge(self, challenger_name: str, challenger_verdict: str,
                             my_tool_result: ToolResult) -> str:
        """규칙 기반 반박"""
        key = f"{self.role.value}_vs_{challenger_name}"

        if key in self.COUNTERARGUMENT_RULES:
            rule = self.COUNTERARGUMENT_RULES[key]
            return rule["response"]

        # 기본 반박: 자신의 증거로 설명
        return f"내 분석 결과 {my_tool_result.verdict.value}입니다. " \
               f"이는 {self.REASONING_RULES.get(my_tool_result.evidence.get('type', ''), '')}에 기반합니다."
```

**장점**:
- ✅ LLM 비용 절감
- ✅ 빠른 응답
- ✅ 예측 가능

**단점**:
- ❌ 규칙을 미리 정의해야 함 (모든 시나리오 불가능)
- ❌ 진정한 추론이 아님 (템플릿 기반)
- ❌ 새로운 케이스 대응 어려움

---

### Option 3: 현재 설계 유지 + Manager LLM 강화

**구조** (현재):
```
Sub-agents (Tool만 사용) → Manager Agent (LLM으로 종합)
```

**개선점**:
```python
class ManagerAgent(BaseAgent):
    def _conduct_llm_debate(self, agent_responses: Dict) -> List[Dict]:
        """LLM을 사용한 토론 시뮬레이션"""

        # Sub-agent들의 주장을 LLM에 제시
        prompt = f"""
        이미지 분석에서 다음과 같은 의견이 나왔습니다:

        {self._format_agent_responses(agent_responses)}

        각 전문가 입장에서, 다른 의견에 대해 어떻게 반박할지 생각해보세요.
        각 에이전트가 자신의 분야 전문성을 바탕으로 논쟁할 때:

        - Frequency vs Noise: 어느 증거가 더 강한가?
        - Watermark vs Spatial: 모순되는 부분은?

        각 입장의 최종 주장을 정리해주세요.
        """

        llm_response = self.llm_client.generate(prompt)
        return self._parse_debate_result(llm_response)
```

**장점**:
- ✅ 진정한 토론 효과 (Manager가 시뮬레이션)
- ✅ Sub-agent 수정 불필요
- ✅ 모든 시나리오 대응 가능

**단점**:
- ❌ 이건 "Sub-agent들의 의견"이 아님
- ❌ Manager가 만든 가상의 토론일 뿐

---

## 🎯 권장 방안: Option 1 + Option 3 혼합

### 단계별 구현

**Phase 5-1: Sub-agents에 LLM 추가** (추천)
```python
# src/agents/specialist_agents.py 수정

from ..llm.claude_client import ClaudeClient

class FrequencyAgent(BaseAgent):
    def __init__(self, llm_model: str = "claude-sonnet-4-20250514", api_key=None, use_llm=False):
        super().__init__(...)
        self._tool = FrequencyAnalysisTool()
        self._use_llm = use_llm
        self._llm = ClaudeClient(api_key=api_key, model=llm_model) if use_llm else None

    def analyze(self, image: np.ndarray, context=None) -> AgentResponse:
        tool_result = self._tool(image)

        if self._use_llm and self._llm.is_available:
            reasoning = self._generate_llm_reasoning(tool_result)
            arguments = self._generate_llm_arguments(tool_result)
        else:
            reasoning = self._generate_rule_based_reasoning(tool_result)
            arguments = self._extract_arguments(tool_result)

        return AgentResponse(...)

    def respond_to_challenge(self, challenge: str) -> str:
        """토론에서 도전에 응답"""
        if self._use_llm and self._llm.is_available:
            return self._generate_llm_response(challenge)
        else:
            return self._generate_rule_based_response(challenge)
```

**Phase 5-2: 토론 시스템 개선**
```python
# src/agents/manager_agent.py 수정

def _conduct_debate(self, responses: Dict[str, AgentResponse]) -> List[Dict]:
    """개선된 토론 시스템"""
    debate_history = []

    # 1단계: 불일치 에이전트 찾기
    verdicts = {name: r.verdict for name, r in responses.items()}
    unique_verdicts = set(verdicts.values())

    if len(unique_verdicts) <= 1:
        return debate_history

    # 2단계: 각 에이전트가 실제로 응답 (LLM 또는 규칙 기반)
    for name_a, verdict_a in verdicts.items():
        for name_b, verdict_b in verdicts.items():
            if name_a < name_b and verdict_a != verdict_b:
                # A가 B의 의견에 도전
                challenge = f"당신은 {verdict_b.value}로 판단했지만, 나는 {verdict_a.value}라고 봅니다. 왜냐하면..."

                # B가 응답 (이제 실제 reasoning 가능)
                if hasattr(responses[name_b], 'respond_to_challenge'):
                    rebuttal = responses[name_b].respond_to_challenge(challenge)
                else:
                    rebuttal = responses[name_b].reasoning

                debate_history.append({
                    "challenger": name_a,
                    "challenged": name_b,
                    "challenge": challenge,
                    "rebuttal": rebuttal
                })

    return debate_history
```

---

## 📊 구현 우선순위

```
1순위: Sub-agents에 LLM 추가 (새로운 메서드 추가)
      → 기존 코드 유지, 선택적 LLM 사용

2순위: 토론 시스템 개선
      → respond_to_challenge() 메서드 추가
      → Manager의 토론 로직 개선

3순위: 규칙 기반 추론 강화 (Fallback)
      → LLM 없을 때 더 나은 응답
```

---

## ✅ 최종 아키텍처

```
이미지
  ↓
Sub-agents (Tool + LLM)
  ├─ FrequencyAgent (LLM)
  ├─ NoiseAgent (LLM)
  ├─ WatermarkAgent (LLM)
  └─ SpatialAgent (LLM)
  ↓
각 agent가 "의견" 생성 + "근거" 제시
  ↓
토론 시스템
  └─ 각 agent가 실제로 다른 의견에 "반박" 가능
  ↓
Manager Agent (LLM)
  └─ 토론 결과를 최종 판정으로 통합
```

이렇게 하면:
- ✅ Sub-agents가 진정한 "전문가" 역할
- ✅ 진정한 토론 구현
- ✅ 해석 가능한 AI (Explainable AI)

---

## 📝 다음 작업

이 개선을 구현하려면:
1. 각 sub-agent에 `_llm` 필드 추가
2. `generate_llm_reasoning()` 메서드 추가
3. `respond_to_challenge()` 메서드 추가
4. Manager의 `_conduct_debate()` 개선
5. 새로운 테스트 추가

구현하시겠습니까?
