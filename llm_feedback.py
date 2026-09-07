"""
LLM integration for AI coaching feedback via DeepSeek API.
Clean formatting - no markdown, no bold, no headers.
"""
import os

_api_key = os.getenv("DEEPSEEK_API_KEY", "")
_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

print(f"[LLM] API Key loaded: {'Yes (' + str(len(_api_key)) + ' chars)' if _api_key else 'NO KEY FOUND'}")
print(f"[LLM] Model: {_model}")

def has_ai_analyzer():
    has_key = bool(_api_key and len(_api_key) > 10 and not _api_key.startswith("sk-YOUR"))
    print(f"[LLM] has_ai_analyzer: {has_key}")
    return has_key

def get_ai_model_name():
    if has_ai_analyzer():
        return f"DeepSeek ({_model})"
    return "Rule-based Fallback (no API key)"

def generate_structured_feedback(prompt, max_new_tokens=500):
    if not has_ai_analyzer():
        print("[LLM] Skipping API call - no key available")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=_api_key, base_url="https://api.deepseek.com/v1")

        print(f"[LLM] Calling DeepSeek... prompt length: {len(prompt)} chars")

        response = client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": "You are an elite badminton coach with 20 years experience. Provide detailed, specific, actionable feedback. Use plain text only. NO markdown, NO asterisks, NO bold, NO headers like # or ##. Use clean paragraphs with numbered lists if needed. Be encouraging but honest."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.7,
        )

        result = response.choices[0].message.content
        # Strip any remaining markdown
        result = result.replace("**", "").replace("#", "").replace("*", "")
        print(f"[LLM] Response received: {len(result)} chars")
        return result

    except Exception as e:
        print(f"[LLM] DeepSeek API ERROR: {e}")
        return None

def generate_chat_reply(player, question, context):
    if not has_ai_analyzer():
        print("[LLM] Chat skipped - no API key")
        return None

    try:
        from openai import OpenAI

        profile = context.get("profile", {})
        feedback = context.get("feedback", {})
        pose_metrics = context.get("pose_metrics", {})
        box_metrics = context.get("box_metrics", {})

        system_msg = (
            "You are CoachAI, an elite badminton coach. Answer questions based on biomechanical data. "
            "Use plain text only. NO markdown, NO asterisks, NO bold, NO headers. "
            "Be specific, reference actual metrics, and give actionable advice. "
            "Write in clean paragraphs. If listing items, use simple numbered lists like 1. 2. 3."
        )

        user_msg = f"""Player: {player}
Question: {question}

BIOMECHANICAL ANALYSIS DATA:
- Posture: {profile.get('posture', 'unknown')}
- Balance: {profile.get('balance', 'unknown')}  
- Stance: {profile.get('stance', 'unknown')}
- Height Ratio: {profile.get('height_ratio', 0):.3f}
- Upright Ratio: {pose_metrics.get('upright_ratio', 0):.1%}
- Centered Ratio: {pose_metrics.get('centred_ratio', 0):.1%}
- Athletic Ready: {pose_metrics.get('athletic_ready_ratio', 0):.1%}
- Movement Consistency: {box_metrics.get('movement_consistency', 0):.2f}
- Total Movement: {box_metrics.get('total_movement_px', 0):.0f}px
- Avg Step: {box_metrics.get('avg_step_px', 0):.1f}px

Identified Strengths:
{chr(10).join(str(i+1) + '. ' + s for i, s in enumerate(feedback.get('strengths', [])))}

Identified Weaknesses:
{chr(10).join(str(i+1) + '. ' + w for i, w in enumerate(feedback.get('weaknesses', [])))}

Recommended Drills:
{chr(10).join(str(i+1) + '. ' + d for i, d in enumerate(feedback.get('drills', [])))}

Instructions:
1. Answer the specific question asked
2. Reference actual metrics from the analysis
3. Give 2-3 specific, actionable recommendations
4. Be encouraging but honest about weaknesses
5. If asked about a specific shot (smash, clear, drop, net), explain the biomechanical factors that affect it"""

        print(f"[LLM] Chat call for player={player}, question={question[:50]}...")

        client = OpenAI(api_key=_api_key, base_url="https://api.deepseek.com/v1")

        response = client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=600,
            temperature=0.7,
        )

        result = response.choices[0].message.content
        result = result.replace("**", "").replace("#", "").replace("*", "")
        print(f"[LLM] Chat response: {len(result)} chars")
        return result

    except Exception as e:
        print(f"[LLM] Chat API ERROR: {e}")
        return None

def is_valid_coaching_feedback(text):
    return text and len(text) > 20
