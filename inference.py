import os
import json
from openai import OpenAI
from environment import CustomerSupportEnv
from models import Action

# 1. Load Environment Variables (Mandatory per Hackathon Rules)
# The judges' server will inject these variables automatically when they run it.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1") 
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") 
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

def parse_action_from_response(response_text: str) -> Action:
    """Cleans the LLM response and parses it into our strict Pydantic Action model."""
    try:
        # Strip markdown formatting if the LLM added it (e.g., ```json ... ```)
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        action_dict = json.loads(clean_text)
        return Action(**action_dict)
    except Exception as e:
        print(f"Failed to parse AI response: {response_text}. Error: {e}")
        # If the AI hallucinates badly, force an escalation to prevent the script from crashing
        return Action(action_type="escalate_to_human")

def main():
    if not HF_TOKEN:
        print("ERROR: Please set your OPENAI_API_KEY or HF_TOKEN environment variable.")
        return

    # 2. Initialize the AI Client using the OpenAI SDK
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # 3. Initialize your OpenEnv game engine
    env = CustomerSupportEnv()
    
    print("Starting Baseline Inference Evaluation...\n")
    
    # Run 3 episodes to prove it works on multiple tasks
    for task_num in range(3): 
        print(f"--- Episode {task_num + 1} ---")
        obs = env.reset()
        print(f"Initial Ticket: {obs.conversation_history[0]}")
        
        for step in range(1, 11): # Maximum of 10 steps per ticket
            # 4. Prompt Engineering: Give the AI the rules
            # 4. Prompt Engineering: Give the AI the rules
            # 4. Prompt Engineering: Give the AI an unbreakable SOP
            system_prompt = (
                "You are an expert AI customer support agent. Look at the current Observation and choose the ONE best next action. "
                "You are evaluated on efficiency. Solve the ticket in the fewest steps possible.\n\n"
                "STRICT STANDARD OPERATING PROCEDURE (SOP):\n"
                "1. CHECK CATEGORY: If 'issue_category' is null or missing, you MUST use 'classify_issue' first.\n"
                "2. CHECK KNOWLEDGE: If the issue is classified but you haven't searched the KB yet, you MUST use 'search_kb'.\n"
                "3. RESOLVE: If you have a relevant 'kb_search_result', you MUST use 'resolve_ticket' and provide a polite 'message_to_customer' containing the solution.\n"
                "4. ESCALATE: If the customer is hostile, asks for something illegal/against policy (like a refund for an already-read eBook), or the KB says to escalate, you MUST use 'escalate_to_human'.\n"
                "5. RULE OF THUMB: NEVER repeat the exact same action twice in a row. If you are stuck, escalate.\n\n"
                "You MUST respond ONLY with a valid JSON object matching this schema:\n"
                "{\n"
                '  "action_type": "ask_clarifying_question" | "classify_issue" | "search_kb" | "resolve_ticket" | "escalate_to_human",\n'
                '  "message_to_customer": "...", \n'
                '  "category_guess": "Billing" | "Technical" | "Refund_Request", \n'
                '  "search_query": "..."\n'
                "}\n"
            )
            
            # 5. Call the LLM with the current Observation
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Current Observation:\n{obs.model_dump_json(indent=2)}"}
                ],
                temperature=0.1 # Keep it low so the AI is deterministic and logical
            )
            
            ai_text = response.choices[0].message.content
            action = parse_action_from_response(ai_text)
            
            print(f"Step {step} - AI Decided to: {action.action_type}")
            
            # 6. Pass the AI's action into our game engine
            obs, reward, done, info = env.step(action)
            
            print(f"  -> Reward Given: {reward.value:.2f} | Reason: {reward.reason}")
            
            if done:
                print(f"Episode Finished! \n")
                break

if __name__ == "__main__":
    main()