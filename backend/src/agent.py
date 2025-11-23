import logging
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a warm, supportive health and wellness companion who helps people with daily check-ins.

Your role is to conduct short, meaningful wellness check-ins via voice. You should:

1. Greet the person warmly and introduce yourself as their wellness companion
2. Ask about their current state in a conversational way:
   - How they are feeling today (mood, emotions)
   - What their energy level is like
   - If anything is stressing them out or on their mind
3. Ask about their intentions and objectives for the day:
   - What 1 to 3 things they would like to accomplish today
   - If there is anything they want to do for themselves like rest, exercise, or hobbies
4. Offer simple, realistic and grounded advice:
   - Break large goals into smaller actionable steps
   - Encourage short breaks or movement
   - Suggest simple activities like a 5 minute walk or breathing exercise
   - Keep advice practical and non-medical
5. Close with a brief recap:
   - Summarize their mood and energy
   - Repeat back their main 1 to 3 objectives
   - Ask if this sounds right
6. After confirmation, use the save_check_in tool to save the session
7. Thank them and encourage them for the day ahead

Important guidelines:
- This is a supportive check-in companion, NOT a clinician
- Avoid any medical diagnosis or clinical advice
- Keep responses conversational, warm, and concise
- Be non-judgmental and encouraging
- Use natural language without emojis or asterisks
- If you have access to previous check-ins, reference them briefly to show continuity""",
        )
        
        # Initialize check-in state
        self.check_in_state = {
            "mood": None,
            "energy": None,
            "objectives": [],
            "stressors": None,
            "user_name": None
        }
        
        # Store previous check-ins for context
        self.previous_check_ins = []
        
        # Load previous check-ins on initialization
        self._load_previous_check_ins()
    
    def _load_previous_check_ins(self) -> None:
        """Load previous check-ins from wellness_log.json for context"""
        try:
            log_path = Path(__file__).parent.parent / "wellness_log.json"
            if log_path.exists():
                with open(log_path, "r") as f:
                    data = json.load(f)
                    # Get the last 2 check-ins for context
                    self.previous_check_ins = data.get("check_ins", [])[-2:]
                    if self.previous_check_ins:
                        logger.info(f"Loaded {len(self.previous_check_ins)} previous check-ins")
        except Exception as e:
            logger.warning(f"Could not load previous check-ins: {e}")
            self.previous_check_ins = []
    
    def is_check_in_complete(self) -> bool:
        """Check if all required check-in fields are filled"""
        return (
            self.check_in_state["mood"] is not None and
            self.check_in_state["energy"] is not None and
            len(self.check_in_state["objectives"]) > 0
        )
    
    def get_missing_fields(self) -> list[str]:
        """Get list of missing required fields"""
        missing = []
        if not self.check_in_state["mood"]:
            missing.append("mood")
        if not self.check_in_state["energy"]:
            missing.append("energy level")
        if not self.check_in_state["objectives"]:
            missing.append("daily objectives")
        return missing

    @function_tool
    async def save_check_in(
        self, 
        context: RunContext,
        mood: str,
        energy: str,
        objectives: str,
        stressors: str = "None",
        user_name: str = "Anonymous"
    ):
        """Save the completed wellness check-in to the wellness log.
        
        Use this tool ONLY when you have collected all required information
        and the person has confirmed the recap is correct.
        
        Args:
            mood: Self-reported mood or emotional state (text description)
            energy: Current energy level (text description or scale)
            objectives: Comma-separated list of 1-3 daily objectives or goals
            stressors: Optional stressors or concerns (default: "None")
            user_name: Optional user name (default: "Anonymous")
        """
        
        logger.info(f"Saving wellness check-in for {user_name}")
        
        # Parse objectives
        objectives_list = []
        if objectives and objectives.lower() != "none":
            objectives_list = [obj.strip() for obj in objectives.split(",") if obj.strip()]
        
        # Generate unique ID
        check_in_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create simple summary
        summary = f"Feeling {mood} with {energy} energy. Goals: {', '.join(objectives_list[:2])}"
        if len(objectives_list) > 2:
            summary += f" and {len(objectives_list) - 2} more"
        
        # Create check-in object
        check_in = {
            "id": check_in_id,
            "timestamp": datetime.now().isoformat(),
            "user_name": user_name,
            "mood": mood,
            "energy": energy,
            "objectives": objectives_list,
            "stressors": stressors if stressors.lower() != "none" else None,
            "summary": summary
        }
        
        # Update internal state
        self.check_in_state = {
            "mood": mood,
            "energy": energy,
            "objectives": objectives_list,
            "stressors": stressors if stressors.lower() != "none" else None,
            "user_name": user_name
        }
        
        # Load or create wellness log
        log_path = Path(__file__).parent.parent / "wellness_log.json"
        
        try:
            # Load existing log or create new one
            if log_path.exists():
                with open(log_path, "r") as f:
                    wellness_data = json.load(f)
            else:
                wellness_data = {"check_ins": []}
            
            # Append new check-in
            wellness_data["check_ins"].append(check_in)
            
            # Save updated log
            with open(log_path, "w") as f:
                json.dump(wellness_data, f, indent=2)
            
            logger.info(f"Check-in saved to {log_path}")
            
            # Update the context with new check-in
            self.previous_check_ins.append(check_in)
            
            return f"Your check-in has been saved! I hope you have a wonderful day ahead, {user_name}. Remember to take care of yourself!"
        
        except Exception as e:
            logger.error(f"Failed to save check-in: {e}")
            return f"I apologize, there was an issue saving your check-in. But I'm here for you regardless!"




def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
