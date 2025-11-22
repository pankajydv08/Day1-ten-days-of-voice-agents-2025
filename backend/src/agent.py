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
            instructions="""You are a friendly and enthusiastic barista at Piku Coffee, a premium coffee shop known for exceptional service and delicious beverages.

Your role is to take coffee orders from customers via voice. You should:

1. Greet customers warmly and introduce yourself as a Piku Coffee barista
2. Ask about their drink preferences systematically
3. Collect all required order information:
   - Drink type (e.g., Latte, Cappuccino, Americano, Espresso, Mocha, Cold Brew, etc.)
   - Size (Small, Medium, or Large)
   - Milk preference (Whole, Skim, Oat, Almond, Soy, or None for black coffee)
   - Any extras (e.g., Extra Shot, Whipped Cream, Vanilla Syrup, Caramel Drizzle, Chocolate Chips, etc.)
   - Customer's name for the order

4. If any information is missing, politely ask clarifying questions
5. Once you have all the information, confirm the complete order with the customer
6. After confirmation, use the save_order tool to finalize the order
7. Thank the customer and let them know their order will be ready soon

Be conversational, friendly, and patient. Keep responses concise since this is a voice interaction.
Your responses should be natural and without complex formatting or punctuation including emojis or asterisks.""",
        )
        
        # Initialize order state
        self.order_state = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None
        }
    
    def is_order_complete(self) -> bool:
        """Check if all required order fields are filled"""
        return (
            self.order_state["drinkType"] is not None and
            self.order_state["size"] is not None and
            self.order_state["milk"] is not None and
            self.order_state["name"] is not None
        )
    
    def get_missing_fields(self) -> list[str]:
        """Get list of missing required fields"""
        missing = []
        if not self.order_state["drinkType"]:
            missing.append("drink type")
        if not self.order_state["size"]:
            missing.append("size")
        if not self.order_state["milk"]:
            missing.append("milk preference")
        if not self.order_state["name"]:
            missing.append("customer name")
        return missing

    @function_tool
    async def save_order(
        self, 
        context: RunContext,
        drink_type: str,
        size: str,
        milk: str,
        extras: str,
        customer_name: str
    ):
        """Save the completed coffee order to a JSON file.
        
        Use this tool ONLY when you have collected ALL required information from the customer
        and they have confirmed their order is correct.
        
        Args:
            drink_type: Type of coffee drink (e.g., Latte, Cappuccino, Americano)
            size: Size of the drink (Small, Medium, or Large)
            milk: Milk preference (Whole, Skim, Oat, Almond, Soy, or None)
            extras: Comma-separated list of extras or "None" if no extras
            customer_name: Customer's name for the order
        """
        
        logger.info(f"Saving order for {customer_name}")
        
        # Parse extras
        extras_list = []
        if extras and extras.lower() != "none":
            extras_list = [e.strip() for e in extras.split(",") if e.strip()]
        
        # Create order object
        order = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras_list,
            "name": customer_name,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Update internal state
        self.order_state = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras_list,
            "name": customer_name
        }
        
        # Create orders directory if it doesn't exist
        orders_dir = Path(__file__).parent.parent / "orders"
        orders_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"order_{timestamp}.json"
        filepath = orders_dir / filename
        
        # Save to JSON file
        try:
            with open(filepath, "w") as f:
                json.dump(order, f, indent=2)
            
            logger.info(f"Order saved to {filepath}")
            
            return f"Perfect! Your order has been saved. Order ID: {timestamp}. Your {size} {drink_type} with {milk} milk will be ready shortly, {customer_name}!"
        
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return f"I apologize, there was an issue saving your order. Please let a staff member know."



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
