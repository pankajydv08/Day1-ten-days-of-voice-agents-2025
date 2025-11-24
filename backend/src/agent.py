import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

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
    RunContext,
    llm
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# Load tutor content
def load_tutor_content():
    """Load the tutor content from JSON file"""
    content_path = Path(__file__).parent.parent / "shared-data" / "day4_tutor_content.json"
    try:
        with open(content_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load tutor content: {e}")
        return []


TUTOR_CONTENT = load_tutor_content()


class GreeterAgent(Agent):
    """Initial agent that greets users and routes to learning modes"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly educational assistant who helps learners choose their learning mode.

Your role is to:
1. Warmly greet the user and introduce yourself as their Active Recall Coach
2. Explain the three available learning modes:
   - LEARN mode: I'll explain programming concepts to you
   - QUIZ mode: I'll ask you questions to test your knowledge
   - TEACH BACK mode: You'll explain concepts back to me and I'll give you feedback

3. Ask which mode they'd like to start with
4. Once they choose, use the switch_to_mode tool to connect them to the appropriate learning agent

Keep your responses warm, encouraging, and concise since this is a voice interaction.
Your responses should be natural and without complex formatting or punctuation including emojis or asterisks.""",
        )

    @function_tool
    async def switch_to_mode(
        self,
        context: RunContext,
        mode: Annotated[str, llm.TypeInfo(description="The learning mode to switch to: 'learn', 'quiz', or 'teach_back'")]
    ):
        """Switch to the requested learning mode.
        
        Args:
            mode: The learning mode - must be 'learn', 'quiz', or 'teach_back'
        """
        mode = mode.lower().strip()
        
        if mode not in ["learn", "quiz", "teach_back"]:
            return "I didn't understand that mode. Please choose: learn, quiz, or teach back."
        
        logger.info(f"Switching to {mode} mode")
        
        # Transfer to the appropriate agent
        if mode == "learn":
            context.transfer_to_agent("learn_agent")
            return "Great! Switching you to Learn mode now."
        elif mode == "quiz":
            context.transfer_to_agent("quiz_agent")
            return "Perfect! Switching you to Quiz mode now."
        elif mode == "teach_back":
            context.transfer_to_agent("teach_back_agent")
            return "Excellent! Switching you to Teach Back mode now."


class LearnAgent(Agent):
    """Agent for LEARN mode - explains concepts to users"""
    
    def __init__(self) -> None:
        available_concepts = ", ".join([c["title"] for c in TUTOR_CONTENT])
        
        super().__init__(
            instructions=f"""You are a patient and enthusiastic teacher in LEARN mode, using the voice of Matthew.

Your role is to explain programming concepts clearly and engagingly.

Available concepts you can teach: {available_concepts}

When a user asks about a concept:
1. Check if it matches one of the available topics
2. Use the get_concept tool to retrieve the explanation
3. Explain it in a clear, friendly way
4. Ask if they'd like to hear about another concept or switch to a different mode
5. If they want to switch modes, use the switch_mode tool

Keep explanations clear and engaging. Use analogies and examples.
Your responses should be natural and without complex formatting or punctuation including emojis or asterisks.""",
        )

    @function_tool
    async def get_concept(
        self,
        context: RunContext,
        concept_id: Annotated[str, llm.TypeInfo(description="The concept ID to explain (e.g., 'variables', 'loops', 'functions')")]
    ):
        """Get the explanation for a specific concept.
        
        Args:
            concept_id: The ID of the concept to explain
        """
        concept_id = concept_id.lower().strip()
        
        for concept in TUTOR_CONTENT:
            if concept["id"] == concept_id or concept["title"].lower() == concept_id:
                logger.info(f"Explaining concept: {concept['title']}")
                return f"Here's what you need to know about {concept['title']}: {concept['summary']}"
        
        available = ", ".join([c["title"] for c in TUTOR_CONTENT])
        return f"I don't have information about that concept. Available topics are: {available}"

    @function_tool
    async def switch_mode(
        self,
        context: RunContext,
        mode: Annotated[str, llm.TypeInfo(description="The mode to switch to: 'quiz' or 'teach_back' or 'greeting'")]
    ):
        """Switch to a different learning mode.
        
        Args:
            mode: The target mode
        """
        mode = mode.lower().strip()
        
        if mode == "quiz":
            context.transfer_to_agent("quiz_agent")
            return "Switching to Quiz mode!"
        elif mode == "teach_back" or mode == "teach":
            context.transfer_to_agent("teach_back_agent")
            return "Switching to Teach Back mode!"
        elif mode == "greeting" or mode == "main" or mode == "menu":
            context.transfer_to_agent("greeter_agent")
            return "Going back to the main menu!"
        else:
            return "I can switch you to quiz mode, teach back mode, or back to the main menu. Which would you like?"


class QuizAgent(Agent):
    """Agent for QUIZ mode - asks questions to test knowledge"""
    
    def __init__(self) -> None:
        available_concepts = ", ".join([c["title"] for c in TUTOR_CONTENT])
        
        super().__init__(
            instructions=f"""You are an encouraging quiz master in QUIZ mode, using the voice of Alicia.

Your role is to test the user's knowledge through questions.

Available topics you can quiz on: {available_concepts}

When quizzing:
1. Ask which topic they'd like to be quizzed on
2. Use the get_quiz_question tool to get a question
3. Ask the question and listen to their answer
4. Provide encouraging feedback on their answer, whether correct or not
5. Offer to quiz them on another topic or switch modes using the switch_mode tool

Be supportive and encouraging. Make learning fun!
Your responses should be natural and without complex formatting or punctuation including emojis or asterisks.""",
        )

    @function_tool
    async def get_quiz_question(
        self,
        context: RunContext,
        concept_id: Annotated[str, llm.TypeInfo(description="The concept to quiz on (e.g., 'variables', 'loops')")]
    ):
        """Get a quiz question for the specified concept.
        
        Args:
            concept_id: The ID of the concept to quiz on
        """
        concept_id = concept_id.lower().strip()
        
        for concept in TUTOR_CONTENT:
            if concept["id"] == concept_id or concept["title"].lower() == concept_id:
                logger.info(f"Quizzing on concept: {concept['title']}")
                return f"Here's your question about {concept['title']}: {concept['sample_question']}"
        
        available = ", ".join([c["title"] for c in TUTOR_CONTENT])
        return f"I don't have a quiz for that topic. Available topics are: {available}"

    @function_tool
    async def switch_mode(
        self,
        context: RunContext,
        mode: Annotated[str, llm.TypeInfo(description="The mode to switch to: 'learn' or 'teach_back' or 'greeting'")]
    ):
        """Switch to a different learning mode.
        
        Args:
            mode: The target mode
        """
        mode = mode.lower().strip()
        
        if mode == "learn":
            context.transfer_to_agent("learn_agent")
            return "Switching to Learn mode!"
        elif mode == "teach_back" or mode == "teach":
            context.transfer_to_agent("teach_back_agent")
            return "Switching to Teach Back mode!"
        elif mode == "greeting" or mode == "main" or mode == "menu":
            context.transfer_to_agent("greeter_agent")
            return "Going back to the main menu!"
        else:
            return "I can switch you to learn mode, teach back mode, or back to the main menu. Which would you like?"


class TeachBackAgent(Agent):
    """Agent for TEACH_BACK mode - user explains concepts"""
    
    def __init__(self) -> None:
        available_concepts = ", ".join([c["title"] for c in TUTOR_CONTENT])
        
        super().__init__(
            instructions=f"""You are a supportive learning coach in TEACH BACK mode, using the voice of Ken.

Your role is to let the user teach YOU concepts and provide constructive feedback.

Available concepts they can teach: {available_concepts}

When in teach-back mode:
1. Ask which concept they'd like to explain to you
2. Use the get_concept_reference tool to get the official explanation
3. Ask them to explain the concept in their own words
4. Listen carefully to their explanation
5. Provide encouraging, constructive feedback comparing to the reference
6. Highlight what they got right and gently note any gaps
7. Offer to let them teach another concept or switch modes using switch_mode

Remember: The goal is active recall. Be supportive and focus on learning, not perfection!
Your responses should be natural and without complex formatting or punctuation including emojis or asterisks.""",
        )

    @function_tool
    async def get_concept_reference(
        self,
        context: RunContext,
        concept_id: Annotated[str, llm.TypeInfo(description="The concept they will teach (e.g., 'variables', 'loops')")]
    ):
        """Get the reference explanation for comparison when user teaches.
        
        Args:
            concept_id: The ID of the concept
        """
        concept_id = concept_id.lower().strip()
        
        for concept in TUTOR_CONTENT:
            if concept["id"] == concept_id or concept["title"].lower() == concept_id:
                logger.info(f"Preparing teach-back for: {concept['title']}")
                # Store the summary in metadata for comparison
                return f"Great! Please explain {concept['title']} to me in your own words. Take your time. Reference for your feedback: {concept['summary']}"
        
        available = ", ".join([c["title"] for c in TUTOR_CONTENT])
        return f"I don't have that concept in my system. Available topics are: {available}"

    @function_tool
    async def switch_mode(
        self,
        context: RunContext,
        mode: Annotated[str, llm.TypeInfo(description="The mode to switch to: 'learn' or 'quiz' or 'greeting'")]
    ):
        """Switch to a different learning mode.
        
        Args:
            mode: The target mode
        """
        mode = mode.lower().strip()
        
        if mode == "learn":
            context.transfer_to_agent("learn_agent")
            return "Switching to Learn mode!"
        elif mode == "quiz":
            context.transfer_to_agent("quiz_agent")
            return "Switching to Quiz mode!"
        elif mode == "greeting" or mode == "main" or mode == "menu":
            context.transfer_to_agent("greeter_agent")
            return "Going back to the main menu!"
        else:
            return "I can switch you to learn mode, quiz mode, or back to the main menu. Which would you like?"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Create the greeting agent session (main entry point)
    greeter_session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-exp"),
        tts=murf.TTS(
            voice="en-US-matthew",  # Matthew for greeter
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Create the learn agent session
    learn_session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-exp"),
        tts=murf.TTS(
            voice="en-US-matthew",  # Matthew for learn
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Create the quiz agent session
    quiz_session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-exp"),
        tts=murf.TTS(
            voice="en-US-alicia",  # Alicia for quiz
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Create the teach-back agent session
    teach_back_session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-exp"),
        tts=murf.TTS(
            voice="en-US-ken",  # Ken for teach-back
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @greeter_session.on("metrics_collected")
    def _on_greeter_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @learn_session.on("metrics_collected")
    def _on_learn_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @quiz_session.on("metrics_collected")
    def _on_quiz_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @teach_back_session.on("metrics_collected")
    def _on_teach_back_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Register all agents
    greeter_session.register_agent("greeter_agent", GreeterAgent())
    greeter_session.register_agent("learn_agent", LearnAgent())
    greeter_session.register_agent("quiz_agent", QuizAgent())
    greeter_session.register_agent("teach_back_agent", TeachBackAgent())

    learn_session.register_agent("greeter_agent", GreeterAgent())
    learn_session.register_agent("learn_agent", LearnAgent())
    learn_session.register_agent("quiz_agent", QuizAgent())
    learn_session.register_agent("teach_back_agent", TeachBackAgent())

    quiz_session.register_agent("greeter_agent", GreeterAgent())
    quiz_session.register_agent("learn_agent", LearnAgent())
    quiz_session.register_agent("quiz_agent", QuizAgent())
    quiz_session.register_agent("teach_back_agent", TeachBackAgent())

    teach_back_session.register_agent("greeter_agent", GreeterAgent())
    teach_back_session.register_agent("learn_agent", LearnAgent())
    teach_back_session.register_agent("quiz_agent", QuizAgent())
    teach_back_session.register_agent("teach_back_agent", TeachBackAgent())

    # Start the greeter session (entry point)
    await greeter_session.start(
        agent=GreeterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
