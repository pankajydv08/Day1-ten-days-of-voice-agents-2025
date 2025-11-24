# Day 4: Teach-the-Tutor Active Recall Coach

## Overview

This is an **Active Recall Coach** voice agent that implements the "learning by teaching" principle. The agent features three distinct learning modes with different AI personalities and voices.

## Features

### Three Learning Modes

1. **LEARN Mode** (Voice: Matthew)
   - Agent explains programming concepts to you
   - Topics: Variables, Loops, Functions, Conditionals, Arrays
   - Clear, engaging explanations with analogies

2. **QUIZ Mode** (Voice: Alicia)
   - Agent asks you questions to test your knowledge
   - Provides encouraging feedback
   - Helps reinforce learning through active recall

3. **TEACH BACK Mode** (Voice: Ken)
   - You explain concepts back to the agent
   - Agent provides constructive feedback
   - Compares your explanation with reference material
   - Focuses on active recall and deeper understanding

### Seamless Mode Switching

- Switch between any mode at any time by simply asking
- Context is preserved between modes
- Natural voice-based interactions

## Architecture

### Agent Handoff System

The implementation uses LiveKit's agent handoff capabilities to create multiple specialized agents:

- **GreeterAgent**: Initial entry point, helps users select their learning mode
- **LearnAgent**: Teaches concepts using explanations
- **QuizAgent**: Tests knowledge through questions
- **TeachBackAgent**: Facilitates teaching back and provides feedback

Each agent can transfer control to any other agent, enabling fluid mode switching.

### Content System

All learning content is stored in `shared-data/day4_tutor_content.json`:

```json
[
  {
    "id": "variables",
    "title": "Variables",
    "summary": "...",
    "sample_question": "..."
  }
]
```

This makes it easy to extend with new topics and concepts.

## How to Use

1. **Start the application** (see main README.md for setup)

2. **Connect in browser** at `http://localhost:3000`

3. **Interact with the agent**:
   - The greeter will ask which mode you want
   - Say "learn", "quiz", or "teach back"
   - The agent will switch to the appropriate mode

4. **Switch modes anytime**:
   - "Switch to quiz mode"
   - "I want to learn now"
   - "Let me teach back"

## Example Interaction

```
Agent (Matthew): "Hi! I'm your Active Recall Coach. Which mode would you like to start with: learn, quiz, or teach back?"

You: "Let's start with learn mode"

Agent (Matthew): "Great! Which concept would you like to learn about? I can teach you about variables, loops, functions, conditionals, or arrays."

You: "Tell me about loops"

Agent (Matthew): "Here's what you need to know about Loops: Loops are programming structures..."

You: "Now let's switch to quiz mode"

Agent (Alicia): "Perfect! Switching to Quiz mode. Which topic would you like to be quizzed on?"
```

## Technical Details

### Voice Configuration

- **Learn Mode**: `en-US-matthew` - Warm, teaching voice
- **Quiz Mode**: `en-US-alicia` - Encouraging, upbeat voice
- **Teach Back Mode**: `en-US-ken` - Supportive, coaching voice

### Agent Tools

Each agent has specific tools:

- **GreeterAgent**: `switch_to_mode(mode)`
- **LearnAgent**: `get_concept(concept_id)`, `switch_mode(mode)`
- **QuizAgent**: `get_quiz_question(concept_id)`, `switch_mode(mode)`
- **TeachBackAgent**: `get_concept_reference(concept_id)`, `switch_mode(mode)`

### Context Preservation

The agent handoff system preserves conversation context, so you can seamlessly switch between modes without losing your place in the learning journey.

## Extending the System

### Adding New Concepts

Edit `shared-data/day4_tutor_content.json` and add new concept objects:

```json
{
  "id": "your_concept_id",
  "title": "Concept Title",
  "summary": "Detailed explanation...",
  "sample_question": "Question to test understanding?"
}
```

### Advanced Features (Optional)

The task specification includes advanced challenges:

1. **Mastery Tracking**: Track scores and progress per concept
2. **Teach-back Evaluator**: Automated scoring of user explanations
3. **Learning Paths**: Progressive difficulty levels
4. **Database Integration**: Persist learning history

## Resources

- [LiveKit Agent Handoffs](https://docs.livekit.io/agents/build/agents-handoffs/#tool-handoff)
- [Context Preservation](https://docs.livekit.io/agents/build/agents-handoffs/#context-preservation)
- [Medical Office Triage Example](https://github.com/livekit-examples/python-agents-examples/blob/main/complex-agents/medical_office_triage/triage.py)

## Completion Checklist

- [x] Create three learning modes (learn, quiz, teach_back)
- [x] Implement agent handoff logic
- [x] Configure different voices (Matthew, Alicia, Ken)
- [x] Create content file with concepts
- [x] Enable seamless mode switching
- [ ] Test all modes in browser
- [ ] Record demonstration video
- [ ] Post on LinkedIn with #MurfAIVoiceAgentsChallenge

---

**Day 4 Challenge**: Successfully demonstrate all three learning modes working together with voice-based mode switching!
