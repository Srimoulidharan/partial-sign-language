"""
Gesture labels and mappings for the sign language recognition system
"""

# Basic static gestures that can be recognized
GESTURE_LABELS = [
    "hello",
    "goodbye", 
    "thank_you",
    "please",
    "yes",
    "no",
    "help",
    "stop",
    "ok",
    "good",
    "bad",
    "water",
    "food",
    "eat",
    "drink",
    "more",
    "finished",
    "sorry",
    "love",
    "peace"
]

# Extended gesture vocabulary for dynamic sequences
DYNAMIC_GESTURE_LABELS = [
    "how_are_you",
    "nice_to_meet_you", 
    "what_is_your_name",
    "where_is_bathroom",
    "i_need_help",
    "thank_you_very_much",
    "have_a_good_day",
    "see_you_later",
    "i_am_hungry",
    "i_am_thirsty",
    "excuse_me",
    "i_am_sorry",
    "i_love_you",
    "good_morning",
    "good_night"
]

# Gesture to text mappings
GESTURE_TO_TEXT = {
    # Basic static gestures
    "hello": "Hello",
    "goodbye": "Goodbye", 
    "thank_you": "Thank you",
    "please": "Please",
    "yes": "Yes",
    "no": "No",
    "help": "Help",
    "stop": "Stop",
    "ok": "OK",
    "good": "Good",
    "bad": "Bad",
    "water": "Water",
    "food": "Food",
    "eat": "Eat",
    "drink": "Drink",
    "more": "More",
    "finished": "Finished",
    "sorry": "Sorry",
    "love": "Love",
    "peace": "Peace",
    
    # Dynamic gestures
    "how_are_you": "How are you?",
    "nice_to_meet_you": "Nice to meet you",
    "what_is_your_name": "What is your name?",
    "where_is_bathroom": "Where is the bathroom?",
    "i_need_help": "I need help",
    "thank_you_very_much": "Thank you very much",
    "have_a_good_day": "Have a good day",
    "see_you_later": "See you later",
    "i_am_hungry": "I am hungry",
    "i_am_thirsty": "I am thirsty",
    "excuse_me": "Excuse me",
    "i_am_sorry": "I am sorry",
    "i_love_you": "I love you",
    "good_morning": "Good morning",
    "good_night": "Good night"
}

# Text to gesture mappings (for speech-to-sign mode)
TEXT_TO_GESTURE = {
    # Reverse mapping for speech-to-sign translation
    "hello": "hello",
    "hi": "hello",
    "hey": "hello",
    "goodbye": "goodbye",
    "bye": "goodbye",
    "see you": "goodbye",
    "thanks": "thank_you",
    "thank you": "thank_you",
    "please": "please",
    "yes": "yes",
    "yeah": "yes",
    "yep": "yes",
    "no": "no",
    "nope": "no",
    "help": "help",
    "stop": "stop",
    "ok": "ok",
    "okay": "ok",
    "good": "good",
    "bad": "bad",
    "water": "water",
    "food": "food",
    "eat": "eat",
    "drink": "drink",
    "more": "more",
    "done": "finished",
    "finished": "finished",
    "sorry": "sorry",
    "love": "love",
    "peace": "peace",
    
    # Common phrases
    "how are you": "how_are_you",
    "nice to meet you": "nice_to_meet_you",
    "what is your name": "what_is_your_name",
    "where is the bathroom": "where_is_bathroom",
    "i need help": "i_need_help",
    "thank you very much": "thank_you_very_much",
    "have a good day": "have_a_good_day",
    "see you later": "see_you_later",
    "i am hungry": "i_am_hungry",
    "i am thirsty": "i_am_thirsty",
    "excuse me": "excuse_me",
    "i am sorry": "i_am_sorry",
    "i love you": "i_love_you",
    "good morning": "good_morning",
    "good night": "good_night"
}

# Gesture categories for organization
GESTURE_CATEGORIES = {
    "greetings": ["hello", "goodbye", "good_morning", "good_night"],
    "politeness": ["please", "thank_you", "sorry", "excuse_me"],
    "responses": ["yes", "no", "ok", "good", "bad"],
    "needs": ["help", "water", "food", "eat", "drink", "more"],
    "emotions": ["love", "peace", "good", "bad"],
    "actions": ["stop", "finished", "eat", "drink"],
    "questions": ["how_are_you", "what_is_your_name", "where_is_bathroom"],
    "phrases": ["nice_to_meet_you", "i_need_help", "have_a_good_day", "see_you_later"]
}

# Priority gestures (most commonly used)
PRIORITY_GESTURES = [
    "hello", "goodbye", "thank_you", "please", "yes", "no", 
    "help", "ok", "water", "food", "sorry"
]

# Gesture difficulty levels (for progressive learning)
GESTURE_DIFFICULTY = {
    "beginner": ["hello", "goodbye", "yes", "no", "ok", "stop"],
    "intermediate": ["thank_you", "please", "help", "good", "bad", "sorry"],
    "advanced": ["water", "food", "eat", "drink", "more", "finished", "love", "peace"],
    "expert": [
        "how_are_you", "nice_to_meet_you", "what_is_your_name", 
        "where_is_bathroom", "i_need_help", "thank_you_very_much"
    ]
}

# Hand shapes associated with gestures (for reference)
HAND_SHAPES = {
    "open_hand": ["hello", "stop", "help"],
    "closed_fist": ["good", "finished"],
    "pointing": ["you", "there"],
    "thumbs_up": ["good", "yes", "ok"],
    "thumbs_down": ["bad", "no"],
    "peace_sign": ["peace", "two"],
    "ok_sign": ["ok", "good"]
}

# Function to get gesture information
def get_gesture_info(gesture_name: str) -> dict:
    """
    Get comprehensive information about a gesture
    
    Args:
        gesture_name: Name of the gesture
        
    Returns:
        Dictionary with gesture information
    """
    info = {
        "name": gesture_name,
        "text": GESTURE_TO_TEXT.get(gesture_name, gesture_name.replace("_", " ")),
        "category": None,
        "difficulty": None,
        "is_static": gesture_name in GESTURE_LABELS,
        "is_dynamic": gesture_name in DYNAMIC_GESTURE_LABELS
    }
    
    # Find category
    for category, gestures in GESTURE_CATEGORIES.items():
        if gesture_name in gestures:
            info["category"] = category
            break
    
    # Find difficulty
    for difficulty, gestures in GESTURE_DIFFICULTY.items():
        if gesture_name in gestures:
            info["difficulty"] = difficulty
            break
    
    return info

def get_gestures_by_category(category: str) -> list:
    """Get gestures by category"""
    return GESTURE_CATEGORIES.get(category, [])

def get_gestures_by_difficulty(difficulty: str) -> list:
    """Get gestures by difficulty level"""
    return GESTURE_DIFFICULTY.get(difficulty, [])

def search_gestures(query: str) -> list:
    """
    Search for gestures by text or name
    
    Args:
        query: Search query
        
    Returns:
        List of matching gesture names
    """
    query_lower = query.lower()
    matches = []
    
    # Search in gesture names
    for gesture in GESTURE_LABELS + DYNAMIC_GESTURE_LABELS:
        if query_lower in gesture.lower():
            matches.append(gesture)
    
    # Search in gesture text
    for gesture, text in GESTURE_TO_TEXT.items():
        if query_lower in text.lower() and gesture not in matches:
            matches.append(gesture)
    
    return matches

# Total number of gestures
TOTAL_STATIC_GESTURES = len(GESTURE_LABELS)
TOTAL_DYNAMIC_GESTURES = len(DYNAMIC_GESTURE_LABELS)
TOTAL_GESTURES = TOTAL_STATIC_GESTURES + TOTAL_DYNAMIC_GESTURES

print(f"✅ Gesture labels loaded: {TOTAL_STATIC_GESTURES} static, {TOTAL_DYNAMIC_GESTURES} dynamic, {TOTAL_GESTURES} total")
