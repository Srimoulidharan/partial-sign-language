import re
import string
from typing import List, Dict, Set
from collections import Counter

class NLPCleaner:
    """Simple NLP text cleanup and processing for sign language translation"""
    
    def __init__(self):
        # Common stop words to remove in certain contexts
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        # Common contractions and their expansions
        self.contractions = {
            "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
            "mightn't": "might not", "mustn't": "must not"
        }
        
        # Sign language specific word mappings
        self.sign_language_mappings = {
            "hello": "hello",
            "hi": "hello", 
            "hey": "hello",
            "goodbye": "goodbye",
            "bye": "goodbye",
            "thanks": "thank you",
            "thx": "thank you",
            "please": "please",
            "sorry": "sorry",
            "yes": "yes",
            "yeah": "yes",
            "yep": "yes",
            "no": "no",
            "nope": "no",
            "help": "help",
            "water": "water",
            "food": "food",
            "eat": "eat",
            "drink": "drink",
            "bathroom": "bathroom",
            "restroom": "bathroom"
        }
        
        # Common gesture-to-word mappings (these would come from your trained models)
        self.gesture_mappings = {
            "thumbs_up": "good",
            "thumbs_down": "bad", 
            "peace": "peace",
            "ok_sign": "ok",
            "pointing": "you",
            "wave": "hello",
            "fist": "stop",
            "open_hand": "help",
            "prayer": "please"
        }
        
        print("✅ NLP Cleaner initialized")
    
    def clean_text(self, text: str, remove_stop_words: bool = False) -> str:
        """
        Clean and normalize input text
        
        Args:
            text: Input text to clean
            remove_stop_words: Whether to remove stop words
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        try:
            # Convert to lowercase
            cleaned = text.lower().strip()
            
            # Expand contractions
            cleaned = self._expand_contractions(cleaned)
            
            # Remove extra whitespace and punctuation
            cleaned = self._normalize_whitespace(cleaned)
            
            # Remove unnecessary punctuation (keep some for meaning)
            cleaned = self._clean_punctuation(cleaned)
            
            # Apply sign language specific mappings
            cleaned = self._apply_sign_mappings(cleaned)
            
            # Remove stop words if requested
            if remove_stop_words:
                cleaned = self._remove_stop_words(cleaned)
            
            # Final cleanup
            cleaned = cleaned.strip()
            
            return cleaned if cleaned else text  # Return original if cleaning failed
            
        except Exception as e:
            print(f"❌ Error cleaning text: {str(e)}")
            return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text"""
        for contraction, expansion in self.contractions.items():
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean unnecessary punctuation while preserving meaning"""
        # Keep question marks, exclamation marks, and periods for sentence structure
        # Remove other punctuation
        text = re.sub(r'[^\w\s\?\!\.]', ' ', text)
        text = re.sub(r'\.+', '.', text)  # Multiple periods to single
        text = re.sub(r'\?+', '?', text)  # Multiple question marks to single
        text = re.sub(r'!+', '!', text)   # Multiple exclamation marks to single
        return text
    
    def _apply_sign_mappings(self, text: str) -> str:
        """Apply sign language specific word mappings"""
        words = text.split()
        mapped_words = []
        
        for word in words:
            # Remove punctuation for mapping lookup
            clean_word = word.strip(string.punctuation)
            
            if clean_word in self.sign_language_mappings:
                mapped_word = self.sign_language_mappings[clean_word]
                # Preserve punctuation
                if word != clean_word:
                    punctuation = word[len(clean_word):]
                    mapped_word += punctuation
                mapped_words.append(mapped_word)
            else:
                mapped_words.append(word)
        
        return ' '.join(mapped_words)
    
    def _remove_stop_words(self, text: str) -> str:
        """Remove stop words from text"""
        words = text.split()
        filtered_words = []
        
        for word in words:
            clean_word = word.strip(string.punctuation).lower()
            if clean_word not in self.stop_words or len(words) <= 2:
                # Keep stop words if sentence is very short
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def build_sentence(self, gestures: List[str], confidence_threshold: float = 0.5) -> str:
        """
        Build a coherent sentence from a sequence of recognized gestures
        
        Args:
            gestures: List of recognized gesture names
            confidence_threshold: Minimum confidence to include gesture
            
        Returns:
            Constructed sentence
        """
        if not gestures:
            return ""
        
        try:
            # Convert gestures to words
            words = []
            for gesture in gestures:
                if gesture in self.gesture_mappings:
                    word = self.gesture_mappings[gesture]
                    words.append(word)
                else:
                    # Try to clean gesture name (remove underscores, etc.)
                    clean_gesture = gesture.replace('_', ' ').replace('-', ' ')
                    words.append(clean_gesture)
            
            if not words:
                return ""
            
            # Remove duplicates while preserving some order
            unique_words = []
            seen = set()
            for word in words:
                if word not in seen:
                    unique_words.append(word)
                    seen.add(word)
            
            # Build sentence
            sentence = ' '.join(unique_words)
            
            # Apply basic grammar rules
            sentence = self._apply_basic_grammar(sentence)
            
            # Clean the final sentence
            sentence = self.clean_text(sentence)
            
            return sentence
            
        except Exception as e:
            print(f"❌ Error building sentence: {str(e)}")
            return ' '.join(gestures[:5])  # Fallback to first 5 gestures
    
    def _apply_basic_grammar(self, sentence: str) -> str:
        """Apply basic grammar rules to improve sentence structure"""
        if not sentence:
            return ""
        
        try:
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            
            # Add period if no ending punctuation
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            # Handle common patterns
            sentence = re.sub(r'\bi\b', 'I', sentence)  # Capitalize 'I'
            sentence = re.sub(r'\bgood\s+bad\b', 'okay', sentence)  # Handle contradictions
            sentence = re.sub(r'\bhello\s+goodbye\b', 'hello', sentence)  # Handle greetings
            
            return sentence
            
        except Exception as e:
            print(f"❌ Error applying grammar: {str(e)}")
            return sentence
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract key words from text
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        try:
            # Clean text and remove stop words
            cleaned = self.clean_text(text, remove_stop_words=True)
            
            # Split into words and count frequency
            words = cleaned.split()
            word_freq = Counter(word.lower().strip(string.punctuation) for word in words)
            
            # Remove empty strings and very short words
            word_freq = {word: freq for word, freq in word_freq.items() 
                        if word and len(word) > 2}
            
            # Get most common words
            keywords = [word for word, freq in word_freq.most_common(max_keywords)]
            
            return keywords
            
        except Exception as e:
            print(f"❌ Error extracting keywords: {str(e)}")
            return []
    
    def similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity score between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Clean both texts
            clean1 = set(self.clean_text(text1).split())
            clean2 = set(self.clean_text(text2).split())
            
            if not clean1 and not clean2:
                return 1.0
            if not clean1 or not clean2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(clean1.intersection(clean2))
            union = len(clean1.union(clean2))
            
            similarity = intersection / union if union > 0 else 0.0
            
            return similarity
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {str(e)}")
            return 0.0
    
    def suggest_corrections(self, text: str) -> List[str]:
        """
        Suggest simple corrections for common errors
        
        Args:
            text: Input text
            
        Returns:
            List of suggested corrections
        """
        suggestions = []
        
        try:
            # Check for common patterns that might need correction
            words = text.lower().split()
            
            # Suggest adding articles
            if len(words) >= 2:
                first_word = words[0]
                if first_word not in ['the', 'a', 'an', 'i', 'you', 'we', 'they']:
                    suggestions.append(f"Consider: 'I {text.lower()}'")
                    suggestions.append(f"Consider: 'The {text.lower()}'")
            
            # Suggest verb forms
            if 'eat' in words and 'want' not in words:
                suggestions.append("Consider: 'I want to eat'")
            
            if 'drink' in words and 'want' not in words:
                suggestions.append("Consider: 'I want to drink'")
            
            # Suggest common phrases
            if 'help' in words:
                suggestions.append("Consider: 'I need help'")
                suggestions.append("Consider: 'Can you help me?'")
            
            return suggestions[:3]  # Return top 3 suggestions
            
        except Exception as e:
            print(f"❌ Error generating suggestions: {str(e)}")
            return []
    
    def format_for_display(self, text: str) -> str:
        """
        Format text for display in the UI
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text
        """
        try:
            if not text:
                return "No translation available"
            
            # Clean and format
            formatted = self.clean_text(text)
            
            # Ensure proper capitalization
            formatted = formatted[0].upper() + formatted[1:] if len(formatted) > 1 else formatted.upper()
            
            # Ensure ending punctuation
            if not formatted.endswith(('.', '!', '?')):
                formatted += '.'
            
            return formatted
            
        except Exception as e:
            print(f"❌ Error formatting text: {str(e)}")
            return text or "Error formatting text"
