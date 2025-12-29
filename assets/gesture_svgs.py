# Simple SVG representations for common sign language gestures
# These are basic vector diagrams of hand positions
# Each SVG is 100x100 pixels, scalable

GESTURE_SVGS = {
    'hello': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Palm -->
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <!-- Fingers (waving/open hand) -->
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/> <!-- Thumb -->
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/> <!-- Index -->
      <line x1="55" y1="50" x2="55" y2="25" stroke="black" stroke-width="2"/> <!-- Middle -->
      <line x1="52" y1="50" x2="52" y2="35" stroke="black" stroke-width="2"/> <!-- Ring (wavy) -->
      <line x1="48" y1="50" x2="48" y2="30" stroke="black" stroke-width="2"/> <!-- Pinky -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Hello (Wave)</text>
    </svg>
    ''',
    
    'goodbye': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Palm -->
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <!-- Fingers waving -->
      <line x1="45" y1="50" x2="45" y2="25" stroke="black" stroke-width="2"/> <!-- Thumb -->
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/> <!-- Index -->
      <line x1="55" y1="50" x2="55" y2="30" stroke="black" stroke-width="2"/> <!-- Middle -->
      <line x1="52" y1="50" x2="52" y2="15" stroke="black" stroke-width="2"/> <!-- Ring -->
      <line x1="48" y1="50" x2="48" y2="35" stroke="black" stroke-width="2"/> <!-- Pinky -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Goodbye (Wave)</text>
    </svg>
    ''',
    
    'thank_you': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Palm cupped -->
      <path d="M40 50 Q50 60 60 50 L60 80 L40 80 Z" fill="none" stroke="black" stroke-width="3"/>
      <!-- Fingers curled -->
      <path d="M45 50 Q45 40 50 35" stroke="black" stroke-width="2" fill="none"/> <!-- Index -->
      <path d="M50 50 Q50 35 55 40" stroke="black" stroke-width="2" fill="none"/> <!-- Middle -->
      <path d="M52 50 Q52 45 48 40" stroke="black" stroke-width="2" fill="none"/> <!-- Ring -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Thank You (Cup)</text>
    </svg>
    ''',
    
    'please': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Palm flat -->
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <!-- Fingers slightly bent (pleading) -->
      <line x1="45" y1="50" x2="45" y2="35" stroke="black" stroke-width="2"/> <!-- Thumb -->
      <line x1="50" y1="50" x2="50" y2="30" stroke="black" stroke-width="2"/> <!-- Index bent -->
      <line x1="55" y1="50" x2="55" y2="40" stroke="black" stroke-width="2"/> <!-- Middle -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Please (Beg)</text>
    </svg>
    ''',
    
    'yes': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Palm -->
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <!-- Thumb up, fingers closed -->
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="3"/> <!-- Thumb up -->
      <line x1="50" y1="50" x2="50" y2="70" stroke="black" stroke-width="2"/> <!-- Closed fingers -->
      <line x1="55" y1="50" x2="55" y2="70" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Yes (Thumbs Up)</text>
    </svg>
    ''',
    
    'no': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Palm flat, fingers closed -->
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="70" stroke="black" stroke-width="2"/> <!-- Closed -->
      <line x1="50" y1="50" x2="50" y2="70" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="70" stroke="black" stroke-width="2"/>
      <!-- Index finger across (no) -->
      <line x1="50" y1="40" x2="60" y2="30" stroke="black" stroke-width="3"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">No (Block)</text>
    </svg>
    ''',
    
    'help': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Palm up, fingers raised -->
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/> <!-- Thumb -->
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/> <!-- Index -->
      <line x1="55" y1="50" x2="55" y2="25" stroke="black" stroke-width="2"/> <!-- Middle raised -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Help (Raise)</text>
    </svg>
    ''',
    
    'stop': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Open palm -->
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="20" stroke="black" stroke-width="2"/> <!-- All fingers extended -->
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="20" stroke="black" stroke-width="2"/>
      <line x1="47" y1="50" x2="47" y2="25" stroke="black" stroke-width="2"/>
      <line x1="53" y1="50" x2="53" y2="25" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Stop (Palm)</text>
    </svg>
    ''',
    
    'ok': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Wrist -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <!-- Palm -->
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <!-- OK circle: thumb and index -->
      <circle cx="48" cy="45" r="5" fill="none" stroke="black" stroke-width="3"/> <!-- Circle -->
      <line x1="45" y1="50" x2="43" y2="45" stroke="black" stroke-width="2"/> <!-- Thumb -->
      <line x1="50" y1="50" x2="48" y2="40" stroke="black" stroke-width="2"/> <!-- Index -->
      <line x1="55" y1="50" x2="55" y2="70" stroke="black" stroke-width="2"/> <!-- Closed fingers -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">OK (Circle)</text>
    </svg>
    ''',
    
    'good': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Similar to thumbs up -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="3"/> <!-- Thumb up -->
      <line x1="50" y1="50" x2="50" y2="70" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="70" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Good (Thumbs)</text>
    </svg>
    ''',
    
    'bad': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <!-- Thumbs down -->
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="70" stroke="black" stroke-width="3"/> <!-- Thumb down -->
      <line x1="50" y1="50" x2="50" y2="30" stroke="black" stroke-width="2"/> <!-- Fingers up -->
      <line x1="55" y1="50" x2="55" y2="30" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Bad (Thumbs Down)</text>
    </svg>
    ''',
    
    'water': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/> <!-- Fingers tapping (W) -->
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="25" stroke="black" stroke-width="2"/>
      <circle cx="50" cy="60" r="2" fill="blue"/> <!-- Water drop -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Water (Tap)</text>
    </svg>
    ''',
    
    'food': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/> <!-- Fingers to mouth -->
      <line x1="50" y1="50" x2="55" y2="35" stroke="black" stroke-width="2"/>
      <circle cx="52" cy="40" r="3" fill="red"/> <!-- Food icon -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Food (Mouth)</text>
    </svg>
    ''',
    
    'eat': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="50" y2="35" stroke="black" stroke-width="2"/> <!-- Pinch to mouth -->
      <line x1="55" y1="50" x2="52" y2="35" stroke="black" stroke-width="2"/>
      <path d="M48 35 Q50 30 52 35" stroke="black" stroke-width="2" fill="none"/> <!-- Mouth -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Eat (Pinch)</text>
    </svg>
    ''',
    
    'drink': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/> <!-- Hand to mouth (C shape) -->
      <line x1="55" y1="50" x2="55" y2="30" stroke="black" stroke-width="2"/>
      <path d="M42 45 Q50 35 58 45" stroke="black" stroke-width="2" fill="none"/> <!-- Cup shape -->
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Drink (Cup)</text>
    </svg>
    ''',
    
    # Additional static gestures
    'more': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="25" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">More</text>
    </svg>
    ''',
    
    'finished': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="70" stroke="black" stroke-width="2"/>
      <line x1="50" y1="50" x2="50" y2="70" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="70" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Finished</text>
    </svg>
    ''',
    
    'sorry': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <path d="M45 50 Q45 40 50 35" stroke="black" stroke-width="2" fill="none"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Sorry</text>
    </svg>
    ''',
    
    'love': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <path d="M45 50 Q50 40 55 50" stroke="red" stroke-width="2" fill="red" opacity="0.7"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Love</text>
    </svg>
    ''',
    
    'peace': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="25" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Peace</text>
    </svg>
    ''',
    
    # Dynamic gestures
    'how_are_you': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="25" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">How Are You</text>
    </svg>
    ''',
    
    'nice_to_meet_you': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="6" fill="black">Nice to Meet</text>
    </svg>
    ''',
    
    'what_is_your_name': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="6" fill="black">Your Name?</text>
    </svg>
    ''',
    
    'where_is_bathroom': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <circle cx="50" cy="40" r="3" fill="blue"/>
      <text x="50" y="95" text-anchor="middle" font-size="6" fill="black">Bathroom?</text>
    </svg>
    ''',
    
    'i_need_help': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="25" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">I Need Help</text>
    </svg>
    ''',
    
    'thank_you_very_much': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <path d="M40 50 Q50 60 60 50 L60 80 L40 80 Z" fill="none" stroke="black" stroke-width="3"/>
      <text x="50" y="95" text-anchor="middle" font-size="6" fill="black">Thank You Much</text>
    </svg>
    ''',
    
    'have_a_good_day': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/>
      <circle cx="50" cy="40" r="5" fill="yellow" opacity="0.5"/>
      <text x="50" y="95" text-anchor="middle" font-size="6" fill="black">Good Day</text>
    </svg>
    ''',
    
    'see_you_later': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="25" stroke="black" stroke-width="2"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="6" fill="black">See You Later</text>
    </svg>
    ''',
    
    'i_am_hungry': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="50" y2="35" stroke="black" stroke-width="2"/>
      <circle cx="52" cy="40" r="3" fill="red"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">I Am Hungry</text>
    </svg>
    ''',
    
    'i_am_thirsty': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <path d="M42 45 Q50 35 58 45" stroke="black" stroke-width="2" fill="none"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">I Am Thirsty</text>
    </svg>
    ''',
    
    'excuse_me': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="50" y1="50" x2="50" y2="30" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">Excuse Me</text>
    </svg>
    ''',
    
    'i_am_sorry': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <path d="M45 50 Q45 40 50 35" stroke="black" stroke-width="2" fill="none"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">I Am Sorry</text>
    </svg>
    ''',
    
    'i_love_you': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="30" stroke="black" stroke-width="2"/>
      <path d="M45 50 Q50 40 55 50" stroke="red" stroke-width="2" fill="red" opacity="0.7"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">I Love You</text>
    </svg>
    ''',
    
    'good_morning': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/>
      <circle cx="50" cy="40" r="5" fill="yellow" opacity="0.5"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">Good Morning</text>
    </svg>
    ''',
    
    'good_night': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/>
      <circle cx="50" cy="40" r="5" fill="navy" opacity="0.5"/>
      <text x="50" y="95" text-anchor="middle" font-size="7" fill="black">Good Night</text>
    </svg>
    ''',
    
    # Generic fallback for unknown words
    'unknown': '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="80" x2="50" y2="90" stroke="black" stroke-width="3"/>
      <rect x="40" y="50" width="20" height="30" fill="none" stroke="black" stroke-width="3"/>
      <line x1="45" y1="50" x2="45" y2="30" stroke="black" stroke-width="2"/>
      <line x1="50" y1="50" x2="50" y2="20" stroke="black" stroke-width="2"/>
      <line x1="55" y1="50" x2="55" y2="25" stroke="black" stroke-width="2"/>
      <text x="50" y="95" text-anchor="middle" font-size="8" fill="black">Generic Hand</text>
    </svg>
    '''
}

# Additional mappings for variations (e.g., "thankyou" -> "thank_you")
WORD_VARIATIONS = {
    'thankyou': 'thank_you',
    'thank you': 'thank_you',
    'goodbye': 'goodbye',
    'helllo': 'hello',  # Typos
    # Add more as needed
}

def get_svg_for_word(word: str) -> str:
    """
    Get SVG for a word, handling variations and fallbacks
    """
    # Normalize word (lowercase, remove punctuation)
    normalized = word.lower().strip('.,!?')
    
    # Check variations
    if normalized in WORD_VARIATIONS:
        normalized = WORD_VARIATIONS[normalized]
    
    # Get SVG
    return GESTURE_SVGS.get(normalized, GESTURE_SVGS['unknown'])


def get_svg_for_gesture(gesture_name: str) -> str:
    """
    Get SVG for a gesture name (with underscores, e.g., 'how_are_you')
    This is used when we have the gesture name directly from the recognition system
    """
    # Normalize gesture name (lowercase, handle underscores)
    normalized = gesture_name.lower().strip()
    
    # Direct lookup for gesture names
    if normalized in GESTURE_SVGS:
        return GESTURE_SVGS[normalized]
    
    # Try replacing underscores with spaces and checking word variations
    word_version = normalized.replace('_', ' ')
    if word_version in WORD_VARIATIONS:
        normalized = WORD_VARIATIONS[word_version]
        return GESTURE_SVGS.get(normalized, GESTURE_SVGS['unknown'])
    
    # Fallback to unknown
    return GESTURE_SVGS.get('unknown', '')
