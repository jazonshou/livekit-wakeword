"""Built-in English word and sentence lists for negative and context clips."""

from __future__ import annotations

# Common one- and two-syllable words and first names. Swapped into each word slot of
# the wake phrase ("hey <word>", "<word> computer") so the model hears the rest of the
# phrase with something that is clearly not the wake word.
# fmt: off
COMMON_SWAP_WORDS: tuple[str, ...] = (
    # everyday words
    "there", "here", "you", "guys", "man", "dude", "buddy", "friend", "babe", "honey",
    "mom", "dad", "kids", "folks", "team", "all", "now", "wait", "stop", "look",
    "listen", "what", "why", "how", "who", "when", "where", "yes", "no", "okay",
    "please", "thanks", "sorry", "hello", "hi", "hey", "well", "so", "just", "come",
    "go", "get", "give", "take", "make", "help", "tell", "call", "play", "turn",
    "check", "find", "show", "open", "close", "start", "set", "put", "read", "write",
    "time", "day", "night", "home", "work", "food", "car", "phone", "music", "light",
    "door", "dog", "cat", "book", "game", "news", "thing", "stuff", "world", "life",
    "good", "great", "nice", "cool", "fine", "sure", "right", "wrong", "really", "maybe",
    "again", "later", "today", "morning", "dinner", "lunch", "water", "coffee", "money",
    "weather", "people", "little", "something", "nothing", "everyone", "sister", "brother",
    # common first names and assistant names
    "jack", "zach", "mike", "mark", "nick", "rick", "chuck", "luke", "john", "james",
    "tom", "tim", "sam", "max", "ben", "dan", "joe", "bob", "bill", "steve",
    "dave", "chris", "matt", "pat", "kate", "jane", "anne", "amy", "emma", "lucy",
    "sarah", "laura", "lisa", "linda", "mary", "julia", "nora", "maya", "zoe", "jess",
    "siri", "alexa", "google", "cortana", "bixby", "jarvis", "computer", "robot",
)
# fmt: on

# Short, generic sentences synthesized as speech heard *before* the wake phrase, so
# the model sees the phrase (and near-misses) right after other talk.
CONTEXT_SENTENCES: tuple[str, ...] = (
    "I think we should leave a little earlier today.",
    "Did you see where I put my keys?",
    "That was the best movie I have seen all year.",
    "Can you pass me the salt, please?",
    "We need to pick up some milk on the way home.",
    "The meeting got moved to three o'clock.",
    "It is supposed to rain all weekend.",
    "I could not find a parking spot anywhere.",
    "Let me know when you are ready to go.",
    "She said she would call back after lunch.",
    "This coffee is way too hot to drink.",
    "Have you finished the report yet?",
    "My phone battery is almost dead.",
    "The kids are already asleep upstairs.",
    "I will be there in about ten minutes.",
    "Do you want pizza or pasta tonight?",
    "He has been working from home all week.",
    "The train was late again this morning.",
    "Could you turn the music down a bit?",
    "We should plan a trip somewhere warm.",
    "I forgot to water the plants yesterday.",
    "What time does the store close today?",
    "That sounds like a really good idea.",
    "Honestly I have no idea what happened.",
    "The game starts in about an hour.",
    "Remind me to buy a birthday card.",
    "I am going to take the dog for a walk.",
    "Our neighbors just got a new puppy.",
    "The traffic on the highway was terrible.",
    "Can you help me carry these boxes?",
    "I need to get some sleep tonight.",
    "Let's order something for dinner.",
    "Wait, did you hear that noise?",
    "Okay, so here is the plan for tomorrow.",
    "Yeah, I totally agree with you.",
    "No, I meant the other one.",
    "Well, that is not what I expected.",
    "Alright, let's get started then.",
    "Hmm, I am not sure about that.",
    "So anyway, how was your day?",
    "The printer is out of paper again.",
    "I left my jacket in the car.",
    "Dinner will be ready in twenty minutes.",
    "Please close the window, it is cold.",
    "Where did you say the restaurant was?",
    "I have to finish this before Friday.",
    "Tell me more about your new job.",
    "Everyone is coming over on Saturday.",
    "I really like the color of that wall.",
    "Did anyone feed the cat this morning?",
    "The internet has been slow all day.",
    "We ran out of bread and eggs.",
    "My sister is visiting next month.",
    "Turn left at the end of the street.",
    "I am almost done, just give me a second.",
    "That show was much better than I thought.",
    "Can we talk about this later?",
    "It is way too early to be awake.",
    "Thanks for coming over tonight.",
    "I will send you the pictures tomorrow.",
)
