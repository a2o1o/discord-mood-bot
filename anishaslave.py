import os, discord, asyncio, json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
print("🔍 Loading environment variables...")

# Discord & Gemini creds
ANISHA_SLAVE_TOKEN   = os.getenv("ANISHA_SLAVE_TOKEN")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")

# Verify credentials are loaded
if not ANISHA_SLAVE_TOKEN:
    print("❌ Error: ANISHA_SLAVE_TOKEN not found in .env file")
    exit(1)
print("✅ Discord token loaded")

if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env file")
    exit(1)
print("✅ Gemini API key loaded")

# Setup clients
print("🤖 Setting up Discord client...")
intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # Enable member intents
intents.presences = True  # Enable presence intents
bot = discord.Client(intents=intents)

print("🔑 Setting up Gemini...")
genai.configure(api_key=GEMINI_API_KEY)

# Peak tracking per channel
peaks = {}  # channel_id → {"intensity":float,"emotion":str}

# Song suggestions for different emotions
EMOTION_SONGS = {
    "roast": {
        "name": "Humble - Kendrick Lamar",
        "url": "https://open.spotify.com/track/7KXjTSCq5nL1LoYtL7XAwS"
    },
    "sad": {
        "name": "Someone Like You - Adele",
        "url": "https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"
    },
    "happy": {
        "name": "It Was A Good Day - Ice Cube",
        "url": "https://open.spotify.com/track/2qOm7ukLyHUXWyR4ZWLwxA"
    },
    "jealousy": {
        "name": "Jealousy,jealousy - Olivia Rodrigo",
        "url": "https://open.spotify.com/track/0MMyJUC3WNnFS1lit5pTjk"
    },
    "excitement": {
        "name": "Can't Stop the Feeling! - Justin Timberlake",
        "url": "https://open.spotify.com/track/3igTxmBCArlfuXwW2sGKlz"
    },
    "anger": {
        "name": "Can't Stop the Feeling! - Justin Timberlake",
        "url": "https://open.spotify.com/track/3igTxmBCArlfuXwW2sGKlz"
    },
    "despair": {
        "name": "Can't Stop the Feeling! - Justin Timberlake",
        "url": "https://open.spotify.com/track/3igTxmBCArlfuXwW2sGKlz"
    },
    "shy": {
        "name": "Can't Stop the Feeling! - Justin Timberlake",
        "url": "https://open.spotify.com/track/3igTxmBCArlfuXwW2sGKlz"
    },
    "neutral": {
        "name": "Chill Vibes Playlist",
        "url": "https://open.spotify.com/playlist/37i9dQZF1DX3Ogo9pFvBkY"
    }
}

async def classify_emotion(text):
    try:
        print(f"🤖 Attempting to classify emotion for text: {text[:50]}...")
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = (
            "You are a sentiment analyzer. Analyze the following message and respond with a JSON object "
            "containing 'emotion' (one of: roast, sad, happy, anger, despair, shy, jealousy, neutral, excitement) and 'intensity' (0-10).\n\n"
            f"Message: {text}\n\n"
            "Respond with ONLY a valid JSON object in this exact format:\n"
            '{"emotion": "emotion_name", "intensity": number}\n'
            "Do not include any other text or explanation."
        )
        print("📝 Sending prompt to Gemini...")
        response = model.generate_content(prompt)
        print(f"✅ Got response from Gemini: {response.text[:100]}...")
        
        # Clean the response text to ensure it's valid JSON
        response_text = response.text.strip()
        # Remove any markdown code block markers if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        try:
            result = json.loads(response_text)
            # Validate the response format
            if not isinstance(result, dict) or 'emotion' not in result or 'intensity' not in result:
                raise ValueError("Invalid response format")
            return result
        except json.JSONDecodeError as je:
            print(f"❌ JSON parsing error: {str(je)}")
            print(f"Raw response: {response_text}")
            return {"emotion": "neutral", "intensity": 0}
            
    except Exception as e:
        print(f"❌ Error in emotion classification: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return {"emotion": "neutral", "intensity": 0}

@bot.event
async def on_ready():
    print(f"✅ {bot.user} is online and ready!")
    print(f"🔗 Connected to {len(bot.guilds)} servers")
    for guild in bot.guilds:
        print(f"   - {guild.name}")

@bot.event
async def on_message(msg):
    if msg.author.bot: return

    ch = msg.channel.id
    try:
        print(f"📨 Received message: {msg.content[:50]}...")
        data = await classify_emotion(msg.content)
        print(f"📊 Classification result: {data}")
        emo, intensity = data["emotion"], float(data["intensity"])
        peak = peaks.get(ch, {"intensity":0})

        # Only send song if intensity is above 3
        if intensity > 3:
            print(f"🎯 Message intensity {intensity} exceeds threshold of 3")
            peaks[ch] = {"emotion":emo,"intensity":intensity}
            
            # Get song suggestion for the emotion
            song = EMOTION_SONGS.get(emo, EMOTION_SONGS["neutral"])
            
            # Send song suggestion as a reply
            await msg.reply(
                f"🎵 Mood: **{emo}** (intensity: {intensity:.1f})\n"
                f"Here's a song suggestion:\n"
                f"**{song['name']}**\n"
                f"{song['url']}"
            )
    except Exception as e:
        print(f"❌ Error processing message: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

print("🚀 Starting bot...")
try:
    bot.run(ANISHA_SLAVE_TOKEN)
except Exception as e:
    print(f"❌ Error starting bot: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
