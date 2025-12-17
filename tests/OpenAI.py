import openai
# import os
# from gtts import gTTS

openai.api_key = "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"

# def speak(text):
#     tts = gTTS(text)
#     print(text)
#     tts.save('to read out loud.mp3')
#     os.system('mpg123 to\ read\ out\ loud.mp3')

# response = openai.Completion.create(
#   model="text-davinci-002",
#   prompt='''create a python code to listen to whatever we say simultaneously''',
#   temperature=0.2,
#   max_tokens=300,
#   top_p=1.0,
#   frequency_penalty=0.0,
#   presence_penalty=0.0
# )

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

response = openai.Completion.create(
  model="text-davinci-002",
  prompt='''Human: is oil vegetarian\n
AI: No, oil is not vegetarian.\n
Human: Why\n
AI:''',
  temperature=0.5,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0.6,
  stop=[" Human:", " AI:"]
)
print(response.choices[0].text)