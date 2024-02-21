from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import hashlib
load_dotenv()
MODEL_NAME = "gpt-3.5-turbo"
now = '.'.join(map(str, [datetime.today().day, datetime.today().month, datetime.today().year]))

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
model = "gpt-3.5-turbo"
cache = dict()

def connect_api(user_name, year, month, day, query):
    keys = user_name.strip() + year + month + day + now + query.strip()
    hash = hashlib.md5(keys.encode()).hexdigest()

    if hash not in cache:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system",
                     "content": f"You are the most talented astrologer in the world.\
                      As long as you know the other person's name and date of birth, \
                      you can predict that person's fortune.\
                       It can tell you not only short-term financial or human relationship predictions, \
                       but also long-term fortune. \
                       You can anticipate and answer all the information the other person is curious about. \
                       Any questions must be answered in Korean.\
                        When answering, first greet the other person by name, then introduce yourself, \
                        summarize the other person's question, and then answer that question. \
                        Today is {now}, "},
                    {"role": "user",
                     "content": f"Hi. My Name is {user_name}, My birth day is {year}.{month}.{day}. {query}"},
                ],
                model=model,
                temperature=1,
            )
            # print(chat_completion.choices[0].message.content)
            cache[hash] = chat_completion

            return chat_completion
        except Exception as e:
            print(e)
    return cache[hash]





