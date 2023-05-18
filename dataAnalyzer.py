import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from textblob import TextBlob
from langdetect import detect_langs
import concurrent.futures

nltk.download('punkt')
nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

with open('chatlogs.jsonl', 'r') as file:
    lines = file.readlines()

def process_line(line):
    total_user_messages = 0
    total_chat_gpt_messages = 0
    total_user_message_length = 0
    total_chat_gpt_message_length = 0
    total_user_sentiment = 0
    total_chat_gpt_sentiment = 0
    all_user_words = []
    all_chat_gpt_words = []
    languages_counter = Counter()

    def preprocess_text(text):
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in english_stopwords]
        return filtered_words

    def detect_language(text):
        try:
            langs = detect_langs(text)
            lang = sorted(langs, key=lambda x: x.prob, reverse=True)[0]
            return lang.lang
        except:
            return None

    conversation_data = json.loads(line)
    conversation = conversation_data['conversation']

    user_messages = [message for message in conversation if message['user'][0] == 'Anonymous']
    chat_gpt_messages = [message for message in conversation if message['user'][0] == 'Chat GPT']

    total_user_messages += len(user_messages)
    total_chat_gpt_messages += len(chat_gpt_messages)

    total_user_message_length += sum([len(message['message']) for message in user_messages])
    total_chat_gpt_message_length += sum([len(message['message']) for message in chat_gpt_messages])

    conversation_text = ' '.join([message['message'] for message in conversation])

    lang = detect_language(conversation_text)
    if lang:
        languages_counter[lang] += 1

    for user_message in user_messages:
        words = preprocess_text(user_message['message'])
        all_user_words.extend(words)
        user_sentiment = TextBlob(user_message['message']).sentiment.polarity
        total_user_sentiment += user_sentiment

    for chat_gpt_message in chat_gpt_messages:
        words = preprocess_text(chat_gpt_message['message'])
        all_chat_gpt_words.extend(words)
        chat_gpt_sentiment = TextBlob(chat_gpt_message['message']).sentiment.polarity
        total_chat_gpt_sentiment += chat_gpt_sentiment

    return {
        'total_user_messages': total_user_messages,
        'total_chat_gpt_messages': total_chat_gpt_messages,
        'total_user_message_length': total_user_message_length,
        'total_chat_gpt_message_length': total_chat_gpt_message_length,
        'total_user_sentiment': total_user_sentiment,
        'total_chat_gpt_sentiment': total_chat_gpt_sentiment,
        'all_user_words': all_user_words,
        'all_chat_gpt_words': all_chat_gpt_words,
        'languages_counter': languages_counter
    }

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_line, lines), total=len(lines), desc="Processing conversations"))

combined_results = {
    'total_user_messages': 0,
    'total_chat_gpt_messages': 0,
    'total_user_message_length': 0,
    'total_chat_gpt_message_length': 0,
    'total_user_sentiment': 0,
    'total_chat_gpt_sentiment': 0,
    'all_user_words': [],
    'all_chat_gpt_words': [],
    'languages_counter': Counter()
}

for result in results:
    combined_results['total_user_messages'] += result['total_user_messages']
    combined_results['total_chat_gpt_messages'] += result['total_chat_gpt_messages']
    combined_results['total_user_message_length'] += result['total_user_message_length']
    combined_results['total_chat_gpt_message_length'] += result['total_chat_gpt_message_length']
    combined_results['total_user_sentiment'] += result['total_user_sentiment']
    combined_results['total_chat_gpt_sentiment'] += result['total_chat_gpt_sentiment']
    combined_results['all_user_words'].extend(result['all_user_words'])
    combined_results['all_chat_gpt_words'].extend(result['all_chat_gpt_words'])
    combined_results['languages_counter'] += result['languages_counter']


total_user_messages = combined_results['total_user_messages']
total_chat_gpt_messages = combined_results['total_chat_gpt_messages']
total_conversations = len(lines)

average_user_messages = total_user_messages / total_conversations
average_chat_gpt_messages = total_chat_gpt_messages / total_conversations

average_user_message_length = combined_results['total_user_message_length'] / total_user_messages
average_chat_gpt_message_length = combined_results['total_chat_gpt_message_length'] / total_chat_gpt_messages

average_user_sentiment = combined_results['total_user_sentiment'] / total_user_messages
average_chat_gpt_sentiment = combined_results['total_chat_gpt_sentiment'] / total_chat_gpt_messages

word_freq_user = Counter(combined_results['all_user_words'])
word_freq_chat_gpt = Counter(combined_results['all_chat_gpt_words'])

most_common_words_user = word_freq_user.most_common(10)
most_common_words_chat_gpt = word_freq_chat_gpt.most_common(10)

total_user_words = len(combined_results['all_user_words'])
total_chat_gpt_words = len(combined_results['all_chat_gpt_words'])

languages_counter = combined_results['languages_counter']

print(f'Total user messages: {total_user_messages}')
print(f'Total Chat GPT messages: {total_chat_gpt_messages}')

print(f'Average user messages per conversation: {average_user_messages}')
print(f'Average Chat GPT messages per conversation: {average_chat_gpt_messages}')

print(f'Average length of user questions: {average_user_message_length}')
print(f'Average length of Chat GPT responses: {average_chat_gpt_message_length}')

print(f'Average sentiment of user questions: {average_user_sentiment}')
print(f'Average sentiment of Chat GPT responses: {average_chat_gpt_sentiment}')

print(f'Total number of words in user messages: {total_user_words}')
print(f'Total number of words in Chat GPT messages: {total_chat_gpt_words}')

print("Top 10 most common words in user messages:")
for word, freq in most_common_words_user:
    print(f"{word}: {freq}")

print("Top 10 most common words in Chat GPT messages:")
for word, freq in most_common_words_chat_gpt:
    print(f"{word}: {freq}")
    
print(f"Total number of languages detected: {len(languages_counter)}")
print("Languages detected in conversations:")
for lang, count in languages_counter.items():
    print(f"{lang}: {count}")
