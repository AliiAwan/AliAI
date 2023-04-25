import nltk
nltk.download('punkt')

def read_conversations(filename):
    conversations = []
    with open(filename, 'r', encoding='utf8') as file:
        lines = file.readlines()
        conversation = {}
        for line in lines:
            if line.startswith('Kontext:'):
                conversation['context'] = line.strip()[9:]
            elif line.startswith('Benutzer:'):
                conversation['user'] = line.strip()[10:]
            elif line.startswith('Ali:'):
                conversation['bot'] = line.strip()[5:]
                conversations.append(conversation)
                conversation = {}
    return conversations

conversations = read_conversations('AliConversations.txt')
tokenizer = nltk.tokenize.sent_tokenize

for conversation in conversations:
    if 'context' in conversation:
        context = conversation['context']
        context_sentences = tokenizer(context)
        print('Context sentences:', context_sentences)
    else:
        print('No context found for this conversation')

    user = conversation['user']
    bot = conversation['bot']
    user_sentences = tokenizer(user)
    bot_sentences = tokenizer(bot)
    print('User sentences:', user_sentences)
    print('Bot sentences:', bot_sentences)
