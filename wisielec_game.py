import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# Przygotowanie danych
words = ["hangman", "python", "keras", "machine", "learning", "artificial", "intelligence", "neural", "network"]
max_word_length = max(len(word) for word in words)
vocab = list("abcdefghijklmnopqrstuvwxyz")

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(vocab)

X_data = []
y_data = []

for word in words:
    for i in range(1, len(word)):
        X_data.append(word[:i])
        y_data.append(word[i])

X_data = tokenizer.texts_to_sequences(X_data)
X_data = [np.pad(x, (0, max_word_length - len(x)), 'constant') for x in X_data]
X_data = np.array(X_data)
y_data = tokenizer.texts_to_sequences(y_data)
y_data = np.array([item[0] for item in y_data])
y_data = to_categorical(y_data, num_classes=len(vocab) + 1)

# Stworzenie modelu
model = Sequential()
model.add(Embedding(input_dim=len(vocab) + 1, output_dim=10, input_length=max_word_length))
model.add(LSTM(50))
model.add(Dense(len(vocab) + 1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_data, y_data, epochs=100, verbose=2)

# Implementacja gry
def get_next_letter_probabilities(model, current_word, tokenizer, max_word_length, vocab):
    current_word_seq = tokenizer.texts_to_sequences([current_word])[0]
    current_word_seq = np.pad(current_word_seq, (0, max_word_length - len(current_word_seq)), 'constant')
    current_word_seq = np.array([current_word_seq])
    probabilities = model.predict(current_word_seq)[0]
    return {vocab[i]: probabilities[i + 1] for i in range(len(vocab))}

def hangman_game(words, model, tokenizer, max_word_length, vocab):
    word = random.choice(words)
    guessed_word = ['_'] * len(word)
    attempts = len(word) + 5
    guessed_letters = []

    while attempts > 0:
        print("Word:", ' '.join(guessed_word))
        print("Attempts left:", attempts)
        next_letter_probabilities = get_next_letter_probabilities(model, ''.join(guessed_word).replace('_', ''), tokenizer, max_word_length, vocab)
        next_letter = max(next_letter_probabilities, key=next_letter_probabilities.get)

        print("Model suggests letter:", next_letter)

        user_input = input("Enter your guess (or press Enter to use model's suggestion): ").lower()
        if user_input == '':
            guess = next_letter
        else:
            guess = user_input

        if guess in guessed_letters:
            print("You already guessed that letter.")
            continue

        guessed_letters.append(guess)

        if guess in word:
            for i, char in enumerate(word):
                if char == guess:
                    guessed_word[i] = guess
            if '_' not in guessed_word:
                print("Congratulations! You guessed the word:", word)
                break
        else:
            attempts -= 1
            print("Wrong guess.")

        if attempts == 0:
            print("Game over. The word was:", word)

hangman_game(words, model, tokenizer, max_word_length, vocab)