import tensorflow as tf


"""
    Character-level RNN for text generation.
    
    How generation works:
    
    START: “Hello”
    
    Iteration 1:
      Input: “Hello” → Model → Predictions for each position
      We take the last [-1] → Probabilities of the next character
      We draw: ‘ ’
      Result: “Hello ”
    
    Iteration 2:
      Input: ‘ ’ (+ hidden states) → Model → Prediction
      We draw: 'w'
      Result: “Hello w”
    
    Iteration 3:
      Input: ‘w’ (+ hidden states) → Model → Prediction
      We draw: ‘o’
      Result: “Hello wo”
    
    ...and so on until generation_length...
    
    RESULT: "Hello world! This is generated..."
    
    Key points:
    - The first iteration processes the entire start_string
    - Subsequent iterations: only 1 character + hidden states from the previous step (the model “remembers” the context through LSTM states, not by reprocessing the entire text).
    - Temperature controls randomness (lower = more predictable)
"""

class CharRNN(tf.keras.Model):
    def __init__(self,
                 char_to_idx: dict,
                 idx_to_char: dict,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 rnn_units: int = 256,
                 num_layers: int = 2):
        super().__init__()

        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers

        self.lstm_layers = []
        for i, rnn_unit in zip(range(num_layers), rnn_units):
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    rnn_unit,
                    return_sequences=True,
                    return_state=True,
                    dropout=0.4
                )
            )

        self.dense = tf.keras.layers.Dense(vocab_size)

        self._built = True

    def call(self, inputs, training=False):
        """for training"""
        x = self.embedding(inputs, training=training)
        for lstm in self.lstm_layers:
            x, _, _ = lstm(x, training=training)
        return self.dense(x, training=training)

    def generate_step(self, inputs, states=None):
        """for generating"""
        x = self.embedding(inputs, training=False)

        all_states = []
        for i, lstm in enumerate(self.lstm_layers):
            if states:
                h_idx = i * 2
                c_idx = i * 2 + 1
                if c_idx < len(states):
                    layer_states = [states[h_idx], states[c_idx]]
                else:
                    layer_states = None
            else:
                layer_states = None

            x, h, c = lstm(x, initial_state=layer_states)
            all_states.extend([h, c])

        return self.dense(x), all_states

    def build(self, input_shape):
        """
        prevent:
        'UserWarning: `build()` was called on layer 'char_rnn_7', however the layer does not have a `build()` method implemented and it looks like it has unbuilt state'
        """
        if self._built:
            return

        super().build(input_shape)

        # Simulate forward pass to build layers
        # Use dummy input with the correct shape
        batch_size = input_shape[0] or 1
        seq_len = input_shape[1] or 10

        dummy_input = tf.zeros((batch_size, seq_len), dtype=tf.int32)

        # Pass through the layers to build them
        x = self.embedding(dummy_input)
        for lstm in self.lstm_layers:
            x, _, _ = lstm(x)
        self.dense(x)

        self._built = True


    def generate(self, start_string: str, generation_length: int = 100, temperature: float = 1.0) -> str:
        """
        Generates text character by character

        temperature: controls “creativity”
        - 0.5 = more predictable
        - 1.0 = balanced
        - 2.0 = more random
        """

        # Convert start string to indexes
        input_eval = [self.char_to_idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        generated = []
        states = None

        for i in range(generation_length):
            predictions, states = self.generate_step(input_eval, states=states)

            # Remove batch dim
            predictions = tf.squeeze(predictions, 0)
            # Use last prediction
            predictions = predictions[-1, :] / temperature

            # Sample from distribution (not argmax!)
            # Sample from distribution will draw an additional character based on softmax
            # Randomly pick next char (higher logit = higher chance)
            predicted_id = tf.random.categorical(
                tf.expand_dims(predictions, 0),
                num_samples=1
            )[-1, 0].numpy()

            # The next input is the generated character
            input_eval = tf.expand_dims([predicted_id], 0)

            generated.append(self.idx_to_char[predicted_id])

        return start_string + ''.join(generated)