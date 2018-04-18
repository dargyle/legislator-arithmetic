
# Set parameters
n_leg = # Number of legislators
n_bills = # Number of bills

n_tokens = # Number of words in vocabulary
max_words = # Maximum document length

k_emb = 50 # word embedding dimension
k_pol = 1 # polarity dimension
k_pop = 1 # popularity dimension


leg_input = Input(shape=(1, ), dtype="int32", name="leg_input")
ideal_point = Embedding(input_dim=n_leg, output_dim=k_pol, input_length=1, name="ideal_point")(leg_input)
bill_input = Input(shape=(1, ), dtype="int32", name="bill_input")
polarity = Embedding(input_dim=n_bills, output_dim=k_pol, input_length=1, name="polarity")(bill_input)
popularity = Embedding(input_dim=n_bills, output_dim=k_pol, input_length=1, name="popularity")(bill_input)

Q = Embedding(input_dim=n_tokens + 1,
              output_dim=k_emb,
              input_length=max_words,
              weights=[initialize_embedding.values],
              trainable=True,
              )(bill_input)
Q2 = SpatialDropout1D(0.5)(Q)
average_Q = AveragePooling1D(pool_size=max_words)(Q2)
transformed_Q = Dense(k_pol, input_shape=(k_emb, ), activation="selu",)(average_Q)
transformed_Q2 = Dropout(0.25)(transformed_Q)
flat_Q = Flatten()(transformed_Q2)
combined = Dot(axes=1)([flat_P, flat_Q])
main_output = Dense(1, activation="sigmoid", name="main_output",
                    # kernel_regularizer=regularizers.L1L2(0.0, 0.01),
                    )(combined)

model = Model(inputs=[leg_input, bill_input], outputs=[main_output])
model.summary()
