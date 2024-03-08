import evaluate
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize, iters, results_save_path, learning_rate_base, warmup_learning_rate, warmup_steps, hold_base_rate_steps,n_class):
    
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch+1))
        # lr = cosine_decay_with_warmup(global_step = epoch,
        #                      learning_rate_base = learning_rate_base,
        #                      total_steps = epochs,
        #                      warmup_learning_rate = warmup_learning_rate,
        #                      warmup_steps=warmup_steps,
        #                      hold_base_rate_steps=hold_base_rate_steps)
        # print(lr)
        # model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1)
        # evaluate.evaluateModel(model,X_test, Y_test, batchSize, iters, results_save_path)
        evaluate.evaluateMultiClass(model,X_test, Y_test, batchSize, iters, results_save_path, n_class, epoch)

    return model