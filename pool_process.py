# working with data imports 
import os
import time
import pretty_midi
import numpy as np
import polars as pl
import pandas as pd
from mido import KeySignatureError
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns


# deep learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report




# midi file path and the composers in the file path
main_dir = 'midiclassics'
composers = ['Chopin', 'Beethoven', 'Bach', 'Mozart']




# function to get chroma features
def get_chroma(midi_data, fs=100):
    chroma = midi_data.get_chroma(fs=fs)
    return chroma



# function to calculate notes density
def calculate_notes_density(midi_data, max_sequence_length=300):
    total_notes = sum(len(instr.notes) for instr in midi_data.instruments)
    duration = midi_data.get_end_time()
    density = total_notes / duration
    return density




def process_composer(composer):
    composer_folder = os.path.join(main_dir, composer)
    composer_data = []
    composer_labels = []
    for file in os.listdir(composer_folder):
        if file.endswith('.midi') or file.endswith('.mid') or file.endswith('.MID'):
            file_path = os.path.join(composer_folder, file)
            start_time = time.time()
            try:
                features, filename = extract_features(file_path)
                end_time = time.time()
                if features is not None:
                    processing_time = end_time - start_time
                    print(f"Processed {filename} by {composer} in {processing_time:.2f} seconds")
                    composer_data.append(features)
                    composer_labels.append(composer)
                else:
                    print(f"Skipping file {file_path} by {composer} due to extraction error.")
            except Exception as e:
                print(f"Failed to process {file_path} by {composer}: {e}")
    return composer_data, composer_labels




def extract_features(midi_file, max_sequence_length=300):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except KeySignatureError as e:
        print(f"Failed to process {midi_file} due to key signature error: {e}")
        return None, None
    except Exception as e:
        print(f"Failed to process {midi_file} due to error: {e}")
        return None, None

    features = {}

    # tempo
    features['tempo'] = midi_data.estimate_tempo()

    # handling key signatures errors that happen with some midi files 
    try:
        features['key_signatures'] = [key.key_number for key in midi_data.key_signature_changes]
    except KeySignatureError as e:
        print(f"Key signature error in file {midi_file}: {e}")
        features['key_signatures'] = [0]  # Default to C major/A minor if error occurs
    except Exception as e:
        print(f"General error in file {midi_file}: {e}")
        features['key_signatures'] = [0]

    # getting the time signatures
    features['time_signatures'] = [(time.numerator, time.denominator) for time in midi_data.time_signature_changes]

    # instrument types
    features['instrument_types'] = [instr.program for instr in midi_data.instruments]

    # notes hist 
    histogram = np.zeros(12)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitch_class = note.pitch % 12
            histogram[pitch_class] += 1
    if np.sum(histogram) > 0:
        histogram /= np.sum(histogram)
    features['notes_histogram'] = histogram

    # notes matrix
    notes = np.zeros((max_sequence_length, 128))
    end_time = midi_data.get_end_time()
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = int(note.start * max_sequence_length / end_time)
            end = int(note.end * max_sequence_length / end_time)
            notes[start:end, note.pitch] = note.velocity / 127
    features['notes_matrix'] = notes

    # Chroma features
    features['chroma'] = get_chroma(midi_data)

    # Notes density
    features['notes_density'] = calculate_notes_density(midi_data)

    return features, os.path.basename(midi_file)





if __name__ == "__main__":
    data = []
    labels = []

    # process each composer sequentially
    for composer in composers:
        composer_data, composer_labels = process_composer(composer)
        data.extend(composer_data)
        labels.extend(composer_labels)

    print("Processing complete!")






# Preprocess features
def preprocess_features(df):
    # Flatten complex features
    df['key_signatures'] = df['key_signatures'].apply(lambda x: x[:1] if len(x) > 0 else [0])  # Use first key signature
    df['time_signatures'] = df['time_signatures'].apply(lambda x: x[0] if len(x) > 0 else (4, 4))  # Use first time signature
    df['instrument_types'] = df['instrument_types'].apply(lambda x: x[:1] if len(x) > 0 else [0])  # Use first instrument type
    df['notes_histogram'] = df['notes_histogram'].apply(lambda x: x[:12] if len(x) >= 12 else np.zeros(12))  # Ensure length 12
    
    # Concatenate all features into a single array
    feature_array = np.hstack([
        df['tempo'].values.reshape(-1, 1),
        np.vstack(df['key_signatures']),
        np.vstack(df['time_signatures']),
        np.vstack(df['instrument_types']),
        np.vstack(df['notes_histogram']),
        np.vstack(df['notes_matrix'].apply(lambda x: x.flatten()))  # Flatten the notes matrix
    ])
    
    return feature_array





data_pl = pl.DataFrame(data)
data_df = data_pl.to_pandas()
labels = np.array(labels)




# prep the data 
X = preprocess_features(data_df)




# encode the labels for the data 
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)



# splitting the data using sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# normalizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



### if you made it this far you get a 🍆