
# imports for the data extraction from the midi files
import os
import time
import pretty_midi
import numpy as np
import pandas as pd
import polars as pl
from mido import KeySignatureError
from concurrent.futures import ProcessPoolExecutor, as_completed

# midi file path and the composers in the file path
main_dir = 'midiclassics'
composers = ['Chopin', 'Beethoven', 'Bach', 'Mozart']


#####################################################################




# function to get chroma features
def get_chroma(midi_data, fs=100):
    chroma = midi_data.get_chroma(fs=fs)
    return chroma



#####################################################################




# function to calculate notes density
def calculate_notes_density(midi_data, max_sequence_length=300):
    total_notes = sum(len(instr.notes) for instr in midi_data.instruments)
    duration = midi_data.get_end_time()
    density = total_notes / duration
    return density



#####################################################################






# function to extract features from the midi files, will need to change when adding more features to model
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


###############################################################################


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


#####################################################################



if __name__ == "__main__":
    data = []
    labels = []

    # process each composer sequentially
    for composer in composers:
        composer_data, composer_labels = process_composer(composer)
        data.extend(composer_data)
        labels.extend(composer_labels)

    print("Processing complete.")
    
#####################################################################


# converting the data to a pandas dataframe and the labels to a numpy array
data_pl = pl.DataFrame(data)
data_df = data_pl.to_pandas()
labels = np.array(labels)


#####################################################################

data_df.head()


#####################################################################



labels




