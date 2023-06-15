from __future__ import annotations

from pathlib import Path, PurePath
from typing import Literal, Sequence

import os
import librosa
import numpy as np
import soundfile
import sys
import torch
import json
from pydub import AudioSegment
from cm_time import timer
from tqdm import tqdm

from so_vits_svc_fork.inference.core import RealtimeVC, RealtimeVC2, Svc
from so_vits_svc_fork.utils import get_optimal_device

# f0_method = "dio" #"crepe" "crepe-tiny" "parselmouth" "dio" "harvest"

device = get_optimal_device()

class ModelDefenition:
    model = ""
    config = ""
    cluster = None

    speaker = 0

    transpose = 0
    auto_predict_f0 = False
    f0_method = "dio"

    cluster_infer_ratio = 0
        
    noise_scale = 0.4
    db_thresh = -40

    pad_seconds = 0.5
    chunk_seconds = 0.5
    absolute_thresh = False
    max_chunk_seconds = 40
    def __init__(self, modelDict):
        if modelDict["autodetectModel"]:
            modelDirectory = Path(modelDict["modelDirLocation"])
            modelName =  Path(modelDict["modelName"])
            modelPath = PurePath(modelDirectory, modelName)
            self.model = modelDict["model"] if "model" in modelDict else Path()
            for filename in os.listdir(modelPath):
                if filename.startswith("G_"):
                    self.model = Path(modelPath, filename)

            self.cluster = modelDict["cluster"] if "cluster" in modelDict else None
            for filename in os.listdir(modelPath):
                if filename.startswith("kmeans_"):
                    self.cluster = Path(modelPath, filename)

            self.config = modelDict["config"] if "model" in modelDict else Path(modelPath, "config.json")
        else:
            self.model = Path(modelDict["model"])
            self.cluster = Path(modelDict["cluster"]) if "cluster" in modelDict else None
            self.config = Path(modelDict["config"])
        self.transpose = modelDict["transpose"]
        self.auto_predict_f0 = modelDict["auto_predict_f0"]
        self.f0_method = modelDict["f0_method"]

        self.cluster_infer_ratio = modelDict["cluster_infer_ratio"]

        self.noise_scale = modelDict["noise_scale"]
        self.db_thresh = modelDict["db_thresh"]

        self.pad_seconds = modelDict["pad_seconds"]
        self.chunk_seconds = modelDict["chunk_seconds"]
        self.absolute_thresh = modelDict["absolute_thresh"]
        self.max_chunk_seconds = modelDict["max_chunk_seconds"]

        self.speaker = modelDict["speaker"]



print("loading pony.json")
pony = {}
with open(sys.argv[1] if len(sys.argv)>1 else "./pony.json", "r") as f:
    pony = json.loads(f.read())

if not os.path.isdir(pony["outputFolder"]):
    os.mkdir(pony["outputFolder"])

print("loading character map")
characterMap = {}
with open(pony["characterMap"], "r") as f:
    characterMapInetermidiate = f.read().split("\n")
    for i in range(len(characterMapInetermidiate)):
        lineInfo = characterMapInetermidiate[i].split("\t")
        if len(lineInfo) != 3:
            continue
        if not lineInfo[2] in characterMap:
            characterMap[lineInfo[2]] = []
        characterMap[lineInfo[2]].append([float(lineInfo[0]),float(lineInfo[1])])
    
for name, modelInfo in pony["characterModels"].items():
    if not name in characterMap:
        print("skipping "+name+" as it is not present in map")
        continue
    if (pony["defaultSettings"] | modelInfo)["skip"]:
        print("skipping "+name+" as it is set to skip")
        continue
    print("setting up "+name)
    try:
        modelDefenition = ModelDefenition(pony["defaultSettings"] | modelInfo)
        svc_model = Svc(
            net_g_path=modelDefenition.model.as_posix(),
            config_path=modelDefenition.config.as_posix(),
            cluster_model_path=modelDefenition.cluster.as_posix()
            if modelDefenition.cluster
            else None,
            device=device
        )
    except Exception as e:
        print(e)
        print("failed to create modelDefenition. skipping...")
        continue
    try:
        print("loading audio")
        audioCombined = None
        characterAudioMap = []
        for strip in characterMap[name]:
            try:
                print("loading audio at "+str(strip[1]))
                duration = strip[1]-strip[0]
                if len(characterAudioMap) < 1:
                    characterAudioMap.append([strip[0], strip[1], 0, duration])
                else:
                    characterAudioMap.append([strip[0], strip[1], characterAudioMap[-1][3], characterAudioMap[-1][3]+duration])
                audio, _ = librosa.load(pony["inputAudio"], sr=svc_model.target_sample, offset=strip[0], duration=duration)
                if audioCombined is None:
                    print("Creating a new audio track for "+name)
                    audioCombined = audio
                else:
                    audioCombined = np.append(audioCombined, audio)
            except Exception as e:
                print(f"Failed to load audio")
                print(e)
                continue
        audioCombined = svc_model.infer_silence(
            audioCombined.astype(np.float32),
            speaker=modelDefenition.speaker,
            transpose=modelDefenition.transpose,
            auto_predict_f0=modelDefenition.auto_predict_f0,
            cluster_infer_ratio=modelDefenition.cluster_infer_ratio,
            noise_scale=modelDefenition.noise_scale,
            f0_method=modelDefenition.f0_method,
            db_thresh=modelDefenition.db_thresh,
            pad_seconds=modelDefenition.pad_seconds,
            chunk_seconds=modelDefenition.chunk_seconds,
            absolute_thresh=modelDefenition.absolute_thresh,
            max_chunk_seconds=modelDefenition.max_chunk_seconds,
        )
        # hack: find a way without write to sdisk
        waveName = "./temp/"+name+"-temp.wav"
        soundfile.write(waveName, audioCombined, svc_model.target_sample)
        characterAudio = AudioSegment.from_wav(waveName)
        finalAudio = AudioSegment.empty()
        timeAccumulation = 0
        prevTime = 0
        print ("creating audio")
        finalAudio+=AudioSegment.silent(duration=(characterAudioMap[0][0])*1000-prevTime*1000)
        for i in range(len(characterAudioMap)):
            print ("processing track "+str(i))
            finalAudio+=characterAudio[characterAudioMap[i][2]*1000:characterAudioMap[i][3]*1000]
            finalAudio+=AudioSegment.silent(duration=(characterAudioMap[i+1][0]-characterAudioMap[i][1])*1000) if i+1 < len(characterAudioMap) else AudioSegment.empty()
        finalAudio.export(Path(pony["outputFolder"],name+".wav"), format="wav")

    finally:
        del svc_model
        torch.cuda.empty_cache()
