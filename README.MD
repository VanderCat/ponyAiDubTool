# Pony AI Dub Tool
based and depends on so-vits-svc-fork

made primarily to make an ai redubbing of russian MLP localization (make russian actors sound like original counterplates)

## installation
```
conda create -n so-vits-svc-fork python=3.10 pip
conda activate so-vits-svc-fork
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U so-vits-svc-fork pydub
```
or you can use exsisting so-vits-svc-fork installation environment and only install pydub
```
pip install pydub
```

## Usage
1. Get a clean voice track to map characters at.
   You can use Ultimate Vocal Remover for that (great tool btw)
2. Load it in Audacity. Start selecting voicelines and mark them 
   by pressing <kbd>Ctrl</kbd>+<kbd>B</kbd> and naming them whatever
   you want (labels can overlap)

   e.g. Celestia's line labeled as `Sunbutt`

   ![Audacity](https://files.catbox.moe/7cevcz.png)
3. When you are done export labels as `Files` > `Export` > `Export Labels`
4. make a copy of ponyExample.json as pony.json and start modifying to your needs
   
   `"settingsDefault"` - will be applied to each character model defenition in `"characterModels"` list

   if your model has a custom checkpoints name autodetect will fail. to avoid this you will need to disable it using `"autodetectModel":false` and providing paths to model, config and cluster (if it exists)  
   e.g.
   ```json
    "characterModels": {
        "Moonbutt": {
            "autodetectModel": false,
            "model": "D://AI//Models//Luna.pth",
            "config": "D://AI/Models//Luna.json",
        }
    }
    ```
5. Activate the environment and run
   ```
   python ponyairedb.py
   ```
   wait for magic to happen
6. you will get a separate syncronised voice tracks for each character that you will need to combine yourself. (if it is not obvious you can add effects like reverb and etc.)

## Note:
if there is no character in character map or there is no characterModel for character the audio track will not render. 

If you do not want to render a track (e.g. for a testing purpouse) you can add a `"skip":"true"` to `"characterModels"`. To solo 1 track you can flip `"skip"` in the default settings and set `"skip"` to false in `"characterModels"`