# Audio2Audio Translate Next

Traduction vocale auto-hébergée : microphone ou fichier audio → transcription → traduction → voix de synthèse.

Le service associe **Nemotron 3.5 ASR**, **MarianMT** et **OmniVoice**, avec une interface web et des API REST et WebSocket. Les modèles s’exécutent sur le serveur ; aucun service cloud d’inférence n’est appelé par le pipeline.

## Fonctionnement

| Étape | Implémentation |
| --- | --- |
| Reconnaissance vocale | `nvidia/nemotron-3.5-asr-streaming-0.6b`, via NeMo |
| Traduction | Modèles `Helsinki-NLP/opus-mt-*` via MarianMT ; paire directe ou passage par l’anglais |
| Synthèse vocale | `k2-fsa/OmniVoice`, sortie WAV mono à 24 kHz |
| Serveur | FastAPI / Uvicorn, port `8741` |
| Interface | HTML / JavaScript, capture du microphone via AudioWorklet |

En direct, le serveur conserve le contexte de l’encodeur ASR pendant chaque énoncé et envoie des transcriptions partielles. La traduction et la synthèse sont déclenchées après environ **0,6 seconde de silence**, ou lorsque l’énoncé atteint **10 secondes**. Les réponses finales sont traitées dans l’ordre, avec un cache vocal par langue pour stabiliser la voix entre les phrases.

La latence totale comprend donc la détection de fin de parole, la traduction et la synthèse ; elle ne correspond pas à la seule latence de l’ASR.

## Installation

### Prérequis

- Linux avec Python **3.12**, `venv` et Git.
- Un GPU NVIDIA avec un pilote compatible avec les roues CUDA choisies est recommandé. Le code prévoit un repli CPU, sans garantie de performances en temps réel.
- Un accès aux dépôts de paquets et à Hugging Face pour télécharger les dépendances et les modèles. Prévoir de l’espace disque et de la mémoire pour l’ASR, le TTS et plusieurs modèles MarianMT.
- Un navigateur prenant en charge `getUserMedia`, Web Audio et AudioWorklet.

Les versions de référence sont consignées dans [backend/requirements.txt](backend/requirements.txt) et [AGENTS.md](AGENTS.md) : Python 3.12, PyTorch 2.14.1 avec CUDA 13.0, torchaudio 2.11.0, NeMo 3.0.0, Transformers 5.19.0 et OmniVoice 0.2.1. Leur disponibilité dépend de la plateforme et de l’index de paquets utilisé.

### Préparer l’environnement

`run.sh` utilise un environnement virtuel à un emplacement fixe :

```bash
git clone https://github.com/ynotopec/audio2audio-translate-next.git
cd audio2audio-translate-next

python3.12 -m venv "$HOME/venv/audio2audio-translate-next-py312"
source "$HOME/venv/audio2audio-translate-next-py312/bin/activate"
python -m pip install --upgrade pip

# Installer PyTorch et torchaudio avant les autres dépendances.
# Commande de référence du dépôt pour CUDA 13.0 :
python -m pip install torch==2.14.1 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip install -r backend/requirements.txt

# Dépendances utilisées par les formulaires FastAPI et les modèles MarianMT,
# qui ne sont pas explicitement listées dans requirements.txt :
python -m pip install python-multipart sentencepiece sacremoses accelerate

python -m pip check
```

Si les roues PyTorch ne sont pas disponibles pour votre architecture, adapter leur installation à la plateforme avant d’installer les autres dépendances. Ces commandes reprennent la configuration du dépôt ; elles ne constituent pas une validation sur toutes les architectures.

### Démarrer

Depuis la racine du dépôt :

```bash
./run.sh
```

Le serveur écoute sur `0.0.0.0:8741`. Au démarrage, il charge l’ASR, six modèles de traduction pour les échanges anglais ↔ français / allemand / espagnol, puis OmniVoice. Le premier démarrage peut prendre du temps en raison des téléchargements ; les autres paires de traduction sont chargées à la demande.

- Interface : [http://localhost:8741/](http://localhost:8741/)
- Documentation REST : [http://localhost:8741/docs](http://localhost:8741/docs)
- État : [http://localhost:8741/health](http://localhost:8741/health)

```bash
curl --fail http://localhost:8741/health
```

Pour choisir un autre port, lancer Uvicorn depuis la racine avec le même environnement :

```bash
"$HOME/venv/audio2audio-translate-next-py312/bin/python3" -m uvicorn \
  backend.main:app --host 127.0.0.1 --port 8742
```

`server.py` est une entrée alternative utilisant l’interpréteur Python courant ; `run.sh` reste l’entrée recommandée.

## Utilisation dans le navigateur

1. Ouvrir l’interface et choisir les langues source et cible.
2. Cliquer sur le microphone et autoriser la capture.
3. Parler, puis laisser une courte pause pour obtenir la traduction audio.
4. Cliquer de nouveau pour arrêter.

Pour accéder au microphone depuis une autre machine, servir l’interface en **HTTPS** et transmettre les connexions WebSocket via **WSS**. L’accès local via `localhost` peut utiliser HTTP.

La capture demande l’annulation d’écho du navigateur ; la réduction de bruit et le gain automatique sont désactivés. La case **anti-écho**, activée par défaut, écarte en plus les échantillons du microphone pendant la lecture TTS et une courte période après celle-ci. La parole prononcée pendant cette fenêtre peut donc être ignorée ; un casque permet de désactiver cette option plus facilement.

Les langues sont envoyées à l’ouverture de la connexion : arrêter puis relancer le microphone après un changement de sélection.

## API REST : traduire un fichier

`POST /api/translate` attend un formulaire `multipart/form-data` :

| Champ | Type | Défaut | Description |
| --- | --- | --- | --- |
| `file` | fichier, obligatoire | — | Audio lisible par SoundFile ; WAV recommandé |
| `src_lang` | texte | `auto` | Locale source, par exemple `en-US` ou `fr-FR` |
| `tgt_lang` | texte | `fr-FR` | Locale cible |

**Les langues doivent être envoyées comme champs de formulaire, pas comme paramètres d’URL.**

```bash
curl --fail-with-body --show-error \
  http://localhost:8741/api/translate \
  -F "file=@input.wav" \
  -F "src_lang=en-US" \
  -F "tgt_lang=fr-FR" \
  --output translated.wav
```

L’entrée est convertie en mono et rééchantillonnée à 16 kHz si nécessaire. La réponse réussie est un fichier `audio/wav` PCM 16 bits à 24 kHz. La limite par défaut est de **50 Mio** ; un dépassement renvoie HTTP `413`.

Avec `src_lang=auto`, le serveur utilise la langue indiquée par la transcription ASR, avec un repli sur `en-US` si cette indication est absente. Une langue explicite sert à choisir la paire de traduction ; elle ne force pas la langue du décodeur ASR dans l’implémentation actuelle.

## API WebSocket : audio en direct

Ouvrir `/ws/translate`, puis envoyer un message JSON de configuration :

```json
{
  "src_lang": "auto",
  "tgt_lang": "fr-FR",
  "sample_rate": 16000
}
```

Envoyer ensuite des messages binaires contenant du **PCM signé 16 bits, mono, little-endian**, sans en-tête WAV. `sample_rate` décrit la fréquence réelle de la capture ; le serveur rééchantillonne vers 16 kHz.

| Message reçu | Contenu |
| --- | --- |
| JSON `{"type":"transcript","text":"…"}` | Transcription partielle ou finale |
| JSON `{"type":"translated","text":"…"}` | Traduction de l’énoncé final |
| Binaire | Fichier WAV complet de la voix synthétisée |
| JSON `{"type":"error","detail":"…"}` | Erreur signalée, par exemple TTS indisponible |

Pour terminer, envoyer `{"command":"stop"}` et laisser la connexion ouverte pour recevoir les dernières réponses. Le protocole ne distingue pas les transcriptions partielles et finales par un champ dédié.

## Configuration

Exporter les variables avant le lancement, ou les préfixer à la commande. **Aucun fichier `.env` n’est chargé automatiquement.**

| Variable | Défaut | Effet |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` via `run.sh` | GPU visible par le processus |
| `TTS_BACKEND` | `omnivoice` | Seul moteur TTS accepté actuellement |
| `TTS_VOICE` | `female, moderate pitch` | Mots-clés anglais OmniVoice pour créer la voix de référence |
| `TTS_VOICE_SEED` | `1234` | Graine de création des références vocales |
| `MAX_UPLOAD_MB` | `50` | Taille maximale des fichiers reçus, en Mio |

Exemple :

```bash
TTS_VOICE="male, moderate pitch" TTS_VOICE_SEED=42 MAX_UPLOAD_MB=25 ./run.sh
```

`TTS_VOICE` accepte le vocabulaire strict d’OmniVoice, pas une instruction libre. Le fichier [backend/tts_engine_qwen.py](backend/tts_engine_qwen.py) est conservé comme ancien moteur : son besoin de Transformers 4.57.3 est incompatible avec l’environnement OmniVoice, et `TTS_BACKEND=qwen` empêche le démarrage.

## Langues et limites actuelles

- L’interface expose six langues : anglais, français, allemand, espagnol, chinois et japonais, ainsi que la détection automatique en source.
- Le serveur déclare 33 locales dans [backend/translation.py](backend/translation.py), consultables via `/health`. **Cette liste ne garantit pas une traduction pour toutes les paires.** Il faut une paire MarianMT directe ou les deux modèles nécessaires au passage par l’anglais.
- Lorsqu’une paire est indisponible, le moteur peut renvoyer le texte source inchangé, puis le transmettre au TTS. Vérifier les journaux et les paires configurées avant de retenir une combinaison.
- La couverture annoncée des modèles ASR et TTS ne représente pas la couverture de bout en bout de l’application.
- Le sens de traduction reste celui de la configuration source → cible. La détection automatique ne met pas en place un routage bidirectionnel vers une langue principale.
- Le sélecteur de latence de l’interface envoie `latency_ms`, mais le serveur ne l’applique pas actuellement.
- Le sous-titre de l’interface mentionne encore Qwen3-TTS ; le moteur effectivement utilisé est **OmniVoice**.
- Les textes sont tronqués à 512 tokens à l’entrée de chaque modèle MarianMT. Les longs fichiers ne sont pas découpés en énoncés par l’API REST.
- Le service ne fournit pas d’authentification et autorise toutes les origines CORS. Pour une exposition réseau, placer le service derrière un reverse proxy avec TLS et contrôle d’accès.

## Dépannage

| Symptôme | Vérification |
| --- | --- |
| `No module named nemo` | Installer `backend/requirements.txt` dans le venv utilisé par `run.sh` |
| Erreur liée aux formulaires FastAPI | Installer `python-multipart` dans ce même venv |
| Erreur de tokenizer MarianMT ou de `device_map` | Vérifier `sentencepiece`, `sacremoses` et `accelerate` |
| Pas d’accès au microphone à distance | Utiliser HTTPS, autoriser le microphone et vérifier le support AudioWorklet |
| Échec WebSocket derrière un proxy | Transmettre `/ws/translate` avec les en-têtes de mise à niveau WebSocket |
| Texte inchangé après traduction | Vérifier la disponibilité des modèles de la paire ou du pivot anglais |
| Mémoire GPU insuffisante | Libérer de la mémoire ; les chargements ASR et TTS réessaient jusqu’à six fois en cas de manque de mémoire |
| Erreur avec `TTS_BACKEND=qwen` | Supprimer la variable ou choisir `omnivoice` |

## Sources et licence

- [Serveur et API](backend/main.py)
- [ASR en streaming](backend/streaming_asr.py)
- [Paires de traduction](backend/translation.py)
- [Synthèse et voix de référence](backend/tts_engine.py)
- [Interface web](static/index.html)

Projet distribué sous [licence MIT](LICENSE).
