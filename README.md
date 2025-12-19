# Analyse-CV - MCP Server

MCP (Model Context Protocol) server exposant le workflow "Analyse-CV" comme outil réutilisable.

## Description

Analyse et compare automatiquement 5 CV par rapport à une fiche de poste, en évaluant chaque candidat selon des critères pondérés et génère un rapport professionnel avec scores et recommandations.

Ce serveur MCP permet d'utiliser ce workflow directement depuis:
- **Claude Desktop** - Assistant IA avec accès aux outils MCP
- **LightOn Paradigm** - Plateforme IA (support MCP à venir)
- **Tout client compatible MCP** - Via le protocole standardisé

## Prérequis

- Python 3.10 ou supérieur
- Clé API LightOn Paradigm (https://paradigm.lighton.ai)
- Accès aux documents dans votre workspace Paradigm

## Installation

### 1. Installer les dépendances communes (une seule fois)

**IMPORTANT:** N'installez les dépendances qu'une seule fois, pas pour chaque workflow !

```bash
pip install mcp aiohttp python-dotenv
```

⚠️ **NE FAITES PAS** `pip install -e .` dans ce dossier ! Cela créerait des conflits si vous avez plusieurs workflows MCP.

### 2. Configurer les variables d'environnement

Créez un fichier `.env` à la racine du package:

```bash
PARADIGM_API_KEY=votre_clé_api_ici
PARADIGM_BASE_URL=https://paradigm.lighton.ai
# Optionnel : Bearer token pour sécuriser le serveur HTTP MCP (Paradigm)
MCP_BEARER_TOKEN=votre_token_secret
```

### 3. Configuration pour LightOn Paradigm (Recommandé)

Pour utiliser ce workflow dans LightOn Paradigm via MCP :

**Étape 1 : Démarrer le serveur HTTP**

```bash
python -m http_server --port 8080
```

Le bearer token sera lu depuis le fichier `.env` (variable `MCP_BEARER_TOKEN`). Si aucun token n'est configuré, le serveur démarrera en mode développement sans authentification.

**Étape 2 : Enregistrer le serveur dans Paradigm**

En tant qu'administrateur système dans Paradigm :
1. Allez dans **Admin > MCP Servers**
2. Ajoutez un nouveau serveur :
   - **Name**: `analyse-cv`
   - **URL**: `http://votre-serveur:8080` (ou l'URL où le serveur est accessible)
   - **Bearer Token**: La valeur de `MCP_BEARER_TOKEN` (si configuré)
3. Activez le serveur dans **Chat Settings > Agent Tools**

Le workflow sera alors disponible comme outil dans vos conversations Paradigm avec le mode Agent activé.

### 4. Configuration pour Claude Desktop (Optionnel)

Ajoutez cette configuration dans le fichier de configuration de Claude Desktop:

**macOS/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:**
```powershell
# Ouvrir directement le fichier avec l'explorateur
start "" "%APPDATA%\Claude"
# Puis ouvrir claude_desktop_config.json avec un éditeur de texte
```

Ou via l'explorateur Windows :
1. Tapez `%APPDATA%\Claude` dans la barre d'adresse de l'explorateur
2. Ouvrez le fichier `claude_desktop_config.json` avec un éditeur de texte

```json
{{
  "mcpServers": {{
    "analyse-cv": {{
      "command": "python",
      "args": ["-m", "server"],
      "cwd": "/chemin/absolu/vers/ce/dossier"
    }}
  }}
}}
```

**Notes:**
- Le chemin `cwd` doit pointer vers le dossier où se trouve ce package
- La commande `command` doit pointer vers votre exécutable Python 3.10+ (sous Windows, utilisez le chemin complet comme `C:\\Users\\VotreNom\\AppData\\Local\\Programs\\Python\\Python310\\python.exe`)
- Le fichier `.env` dans le dossier `cwd` contient déjà les variables d'environnement (pas besoin de les mettre dans `env`)

### 5. Redémarrer Claude Desktop

Si vous utilisez Claude Desktop, fermez-le complètement et relancez-le. Le nouvel outil MCP sera disponible.

## Utilisation

### Dans Claude Desktop

Une fois configuré, vous pouvez utiliser le workflow directement dans vos conversations:

Exemple de demande dans Claude Desktop:

```
Utilise le workflow Analyse-CV avec les parametres suivants: file_paths: ["fichier1.pdf", "fichier2.pdf"], query: "votre texte ici"
```

Claude utilisera automatiquement l'outil MCP pour executer le workflow.

### Test en ligne de commande

Pour tester le serveur MCP en ligne de commande:

```bash
python -m server
```

Le serveur démarre et attend les commandes MCP via stdin/stdout.

## Structure du projet

```
analyse-cv/
├── server.py              # Serveur MCP stdio (Claude Desktop)
├── http_server.py         # Serveur MCP HTTP (Paradigm)
├── workflow.py            # Logique du workflow
├── paradigm_client.py     # Client API Paradigm
├── pyproject.toml         # Configuration Python
├── README.md              # Ce fichier
└── .env                   # Variables d'environnement (à créer)
```

## Paramètres du workflow

- `file_paths` (files): **Obligatoire** - Chemins complets des fichiers locaux a analyser (ex: C:\\Documents\\cv.pdf)
- `query` (text): *Optionnel* - Question ou demande d'analyse (optionnel)

## Format de sortie

JSON object containing the workflow results, including any analysis, extracted data, or generated content.

## Dépannage

### Le serveur ne démarre pas

- Vérifiez que Python 3.10+ est installé: `python --version`
- Vérifiez que les dépendances sont installées: `pip install -e .`
- Vérifiez les logs dans Claude Desktop (menu Developer > Show Logs)

### Erreur d'authentification Paradigm

- Vérifiez que votre clé API est correcte dans `.env` ou `claude_desktop_config.json`
- Testez la clé avec: `curl -H "Authorization: Bearer VOTRE_CLE" https://paradigm.lighton.ai/api/v2/health`

### Le workflow échoue

- Vérifiez que les documents nécessaires sont bien dans votre workspace Paradigm
- Consultez les logs pour voir les détails de l'erreur
- Vérifiez que les paramètres d'entrée respectent le schéma attendu

## Support

Pour toute question ou problème:
- Documentation MCP: https://modelcontextprotocol.io
- Documentation Paradigm: https://docs.lighton.ai
- Support LightOn: support@lighton.ai

## Licence

Généré par LightOn Workflow Builder
