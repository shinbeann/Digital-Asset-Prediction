
### 📂 `data/`
Contains all data-related files for the project.
- **`raw/`**: Contains unprocessed datasets (e.g., raw podcast audio files, market price data, etc.).
  - **`podcasts/`**: Raw audio files of podcasts.
  - **`podcast_transcripts/`**: Raw, unprocessed transcripts extracted from the podcasts.
- **`processed/`**: Contains cleaned and preprocessed data.

### 📂 `notebooks/`
Contains Jupyter notebooks for different stages of the project.


### 📂 `src/`
- `data_collection/`: 
    - **`fetch_podcast_transcripts.py`**: Script to convert podcast audio files to text
    - **`web_scraper.py`**: Scrapes data from external sources

- `preprocessing/`
    - **`clean_transcripts.py`**: NLP processing of podcast transcripts, including text cleaning and normalization


### 📂 `utils/`
Contains utility functions and configurations
- **`config.py`**: Stores configuration settings like API keys



### others
 [tps://www.assemblyai.com/app](https://www.assemblyai.com/app) - check credits used